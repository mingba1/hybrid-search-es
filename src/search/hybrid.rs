use std::sync::Arc;
use std::time::Instant;

use serde_json::Value;

use super::decide::{allows_bm25_fallback, decide, RetrievalChoice};
use super::dsl::{
    bm25_body, compile_filters, knn_body, native_retriever, native_sub_searches,
    weighted_function_score_body, weighted_knn_body, LexicalSpec, RrfParams, VectorSpec,
};
use super::rrf::{es_hits, es_took_ms, leg_to_hit, reciprocal_rank_fusion};
use crate::config::{RrfMode, SearchSettings, WeightedSyntax};
use crate::embedding::{query_input, EmbeddingService};
use crate::error::AppError;
use crate::es::EsClient;
use crate::model::{FusionMode, Health, RrfSyntax, SearchHit, SearchRequest, SearchResponse};

#[derive(Debug, Clone)]
pub struct SearchPlan {
    pub fusion: FusionMode,
    pub rrf_syntax: Option<RrfSyntax>,
    pub degraded: bool,
    pub degrade_reason: Option<String>,
    pub embed_ms: Option<u128>,
    pub limit: i64,
    pub request: Value,
    pub knn_request: Option<Value>,
}

pub struct HybridSearchService {
    es: EsClient,
    embedding: Arc<dyn EmbeddingService>,
    settings: SearchSettings,
}

impl HybridSearchService {
    pub fn new(
        es: EsClient,
        embedding: Arc<dyn EmbeddingService>,
        settings: SearchSettings,
    ) -> Self {
        Self {
            es,
            embedding,
            settings,
        }
    }

    pub fn index_name(&self) -> &str {
        self.es.index_name()
    }

    pub async fn health(&self) -> Result<Health, AppError> {
        let version = self.es.version().await?;
        let vector_search = version.supports_knn();
        Ok(Health {
            status: "ok".into(),
            es_version: version.raw,
            index: self.es.index_name().to_string(),
            embedding_model: self.embedding.model_id().to_string(),
            embedding_dims: self.embedding.dims(),
            rrf_mode: self.settings.rrf_mode.as_str().to_string(),
            vector_search,
        })
    }

    #[tracing::instrument(skip(self, request), fields(query_len = request.query.chars().count()))]
    pub async fn search(&self, request: SearchRequest) -> Result<SearchResponse, AppError> {
        let started = Instant::now();
        let plan = self.plan(&request).await?;
        let executed = self.execute(&plan).await?;
        if plan.degraded {
            tracing::warn!(
                reason = plan.degrade_reason.as_deref().unwrap_or(""),
                fusion = ?executed.fusion,
                "hybrid search degraded"
            );
        } else {
            tracing::info!(
                fusion = ?executed.fusion,
                hits = executed.hits.len(),
                took_ms = started.elapsed().as_millis() as u64,
                "hybrid search completed"
            );
        }
        Ok(SearchResponse {
            hits: executed.hits,
            took_ms: started.elapsed().as_millis(),
            embed_ms: plan.embed_ms,
            es_took_ms: executed.es_took_ms,
            fusion: executed.fusion,
            rrf_syntax: executed.rrf_syntax,
            degraded: executed.degraded,
            degrade_reason: executed.degrade_reason,
        })
    }

    pub async fn plan(&self, request: &SearchRequest) -> Result<SearchPlan, AppError> {
        let query = request.query.trim();
        if query.is_empty() {
            return Err(AppError::BadRequest("query 不能为空".into()));
        }
        if query.chars().count() > 2000 {
            return Err(AppError::BadRequest("query 超过 2000 字".into()));
        }
        let top_n = resolve_top_n(request.top_n, self.settings.default_top_n)?;
        let filters = compile_filters(
            request.doc_type.as_deref(),
            request.created_from,
            request.created_to,
            &request.terms,
            &self.settings.keyword_fields,
        )?;
        let (window, knn_k, num_candidates) = self.recall_shape(top_n);
        let version = if needs_cluster_version(self.settings.rrf_mode, request.fusion) {
            Some(self.es.version().await?)
        } else {
            None
        };
        let choice = decide(self.settings.rrf_mode, request.fusion, version.as_ref())?;
        let lexical = LexicalSpec {
            text: query.to_string(),
            filters: filters.clone(),
            title_boost: self.settings.title_boost,
            fuzziness: self.settings.fuzziness.clone(),
            boost: None,
        };

        if let RetrievalChoice::Bm25 { degraded } = &choice {
            return Ok(SearchPlan {
                fusion: FusionMode::Bm25,
                rrf_syntax: None,
                degraded: degraded.is_some(),
                degrade_reason: degraded.clone(),
                embed_ms: None,
                limit: top_n,
                request: bm25_body(&lexical, top_n),
                knn_request: None,
            });
        }

        let embed = self.embed_query(query).await;
        let fallback = allows_bm25_fallback(request.fusion);
        let vector = match embed.result {
            Ok(vector) => vector,
            Err(err) if fallback && is_degradable(&err) => {
                return Ok(SearchPlan {
                    fusion: FusionMode::Bm25,
                    rrf_syntax: None,
                    degraded: true,
                    degrade_reason: Some(degrade_reason(&err)),
                    embed_ms: Some(embed.elapsed_ms),
                    limit: top_n,
                    request: bm25_body(&lexical, top_n),
                    knn_request: None,
                });
            }
            Err(err) => return Err(err),
        };
        let vector_spec = VectorSpec {
            vector,
            filters,
            k: knn_k,
            num_candidates,
            boost: None,
        };
        let rrf = RrfParams {
            rank_constant: self.settings.rank_constant,
            window_size: window,
            size: top_n,
        };

        let plan = match choice {
            RetrievalChoice::Bm25 { .. } => unreachable!("BM25 已在向量化之前返回"),
            RetrievalChoice::Knn => {
                let mut spec = vector_spec.clone();
                if spec.k < top_n {
                    spec.k = top_n;
                }
                if spec.num_candidates < spec.k {
                    spec.num_candidates = spec.k;
                }
                SearchPlan {
                    fusion: FusionMode::Knn,
                    rrf_syntax: None,
                    degraded: false,
                    degrade_reason: None,
                    embed_ms: Some(embed.elapsed_ms),
                    limit: top_n,
                    request: knn_body(&spec, top_n),
                    knn_request: None,
                }
            }
            RetrievalChoice::Weighted => SearchPlan {
                fusion: FusionMode::Weighted,
                rrf_syntax: None,
                degraded: false,
                degrade_reason: None,
                embed_ms: Some(embed.elapsed_ms),
                limit: top_n,
                request: self.weighted_body(&lexical, &vector_spec, top_n),
                knn_request: None,
            },
            RetrievalChoice::Native { syntax } => SearchPlan {
                fusion: FusionMode::NativeRrf,
                rrf_syntax: Some(syntax),
                degraded: false,
                degrade_reason: None,
                embed_ms: Some(embed.elapsed_ms),
                limit: top_n,
                request: match syntax {
                    RrfSyntax::Retriever => native_retriever(&lexical, &vector_spec, &rrf),
                    RrfSyntax::SubSearches => native_sub_searches(&lexical, &vector_spec, &rrf),
                    RrfSyntax::Application => {
                        return Err(AppError::Internal("原生 RRF 收到了应用层语法".into()))
                    }
                },
                knn_request: None,
            },
            RetrievalChoice::App => SearchPlan {
                fusion: FusionMode::AppRrf,
                rrf_syntax: Some(RrfSyntax::Application),
                degraded: false,
                degrade_reason: None,
                embed_ms: Some(embed.elapsed_ms),
                limit: top_n,
                request: bm25_body(&lexical, window),
                knn_request: Some(knn_body(&vector_spec, knn_k)),
            },
        };
        Ok(plan)
    }

    async fn execute(&self, plan: &SearchPlan) -> Result<Executed, AppError> {
        let mut executed = if plan.fusion == FusionMode::AppRrf {
            if let Some(knn_request) = &plan.knn_request {
                self.execute_app(&plan.request, knn_request).await?
            } else {
                self.execute_single(plan).await?
            }
        } else {
            self.execute_single(plan).await?
        };
        let limit = plan.limit.max(0) as usize;
        executed.hits.truncate(limit);
        Ok(executed)
    }

    async fn execute_single(&self, plan: &SearchPlan) -> Result<Executed, AppError> {
        let body = self.es.search(&plan.request).await?;
        let hits = es_hits(&body)?
            .into_iter()
            .map(|hit| leg_to_hit(&hit))
            .collect();
        Ok(Executed {
            hits,
            es_took_ms: es_took_ms(&body),
            fusion: plan.fusion,
            rrf_syntax: plan.rrf_syntax,
            degraded: plan.degraded,
            degrade_reason: plan.degrade_reason.clone(),
        })
    }

    async fn execute_app(&self, bm25_body: &Value, knn_body: &Value) -> Result<Executed, AppError> {
        let (bm25_result, knn_result) =
            tokio::join!(self.es.search(bm25_body), self.es.search(knn_body));
        match (bm25_result, knn_result) {
            (Ok(bm25), Ok(knn)) => {
                let bm25_hits = es_hits(&bm25)?;
                let knn_hits = es_hits(&knn)?;
                tracing::info!(
                    bm25_hits = bm25_hits.len(),
                    knn_hits = knn_hits.len(),
                    "app rrf legs"
                );
                let window = bm25_body.get("size").and_then(Value::as_u64).unwrap_or(0) as usize;
                let hits = reciprocal_rank_fusion(
                    &bm25_hits,
                    &knn_hits,
                    self.settings.rank_constant as f64,
                    window,
                );
                let es_took_ms = match (es_took_ms(&bm25), es_took_ms(&knn)) {
                    (Some(left), Some(right)) => Some(left.max(right)),
                    (Some(value), None) | (None, Some(value)) => Some(value),
                    (None, None) => None,
                };
                Ok(Executed {
                    hits,
                    es_took_ms,
                    fusion: FusionMode::AppRrf,
                    rrf_syntax: Some(RrfSyntax::Application),
                    degraded: false,
                    degrade_reason: None,
                })
            }
            (Ok(bm25), Err(err)) => {
                tracing::warn!(error = %err, "knn leg failed, returning bm25");
                Ok(self.single_leg(bm25, FusionMode::Bm25, &err)?)
            }
            (Err(err), Ok(knn)) => {
                tracing::warn!(error = %err, "bm25 leg failed, returning knn");
                Ok(self.single_leg(knn, FusionMode::Knn, &err)?)
            }
            (Err(left), Err(right)) => Err(AppError::Elasticsearch {
                status: 502,
                body: format!("两路召回都失败: {left}; {right}"),
            }),
        }
    }

    fn single_leg(
        &self,
        body: Value,
        fusion: FusionMode,
        err: &AppError,
    ) -> Result<Executed, AppError> {
        let hits = es_hits(&body)?
            .into_iter()
            .map(|hit| leg_to_hit(&hit))
            .collect();
        Ok(Executed {
            hits,
            es_took_ms: es_took_ms(&body),
            fusion,
            rrf_syntax: None,
            degraded: true,
            degrade_reason: Some(format!("一路召回失败，已返回另一路：{err}")),
        })
    }

    fn weighted_body(&self, lexical: &LexicalSpec, vector: &VectorSpec, top_n: i64) -> Value {
        match self.settings.weighted_syntax {
            WeightedSyntax::KnnBoost => {
                let mut lexical = lexical.clone();
                lexical.boost = Some(self.settings.bm25_weight);
                let mut vector = vector.clone();
                vector.boost = Some(self.settings.knn_weight);
                weighted_knn_body(&lexical, &vector, top_n)
            }
            WeightedSyntax::FunctionScore => weighted_function_score_body(
                lexical,
                vector,
                self.settings.bm25_weight,
                self.settings.knn_weight,
                top_n,
            ),
        }
    }

    fn recall_shape(&self, top_n: i64) -> (i64, i64, i64) {
        let mut window = self.settings.window_size;
        if window < top_n {
            tracing::warn!(window, top_n, "window_size 小于 size，已抬到 size");
            window = top_n;
        }
        let knn_k = self.settings.knn_k;
        let mut num_candidates = self.settings.num_candidates;
        if num_candidates < knn_k {
            tracing::warn!(num_candidates, knn_k, "num_candidates 小于 k，已抬到 k");
            num_candidates = knn_k;
        }
        (window, knn_k, num_candidates)
    }

    async fn embed_query(&self, query: &str) -> EmbedAttempt {
        let input = query_input(&self.settings.query_instruction, query);
        let started = Instant::now();
        let result = tokio::time::timeout(
            self.settings.embedding_timeout,
            self.embedding.embed(&input),
        )
        .await;
        let elapsed_ms = started.elapsed().as_millis();
        let result = match result {
            Ok(Ok(vector)) => {
                if vector.len() == self.embedding.dims() {
                    Ok(vector)
                } else {
                    Err(AppError::BadRequest(format!(
                        "查询向量维度 {} 与模型维度 {} 不一致",
                        vector.len(),
                        self.embedding.dims()
                    )))
                }
            }
            Ok(Err(err)) => Err(err),
            Err(_) => Err(AppError::EmbeddingTimeout),
        };
        EmbedAttempt { result, elapsed_ms }
    }
}

struct EmbedAttempt {
    result: Result<Vec<f32>, AppError>,
    elapsed_ms: u128,
}

struct Executed {
    hits: Vec<SearchHit>,
    es_took_ms: Option<u64>,
    fusion: FusionMode,
    rrf_syntax: Option<RrfSyntax>,
    degraded: bool,
    degrade_reason: Option<String>,
}

fn needs_cluster_version(mode: RrfMode, explicit: Option<FusionMode>) -> bool {
    mode == RrfMode::Auto && !matches!(explicit, Some(FusionMode::Bm25))
}

fn resolve_top_n(requested: Option<i64>, default: i64) -> Result<i64, AppError> {
    let top_n = requested.unwrap_or(default);
    if !(1..=100).contains(&top_n) {
        return Err(AppError::BadRequest("top_n 必须在 1 到 100".into()));
    }
    Ok(top_n)
}

fn is_degradable(err: &AppError) -> bool {
    matches!(err, AppError::Embedding(_) | AppError::EmbeddingTimeout)
}

fn degrade_reason(err: &AppError) -> String {
    match err {
        AppError::EmbeddingTimeout => "查询向量超时，已退回 BM25".into(),
        AppError::Embedding(message) => format!("查询向量失败，已退回 BM25：{message}"),
        other => format!("查询向量失败，已退回 BM25：{other}"),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use async_trait::async_trait;

    use super::HybridSearchService;
    use crate::config::{RrfMode, SearchSettings, WeightedSyntax};
    use crate::embedding::EmbeddingService;
    use crate::error::AppError;
    use crate::es::EsClient;
    use crate::model::{FusionMode, RrfSyntax, SearchRequest};

    struct RecordingEmbedding {
        dims: usize,
        vector: Vec<f32>,
        seen: Arc<Mutex<Vec<String>>>,
        mode: EmbedMode,
    }

    enum EmbedMode {
        Ok,
        Down,
        Slow,
        WrongDims,
    }

    #[async_trait]
    impl EmbeddingService for RecordingEmbedding {
        fn dims(&self) -> usize {
            self.dims
        }

        fn model_id(&self) -> &str {
            "bge-small-zh-v1.5"
        }

        async fn embed(&self, text: &str) -> Result<Vec<f32>, AppError> {
            self.seen.lock().unwrap().push(text.to_string());
            match self.mode {
                EmbedMode::Ok => Ok(self.vector.clone()),
                EmbedMode::Down => Err(AppError::Embedding("connection refused".into())),
                EmbedMode::Slow => {
                    tokio::time::sleep(Duration::from_secs(2)).await;
                    Ok(self.vector.clone())
                }
                EmbedMode::WrongDims => Ok(vec![1.0]),
            }
        }

        async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, AppError> {
            let mut out = Vec::new();
            for text in texts {
                out.push(self.embed(text).await?);
            }
            Ok(out)
        }
    }

    fn service(embedding: RecordingEmbedding, mut settings: SearchSettings) -> HybridSearchService {
        settings.rrf_mode = RrfMode::SubSearches;
        settings.query_instruction = "为这个句子生成表示以用于检索相关文章：".into();
        let es = EsClient::new(
            "http://127.0.0.1:9200",
            "hybrid_doc_index",
            Duration::from_secs(1),
            None,
        )
        .unwrap();
        HybridSearchService::new(es, Arc::new(embedding), settings)
    }

    fn request(fusion: FusionMode) -> SearchRequest {
        SearchRequest {
            query: "跨境汇款报文出错怎么办".into(),
            doc_type: Some("tech_doc".into()),
            top_n: Some(10),
            fusion: Some(fusion),
            created_from: None,
            created_to: None,
            terms: HashMap::new(),
        }
    }

    fn ready_embedding(mode: EmbedMode) -> RecordingEmbedding {
        RecordingEmbedding {
            dims: 4,
            vector: vec![1.0, 0.0, 0.0, 0.0],
            seen: Arc::new(Mutex::new(Vec::new())),
            mode,
        }
    }

    #[tokio::test]
    async fn native_plan_prefixes_the_query_and_keeps_bm25_text_raw() {
        let embedding = Arc::new(ready_embedding(EmbedMode::Ok));
        let seen = Arc::clone(&embedding.seen);
        let es = EsClient::new(
            "http://127.0.0.1:9200",
            "hybrid_doc_index",
            Duration::from_secs(1),
            None,
        )
        .unwrap();
        let settings = SearchSettings {
            rrf_mode: RrfMode::SubSearches,
            query_instruction: "为这个句子生成表示以用于检索相关文章：".into(),
            ..SearchSettings::default()
        };
        let svc = HybridSearchService::new(es, embedding, settings);
        let plan = svc.plan(&request(FusionMode::NativeRrf)).await.unwrap();
        assert_eq!(plan.fusion, FusionMode::NativeRrf);
        assert_eq!(plan.rrf_syntax, Some(RrfSyntax::SubSearches));
        assert!(!plan.degraded);
        let subs = &plan.request["sub_searches"];
        assert_eq!(
            subs[0]["query"]["bool"]["must"][0]["multi_match"]["query"],
            "跨境汇款报文出错怎么办"
        );
        assert_eq!(
            subs[1]["knn"]["query_vector"],
            serde_json::json!([1.0, 0.0, 0.0, 0.0])
        );
        assert_eq!(
            subs[0]["query"]["bool"]["filter"][0]["term"]["doc_type"],
            "tech_doc"
        );
        let seen = seen.lock().unwrap();
        assert_eq!(
            seen[0],
            "为这个句子生成表示以用于检索相关文章：跨境汇款报文出错怎么办"
        );
    }

    #[tokio::test]
    async fn embedding_outage_falls_back_to_bm25_for_hybrid_only() {
        let svc = service(ready_embedding(EmbedMode::Down), SearchSettings::default());
        let plan = svc.plan(&request(FusionMode::NativeRrf)).await.unwrap();
        assert!(plan.degraded);
        assert_eq!(plan.fusion, FusionMode::Bm25);
        assert!(plan.request.get("knn").is_none());
        assert!(plan.request.get("sub_searches").is_none());
        assert!(plan.degrade_reason.unwrap().contains("connection refused"));

        let explicit = svc.plan(&request(FusionMode::Knn)).await;
        assert!(explicit.is_err());
    }

    #[tokio::test]
    async fn embedding_timeout_falls_back_without_waiting_for_the_model() {
        let settings = SearchSettings {
            embedding_timeout: Duration::from_millis(30),
            ..SearchSettings::default()
        };
        let svc = service(ready_embedding(EmbedMode::Slow), settings);
        let started = std::time::Instant::now();
        let plan = svc.plan(&request(FusionMode::Weighted)).await.unwrap();
        assert!(started.elapsed() < Duration::from_millis(500));
        assert!(plan.degraded);
        assert_eq!(plan.fusion, FusionMode::Bm25);
    }

    #[tokio::test]
    async fn dimension_mismatch_is_not_hidden_by_fallback() {
        let svc = service(
            ready_embedding(EmbedMode::WrongDims),
            SearchSettings::default(),
        );
        let err = svc.plan(&request(FusionMode::NativeRrf)).await.unwrap_err();
        assert!(matches!(err, AppError::BadRequest(_)));
    }

    #[tokio::test]
    async fn retriever_weighted_and_app_plans_build_distinct_bodies() {
        let embedding = Arc::new(ready_embedding(EmbedMode::Ok));
        let es = EsClient::new(
            "http://127.0.0.1:9200",
            "hybrid_doc_index",
            Duration::from_secs(1),
            None,
        )
        .unwrap();
        let settings = SearchSettings {
            rrf_mode: RrfMode::Retriever,
            weighted_syntax: WeightedSyntax::KnnBoost,
            query_instruction: String::new(),
            window_size: 10,
            ..SearchSettings::default()
        };
        let svc = HybridSearchService::new(es, embedding, settings);

        let mut native = request(FusionMode::NativeRrf);
        native.top_n = Some(20);
        let native = svc.plan(&native).await.unwrap();
        assert_eq!(native.rrf_syntax, Some(RrfSyntax::Retriever));
        assert_eq!(native.request["retriever"]["rrf"]["window_size"], 20);
        assert!(native.request["retriever"]["rrf"]["retrievers"].is_array());

        let weighted = svc.plan(&request(FusionMode::Weighted)).await.unwrap();
        assert_eq!(weighted.request["knn"]["boost"], serde_json::json!(0.3));

        let app = svc.plan(&request(FusionMode::AppRrf)).await.unwrap();
        assert!(app.knn_request.is_some());
        assert_eq!(app.request["size"], 10);
        assert_eq!(app.limit, 10);
        assert_eq!(app.knn_request.unwrap()["knn"]["k"], 20);
    }
}
