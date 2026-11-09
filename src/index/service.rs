use std::sync::Arc;

use chrono::{NaiveDate, Utc};
use serde_json::{json, Value};
use tokio::sync::OnceCell;
use uuid::Uuid;

use super::mapping::{extract_embedding_meta, index_definition};
use crate::config::IndexSettings;
use crate::embedding::{document_passage, finalize_vector, EmbeddingService};
use crate::error::AppError;
use crate::es::EsClient;
use crate::model::{BulkFailure, BulkReport, DocInput, DocView};

const EMBED_CHUNK: usize = 32;
const BULK_CHUNK: usize = 200;

pub struct DocIndexService {
    es: EsClient,
    embedding: Arc<dyn EmbeddingService>,
    settings: IndexSettings,
    meta_ok: OnceCell<()>,
}

impl DocIndexService {
    pub fn new(
        es: EsClient,
        embedding: Arc<dyn EmbeddingService>,
        settings: IndexSettings,
    ) -> Self {
        Self {
            es,
            embedding,
            settings,
            meta_ok: OnceCell::new(),
        }
    }

    pub async fn ensure_index(&self) -> Result<bool, AppError> {
        if self.es.index_exists().await? {
            self.check_meta().await?;
            let _ = self.meta_ok.set(());
            return Ok(false);
        }
        match self
            .es
            .create_index(&index_definition(&self.settings))
            .await
        {
            Ok(()) => {
                let _ = self.meta_ok.set(());
                Ok(true)
            }
            Err(AppError::Elasticsearch { body, .. })
                if body.contains("resource_already_exists_exception") =>
            {
                self.check_meta().await?;
                let _ = self.meta_ok.set(());
                Ok(false)
            }
            Err(err) => Err(err),
        }
    }

    pub async fn upsert(&self, mut doc: DocInput) -> Result<String, AppError> {
        if doc.id.is_none() {
            doc.id = Some(Uuid::new_v4().to_string());
        }
        let id = doc
            .id
            .clone()
            .ok_or_else(|| AppError::Internal("文档 id 缺失".into()))?;
        let report = self.write_docs(vec![doc], true).await?;
        if let Some(failure) = report.failed.into_iter().next() {
            return Err(AppError::Elasticsearch {
                status: 400,
                body: failure.reason,
            });
        }
        Ok(id)
    }

    pub async fn bulk(&self, docs: Vec<DocInput>, refresh: bool) -> Result<BulkReport, AppError> {
        if docs.is_empty() {
            return Err(AppError::BadRequest("docs 不能为空".into()));
        }
        if docs.len() > 500 {
            return Err(AppError::BadRequest("单次批量最多 500 篇".into()));
        }
        self.write_docs(docs, refresh).await
    }

    pub async fn delete(&self, id: &str) -> Result<(), AppError> {
        validate_id(id)?;
        self.es.delete_doc(id).await
    }

    pub async fn get(&self, id: &str) -> Result<DocView, AppError> {
        validate_id(id)?;
        let body = self.es.get_doc(id).await?;
        let source = body.get("_source").cloned().unwrap_or(Value::Null);
        let embedding_dims = source
            .get("embedding")
            .and_then(Value::as_array)
            .map(Vec::len);
        Ok(DocView {
            id: body
                .get("_id")
                .and_then(Value::as_str)
                .unwrap_or(id)
                .to_string(),
            title: text_field(&source, "title"),
            content: text_field(&source, "content"),
            doc_type: text_field(&source, "doc_type"),
            created_at: source
                .get("created_at")
                .and_then(Value::as_str)
                .map(str::to_string),
            embedding_model: source
                .get("embedding_model")
                .and_then(Value::as_str)
                .map(str::to_string),
            embedding_dims,
        })
    }

    async fn write_docs(
        &self,
        mut docs: Vec<DocInput>,
        refresh: bool,
    ) -> Result<BulkReport, AppError> {
        self.ready().await?;
        for doc in &mut docs {
            validate_document(&doc.title, &doc.content, &doc.doc_type)?;
            if doc.id.is_none() {
                doc.id = Some(Uuid::new_v4().to_string());
            }
            validate_id(doc.id.as_deref().unwrap_or(""))?;
            if doc.created_at.is_none() {
                doc.created_at = Some(Utc::now().date_naive());
            }
        }
        let vectors = self.vectors_for(&docs).await?;
        let model = self.embedding.model_id().to_string();
        let prepared: Vec<(String, Value)> = docs
            .iter()
            .zip(vectors)
            .map(|(doc, vector)| {
                let id = doc.id.clone().unwrap_or_default();
                let body = stored_document(
                    &doc.title,
                    &doc.content,
                    &doc.doc_type,
                    doc.created_at.unwrap_or_else(|| Utc::now().date_naive()),
                    &model,
                    &vector,
                );
                (id, body)
            })
            .collect();

        let mut indexed = 0;
        let mut failed = Vec::new();
        let chunks: Vec<_> = prepared.chunks(BULK_CHUNK).collect();
        for (position, chunk) in chunks.iter().enumerate() {
            let refresh_chunk = refresh && position + 1 == chunks.len();
            let ids: Vec<String> = chunk.iter().map(|(id, _)| id.clone()).collect();
            let ndjson = to_ndjson(chunk)?;
            let response = self.es.bulk(&ndjson, refresh_chunk).await?;
            let report = parse_bulk(&response, &ids);
            indexed += report.indexed;
            failed.extend(report.failed);
        }
        Ok(BulkReport { indexed, failed })
    }

    async fn vectors_for(&self, docs: &[DocInput]) -> Result<Vec<Vec<f32>>, AppError> {
        let dims = self.embedding.dims();
        let mut vectors = vec![None; docs.len()];
        let mut missing = Vec::new();
        for (index, doc) in docs.iter().enumerate() {
            if let Some(vector) = &doc.embedding {
                vectors[index] = Some(finalize_vector(vector.clone(), dims)?);
            } else {
                missing.push(index);
            }
        }
        for chunk in missing.chunks(EMBED_CHUNK) {
            let texts: Vec<String> = chunk
                .iter()
                .map(|index| document_passage(&docs[*index].title, &docs[*index].content))
                .collect();
            let embedded = self.embedding.embed_batch(&texts).await?;
            if embedded.len() != chunk.len() {
                return Err(AppError::Embedding("批量向量条数与文本条数不一致".into()));
            }
            for (index, vector) in chunk.iter().zip(embedded) {
                vectors[*index] = Some(finalize_vector(vector, dims)?);
            }
        }
        vectors
            .into_iter()
            .map(|vector| vector.ok_or_else(|| AppError::Internal("向量缺失".into())))
            .collect()
    }

    async fn ready(&self) -> Result<(), AppError> {
        if self.meta_ok.get().is_some() {
            return Ok(());
        }
        if !self.es.index_exists().await? {
            return Err(AppError::BadRequest(format!(
                "索引 {} 不存在，请先调用 PUT /index",
                self.es.index_name()
            )));
        }
        self.check_meta().await?;
        let _ = self.meta_ok.set(());
        Ok(())
    }

    async fn check_meta(&self) -> Result<(), AppError> {
        let mapping = self.es.mapping().await?;
        let Some(meta) = extract_embedding_meta(&mapping) else {
            tracing::warn!(
                index = self.es.index_name(),
                "索引没有 embedding _meta，无法确认是否与当前模型混写"
            );
            return Ok(());
        };
        if meta.model != self.settings.embedding_model || meta.dims != self.settings.dims {
            return Err(AppError::BadRequest(format!(
                "索引已绑定模型 {} dims {}，当前配置是 {} dims {}。换模型必须全量重建，不能混写",
                meta.model, meta.dims, self.settings.embedding_model, self.settings.dims
            )));
        }
        Ok(())
    }
}

fn stored_document(
    title: &str,
    content: &str,
    doc_type: &str,
    created_at: NaiveDate,
    model: &str,
    embedding: &[f32],
) -> Value {
    json!({
        "title": title,
        "content": content,
        "doc_type": doc_type,
        "created_at": created_at.to_string(),
        "embedding_model": model,
        "embedding": embedding,
    })
}

fn to_ndjson(docs: &[(String, Value)]) -> Result<String, AppError> {
    let mut body = String::new();
    for (id, doc) in docs {
        let meta = json!({ "index": { "_id": id } });
        body.push_str(
            &serde_json::to_string(&meta).map_err(|err| AppError::Decode(err.to_string()))?,
        );
        body.push('\n');
        body.push_str(
            &serde_json::to_string(doc).map_err(|err| AppError::Decode(err.to_string()))?,
        );
        body.push('\n');
    }
    Ok(body)
}

fn parse_bulk(body: &Value, ids: &[String]) -> BulkReport {
    let Some(items) = body.get("items").and_then(Value::as_array) else {
        return BulkReport {
            indexed: 0,
            failed: ids
                .iter()
                .map(|id| BulkFailure {
                    id: id.clone(),
                    reason: "响应缺少 items".into(),
                })
                .collect(),
        };
    };
    let mut indexed = 0;
    let mut failed = Vec::new();
    for (position, item) in items.iter().enumerate() {
        let id = ids
            .get(position)
            .cloned()
            .unwrap_or_else(|| format!("#{position}"));
        let action = item.get("index").unwrap_or(item);
        let status = action.get("status").and_then(Value::as_u64).unwrap_or(500);
        if (200..300).contains(&status) {
            indexed += 1;
        } else {
            let reason = action
                .pointer("/error/reason")
                .and_then(Value::as_str)
                .unwrap_or("写入失败")
                .to_string();
            failed.push(BulkFailure { id, reason });
        }
    }
    BulkReport { indexed, failed }
}

fn validate_document(title: &str, content: &str, doc_type: &str) -> Result<(), AppError> {
    if title.trim().is_empty() {
        return Err(AppError::BadRequest("title 不能为空".into()));
    }
    if title.chars().count() > 512 {
        return Err(AppError::BadRequest("title 超过 512 字".into()));
    }
    if content.chars().count() > 200_000 {
        return Err(AppError::BadRequest("content 超过 200000 字".into()));
    }
    if doc_type.trim().is_empty() || doc_type.len() > 128 || doc_type.chars().any(char::is_control)
    {
        return Err(AppError::BadRequest(
            "doc_type 需为 1 到 128 字符且不含控制字符".into(),
        ));
    }
    Ok(())
}

pub(crate) fn validate_id(id: &str) -> Result<(), AppError> {
    if id.is_empty() || id.len() > 512 {
        return Err(AppError::BadRequest(
            "文档 id 长度必须在 1 到 512 字节之间".into(),
        ));
    }
    if id
        .chars()
        .any(|ch| ch.is_control() || matches!(ch, '/' | '\\' | '?' | '#'))
    {
        return Err(AppError::BadRequest("文档 id 含有非法字符".into()));
    }
    Ok(())
}

fn text_field(source: &Value, name: &str) -> String {
    source
        .get(name)
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::{parse_bulk, to_ndjson, validate_document, validate_id};
    use serde_json::json;

    #[test]
    fn ndjson_has_action_and_source_lines() {
        let body = to_ndjson(&[("1".into(), json!({"title": "外汇"}))]).unwrap();
        let lines: Vec<_> = body.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("\"_id\":\"1\""));
        assert!(lines[1].contains("外汇"));
    }

    #[test]
    fn bulk_parser_splits_success_and_failure() {
        let body = json!({
            "items": [
                { "index": { "_id": "1", "status": 201 } },
                { "index": { "_id": "2", "status": 400, "error": { "reason": "mapper" } } }
            ]
        });
        let report = parse_bulk(&body, &["1".into(), "2".into()]);
        assert_eq!(report.indexed, 1);
        assert_eq!(report.failed[0].id, "2");
        assert_eq!(report.failed[0].reason, "mapper");
    }

    #[test]
    fn document_rules_reject_blank_title_and_bad_id() {
        assert!(validate_document("", "正文", "tech_doc").is_err());
        assert!(validate_document("标题", "正文", "tech_doc").is_ok());
        assert!(validate_id("HT-2024-001").is_ok());
        assert!(validate_id("a/b").is_err());
    }
}
