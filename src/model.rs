use std::collections::HashMap;

use chrono::NaiveDate;
use serde::{Deserialize, Serialize};

/// 实际执行的融合方式。显式请求 `knn` / `bm25` 用于三路效果对比。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FusionMode {
    NativeRrf,
    AppRrf,
    Weighted,
    Bm25,
    Knn,
}

/// 发给 Elasticsearch 的 RRF 语法。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RrfSyntax {
    /// ES 8.16+ 的 `retriever.rrf`。
    Retriever,
    /// ES 8.8–8.15 的 `sub_searches` + `rank.rrf`。
    SubSearches,
    /// 应用层取两路 Top-N 再按名次合并，供 8.0–8.7。
    Application,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DocInput {
    pub id: Option<String>,
    pub title: String,
    pub content: String,
    pub doc_type: String,
    #[serde(default)]
    pub created_at: Option<NaiveDate>,
    /// 调用方已经用同一模型算好的向量。缺省时由本服务生成。
    #[serde(default)]
    pub embedding: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BulkInput {
    pub docs: Vec<DocInput>,
    #[serde(default)]
    pub refresh: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct BulkFailure {
    pub id: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct BulkReport {
    pub indexed: usize,
    pub failed: Vec<BulkFailure>,
}

#[derive(Debug, Clone, Serialize)]
pub struct IndexedDoc {
    pub id: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct EnsureIndexResponse {
    pub index: String,
    pub created: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct DocView {
    pub id: String,
    pub title: String,
    pub content: String,
    pub doc_type: String,
    pub created_at: Option<String>,
    pub embedding_model: Option<String>,
    pub embedding_dims: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    #[serde(default)]
    pub doc_type: Option<String>,
    #[serde(default)]
    pub top_n: Option<i64>,
    #[serde(default)]
    pub fusion: Option<FusionMode>,
    #[serde(default)]
    pub created_from: Option<NaiveDate>,
    #[serde(default)]
    pub created_to: Option<NaiveDate>,
    /// 额外 keyword 过滤，字段必须预先出现在 `INDEX_KEYWORDS` 里。
    #[serde(default)]
    pub terms: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SearchHit {
    pub id: String,
    pub score: f64,
    pub title: String,
    pub content: String,
    pub doc_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bm25_score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub knn_score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bm25_rank: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub knn_rank: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SearchResponse {
    pub hits: Vec<SearchHit>,
    pub took_ms: u128,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embed_ms: Option<u128>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub es_took_ms: Option<u64>,
    pub fusion: FusionMode,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rrf_syntax: Option<RrfSyntax>,
    pub degraded: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub degrade_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Health {
    pub status: String,
    pub es_version: String,
    pub index: String,
    pub embedding_model: String,
    pub embedding_dims: usize,
    pub rrf_mode: String,
    pub vector_search: bool,
}
