//! Elasticsearch 混合检索框架。
//!
//! 一次查询并行走 BM25 与 dense_vector 两路召回，再用倒数排名融合（RRF）合成结果。
//! BM25 负责编码、专名、条款号；向量负责同义改写。两路分数的值域不同，所以主路径
//! 使用名次而不是把 `_score` 直接相加。
//!
//! 分层：
//! - [`api`] 接入
//! - [`search`] 检索编排与降级
//! - [`embedding`] 向量实现（HTTP 或本地 ONNX）
//! - [`index`] 索引创建、写入、批量重建

pub mod api;
pub mod config;
pub mod embedding;
pub mod error;
pub mod es;
pub mod fields;
pub mod index;
pub mod model;
pub mod search;

pub use config::AppConfig;
pub use error::AppError;
pub use model::{DocInput, FusionMode, SearchRequest, SearchResponse};
