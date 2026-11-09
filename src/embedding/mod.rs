mod http;

#[cfg(feature = "local-onnx")]
mod local;

use std::sync::Arc;

use async_trait::async_trait;

use crate::config::EmbeddingConfig;
use crate::error::AppError;

pub use http::HttpEmbeddingService;

#[cfg(feature = "local-onnx")]
pub use local::LocalOnnxEmbedding;

#[async_trait]
pub trait EmbeddingService: Send + Sync {
    fn dims(&self) -> usize;
    fn model_id(&self) -> &str;
    async fn embed(&self, text: &str) -> Result<Vec<f32>, AppError>;
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, AppError>;
}

pub async fn build(config: &EmbeddingConfig) -> Result<Arc<dyn EmbeddingService>, AppError> {
    match config.provider {
        crate::config::EmbeddingProvider::Http => Ok(Arc::new(HttpEmbeddingService::new(config)?)),
        crate::config::EmbeddingProvider::Local => load_local(config).await,
    }
}

#[cfg(feature = "local-onnx")]
async fn load_local(config: &EmbeddingConfig) -> Result<Arc<dyn EmbeddingService>, AppError> {
    let config = config.clone();
    let service = tokio::task::spawn_blocking(move || LocalOnnxEmbedding::load(&config))
        .await
        .map_err(|err| AppError::Internal(err.to_string()))??;
    Ok(Arc::new(service))
}

#[cfg(not(feature = "local-onnx"))]
async fn load_local(_config: &EmbeddingConfig) -> Result<Arc<dyn EmbeddingService>, AppError> {
    Err(AppError::BadRequest(
        "EMBEDDING_PROVIDER=local 需要在编译时打开 local-onnx：cargo run --features local-onnx"
            .into(),
    ))
}

/// 入库文本不加查询指令。BGE 只在查询侧加指令。
pub fn document_passage(title: &str, content: &str) -> String {
    let title = title.trim();
    let content = content.trim();
    if content.is_empty() {
        title.to_string()
    } else {
        format!("{title}\n{content}")
    }
}

pub fn query_input(instruction: &str, query: &str) -> String {
    let query = query.trim();
    if instruction.is_empty() {
        query.to_string()
    } else {
        format!("{instruction}{query}")
    }
}

pub fn finalize_vector(mut vector: Vec<f32>, expected_dims: usize) -> Result<Vec<f32>, AppError> {
    if vector.len() != expected_dims {
        return Err(AppError::BadRequest(format!(
            "向量维度 {} 与配置 dims {expected_dims} 不一致",
            vector.len()
        )));
    }
    if vector.iter().any(|value| !value.is_finite()) {
        return Err(AppError::BadRequest("向量包含 NaN 或 Inf".into()));
    }
    l2_normalize(&mut vector)?;
    Ok(vector)
}

fn l2_normalize(vector: &mut [f32]) -> Result<(), AppError> {
    let sum: f32 = vector.iter().map(|value| value * value).sum();
    if sum == 0.0 || !sum.is_finite() {
        return Err(AppError::BadRequest(
            "embedding 为零向量，无法归一化".into(),
        ));
    }
    let norm = sum.sqrt();
    for value in vector.iter_mut() {
        *value /= norm;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{document_passage, finalize_vector, query_input};

    #[test]
    fn passage_keeps_document_free_of_query_instruction() {
        assert_eq!(document_passage("标题", "正文"), "标题\n正文");
        assert_eq!(document_passage(" 只有标题 ", "  "), "只有标题");
    }

    #[test]
    fn query_input_prefixes_instruction_only_when_present() {
        assert_eq!(query_input("查询：", "报文"), "查询：报文");
        assert_eq!(query_input("", " 报文 "), "报文");
    }

    #[test]
    fn l2_normalizes_three_four_to_unit_vector() {
        let vector = finalize_vector(vec![3.0, 4.0], 2).unwrap();
        assert!((vector[0] - 0.6).abs() < 1e-6);
        assert!((vector[1] - 0.8).abs() < 1e-6);
        let norm: f32 = vector.iter().map(|value| value * value).sum();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn rejects_wrong_dimension_and_zero_vector() {
        assert!(finalize_vector(vec![1.0], 2).is_err());
        assert!(finalize_vector(vec![0.0, 0.0], 2).is_err());
    }
}
