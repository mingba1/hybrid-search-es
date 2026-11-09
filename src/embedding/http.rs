use std::time::Duration;

use async_trait::async_trait;
use reqwest::header::AUTHORIZATION;
use serde_json::{json, Value};
use url::Url;

use super::{finalize_vector, EmbeddingService};
use crate::config::{EmbeddingApiStyle, EmbeddingConfig};
use crate::error::{truncate_chars, AppError};

pub struct HttpEmbeddingService {
    http: reqwest::Client,
    endpoint: Url,
    model: String,
    dims: usize,
    style: EmbeddingApiStyle,
    api_key: Option<String>,
}

impl HttpEmbeddingService {
    pub fn new(config: &EmbeddingConfig) -> Result<Self, AppError> {
        let endpoint = Url::parse(&config.endpoint)
            .map_err(|err| AppError::BadRequest(format!("EMBEDDING_ENDPOINT 不合法: {err}")))?;
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .connect_timeout(Duration::from_secs(2))
            .user_agent("hybrid-search-es/0.1")
            .build()
            .map_err(|err| AppError::Embedding(err.to_string()))?;
        Ok(Self {
            http,
            endpoint,
            model: config.model.clone(),
            dims: config.dims,
            style: config.api_style,
            api_key: config.api_key.clone(),
        })
    }

    async fn post(&self, body: Value) -> Result<Value, AppError> {
        let mut request = self.http.post(self.endpoint.clone()).json(&body);
        if let Some(api_key) = &self.api_key {
            request = request.header(AUTHORIZATION, format!("Bearer {api_key}"));
        }
        let response = request
            .send()
            .await
            .map_err(|err| AppError::Embedding(err.to_string()))?;
        let status = response.status();
        let text = response
            .text()
            .await
            .map_err(|err| AppError::Embedding(err.to_string()))?;
        if !status.is_success() {
            return Err(AppError::Embedding(format!(
                "HTTP {}: {}",
                status.as_u16(),
                truncate_chars(&text.replace('\n', " "), 400)
            )));
        }
        serde_json::from_str(&text).map_err(|err| AppError::Embedding(err.to_string()))
    }
}

#[async_trait]
impl EmbeddingService for HttpEmbeddingService {
    fn dims(&self) -> usize {
        self.dims
    }

    fn model_id(&self) -> &str {
        &self.model
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>, AppError> {
        let mut vectors = self.embed_batch(&[text.to_string()]).await?;
        vectors
            .pop()
            .ok_or_else(|| AppError::Embedding("向量服务返回空结果".into()))
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, AppError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        if texts.iter().any(|text| text.trim().is_empty()) {
            return Err(AppError::BadRequest("待向量化的文本不能为空".into()));
        }
        let body = match self.style {
            EmbeddingApiStyle::OpenAi => json!({ "model": self.model, "input": texts }),
            EmbeddingApiStyle::Tei => json!({ "inputs": texts }),
        };
        let payload = self.post(body).await?;
        let parsed = match self.style {
            EmbeddingApiStyle::OpenAi => parse_openai(&payload, texts.len())?,
            EmbeddingApiStyle::Tei => parse_tei(&payload, texts.len())?,
        };
        parsed
            .into_iter()
            .map(|vector| finalize_vector(vector, self.dims))
            .collect()
    }
}

pub(crate) fn parse_openai(payload: &Value, expected: usize) -> Result<Vec<Vec<f32>>, AppError> {
    let data = payload
        .get("data")
        .and_then(Value::as_array)
        .ok_or_else(|| AppError::Embedding("OpenAI 响应缺少 data".into()))?;
    if data.len() != expected {
        return Err(AppError::Embedding(format!(
            "向量条数 {} 与文本条数 {expected} 不一致",
            data.len()
        )));
    }
    let mut rows = Vec::with_capacity(data.len());
    for (position, item) in data.iter().enumerate() {
        let embedding = item
            .get("embedding")
            .ok_or_else(|| AppError::Embedding("OpenAI 响应缺少 embedding".into()))?;
        let index = item
            .get("index")
            .and_then(Value::as_u64)
            .map(|value| value as usize)
            .unwrap_or(position);
        rows.push((index, json_floats(embedding)?));
    }
    rows.sort_by_key(|(index, _)| *index);
    Ok(rows.into_iter().map(|(_, vector)| vector).collect())
}

pub(crate) fn parse_tei(payload: &Value, expected: usize) -> Result<Vec<Vec<f32>>, AppError> {
    if let Some(nested) = payload.get("embeddings") {
        return parse_vector_list(nested, expected);
    }
    parse_vector_list(payload, expected)
}

fn parse_vector_list(payload: &Value, expected: usize) -> Result<Vec<Vec<f32>>, AppError> {
    let items = payload
        .as_array()
        .ok_or_else(|| AppError::Embedding("TEI 响应不是数组".into()))?;
    if expected == 1 && items.first().and_then(Value::as_f64).is_some() {
        return Ok(vec![json_floats(payload)?]);
    }
    if items.len() != expected {
        return Err(AppError::Embedding(format!(
            "向量条数 {} 与文本条数 {expected} 不一致",
            items.len()
        )));
    }
    items.iter().map(json_floats).collect()
}

fn json_floats(value: &Value) -> Result<Vec<f32>, AppError> {
    let items = value
        .as_array()
        .ok_or_else(|| AppError::Embedding("向量不是数组".into()))?;
    items
        .iter()
        .map(|item| {
            item.as_f64()
                .map(|number| number as f32)
                .ok_or_else(|| AppError::Embedding("向量元素不是数字".into()))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{parse_openai, parse_tei};
    use serde_json::json;

    #[test]
    fn openai_rows_follow_index_not_payload_order() {
        let payload = json!({
            "data": [
                { "index": 1, "embedding": [0.0, 1.0] },
                { "index": 0, "embedding": [1.0, 0.0] }
            ]
        });
        let vectors = parse_openai(&payload, 2).unwrap();
        assert_eq!(vectors[0], vec![1.0, 0.0]);
        assert_eq!(vectors[1], vec![0.0, 1.0]);
    }

    #[test]
    fn tei_accepts_batch_and_single_vector() {
        let batch = parse_tei(&json!([[1.0, 0.0], [0.0, 1.0]]), 2).unwrap();
        assert_eq!(batch.len(), 2);
        let single = parse_tei(&json!([0.2, 0.4]), 1).unwrap();
        assert_eq!(single, vec![vec![0.2, 0.4]]);
    }
}
