use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};

use super::{finalize_vector, EmbeddingService};
use crate::config::EmbeddingConfig;
use crate::error::AppError;

pub struct LocalOnnxEmbedding {
    inner: Arc<Mutex<TextEmbedding>>,
    dims: usize,
    model_id: String,
}

impl LocalOnnxEmbedding {
    pub fn load(config: &EmbeddingConfig) -> Result<Self, AppError> {
        let (model, dims) = resolve_model(&config.model)?;
        if dims != config.dims {
            return Err(AppError::BadRequest(format!(
                "模型 {} 的维度是 {dims}，EMBEDDING_DIMS={}。两边必须一致，写错会在入库时失败",
                config.model, config.dims
            )));
        }
        let mut options = TextInitOptions::new(model).with_show_download_progress(true);
        if let Some(cache_dir) = &config.cache_dir {
            options = options.with_cache_dir(PathBuf::from(cache_dir));
        }
        let text_model = TextEmbedding::try_new(options)
            .map_err(|err| AppError::Embedding(format!("加载本地模型失败: {err}")))?;
        Ok(Self {
            inner: Arc::new(Mutex::new(text_model)),
            dims,
            model_id: config.model.clone(),
        })
    }
}

fn resolve_model(name: &str) -> Result<(EmbeddingModel, usize), AppError> {
    let needle = name
        .trim()
        .trim_start_matches("BAAI/")
        .trim_start_matches("Xenova/")
        .to_ascii_lowercase();
    let found = TextEmbedding::list_supported_models()
        .into_iter()
        .find(|info| {
            let code = info.model_code.to_ascii_lowercase();
            let short = code.rsplit('/').next().unwrap_or(&code);
            short == needle
                || code == name.trim().to_ascii_lowercase()
                || format!("{:?}", info.model).eq_ignore_ascii_case(name.trim())
        });
    match found {
        Some(info) => Ok((info.model, info.dim)),
        None => Err(AppError::BadRequest(format!(
            "本地模型 {name} 不在 fastembed 支持列表。中文场景使用 bge-small-zh-v1.5（512 维）"
        ))),
    }
}

#[async_trait]
impl EmbeddingService for LocalOnnxEmbedding {
    fn dims(&self) -> usize {
        self.dims
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>, AppError> {
        let mut vectors = self.embed_batch(&[text.to_string()]).await?;
        vectors
            .pop()
            .ok_or_else(|| AppError::Embedding("本地模型返回空结果".into()))
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, AppError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        if texts.iter().any(|text| text.trim().is_empty()) {
            return Err(AppError::BadRequest("待向量化的文本不能为空".into()));
        }
        let texts = texts.to_vec();
        let inner = Arc::clone(&self.inner);
        let dims = self.dims;
        let vectors = tokio::task::spawn_blocking(move || {
            let mut model = inner
                .lock()
                .map_err(|_| AppError::Internal("本地模型锁已损坏".into()))?;
            model
                .embed(texts, None)
                .map_err(|err| AppError::Embedding(err.to_string()))
        })
        .await
        .map_err(|err| AppError::Internal(err.to_string()))??;
        vectors
            .into_iter()
            .map(|vector| finalize_vector(vector, dims))
            .collect()
    }
}
