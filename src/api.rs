use std::sync::Arc;

use axum::extract::{DefaultBodyLimit, Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post, put};
use axum::{Json, Router};
use serde_json::json;

use crate::config::AppConfig;
use crate::embedding::EmbeddingService;
use crate::error::AppError;
use crate::es::EsClient;
use crate::index::DocIndexService;
use crate::model::{
    BulkInput, BulkReport, DocInput, DocView, EnsureIndexResponse, Health, IndexedDoc,
    SearchRequest, SearchResponse,
};
use crate::search::HybridSearchService;

#[derive(Clone)]
pub struct AppState {
    pub search: Arc<HybridSearchService>,
    pub docs: Arc<DocIndexService>,
}

impl AppState {
    pub fn new(config: &AppConfig, embedding: Arc<dyn EmbeddingService>) -> Result<Self, AppError> {
        if embedding.dims() != config.index.dims {
            return Err(AppError::BadRequest(format!(
                "Embedding 维度 {} 与索引 dims {} 不一致",
                embedding.dims(),
                config.index.dims
            )));
        }
        if embedding.model_id() != config.index.embedding_model {
            return Err(AppError::BadRequest(format!(
                "Embedding 模型 {} 与索引模型 {} 不一致",
                embedding.model_id(),
                config.index.embedding_model
            )));
        }
        let basic = match (&config.es.username, &config.es.password) {
            (Some(username), Some(password)) => Some((username.clone(), password.clone())),
            _ => None,
        };
        let es = EsClient::new(&config.es.url, &config.es.index, config.es.timeout, basic)?;
        Ok(Self {
            search: Arc::new(HybridSearchService::new(
                es.clone(),
                Arc::clone(&embedding),
                config.search.clone(),
            )),
            docs: Arc::new(DocIndexService::new(es, embedding, config.index.clone())),
        })
    }
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/index", put(ensure_index))
        .route("/docs", post(create_doc))
        .route("/docs/bulk", post(bulk_docs))
        .route(
            "/docs/{id}",
            get(get_doc).put(update_doc).delete(remove_doc),
        )
        .route("/search", post(search))
        .layer(DefaultBodyLimit::max(32 * 1024 * 1024))
        .with_state(state)
}

async fn health(State(state): State<AppState>) -> Result<Json<Health>, ApiError> {
    Ok(Json(state.search.health().await?))
}

async fn ensure_index(
    State(state): State<AppState>,
) -> Result<Json<EnsureIndexResponse>, ApiError> {
    let created = state.docs.ensure_index().await?;
    Ok(Json(EnsureIndexResponse {
        index: state.search.index_name().to_string(),
        created,
    }))
}

async fn create_doc(
    State(state): State<AppState>,
    Json(doc): Json<DocInput>,
) -> Result<(StatusCode, Json<IndexedDoc>), ApiError> {
    let id = state.docs.upsert(doc).await?;
    Ok((StatusCode::CREATED, Json(IndexedDoc { id })))
}

async fn update_doc(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(mut doc): Json<DocInput>,
) -> Result<Json<IndexedDoc>, ApiError> {
    doc.id = Some(id);
    let id = state.docs.upsert(doc).await?;
    Ok(Json(IndexedDoc { id }))
}

async fn remove_doc(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<StatusCode, ApiError> {
    state.docs.delete(&id).await?;
    Ok(StatusCode::NO_CONTENT)
}

async fn get_doc(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<DocView>, ApiError> {
    Ok(Json(state.docs.get(&id).await?))
}

async fn bulk_docs(
    State(state): State<AppState>,
    Json(body): Json<BulkInput>,
) -> Result<Json<BulkReport>, ApiError> {
    Ok(Json(state.docs.bulk(body.docs, body.refresh).await?))
}

async fn search(
    State(state): State<AppState>,
    Json(request): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, ApiError> {
    Ok(Json(state.search.search(request).await?))
}

struct ApiError(AppError);

impl From<AppError> for ApiError {
    fn from(value: AppError) -> Self {
        Self(value)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = match &self.0 {
            AppError::BadRequest(_) => StatusCode::BAD_REQUEST,
            AppError::NotFound(_) => StatusCode::NOT_FOUND,
            AppError::EmbeddingTimeout => StatusCode::GATEWAY_TIMEOUT,
            AppError::Embedding(_)
            | AppError::Elasticsearch { .. }
            | AppError::EsTransport(_)
            | AppError::Decode(_) => StatusCode::BAD_GATEWAY,
            AppError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };
        let body = Json(json!({ "error": self.0.to_string() }));
        (status, body).into_response()
    }
}
