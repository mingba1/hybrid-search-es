use std::process::ExitCode;

use hybrid_search_es::api::{router, AppState};
use hybrid_search_es::config::AppConfig;
use hybrid_search_es::embedding;
use hybrid_search_es::error::AppError;

#[tokio::main]
async fn main() -> ExitCode {
    init_tracing();
    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            tracing::error!(error = %err, "hybrid search stopped");
            eprintln!("启动失败: {err}");
            ExitCode::FAILURE
        }
    }
}

async fn run() -> Result<(), AppError> {
    let config = AppConfig::from_env()?;
    tracing::info!(
        es = %config.es.url,
        index = %config.es.index,
        model = %config.embedding.model,
        dims = config.embedding.dims,
        provider = ?config.embedding.provider,
        rrf = config.search.rrf_mode.as_str(),
        addr = %config.http_addr,
        "hybrid search starting"
    );
    let embedding = embedding::build(&config.embedding).await?;
    let state = AppState::new(&config, embedding)?;
    let app = router(state);
    let listener = tokio::net::TcpListener::bind(&config.http_addr)
        .await
        .map_err(|err| AppError::Internal(format!("监听 {} 失败: {err}", config.http_addr)))?;
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            if tokio::signal::ctrl_c().await.is_err() {
                tracing::warn!("ctrl_c listener failed");
            }
        })
        .await
        .map_err(|err| AppError::Internal(format!("HTTP 服务退出: {err}")))?;
    Ok(())
}

fn init_tracing() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();
}
