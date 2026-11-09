use std::{env, str::FromStr, time::Duration};

use crate::error::AppError;
use crate::fields::{is_keyword_name, is_reserved_field};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingProvider {
    Http,
    Local,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingApiStyle {
    OpenAi,
    Tei,
}

/// 集群能力未知时用 Auto：8.8–8.15 走 sub_searches，8.16+ 走 retriever，8.0–8.7 走应用层 RRF。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RrfMode {
    Auto,
    Retriever,
    SubSearches,
    App,
}

impl RrfMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Retriever => "retriever",
            Self::SubSearches => "sub_searches",
            Self::App => "app",
        }
    }
}

/// 加权对照的 DSL。默认是 ES 能执行的 query + knn boost。
/// `function_score` 保留方案文档里的写法，knn 并不是 function_score 的合法函数，只用于对照实验。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightedSyntax {
    KnnBoost,
    FunctionScore,
}

#[derive(Clone)]
pub struct EsConfig {
    pub url: String,
    pub index: String,
    pub username: Option<String>,
    pub password: Option<String>,
    pub timeout: Duration,
}

impl std::fmt::Debug for EsConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EsConfig")
            .field("url", &self.url)
            .field("index", &self.index)
            .field("username", &self.username)
            .field("password", &self.password.as_ref().map(|_| "***"))
            .field("timeout", &self.timeout)
            .finish()
    }
}

#[derive(Clone)]
pub struct EmbeddingConfig {
    pub provider: EmbeddingProvider,
    pub endpoint: String,
    pub api_style: EmbeddingApiStyle,
    pub api_key: Option<String>,
    pub model: String,
    pub dims: usize,
    pub cache_dir: Option<String>,
}

impl std::fmt::Debug for EmbeddingConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddingConfig")
            .field("provider", &self.provider)
            .field("endpoint", &self.endpoint)
            .field("api_style", &self.api_style)
            .field("api_key", &self.api_key.as_ref().map(|_| "***"))
            .field("model", &self.model)
            .field("dims", &self.dims)
            .field("cache_dir", &self.cache_dir)
            .finish()
    }
}

#[derive(Debug, Clone)]
pub struct IndexSettings {
    pub shards: u32,
    pub replicas: u32,
    pub bm25_k1: f64,
    pub bm25_b: f64,
    pub analyzer: String,
    pub dims: usize,
    pub embedding_model: String,
    pub hnsw_m: u32,
    pub hnsw_ef_construction: u32,
    pub keyword_fields: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct SearchSettings {
    pub rank_constant: i64,
    pub window_size: i64,
    pub knn_k: i64,
    pub num_candidates: i64,
    pub title_boost: f64,
    pub fuzziness: Option<String>,
    pub bm25_weight: f64,
    pub knn_weight: f64,
    pub embedding_timeout: Duration,
    pub rrf_mode: RrfMode,
    pub weighted_syntax: WeightedSyntax,
    pub query_instruction: String,
    pub default_top_n: i64,
    pub keyword_fields: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct AppConfig {
    pub es: EsConfig,
    pub embedding: EmbeddingConfig,
    pub index: IndexSettings,
    pub search: SearchSettings,
    pub http_addr: String,
}

impl Default for IndexSettings {
    fn default() -> Self {
        Self {
            shards: 1,
            replicas: 0,
            bm25_k1: 1.2,
            bm25_b: 0.6,
            analyzer: "ik_max_word".into(),
            dims: 512,
            embedding_model: "bge-small-zh-v1.5".into(),
            hnsw_m: 16,
            hnsw_ef_construction: 100,
            keyword_fields: Vec::new(),
        }
    }
}

impl Default for SearchSettings {
    fn default() -> Self {
        Self {
            rank_constant: 60,
            window_size: 100,
            knn_k: 20,
            num_candidates: 50,
            title_boost: 3.0,
            fuzziness: Some("AUTO".into()),
            bm25_weight: 0.7,
            knn_weight: 0.3,
            embedding_timeout: Duration::from_millis(80),
            rrf_mode: RrfMode::Auto,
            weighted_syntax: WeightedSyntax::KnnBoost,
            query_instruction: default_query_instruction("bge-small-zh-v1.5"),
            default_top_n: 10,
            keyword_fields: Vec::new(),
        }
    }
}

impl AppConfig {
    pub fn from_env() -> Result<Self, AppError> {
        let model = env_or("EMBEDDING_MODEL", "bge-small-zh-v1.5");
        if model.is_empty() {
            return Err(AppError::BadRequest("EMBEDDING_MODEL 不能为空".into()));
        }
        let dims = env_parse("EMBEDDING_DIMS", 512usize)?;
        if !(1..=4096).contains(&dims) {
            return Err(AppError::BadRequest(
                "EMBEDDING_DIMS 必须在 1 到 4096".into(),
            ));
        }
        let provider = parse_provider(&env_or("EMBEDDING_PROVIDER", "http"))?;
        let timeout_ms = match env::var("EMBEDDING_TIMEOUT_MS") {
            Ok(value) => parse_u64("EMBEDDING_TIMEOUT_MS", value.trim())?,
            Err(_) => match provider {
                EmbeddingProvider::Http => 80,
                EmbeddingProvider::Local => 1500,
            },
        };
        if timeout_ms == 0 {
            return Err(AppError::BadRequest(
                "EMBEDDING_TIMEOUT_MS 必须大于 0".into(),
            ));
        }

        let keyword_fields = keyword_fields_from_env()?;
        let bm25_k1 = env_parse("BM25_K1", 1.2f64)?;
        let bm25_b = env_parse("BM25_B", 0.6f64)?;
        if bm25_k1 <= 0.0 || bm25_k1 > 10.0 {
            return Err(AppError::BadRequest("BM25_K1 必须在 (0, 10]".into()));
        }
        if !(0.0..=1.0).contains(&bm25_b) {
            return Err(AppError::BadRequest("BM25_B 必须在 [0, 1]".into()));
        }
        let analyzer = env_or("ANALYZER", "ik_max_word");
        if !analyzer
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_')
        {
            return Err(AppError::BadRequest(format!("ANALYZER 不合法: {analyzer}")));
        }
        let hnsw_m = env_parse("HNSW_M", 16u32)?;
        let hnsw_ef = env_parse("HNSW_EF_CONSTRUCTION", 100u32)?;
        if hnsw_m < 2 {
            return Err(AppError::BadRequest("HNSW_M 至少为 2".into()));
        }
        if hnsw_ef == 0 {
            return Err(AppError::BadRequest(
                "HNSW_EF_CONSTRUCTION 必须大于 0".into(),
            ));
        }
        let shards = env_parse("ES_SHARDS", 1u32)?;
        if shards == 0 {
            return Err(AppError::BadRequest("ES_SHARDS 必须大于 0".into()));
        }

        let rank_constant = env_parse("RANK_CONSTANT", 60i64)?;
        let window_size = env_parse("WINDOW_SIZE", 100i64)?;
        let knn_k = env_parse("KNN_K", 20i64)?;
        let num_candidates = env_parse("KNN_NUM_CANDIDATES", 50i64)?;
        let title_boost = env_parse("TITLE_BOOST", 3.0f64)?;
        let bm25_weight = env_parse("BM25_WEIGHT", 0.7f64)?;
        let knn_weight = env_parse("KNN_WEIGHT", 0.3f64)?;
        let default_top_n = env_parse("DEFAULT_TOP_N", 10i64)?;
        if rank_constant < 1 {
            return Err(AppError::BadRequest("RANK_CONSTANT 至少为 1".into()));
        }
        if window_size < 1 || knn_k < 1 || num_candidates < 1 {
            return Err(AppError::BadRequest(
                "WINDOW_SIZE、KNN_K、KNN_NUM_CANDIDATES 都必须大于 0".into(),
            ));
        }
        if title_boost <= 0.0 || title_boost > 20.0 {
            return Err(AppError::BadRequest("TITLE_BOOST 必须在 (0, 20]".into()));
        }
        if bm25_weight < 0.0 || knn_weight < 0.0 || (bm25_weight == 0.0 && knn_weight == 0.0) {
            return Err(AppError::BadRequest(
                "BM25_WEIGHT 与 KNN_WEIGHT 不能为负，且不能同时为 0".into(),
            ));
        }
        if !(1..=100).contains(&default_top_n) {
            return Err(AppError::BadRequest("DEFAULT_TOP_N 必须在 1 到 100".into()));
        }

        let username = env_nonempty("ES_USERNAME");
        let password = env_nonempty("ES_PASSWORD");
        if username.is_some() != password.is_some() {
            return Err(AppError::BadRequest(
                "ES_USERNAME 和 ES_PASSWORD 需要同时设置".into(),
            ));
        }

        let query_instruction = match env::var("EMBEDDING_QUERY_INSTRUCTION") {
            Ok(value) => value,
            Err(_) => default_query_instruction(&model),
        };
        let fuzziness = match env::var("BM25_FUZZINESS") {
            Ok(value) if value.trim().is_empty() => None,
            Ok(value) => Some(value.trim().to_string()),
            Err(_) => Some("AUTO".into()),
        };

        let es_timeout_ms = env_parse("ES_TIMEOUT_MS", 30_000u64)?;
        if es_timeout_ms == 0 {
            return Err(AppError::BadRequest("ES_TIMEOUT_MS 必须大于 0".into()));
        }

        Ok(Self {
            es: EsConfig {
                url: env_or("ES_URL", "http://127.0.0.1:9200"),
                index: env_or("ES_INDEX", "hybrid_doc_index"),
                username,
                password,
                timeout: Duration::from_millis(es_timeout_ms),
            },
            embedding: EmbeddingConfig {
                provider,
                endpoint: env_or("EMBEDDING_ENDPOINT", "http://127.0.0.1:8080/v1/embeddings"),
                api_style: parse_api_style(&env_or("EMBEDDING_API_STYLE", "openai"))?,
                api_key: env_nonempty("EMBEDDING_API_KEY"),
                model: model.clone(),
                dims,
                cache_dir: env_nonempty("EMBEDDING_CACHE_DIR"),
            },
            index: IndexSettings {
                shards,
                replicas: env_parse("ES_REPLICAS", 0u32)?,
                bm25_k1,
                bm25_b,
                analyzer,
                dims,
                embedding_model: model.clone(),
                hnsw_m,
                hnsw_ef_construction: hnsw_ef,
                keyword_fields: keyword_fields.clone(),
            },
            search: SearchSettings {
                rank_constant,
                window_size,
                knn_k,
                num_candidates,
                title_boost,
                fuzziness,
                bm25_weight,
                knn_weight,
                embedding_timeout: Duration::from_millis(timeout_ms),
                rrf_mode: parse_rrf_mode(&env_or("RRF_SYNTAX", "auto"))?,
                weighted_syntax: parse_weighted(&env_or("WEIGHTED_SYNTAX", "knn_boost"))?,
                query_instruction,
                default_top_n,
                keyword_fields,
            },
            http_addr: env_or("HTTP_ADDR", "0.0.0.0:3000"),
        })
    }
}

/// BGE 查询侧需要指令，文档侧不加。换模型时用环境变量覆盖，两边必须仍是同一个模型。
pub fn default_query_instruction(model: &str) -> String {
    let name = model.to_ascii_lowercase();
    if name.contains("bge") && (name.contains("zh") || name.contains("chinese")) {
        "为这个句子生成表示以用于检索相关文章：".into()
    } else if name.contains("bge") {
        "Represent this sentence for searching relevant passages: ".into()
    } else {
        String::new()
    }
}

fn keyword_fields_from_env() -> Result<Vec<String>, AppError> {
    let raw = env_or("INDEX_KEYWORDS", "");
    if raw.is_empty() {
        return Ok(Vec::new());
    }
    let mut fields = Vec::new();
    for part in raw.split(',') {
        let name = part.trim();
        if name.is_empty() {
            continue;
        }
        if !is_keyword_name(name) || is_reserved_field(name) {
            return Err(AppError::BadRequest(format!(
                "INDEX_KEYWORDS 含有非法字段 {name}"
            )));
        }
        if !fields.iter().any(|existing: &String| existing == name) {
            fields.push(name.to_string());
        }
    }
    Ok(fields)
}

fn parse_provider(value: &str) -> Result<EmbeddingProvider, AppError> {
    match value.to_ascii_lowercase().as_str() {
        "http" => Ok(EmbeddingProvider::Http),
        "local" => Ok(EmbeddingProvider::Local),
        _ => Err(AppError::BadRequest(
            "EMBEDDING_PROVIDER 只能是 http 或 local".into(),
        )),
    }
}

fn parse_api_style(value: &str) -> Result<EmbeddingApiStyle, AppError> {
    match value.to_ascii_lowercase().as_str() {
        "openai" => Ok(EmbeddingApiStyle::OpenAi),
        "tei" => Ok(EmbeddingApiStyle::Tei),
        _ => Err(AppError::BadRequest(
            "EMBEDDING_API_STYLE 只能是 openai 或 tei".into(),
        )),
    }
}

fn parse_rrf_mode(value: &str) -> Result<RrfMode, AppError> {
    match value.to_ascii_lowercase().as_str() {
        "auto" => Ok(RrfMode::Auto),
        "retriever" => Ok(RrfMode::Retriever),
        "sub_searches" => Ok(RrfMode::SubSearches),
        "app" => Ok(RrfMode::App),
        _ => Err(AppError::BadRequest(
            "RRF_SYNTAX 只能是 auto、retriever、sub_searches 或 app".into(),
        )),
    }
}

fn parse_weighted(value: &str) -> Result<WeightedSyntax, AppError> {
    match value.to_ascii_lowercase().as_str() {
        "knn_boost" => Ok(WeightedSyntax::KnnBoost),
        "function_score" => Ok(WeightedSyntax::FunctionScore),
        _ => Err(AppError::BadRequest(
            "WEIGHTED_SYNTAX 只能是 knn_boost 或 function_score".into(),
        )),
    }
}

fn env_or(key: &str, default: &str) -> String {
    env_nonempty(key).unwrap_or_else(|| default.to_string())
}

fn env_nonempty(key: &str) -> Option<String> {
    env::var(key)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn env_parse<T>(key: &str, default: T) -> Result<T, AppError>
where
    T: FromStr,
    T::Err: std::fmt::Display,
{
    match env_nonempty(key) {
        None => Ok(default),
        Some(value) => value
            .parse()
            .map_err(|err| AppError::BadRequest(format!("{key} 无法解析 {value}: {err}"))),
    }
}

fn parse_u64(key: &str, value: &str) -> Result<u64, AppError> {
    value
        .parse()
        .map_err(|err| AppError::BadRequest(format!("{key} 无法解析 {value}: {err}")))
}

#[cfg(test)]
mod tests {
    use super::default_query_instruction;

    #[test]
    fn bge_zh_gets_chinese_query_instruction() {
        let text = default_query_instruction("BAAI/bge-small-zh-v1.5");
        assert!(text.starts_with("为这个句子生成表示"));
        assert!(default_query_instruction("bge-base-en-v1.5").starts_with("Represent"));
        assert!(default_query_instruction("custom-model").is_empty());
    }
}
