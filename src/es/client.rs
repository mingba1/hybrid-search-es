use std::sync::Arc;
use std::time::Duration;

use reqwest::{RequestBuilder, StatusCode};
use serde_json::Value;
use tokio::sync::OnceCell;
use url::Url;

use crate::error::{truncate_chars, AppError};
use crate::model::RrfSyntax;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClusterVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
    pub raw: String,
}

impl ClusterVersion {
    pub fn parse(raw: &str) -> Result<Self, AppError> {
        let core = raw.split('-').next().unwrap_or(raw);
        let mut parts = core.split('.');
        let major = parse_part(parts.next().unwrap_or("0"), raw)?;
        let minor = parse_part(parts.next().unwrap_or("0"), raw)?;
        let patch = parse_part(parts.next().unwrap_or("0"), raw)?;
        Ok(Self {
            major,
            minor,
            patch,
            raw: raw.to_string(),
        })
    }

    pub fn supports_knn(&self) -> bool {
        self.major >= 8
    }
}

fn parse_part(part: &str, raw: &str) -> Result<u32, AppError> {
    part.parse()
        .map_err(|_| AppError::Decode(format!("无法解析 Elasticsearch 版本 {raw}")))
}

/// 8.8 起有原生 RRF。8.16 起改用已经稳定的 retriever 语法，更早的 8.8–8.15 用 sub_searches。
pub fn select_rrf_syntax(version: &ClusterVersion) -> Option<RrfSyntax> {
    if version.major < 8 {
        None
    } else if version.major == 8 && version.minor < 8 {
        Some(RrfSyntax::Application)
    } else if version.major == 8 && version.minor < 16 {
        Some(RrfSyntax::SubSearches)
    } else {
        Some(RrfSyntax::Retriever)
    }
}

struct EsInner {
    http: reqwest::Client,
    base: Url,
    index: String,
    basic: Option<(String, String)>,
    version: OnceCell<ClusterVersion>,
}

#[derive(Clone)]
pub struct EsClient {
    inner: Arc<EsInner>,
}

impl EsClient {
    pub fn new(
        base: &str,
        index: &str,
        timeout: Duration,
        basic: Option<(String, String)>,
    ) -> Result<Self, AppError> {
        validate_index_name(index)?;
        let base = Url::parse(base)
            .map_err(|err| AppError::BadRequest(format!("ES_URL 不合法: {err}")))?;
        let http = reqwest::Client::builder()
            .timeout(timeout)
            .user_agent("hybrid-search-es/0.1")
            .build()
            .map_err(|err| AppError::EsTransport(err.to_string()))?;
        Ok(Self {
            inner: Arc::new(EsInner {
                http,
                base,
                index: index.to_string(),
                basic,
                version: OnceCell::new(),
            }),
        })
    }

    pub fn index_name(&self) -> &str {
        &self.inner.index
    }

    pub async fn version(&self) -> Result<ClusterVersion, AppError> {
        if let Some(version) = self.inner.version.get() {
            return Ok(version.clone());
        }
        let fetched = self.fetch_version().await?;
        match self.inner.version.set(fetched.clone()) {
            Ok(()) => Ok(fetched),
            Err(_) => Ok(self.inner.version.get().cloned().unwrap_or(fetched)),
        }
    }

    pub async fn index_exists(&self) -> Result<bool, AppError> {
        let url = self.index_url(&[])?;
        let (status, text) = self
            .execute(self.decorate(self.inner.http.head(url)))
            .await?;
        match status.as_u16() {
            200 => Ok(true),
            404 => Ok(false),
            _ => Err(es_error(status, &text)),
        }
    }

    pub async fn create_index(&self, body: &Value) -> Result<(), AppError> {
        let url = self.index_url(&[])?;
        let (status, text) = self
            .execute(self.decorate(self.inner.http.put(url).json(body)))
            .await?;
        ensure_success(status, &text).map(|_| ())
    }

    pub async fn mapping(&self) -> Result<Value, AppError> {
        let url = self.index_url(&["_mapping"])?;
        let (status, text) = self
            .execute(self.decorate(self.inner.http.get(url)))
            .await?;
        ensure_success(status, &text)
    }

    pub async fn put_doc(&self, id: &str, doc: &Value, refresh: bool) -> Result<(), AppError> {
        let mut url = self.index_url(&["_doc", id])?;
        url.query_pairs_mut()
            .append_pair("refresh", if refresh { "wait_for" } else { "false" });
        let (status, text) = self
            .execute(self.decorate(self.inner.http.put(url).json(doc)))
            .await?;
        ensure_success(status, &text).map(|_| ())
    }

    pub async fn bulk(&self, ndjson: &str, refresh: bool) -> Result<Value, AppError> {
        let mut url = self.index_url(&["_bulk"])?;
        url.query_pairs_mut()
            .append_pair("refresh", if refresh { "wait_for" } else { "false" });
        let request = self.decorate(
            self.inner
                .http
                .post(url)
                .header(reqwest::header::CONTENT_TYPE, "application/x-ndjson")
                .body(ndjson.to_string()),
        );
        let (status, text) = self.execute(request).await?;
        ensure_success(status, &text)
    }

    pub async fn delete_doc(&self, id: &str) -> Result<(), AppError> {
        let mut url = self.index_url(&["_doc", id])?;
        url.query_pairs_mut().append_pair("refresh", "wait_for");
        let (status, text) = self
            .execute(self.decorate(self.inner.http.delete(url)))
            .await?;
        if status.as_u16() == 404 {
            return Err(AppError::NotFound(id.to_string()));
        }
        ensure_success(status, &text).map(|_| ())
    }

    pub async fn get_doc(&self, id: &str) -> Result<Value, AppError> {
        let url = self.index_url(&["_doc", id])?;
        let (status, text) = self
            .execute(self.decorate(self.inner.http.get(url)))
            .await?;
        if status.as_u16() == 404 {
            return Err(AppError::NotFound(id.to_string()));
        }
        let body = ensure_success(status, &text)?;
        if body.get("found").and_then(Value::as_bool) == Some(false) {
            return Err(AppError::NotFound(id.to_string()));
        }
        Ok(body)
    }

    pub async fn search(&self, body: &Value) -> Result<Value, AppError> {
        let url = self.index_url(&["_search"])?;
        let (status, text) = self
            .execute(self.decorate(self.inner.http.post(url).json(body)))
            .await?;
        ensure_success(status, &text)
    }

    async fn fetch_version(&self) -> Result<ClusterVersion, AppError> {
        let (status, text) = self
            .execute(self.decorate(self.inner.http.get(self.inner.base.clone())))
            .await?;
        let body = ensure_success(status, &text)?;
        let distribution = body
            .pointer("/version/distribution")
            .and_then(Value::as_str)
            .unwrap_or("");
        if distribution.eq_ignore_ascii_case("opensearch") {
            return Err(AppError::BadRequest(
                "当前集群是 OpenSearch，本框架只对接 Elasticsearch".into(),
            ));
        }
        let number = body
            .pointer("/version/number")
            .and_then(Value::as_str)
            .ok_or_else(|| AppError::Decode("缺少 version.number".into()))?;
        ClusterVersion::parse(number)
    }

    pub(crate) fn index_url(&self, rest: &[&str]) -> Result<Url, AppError> {
        let mut segments = Vec::with_capacity(rest.len() + 1);
        segments.push(self.inner.index.as_str());
        segments.extend_from_slice(rest);
        self.url(&segments)
    }

    fn url(&self, segments: &[&str]) -> Result<Url, AppError> {
        let mut url = self.inner.base.clone();
        {
            let mut path = url
                .path_segments_mut()
                .map_err(|_| AppError::BadRequest("ES_URL 不能是 cannot-be-a-base URL".into()))?;
            for segment in segments {
                path.push(segment);
            }
        }
        Ok(url)
    }

    fn decorate(&self, request: RequestBuilder) -> RequestBuilder {
        let request = match &self.inner.basic {
            Some((username, password)) => request.basic_auth(username, Some(password)),
            None => request,
        };
        request.header(reqwest::header::ACCEPT, "application/json")
    }

    async fn execute(&self, request: RequestBuilder) -> Result<(StatusCode, String), AppError> {
        let response = request
            .send()
            .await
            .map_err(|err| AppError::EsTransport(err.to_string()))?;
        let status = response.status();
        let text = response
            .text()
            .await
            .map_err(|err| AppError::EsTransport(err.to_string()))?;
        Ok((status, text))
    }
}

fn ensure_success(status: StatusCode, text: &str) -> Result<Value, AppError> {
    if !status.is_success() {
        return Err(es_error(status, text));
    }
    if text.trim().is_empty() {
        return Ok(Value::Null);
    }
    serde_json::from_str(text).map_err(|err| AppError::Decode(err.to_string()))
}

fn es_error(status: StatusCode, text: &str) -> AppError {
    AppError::Elasticsearch {
        status: status.as_u16(),
        body: truncate_chars(&text.replace(['\n', '\r'], " "), 800),
    }
}

fn validate_index_name(name: &str) -> Result<(), AppError> {
    let ok = !name.is_empty()
        && name.len() <= 255
        && name.chars().next().is_some_and(|c| c.is_ascii_lowercase())
        && name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-');
    if ok {
        Ok(())
    } else {
        Err(AppError::BadRequest(format!(
            "索引名 {name} 不合法，需为小写字母开头，仅含小写字母、数字、_ 和 -"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::{select_rrf_syntax, ClusterVersion, EsClient};
    use crate::model::RrfSyntax;
    use std::time::Duration;

    #[test]
    fn parses_release_and_snapshot_versions() {
        let release = ClusterVersion::parse("8.15.0").unwrap();
        assert_eq!((release.major, release.minor, release.patch), (8, 15, 0));
        let snapshot = ClusterVersion::parse("8.15.0-SNAPSHOT").unwrap();
        assert_eq!(snapshot.minor, 15);
        assert!(!ClusterVersion::parse("7.17.18").unwrap().supports_knn());
    }

    #[test]
    fn picks_rrf_syntax_from_cluster_version() {
        let syntax = |raw: &str| select_rrf_syntax(&ClusterVersion::parse(raw).unwrap());
        assert_eq!(syntax("7.17.18"), None);
        assert_eq!(syntax("8.4.0"), Some(RrfSyntax::Application));
        assert_eq!(syntax("8.8.0"), Some(RrfSyntax::SubSearches));
        assert_eq!(syntax("8.15.3"), Some(RrfSyntax::SubSearches));
        assert_eq!(syntax("8.16.0"), Some(RrfSyntax::Retriever));
        assert_eq!(syntax("9.1.0"), Some(RrfSyntax::Retriever));
    }

    #[test]
    fn encodes_index_path_and_rejects_bad_index_name() {
        let client = EsClient::new(
            "http://127.0.0.1:9200",
            "hybrid_doc_index",
            Duration::from_secs(1),
            None,
        )
        .unwrap();
        let url = client.index_url(&["_doc", "a b"]).unwrap();
        assert_eq!(url.path(), "/hybrid_doc_index/_doc/a%20b");

        let slashed = EsClient::new(
            "http://127.0.0.1:9200/",
            "hybrid_doc_index",
            Duration::from_secs(1),
            None,
        )
        .unwrap();
        let url = slashed.index_url(&["_search"]).unwrap();
        assert_eq!(url.path(), "/hybrid_doc_index/_search");
        assert!(EsClient::new(
            "http://127.0.0.1:9200",
            "Hybrid",
            Duration::from_secs(1),
            None
        )
        .is_err());
    }
}
