use thiserror::Error;

#[derive(Debug, Clone, Error)]
pub enum AppError {
    #[error("请求不合法: {0}")]
    BadRequest(String),
    #[error("文档不存在: {0}")]
    NotFound(String),
    #[error("向量服务失败: {0}")]
    Embedding(String),
    #[error("向量服务超时")]
    EmbeddingTimeout,
    #[error("Elasticsearch 请求失败 ({status}): {body}")]
    Elasticsearch { status: u16, body: String },
    #[error("Elasticsearch 连不上: {0}")]
    EsTransport(String),
    #[error("响应解析失败: {0}")]
    Decode(String),
    #[error("内部错误: {0}")]
    Internal(String),
}

pub fn truncate_chars(text: &str, max_chars: usize) -> String {
    let count = text.chars().count();
    if count <= max_chars {
        return text.to_string();
    }
    let head: String = text.chars().take(max_chars).collect();
    format!("{head}...")
}

#[cfg(test)]
mod tests {
    use super::truncate_chars;

    #[test]
    fn truncate_keeps_short_text_and_marks_long_text() {
        assert_eq!(truncate_chars("abc", 3), "abc");
        assert_eq!(truncate_chars("abcd", 3), "abc...");
        assert_eq!(truncate_chars("外汇支付", 2), "外汇...");
    }
}
