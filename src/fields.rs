/// 额外 keyword 字段名。只接受标识符，避免把查询拼进任意路径。
pub fn is_keyword_name(name: &str) -> bool {
    let mut chars = name.chars();
    match chars.next() {
        Some(first) if first.is_ascii_alphabetic() || first == '_' => {}
        _ => return false,
    }
    if name.len() > 64 {
        return false;
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

pub fn is_reserved_field(name: &str) -> bool {
    matches!(
        name,
        "doc_type" | "created_at" | "title" | "content" | "embedding" | "embedding_model"
    )
}

#[cfg(test)]
mod tests {
    use super::{is_keyword_name, is_reserved_field};

    #[test]
    fn accepts_identifier_and_rejects_paths() {
        assert!(is_keyword_name("department"));
        assert!(is_keyword_name("acl_group"));
        assert!(!is_keyword_name("a.b"));
        assert!(!is_keyword_name("1abc"));
        assert!(!is_keyword_name(""));
        assert!(is_reserved_field("embedding"));
    }
}
