use serde_json::{json, Value};

use crate::config::IndexSettings;

/// 原始 float32 向量占用的字节数，不含 HNSW 图。m=16 时图的内存通常再增加同量级。
pub fn dense_vector_payload_bytes(doc_count: u64, dims: u64) -> u64 {
    doc_count
        .saturating_mul(dims)
        .saturating_mul(std::mem::size_of::<f32>() as u64)
}

pub fn index_definition(settings: &IndexSettings) -> Value {
    let mut body = json!({
        "settings": {
            "number_of_shards": settings.shards,
            "number_of_replicas": settings.replicas,
            "index": {
                "similarity": {
                    "biz_bm25": {
                        "type": "BM25",
                        "k1": settings.bm25_k1,
                        "b": settings.bm25_b
                    }
                }
            }
        },
        "mappings": {
            "dynamic": "strict",
            "_meta": {
                "embedding_model": settings.embedding_model,
                "embedding_dims": settings.dims,
                "normalized": "l2",
                "similarity": "cosine"
            },
            "properties": {
                "title": {
                    "type": "text",
                    "similarity": "biz_bm25",
                    "analyzer": settings.analyzer
                },
                "content": {
                    "type": "text",
                    "similarity": "biz_bm25",
                    "analyzer": settings.analyzer
                },
                "doc_type": { "type": "keyword" },
                "created_at": { "type": "date" },
                "embedding_model": { "type": "keyword" },
                "embedding": {
                    "type": "dense_vector",
                    "dims": settings.dims,
                    "index": true,
                    "similarity": "cosine",
                    "index_options": {
                        "type": "hnsw",
                        "m": settings.hnsw_m,
                        "ef_construction": settings.hnsw_ef_construction
                    }
                }
            }
        }
    });
    if let Some(properties) = body
        .pointer_mut("/mappings/properties")
        .and_then(Value::as_object_mut)
    {
        for field in &settings.keyword_fields {
            properties.insert(field.clone(), json!({ "type": "keyword" }));
        }
    }
    body
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddingMeta {
    pub model: String,
    pub dims: usize,
}

pub fn extract_embedding_meta(mapping: &Value) -> Option<EmbeddingMeta> {
    let meta = mapping
        .as_object()?
        .values()
        .find_map(|value| value.pointer("/mappings/_meta"))?;
    let model = meta.get("embedding_model")?.as_str()?.to_string();
    let dims = meta.get("embedding_dims")?.as_u64()? as usize;
    Some(EmbeddingMeta { model, dims })
}

#[cfg(test)]
mod tests {
    use super::{dense_vector_payload_bytes, extract_embedding_meta, index_definition};
    use crate::config::IndexSettings;

    #[test]
    fn mapping_keeps_bm25_ik_and_hnsw() {
        let settings = IndexSettings {
            keyword_fields: vec!["department".into()],
            ..IndexSettings::default()
        };
        let body = index_definition(&settings);
        assert_eq!(body["settings"]["number_of_shards"], 1);
        assert_eq!(body["settings"]["number_of_replicas"], 0);
        let similarity = &body["settings"]["index"]["similarity"]["biz_bm25"];
        assert_eq!(similarity["type"], "BM25");
        assert!((similarity["k1"].as_f64().unwrap() - 1.2).abs() < 1e-9);
        assert!((similarity["b"].as_f64().unwrap() - 0.6).abs() < 1e-9);
        let title = &body["mappings"]["properties"]["title"];
        assert_eq!(title["analyzer"], "ik_max_word");
        assert_eq!(title["similarity"], "biz_bm25");
        let embedding = &body["mappings"]["properties"]["embedding"];
        assert_eq!(embedding["type"], "dense_vector");
        assert_eq!(embedding["dims"], 512);
        assert_eq!(embedding["index"], true);
        assert_eq!(embedding["similarity"], "cosine");
        assert_eq!(embedding["index_options"]["type"], "hnsw");
        assert_eq!(embedding["index_options"]["m"], 16);
        assert_eq!(embedding["index_options"]["ef_construction"], 100);
        assert_eq!(
            body["mappings"]["_meta"]["embedding_model"],
            "bge-small-zh-v1.5"
        );
        assert_eq!(
            body["mappings"]["properties"]["department"]["type"],
            "keyword"
        );
        assert_eq!(dense_vector_payload_bytes(1_000_000, 512), 2_048_000_000);
    }

    #[test]
    fn reads_meta_from_mapping_response() {
        let body = index_definition(&IndexSettings::default());
        let wrapped =
            serde_json::json!({ "hybrid_doc_index": { "mappings": body["mappings"].clone() } });
        let meta = extract_embedding_meta(&wrapped).unwrap();
        assert_eq!(meta.model, "bge-small-zh-v1.5");
        assert_eq!(meta.dims, 512);
    }
}
