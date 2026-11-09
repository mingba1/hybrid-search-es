use std::collections::HashMap;

use serde_json::Value;

use crate::error::AppError;
use crate::model::SearchHit;

#[derive(Debug, Clone)]
pub struct LegHit {
    pub id: String,
    pub rank: usize,
    pub score: f64,
    pub title: String,
    pub content: String,
    pub doc_type: String,
}

/// `score = Σ 1 / (rank_constant + rank)`，rank 从 1 开始。只使用名次，不使用原始分数。
pub fn reciprocal_rank_fusion(
    bm25: &[LegHit],
    knn: &[LegHit],
    rank_constant: f64,
    window_size: usize,
) -> Vec<SearchHit> {
    let mut rows: HashMap<String, Acc> = HashMap::new();
    add_leg(&mut rows, bm25, rank_constant, window_size, true);
    add_leg(&mut rows, knn, rank_constant, window_size, false);
    let mut hits: Vec<SearchHit> = rows
        .into_values()
        .map(|acc| SearchHit {
            id: acc.id,
            score: acc.score,
            title: acc.title,
            content: acc.content,
            doc_type: acc.doc_type,
            bm25_score: acc.bm25_score,
            knn_score: acc.knn_score,
            bm25_rank: acc.bm25_rank,
            knn_rank: acc.knn_rank,
        })
        .collect();
    hits.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.id.cmp(&right.id))
    });
    hits
}

struct Acc {
    id: String,
    score: f64,
    title: String,
    content: String,
    doc_type: String,
    bm25_score: Option<f64>,
    knn_score: Option<f64>,
    bm25_rank: Option<usize>,
    knn_rank: Option<usize>,
}

fn add_leg(
    rows: &mut HashMap<String, Acc>,
    hits: &[LegHit],
    rank_constant: f64,
    window_size: usize,
    bm25: bool,
) {
    for hit in hits {
        if hit.rank == 0 || hit.rank > window_size {
            continue;
        }
        let entry = rows.entry(hit.id.clone()).or_insert_with(|| Acc {
            id: hit.id.clone(),
            score: 0.0,
            title: hit.title.clone(),
            content: hit.content.clone(),
            doc_type: hit.doc_type.clone(),
            bm25_score: None,
            knn_score: None,
            bm25_rank: None,
            knn_rank: None,
        });
        if entry.title.is_empty() {
            entry.title = hit.title.clone();
            entry.content = hit.content.clone();
            entry.doc_type = hit.doc_type.clone();
        }
        entry.score += 1.0 / (rank_constant + hit.rank as f64);
        if bm25 {
            entry.bm25_score = Some(hit.score);
            entry.bm25_rank = Some(hit.rank);
        } else {
            entry.knn_score = Some(hit.score);
            entry.knn_rank = Some(hit.rank);
        }
    }
}

pub fn es_hits(body: &Value) -> Result<Vec<LegHit>, AppError> {
    let hits = body
        .pointer("/hits/hits")
        .and_then(Value::as_array)
        .ok_or_else(|| AppError::Decode("Elasticsearch 响应缺少 hits.hits".into()))?;
    let mut parsed = Vec::with_capacity(hits.len());
    for (offset, hit) in hits.iter().enumerate() {
        let id = hit
            .get("_id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
            .ok_or_else(|| AppError::Decode("命中缺少 _id".into()))?
            .to_string();
        let source = hit.get("_source").cloned().unwrap_or(Value::Null);
        parsed.push(LegHit {
            id,
            rank: offset + 1,
            score: hit.get("_score").and_then(Value::as_f64).unwrap_or(0.0),
            title: source_text(&source, "title"),
            content: source_text(&source, "content"),
            doc_type: source_text(&source, "doc_type"),
        });
    }
    Ok(parsed)
}

pub fn es_took_ms(body: &Value) -> Option<u64> {
    body.get("took").and_then(Value::as_u64)
}

fn source_text(source: &Value, field: &str) -> String {
    source
        .get(field)
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string()
}

pub fn leg_to_hit(hit: &LegHit) -> SearchHit {
    SearchHit {
        id: hit.id.clone(),
        score: hit.score,
        title: hit.title.clone(),
        content: hit.content.clone(),
        doc_type: hit.doc_type.clone(),
        bm25_score: None,
        knn_score: None,
        bm25_rank: None,
        knn_rank: None,
    }
}

#[cfg(test)]
mod tests {
    use super::{es_hits, reciprocal_rank_fusion, LegHit};
    use serde_json::json;

    fn hit(id: &str, rank: usize, score: f64) -> LegHit {
        LegHit {
            id: id.into(),
            rank,
            score,
            title: id.into(),
            content: String::new(),
            doc_type: "tech_doc".into(),
        }
    }

    #[test]
    fn documents_in_both_lists_outrank_single_list_hits() {
        let bm25 = vec![hit("a", 1, 12.0), hit("b", 2, 8.0)];
        let knn = vec![hit("a", 1, 0.9), hit("c", 2, 0.4)];
        let fused = reciprocal_rank_fusion(&bm25, &knn, 60.0, 100);
        let ids: Vec<_> = fused.iter().map(|hit| hit.id.as_str()).collect();
        assert_eq!(ids, vec!["a", "b", "c"]);
        assert!((fused[0].score - (2.0 / 61.0)).abs() < 1e-12);
        assert!((fused[1].score - (1.0 / 62.0)).abs() < 1e-12);
        assert_eq!(fused[0].bm25_rank, Some(1));
        assert_eq!(fused[0].knn_rank, Some(1));
        assert_eq!(fused[1].knn_rank, None);
        assert_eq!(fused[2].bm25_rank, None);
    }

    #[test]
    fn window_drops_ranks_beyond_the_cutoff() {
        let bm25 = vec![hit("a", 1, 1.0), hit("b", 2, 1.0)];
        let knn = vec![hit("a", 1, 1.0), hit("c", 2, 1.0)];
        let fused = reciprocal_rank_fusion(&bm25, &knn, 60.0, 1);
        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].id, "a");
    }

    #[test]
    fn parses_es_hits_into_one_based_ranks() {
        let body = json!({
            "took": 4,
            "hits": {
                "hits": [
                    { "_id": "1", "_score": 11.5, "_source": { "title": "外汇", "content": "MT103", "doc_type": "tech_doc" } },
                    { "_id": "2", "_score": null, "_source": { "title": "清算" } }
                ]
            }
        });
        let hits = es_hits(&body).unwrap();
        assert_eq!(hits[0].rank, 1);
        assert_eq!(hits[0].title, "外汇");
        assert!((hits[0].score - 11.5).abs() < 1e-9);
        assert_eq!(hits[1].rank, 2);
        assert_eq!(hits[1].score, 0.0);
    }
}
