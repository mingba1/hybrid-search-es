use std::collections::HashMap;

use chrono::NaiveDate;
use serde_json::{json, Map, Value};

use crate::error::AppError;
use crate::fields::{is_keyword_name, is_reserved_field};

const SOURCE_FIELDS: &[&str] = &["title", "content", "doc_type"];

#[derive(Debug, Clone)]
pub struct LexicalSpec {
    pub text: String,
    pub filters: Vec<Value>,
    pub title_boost: f64,
    pub fuzziness: Option<String>,
    pub boost: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct VectorSpec {
    pub vector: Vec<f32>,
    pub filters: Vec<Value>,
    pub k: i64,
    pub num_candidates: i64,
    pub boost: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct RrfParams {
    pub rank_constant: i64,
    pub window_size: i64,
    pub size: i64,
}

pub fn compile_filters(
    doc_type: Option<&str>,
    created_from: Option<NaiveDate>,
    created_to: Option<NaiveDate>,
    terms: &HashMap<String, String>,
    keyword_fields: &[String],
) -> Result<Vec<Value>, AppError> {
    if let (Some(from), Some(to)) = (created_from, created_to) {
        if from > to {
            return Err(AppError::BadRequest(
                "created_from 不能晚于 created_to".into(),
            ));
        }
    }
    let mut filters = Vec::new();
    if let Some(doc_type) = doc_type.map(str::trim).filter(|value| !value.is_empty()) {
        filters.push(json!({ "term": { "doc_type": doc_type } }));
    }
    if created_from.is_some() || created_to.is_some() {
        let mut range = Map::new();
        if let Some(from) = created_from {
            range.insert("gte".into(), json!(from.to_string()));
        }
        if let Some(to) = created_to {
            range.insert("lte".into(), json!(to.to_string()));
        }
        filters.push(json!({ "range": { "created_at": Value::Object(range) } }));
    }
    let mut names: Vec<&String> = terms.keys().collect();
    names.sort();
    for name in names {
        let value = terms.get(name).map(String::as_str).unwrap_or("").trim();
        if value.is_empty() {
            return Err(AppError::BadRequest(format!("{name} 的过滤值不能为空")));
        }
        if is_reserved_field(name) {
            return Err(AppError::BadRequest(format!(
                "{name} 请使用对应的请求字段，不要放进 terms"
            )));
        }
        if !is_keyword_name(name) || !keyword_fields.iter().any(|field| field == name) {
            return Err(AppError::BadRequest(format!(
                "字段 {name} 未在 INDEX_KEYWORDS 中声明，过滤不能后置到结果集"
            )));
        }
        filters.push(json!({ "term": { name: value } }));
    }
    Ok(filters)
}

pub fn bm25_body(lexical: &LexicalSpec, size: i64) -> Value {
    json!({
        "size": size,
        "query": bool_query(lexical),
        "_source": SOURCE_FIELDS
    })
}

pub fn knn_body(vector: &VectorSpec, size: i64) -> Value {
    json!({
        "size": size,
        "knn": knn_object(vector),
        "_source": SOURCE_FIELDS
    })
}

/// ES 8.8–8.15：`sub_searches` + `rank.rrf`。
pub fn native_sub_searches(lexical: &LexicalSpec, vector: &VectorSpec, rrf: &RrfParams) -> Value {
    json!({
        "sub_searches": [
            { "query": bool_query(lexical) },
            { "knn": knn_object(vector) }
        ],
        "rank": {
            "rrf": {
                "rank_constant": rrf.rank_constant,
                "window_size": rrf.window_size
            }
        },
        "size": rrf.size,
        "_source": SOURCE_FIELDS
    })
}

/// ES 8.16+：稳定的 retriever RRF，公式与 sub_searches 相同。
pub fn native_retriever(lexical: &LexicalSpec, vector: &VectorSpec, rrf: &RrfParams) -> Value {
    json!({
        "retriever": {
            "rrf": {
                "retrievers": [
                    { "standard": { "query": bool_query(lexical) } },
                    { "knn": knn_object(vector) }
                ],
                "rank_constant": rrf.rank_constant,
                "window_size": rrf.window_size
            }
        },
        "size": rrf.size,
        "_source": SOURCE_FIELDS
    })
}

/// 可执行的加权对照。boost 只改变两路的相对倍率，分数量纲仍然不同。
pub fn weighted_knn_body(lexical: &LexicalSpec, vector: &VectorSpec, size: i64) -> Value {
    json!({
        "size": size,
        "query": bool_query(lexical),
        "knn": knn_object(vector),
        "_source": SOURCE_FIELDS
    })
}

/// 方案文档中的 function_score 体。Elasticsearch 的 function_score 不接受 knn 函数，
/// 保留这个构造器是为了对照实验，线上默认走 [`weighted_knn_body`]。
pub fn weighted_function_score_body(
    lexical: &LexicalSpec,
    vector: &VectorSpec,
    bm25_weight: f64,
    knn_weight: f64,
    size: i64,
) -> Value {
    let mut lexical = lexical.clone();
    lexical.boost = None;
    let mut knn = knn_object(vector);
    if let Some(object) = knn.as_object_mut() {
        object.remove("boost");
    }
    json!({
        "size": size,
        "_source": SOURCE_FIELDS,
        "query": {
            "function_score": {
                "query": bool_query(&lexical),
                "functions": [
                    { "knn": knn, "weight": knn_weight }
                ],
                "boost": bm25_weight,
                "score_mode": "sum"
            }
        }
    })
}

pub fn boosted_field(name: &str, boost: f64) -> String {
    if (boost - boost.round()).abs() < 1e-9 {
        format!("{name}^{}", boost.round() as i64)
    } else {
        format!("{name}^{boost}")
    }
}

fn bool_query(spec: &LexicalSpec) -> Value {
    let mut query = Map::new();
    query.insert("must".into(), json!([multi_match(spec)]));
    if !spec.filters.is_empty() {
        query.insert("filter".into(), json!(spec.filters));
    }
    json!({ "bool": Value::Object(query) })
}

fn multi_match(spec: &LexicalSpec) -> Value {
    let mut matching = Map::new();
    matching.insert("query".into(), json!(spec.text));
    matching.insert(
        "fields".into(),
        json!([boosted_field("title", spec.title_boost), "content"]),
    );
    if let Some(fuzziness) = spec.fuzziness.as_deref().filter(|value| !value.is_empty()) {
        matching.insert("fuzziness".into(), json!(fuzziness));
    }
    if let Some(boost) = spec.boost {
        matching.insert("boost".into(), json!(boost));
    }
    json!({ "multi_match": Value::Object(matching) })
}

fn knn_object(spec: &VectorSpec) -> Value {
    let mut knn = Map::new();
    knn.insert("field".into(), json!("embedding"));
    knn.insert("query_vector".into(), json!(spec.vector));
    knn.insert("k".into(), json!(spec.k));
    knn.insert("num_candidates".into(), json!(spec.num_candidates));
    if let Some(filter) = filter_query(&spec.filters) {
        knn.insert("filter".into(), filter);
    }
    if let Some(boost) = spec.boost {
        knn.insert("boost".into(), json!(boost));
    }
    Value::Object(knn)
}

fn filter_query(filters: &[Value]) -> Option<Value> {
    match filters {
        [] => None,
        [one] => Some(one.clone()),
        many => Some(json!({ "bool": { "filter": many } })),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use chrono::NaiveDate;
    use serde_json::json;

    use super::{
        boosted_field, compile_filters, native_retriever, native_sub_searches,
        weighted_function_score_body, weighted_knn_body, LexicalSpec, RrfParams, VectorSpec,
    };

    fn lexical(filters: Vec<serde_json::Value>) -> LexicalSpec {
        LexicalSpec {
            text: "SWIFT报文 外汇支付清算".into(),
            filters,
            title_boost: 3.0,
            fuzziness: Some("AUTO".into()),
            boost: None,
        }
    }

    fn vector(filters: Vec<serde_json::Value>) -> VectorSpec {
        VectorSpec {
            vector: vec![0.021, -0.033],
            filters,
            k: 20,
            num_candidates: 50,
            boost: None,
        }
    }

    fn rrf() -> RrfParams {
        RrfParams {
            rank_constant: 60,
            window_size: 100,
            size: 10,
        }
    }

    #[test]
    fn sub_searches_body_matches_the_primary_dsl() {
        let filters = vec![json!({ "term": { "doc_type": "tech_doc" } })];
        let body = native_sub_searches(&lexical(filters.clone()), &vector(filters), &rrf());
        assert_eq!(body["size"], 10);
        assert_eq!(body["rank"]["rrf"]["rank_constant"], 60);
        assert_eq!(body["rank"]["rrf"]["window_size"], 100);
        assert_eq!(body["_source"], json!(["title", "content", "doc_type"]));
        let subs = body["sub_searches"].as_array().unwrap();
        assert_eq!(subs.len(), 2);
        let matching = &subs[0]["query"]["bool"]["must"][0]["multi_match"];
        assert_eq!(matching["query"], "SWIFT报文 外汇支付清算");
        assert_eq!(matching["fields"][0], "title^3");
        assert_eq!(matching["fields"][1], "content");
        assert_eq!(matching["fuzziness"], "AUTO");
        assert_eq!(
            subs[0]["query"]["bool"]["filter"][0]["term"]["doc_type"],
            "tech_doc"
        );
        assert_eq!(subs[1]["knn"]["field"], "embedding");
        assert_eq!(subs[1]["knn"]["k"], 20);
        assert_eq!(subs[1]["knn"]["num_candidates"], 50);
        let query_vector = subs[1]["knn"]["query_vector"].as_array().unwrap();
        assert!((query_vector[0].as_f64().unwrap() - 0.021).abs() < 1e-6);
        assert!((query_vector[1].as_f64().unwrap() - (-0.033)).abs() < 1e-6);
        assert_eq!(subs[1]["knn"]["filter"]["term"]["doc_type"], "tech_doc");
    }

    #[test]
    fn retriever_body_uses_the_same_rank_constant() {
        let body = native_retriever(&lexical(Vec::new()), &vector(Vec::new()), &rrf());
        assert_eq!(body["retriever"]["rrf"]["rank_constant"], 60);
        assert_eq!(body["retriever"]["rrf"]["window_size"], 100);
        assert!(body["retriever"]["rrf"]["retrievers"][0]["standard"]["query"].is_object());
        assert_eq!(body["retriever"]["rrf"]["retrievers"][1]["knn"]["k"], 20);
        assert!(body["retriever"]["rrf"]["retrievers"][1]["knn"]
            .get("filter")
            .is_none());
    }

    #[test]
    fn weighted_bodies_keep_both_scores_addressable() {
        let mut lexical = lexical(Vec::new());
        lexical.boost = Some(0.7);
        let mut vector = vector(Vec::new());
        vector.boost = Some(0.3);
        let knn = weighted_knn_body(&lexical, &vector, 10);
        assert_eq!(
            knn["query"]["bool"]["must"][0]["multi_match"]["boost"],
            json!(0.7)
        );
        assert_eq!(knn["knn"]["boost"], json!(0.3));

        let scored = weighted_function_score_body(&lexical, &vector, 0.7, 0.3, 10);
        assert_eq!(scored["query"]["function_score"]["boost"], json!(0.7));
        assert_eq!(scored["query"]["function_score"]["score_mode"], "sum");
        assert_eq!(
            scored["query"]["function_score"]["functions"][0]["weight"],
            json!(0.3)
        );
        assert!(scored["query"]["function_score"]["functions"][0]["knn"]
            .get("boost")
            .is_none());
    }

    #[test]
    fn multiple_filters_wrap_knn_in_bool() {
        let filters = compile_filters(
            Some("tech_doc"),
            Some(NaiveDate::from_ymd_opt(2026, 1, 1).unwrap()),
            Some(NaiveDate::from_ymd_opt(2026, 11, 8).unwrap()),
            &HashMap::from([("department".into(), "fx".into())]),
            &["department".into()],
        )
        .unwrap();
        assert_eq!(filters.len(), 3);
        let body = native_sub_searches(&lexical(filters.clone()), &vector(filters), &rrf());
        assert_eq!(
            body["sub_searches"][1]["knn"]["filter"]["bool"]["filter"][2]["term"]["department"],
            "fx"
        );
    }

    #[test]
    fn rejects_unknown_term_and_reversed_dates() {
        let err = compile_filters(
            None,
            None,
            None,
            &HashMap::from([("department".into(), "fx".into())]),
            &[],
        );
        assert!(err.is_err());
        let dates = compile_filters(
            None,
            Some(NaiveDate::from_ymd_opt(2026, 12, 1).unwrap()),
            Some(NaiveDate::from_ymd_opt(2026, 1, 1).unwrap()),
            &HashMap::new(),
            &[],
        );
        assert!(dates.is_err());
    }

    #[test]
    fn integer_boost_drops_the_decimal() {
        assert_eq!(boosted_field("title", 3.0), "title^3");
        assert_eq!(boosted_field("title", 2.5), "title^2.5");
    }
}
