# Elasticsearch 混合检索（BM25 + Dense Vector）

Rust 实现的混合检索服务。一次查询并行走关键词和向量两路，主路径用 Elasticsearch 的倒数排名融合（RRF）合成结果。BM25 分数会随查询词和文档长度甩到 0～20 以上，余弦相似度固定在 0～1，所以这里不把两路 `_score` 直接加权当作长期方案。

BM25 保住实体、专名和业务编码，例如 `SWIFT MT103`、`pacs.008`、`HT-2024-001`。向量处理同义改写，例如问「跨境汇款报文出错怎么办」，文档标题是「外汇支付报文异常处理」。两路都靠前的文档最终分最高，只在一路靠前的文档仍能进入结果。

## 分层

```
src/
├── api.rs            接入：/search、/docs、/index
├── search/           编排：DSL、版本选择、超时降级、应用层 RRF
├── embedding/        向量：HTTP（OpenAI / TEI）或本地 ONNX
├── index/            索引 mapping、写入、批量重建、模型混写检查
└── es/               Elasticsearch REST 客户端
```

检索编排只依赖 `EmbeddingService`。查询侧和入库侧使用同一个模型、同一套 L2 归一化。BGE 中文模型默认只在查询前加上「为这个句子生成表示以用于检索相关文章：」，文档原文不加这句。

## 版本

`RRF_SYNTAX=auto` 时按集群版本选语法：

| 集群 | 行为 |
| --- | --- |
| 8.8–8.15 | `sub_searches` + `rank.rrf` |
| 8.16 及以上 | `retriever.rrf`，公式相同 |
| 8.0–8.7 | 两路 Top-N 后在应用层做 RRF，响应里带上两路原始分和名次 |
| 7.x | 没有 dense_vector knn，默认退回 BM25 |

显式 `fusion=weighted` 是灰度对照，默认用 `query` boost + `knn` boost。方案里的 `function_score.knn` 不是合法的 function_score 函数，仍保留为 `WEIGHTED_SYNTAX=function_score`，只用于对照，不作为默认。

向量超时或失败时，混合查询退回 BM25，接口仍返回 200，`degraded=true`。显式 `fusion=knn` 失败则直接报错，避免效果评估被静默换成关键词结果。维度不一致不会降级。

## 运行

本机没有 IK 插件时，把分词器改成 `standard`。生产中文索引用 `ik_max_word`。

```bash
export ES_URL=http://127.0.0.1:9200
export ES_INDEX=hybrid_doc_index
export ANALYZER=standard
export EMBEDDING_ENDPOINT=http://127.0.0.1:8080/v1/embeddings
export EMBEDDING_MODEL=bge-small-zh-v1.5
export EMBEDDING_DIMS=512
export EMBEDDING_API_STYLE=openai
export INDEX_KEYWORDS=department

cargo run
```

内网推理节点和远程服务都走 HTTP，只改 `EMBEDDING_ENDPOINT`。进程内 BGE 是可选编译：

```bash
# Apple Silicon、Linux：首次运行会下载 ONNX 模型
EMBEDDING_PROVIDER=local cargo run --features local-onnx

# 系统已安装 libonnxruntime 时
EMBEDDING_PROVIDER=local cargo run --features local-onnx-dynamic
```

当前这台 x86_64 macOS 没有官方预编译 ONNX Runtime，`local-onnx` 在这里编不过。本地推理默认超时 1500ms；HTTP 向量默认 80ms。入库不走这个短超时。

```bash
curl -X PUT http://127.0.0.1:3000/index

curl -X POST http://127.0.0.1:3000/docs \
  -H 'content-type: application/json' \
  -d '{"id":"1","title":"外汇支付系统SWIFT报文处理流程","content":"外汇支付系统对接SWIFT，支持MT103、MT202报文收发","doc_type":"tech_doc","created_at":"2026-11-08"}'

curl -X POST http://127.0.0.1:3000/search \
  -H 'content-type: application/json' \
  -d '{"query":"跨境汇款报文出错怎么办","doc_type":"tech_doc","top_n":10}'
```

`fusion` 可取 `native_rrf`、`app_rrf`、`weighted`、`bm25`、`knn`。后两个用于同一批标注查询对比 P@10。

## 主要参数

| 变量 | 默认 | 作用 |
| --- | --- | --- |
| `RANK_CONSTANT` | 60 | RRF 平滑常数。调小会放大头部名次差 |
| `WINDOW_SIZE` | 100 | 参与融合的文档数，小于 `top_n` 时会抬到 `top_n` |
| `KNN_K` / `KNN_NUM_CANDIDATES` | 20 / 50 | 向量路条数；candidates 小于 k 时抬到 k |
| `TITLE_BOOST` | 3 | 标题字段权重 |
| `BM25_K1` / `BM25_B` | 1.2 / 0.6 | 索引里的 `biz_bm25` |
| `HNSW_M` / `HNSW_EF_CONSTRUCTION` | 16 / 100 | 向量图邻居数和建索引候选数 |
| `BM25_WEIGHT` / `KNN_WEIGHT` | 0.7 / 0.3 | 只在 `fusion=weighted` 时使用 |
| `BM25_FUZZINESS` | AUTO | 设为空字符串可关掉模糊匹配 |

过滤放进 `doc_type`、`created_from`、`created_to` 和 `terms`。`terms` 的字段必须先写进 `INDEX_KEYWORDS`，建索引时会变成 keyword。不要在结果集上再滤权限和时间。

换模型或维度后必须重建索引。服务会读取 mapping `_meta`，模型和维度对不上时拒绝写入。

100 万文档、512 维原始向量约 2 GB，HNSW 图还要另计内存。单分片控制在百万级以内。

原生 RRF 的 `score` 是融合分，不是 BM25 分，也不是余弦相似度。需要两路原始分时用 `fusion=app_rrf`，响应里有 `bm25_score`、`knn_score` 和对应名次。
