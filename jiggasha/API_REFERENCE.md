# Jiggasha API Reference

**Version:** 2026-06-03  
**Service:** Bengali dense passage retrieval over Qdrant (`bnwiki_chunks`).  
**Protocol:** HTTP / JSON (FastAPI).  
**Single-query only.** No authentication is required on the API surface (service runs inside a trusted network).

---

## Table of Contents

- [Base URL](#base-url)
- [GET /health](#get-health)
- [POST /search](#post-search)
  - [Request schema](#request-schema)
  - [Response schema](#response-schema)
  - [Result object](#result-object)
  - [Two operating modes](#two-operating-modes)
  - [Error responses](#error-responses)
- [cURL examples](#curl-examples)
- [Performance & latency](#performance--latency)
- [Config defaults that affect behavior](#config-defaults-that-affect-behavior)
- [Architecture cheat sheet](#architecture-cheat-sheet)

---

## Base URL

```
http://<host>:10000
```

Default port is `10000`.  Override with `--port` when starting `service.py`.

All endpoints return `Content-Type: application/json`.

---

## GET /health

Liveness / readiness probe.  Checks connectivity to Qdrant, the embedder, and the secondary LLM.

### Request

```http
GET /health HTTP/1.1
```

### Response (200 OK)

```json
{
  "qdrant": {"status": "ok", "points": 932788},
  "embedder": {"status": "ok"},
  "secondary_llm": {"status": "ok", "model": "qwen36"}
}
```

| Field | Description |
|-------|-------------|
| `qdrant.status` | `"ok"` or `"error"` |
| `qdrant.points` | Number of vectors in the configured collection |
| `embedder.status` | `"ok"` or `"error"` |
| `secondary_llm.status` | `"ok"` or `"unavailable"` |
| `secondary_llm.model` | Model name currently loaded |

If a backend is unhealthy, the corresponding block contains `"detail": "..."` with the exception text.  The HTTP status is still `200` so Kubernetes-style probes do not flap; inspect the JSON fields for actual health.

---

## POST /search

The single retrieval endpoint.  Accepts one Bengali query string and returns ranked passages.

### Request schema

```json
{
  "query": "string (required)",
  "top_k": 20,
  "retrieval_instruction": null,
  "use_instruction": null,
  "cosine_threshold": null,
  "rerank_threshold": null,
  "token_budget": null
}
```

| Field | Type | Default | Constraints | Description |
|-------|------|---------|-------------|-------------|
| `query` | `string` | — | Required. Max practical length ~2 k chars. | The Bengali (or English) user query. |
| `top_k` | `integer` | `20` | `≥ 1` | **Raw mode only:** how many results to fetch from Qdrant.  In instruction mode `fetch_k=50` is used internally; `top_k` is ignored unless `use_instruction=false`. |
| `retrieval_instruction` | `string \| null` | `null` | Max ~500 chars. | Caller-supplied English instruction prefixed to the query before embedding.  **Highest priority.** If provided, the secondary LLM instruction generator is skipped entirely. |
| `use_instruction` | `boolean \| null` | `null` | `true`, `false`, or `null` | `true` = force instruction mode ON. `false` = force raw cosine mode. `null` = follow `config.yml` default (`retrieval.use_instruction`). |
| `cosine_threshold` | `number \| null` | `null` | `0.0 – 1.0` | Minimum cosine score to keep after Qdrant retrieval.  `null` = use config default (`0.70`).  Only applies when instruction mode is ON. |
| `rerank_threshold` | `number \| null` | `null` | `0.0 – 1.0` | Minimum `rerank_score` (0–1) a passage must achieve to be returned.  `null` = use config default (`0.50`).  Only applies when instruction mode is ON. |
| `token_budget` | `integer \| null` | `null` | — | **Deprecated.**  No longer enforced.  Kept for backward compatibility. |

#### Notes on field interactions

- If `use_instruction` is `false`, the service runs in **raw cosine mode**: no instruction generation, no cosine threshold, no reranker.  It simply returns the top-`top_k` Qdrant results sorted by cosine similarity.
- If `use_instruction` is `true` (or `null` and config default is `true`), the full pipeline runs: instruction generation → embed → Qdrant → cosine threshold → LLM rerank → rerank threshold.
- `retrieval_instruction` overrides everything: if you provide it, instruction mode is effectively ON and the secondary LLM is bypassed for instruction generation.

---

### Response schema

```json
{
  "query": "string",
  "results": [ ... ],
  "hits_total": 0,
  "instruction": "string | null",
  "elapsed_ms": 1234,
  "timing_ms": {
    "instruction": 590,
    "embedding": 46,
    "qdrant": 40,
    "rerank": 557
  },
  "token_usage": {
    "instruction_prompt": 0,
    "instruction_completion": 0,
    "rerank_prompt": 8845,
    "rerank_completion": 6
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `query` | `string` | Echo of the input query. |
| `results` | `array` | Ordered list of passages.  See [Result object](#result-object) below. |
| `hits_total` | `integer` | Length of `results`. |
| `instruction` | `string \| null` | The instruction that was prefixed to the query before embedding.  `null` in raw mode. |
| `elapsed_ms` | `integer` | Wall-clock time for the entire request (ms). |
| `timing_ms` | `object` | Breakdown of latency per stage.  Keys may be missing if a stage was skipped (e.g. `rerank` is absent in raw mode). |
| `timing_ms.instruction` | `integer` | Time spent generating the retrieval instruction (ms). `0` if caller-supplied or skipped. |
| `timing_ms.embedding` | `integer` | Time spent calling the embedder (ms). |
| `timing_ms.qdrant` | `integer` | Time spent in Qdrant vector search (ms). |
| `timing_ms.rerank` | `integer` | Time spent in the LLM reranker (ms). `0` if skipped. |
| `token_usage` | `object` | Token counters from the secondary LLM.  **Not** the embedder. |
| `token_usage.instruction_prompt` | `integer` | Prompt tokens for instruction generation.  Currently `0` (not instrumented in the upstream library). |
| `token_usage.instruction_completion` | `integer` | Completion tokens for instruction generation.  Currently `0`. |
| `token_usage.rerank_prompt` | `integer` | Total prompt tokens consumed by the reranker across all parallel calls. |
| `token_usage.rerank_completion` | `integer` | Total completion tokens consumed by the reranker (one token per passage). |

---

### Result object

Each item in `results`:

```json
{
  "passage_id": 1898059397,
  "text": "স্মার্ট কার্ড ও জাতীয়পরিচয়পত্র > স্মার্ট কার্ড > নতুন এনআইডি কার্ড চেক করার ধাপ\n\nনতুন এনআইডি কার্ড বা স্মার্ট কার্ড চেক করার জন্য ...",
  "category": "স্মার্ট কার্ড ও জাতীয়পরিচয়পত্র",
  "sub_category": "স্মার্ট কার্ড",
  "service": "নতুন এনআইডি কার্ড চেক করার ধাপ",
  "topic": "স্মার্ট কার্ড ও জাতীয়পরিচয়পত্র",
  "chunk_type": "govt_service",
  "llm_token_count": 1167,
  "score": 0.7930736,
  "rerank_score": 1.0
}
```

| Field | Type | Description |
|-------|------|-------------|
| `passage_id` | `integer` | Stable ID.  Derived from payload `passage_id` or a hash of the Qdrant point ID. |
| `text` | `string` | Full passage text.  Always starts with a breadcrumb line `Category > Sub-category > Service` when available. |
| `category` | `string` | Top-level category (e.g. wiki page title or government service category). |
| `sub_category` | `string` | Section or sub-category. |
| `service` | `string` | Specific service name when applicable. |
| `topic` | `string` | Topic label (often same as `category`). |
| `chunk_type` | `string` | `"wiki"` or `"govt_service"`.  **Informational only** — the API no longer filters by this field. |
| `llm_token_count` | `integer` | Pre-computed token count of the passage (for downstream LLM context budgeting). |
| `score` | `number` | Original Qdrant cosine similarity score.  Range roughly `0.4 – 0.95`. |
| `rerank_score` | `number` | LLM-based relevance score **0.0 – 1.0**.  Only present in instruction mode.  Higher is better. |

**Ordering:** In instruction mode, results are sorted by `rerank_score` descending.  In raw mode, they are sorted by `score` (cosine) descending.

---

## Two operating modes

### Mode A — Instruction-based retrieval (recommended)

Triggered when `use_instruction=true` (or `null` with config default `true`).

Pipeline:
1. **Instruction generation** — dynamic English instruction produced by `qwen36` (or caller-supplied / static / fallback).
2. **Embedding** — `instruction + "\n" + query` is embedded by `Qwen3-Embedding-8B` (4096-dim).
3. **Qdrant search** — fetch top `50` candidates by cosine similarity.
4. **Cosine threshold** — drop anything below `cosine_threshold` (default `0.70`).  If nothing passes, fall back to top-3 raw.
5. **LLM rerank** — `qwen36` scores each passage `0–10` in parallel (`asyncio.gather`).  Normalized to `rerank_score = score / 10.0`.
6. **Rerank threshold** — drop anything below `rerank_threshold` (default `0.50`).
7. **Return** — sorted by `rerank_score` descending.

Typical latency: **500 ms – 2 s** depending on how many passages pass the cosine threshold (reranker processes them in parallel).

### Mode B — Raw cosine retrieval

Triggered when `use_instruction=false`.

Pipeline:
1. **Embedding** — raw query text only.
2. **Qdrant search** — fetch top `top_k` (default `20`) by cosine similarity.
3. **Return** — sorted by `score` descending.  No reranker, no threshold, no instruction.

Typical latency: **30 – 100 ms**.

Use raw mode when:
- You want maximum speed.
- You do your own reranking downstream.
- The query is very simple and cosine similarity is sufficient.

---

## Error responses

| HTTP status | `detail` | When |
|-------------|----------|------|
| `400` | `` `query` must be provided. `` | `POST /search` missing `query`. |
| `503` | `Embedder not initialised` | Service startup failed or embedder unreachable. |
| `503` | `Embedding failed: ...` | Embedder HTTP error (timeout, 5xx, etc.). |
| `503` | `Qdrant search failed: ...` | Qdrant client error (timeout, collection missing, etc.). |

FastAPI validation errors (malformed JSON, wrong types) return `422` with a Pydantic error body.

---

## cURL examples

### 1. Basic instruction-based search

```bash
curl -X POST http://localhost:10000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "এনআইডি কার্ডের স্ট্যাটাস কিভাবে জানব?"
  }'
```

### 2. Raw cosine search (fast, no rerank)

```bash
curl -X POST http://localhost:10000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "এনআইডি কার্ডের স্ট্যাটাস কিভাবে জানব?",
    "use_instruction": false,
    "top_k": 10
  }'
```

### 3. Strict filtering — high rerank threshold

```bash
curl -X POST http://localhost:10000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "পাসপোর্ট করার নিয়ম কি?",
    "use_instruction": true,
    "cosine_threshold": 0.70,
    "rerank_threshold": 0.90
  }'
```

### 4. Caller-supplied instruction (bypasses LLM instruction generator)

```bash
curl -X POST http://localhost:10000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "জন্ম সনদ পাওয়ার প্রক্রিয়া?",
    "retrieval_instruction": "Find official government procedures for obtaining a birth certificate.",
    "use_instruction": true
  }'
```

### 5. Health check

```bash
curl http://localhost:10000/health
```

---

## Performance & latency

| Stage | Typical latency | Notes |
|-------|-----------------|-------|
| Instruction generation | 300 – 800 ms | One LLM call. Cached for repeated queries. |
| Embedding | 30 – 100 ms | HTTP call to vLLM embedding endpoint. |
| Qdrant search | 20 – 80 ms | Local network call. |
| Reranker | 300 – 1200 ms | Parallel LLM calls.  Scales with number of passages that pass cosine threshold (usually 3–15). |
| **Total (instruction mode)** | **500 ms – 2 s** | |
| **Total (raw mode)** | **30 – 100 ms** | |

**Reranker throughput:** All passage evaluations are fired concurrently via `asyncio.gather`.  vLLM prefix caching means the identical system prompt is processed once; only the user message varies per passage.

**Token usage (reranker):** ~1,500–2,000 prompt tokens per passage × number of passages.  Completion tokens = 1 per passage (single integer output).

---

## Config defaults that affect behavior

These live in `config.yml` and can only be changed by restarting the service.

| Config key | Default | Meaning |
|------------|---------|---------|
| `retrieval.use_instruction` | `true` | Whether instruction mode is ON when `req.use_instruction` is `null`. |
| `retrieval.cosine_threshold` | `0.70` | Default cosine cutoff when `req.cosine_threshold` is `null`. |
| `retrieval.top_k_fetch` | `50` | How many candidates Qdrant returns **before** threshold / rerank. |
| `retrieval.fallback_instruction` | `"Retrieve passages that are directly relevant..."` | Used when the dynamic instruction LLM call fails or times out. |
| `qdrant.collection` | `bnwiki_chunks` | Qdrant collection name. |

The `rerank_threshold` default (`0.50`) is currently hard-coded in `service.py` (not in `config.yml`).  You can override it per-request via `req.rerank_threshold`.

---

## Architecture cheat sheet

```
Client
  │ POST /search {query, ...}
  ▼
┌─────────────────────────────────────────────┐
│  1. Instruction Builder                     │
│     • caller-supplied  → use directly       │
│     • static config    → zero latency       │
│     • dynamic LLM      → qwen36, ~600 ms    │
│     • fallback         → if LLM fails       │
├─────────────────────────────────────────────┤
│  2. Embedder (Qwen3-Embedding-8B)           │
│     • instruction + query → 4096-dim vector │
├─────────────────────────────────────────────┤
│  3. Qdrant (bnwiki_chunks, 932k points)     │
│     • cosine similarity, fetch top 50       │
├─────────────────────────────────────────────┤
│  4. Cosine Threshold Filter                 │
│     • drop score < 0.70                     │
│     • fallback to top-3 raw if none pass    │
├─────────────────────────────────────────────┤
│  5. LLM Reranker (reranker.py)              │
│     • qwen36 scores each passage 0–10       │
│     • parallel asyncio.gather               │
│     • normalized to rerank_score 0.0–1.0    │
├─────────────────────────────────────────────┤
│  6. Rerank Threshold Filter                 │
│     • drop rerank_score < 0.50              │
│     • sort descending by rerank_score       │
└─────────────────────────────────────────────┘
  │ JSON response
  ▼
Client
```

**Key files:**
- `service.py` — FastAPI app, orchestration, request/response models
- `reranker.py` — LLM cross-encoder reranker with 0–10 scoring
- `instruction.py` — Dynamic instruction generation (cached, timeout-guarded)
- `embedder.py` — vLLM embedding client (Qwen3-Embedding-8B)
- `config.yml` — Service configuration
