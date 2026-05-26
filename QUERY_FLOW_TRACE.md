# Step-by-Step Query Flow Trace

> **System:** Unified Gov + Wiki Retrieval  
> **Qdrant Collection:** `bnwiki_chunks` (931,431 `wiki` + 1,357 `govt_service`)  
> **Jiggasha Endpoint:** `http://localhost:10000/search`  
> **Embedder:** `Qwen/Qwen3-Embedding-8B` (4096-dim)  
> **LLMs:** Primary + Secondary `qwen36`  
> **Last Updated:** 2026-05-24

---

## Overview

This document traces a single user query from the HTTP boundary all the way down to the Qdrant vector search and back up to the streamed response. Two example queries are used:

- **Example A (Govt):** `এনআইডি কার্ডের স্ট্যাটাস কিভাবে জানব?`
- **Example B (Wiki):** `বাংলাদেশের স্বাধীনতা যুদ্ধ কবে শুরু হয়?`
- **Example C (Mixed):** `বাংলাদেশের প্রধানমন্ত্রী কে?`

---

## Phase 0: HTTP Ingestion

### Step 1: User sends query

```http
POST /chat/stream HTTP/1.1
Host: localhost:9000
Content-Type: application/json

{
  "query": "এনআইডি কার্ডের স্ট্যাটাস কিভাবে জানব?",
  "session_id": "sess_abc123",
  "history": []
}
```

### Step 2: FastAPI receives request

`api.py` → `ChatRequest` Pydantic model validates the payload.

### Step 3: Session / history load

`api.py` loads prior conversation turns from Redis (`redis://localhost:6379/0`) keyed by `session_id`.

---

## Phase 1: Orchestrator Entry (`cogops/agents/orchestrator.py`)

### Step 4: `Orchestrator.process_query()` begins

```python
turn_id = str(uuid.uuid4())[:8]   # e.g., "a3f7b2d9"
```

### Step 5: Stage 0 — Sanitize (`cogops/pipeline/sanitize.py`)

```python
clean = unicodedata.normalize("NFC", query).strip()
```

- Applies **Unicode NFC normalization** (critical for Bengali: U+09AF+U+09BC → U+09DF)
- Checks length bounds (max 500 chars)
- Rejects binary / injection / spam patterns
- If invalid → yields `INPUT_INVALID_REFUSAL_BN` and returns

**Example A:** `"এনআইডি কার্ডের স্ট্যাটাস কিভাবে জানব?"` → valid, passes.

---

## Phase 2: Router (`cogops/pipeline/router.py`)

### Step 6: Hard shortcuts (checked first)

| Shortcut | Trigger | Example A | Example B | Example C |
|---|---|---|---|---|
| `personal_law_refuse` | `_hard_personal_law_match()` — religious/judgment framing | ❌ | ❌ | ❌ |
| `political_refuse` | `_hard_political_match()` — partisan keywords | ❌ | ❌ | ❌ |

**Example A:** No hard shortcuts match. Proceeds to LLM.

### Step 7: LLM Router Call

```python
await secondary_client.chat.completions.create(
    model="qwen36",
    messages=[
        {"role": "system", "content": _ROUTER_SYSTEM_PROMPT},
        {"role": "user", "content": clean_query},
    ],
    response_format={"type": "json_object"},
    temperature=0.0,
    max_tokens=400,
)
```

**LLM returns JSON:**

```json
// Example A
{"intent": "factual_govt", "sub_queries_bengali": ["এনআইডি কার্ড স্ট্যাটাস চেক"]}

// Example B
{"intent": "factual_wiki", "sub_queries_bengali": ["বাংলাদেশের স্বাধীনতা যুদ্ধ কবে শুরু হয়"]}

// Example C
{"intent": "factual_mixed", "sub_queries_bengali": ["বাংলাদেশের প্রধানমন্ত্রী কে"]}
```

### Step 8: Domain-vocab override

```python
if intent not in ("factual_govt", "factual_mixed") and _DOMAIN_RE.search(text):
    intent = "factual_govt"
```

- `_DOMAIN_RE` matches govt-service vocabulary (NID, passport, tax, etc.)
- **Example A:** `"এনআইডি"` matches → stays `factual_govt`
- **Example B:** No govt vocab → stays `factual_wiki`
- **Example C:** `"প্রধানমন্ত্রী"` does NOT match domain vocab → stays `factual_mixed`

### Step 9: `RouterResult` emitted

```python
yield {"type": "router_done", "intent": "factual_govt", ...}
```

---

## Phase 3: Orchestrator Branching

### Step 10: Intent → `chunk_type` mapping

```python
if router_result.intent == "factual_govt":
    self.pipeline_cfg.chunk_type = "govt_service"
elif router_result.intent == "factual_wiki":
    self.pipeline_cfg.chunk_type = "wiki"
elif router_result.intent == "factual_mixed":
    self.pipeline_cfg.chunk_type = None   # no filter
```

| Example | Intent | `chunk_type` | Qdrant Filter |
|---|---|---|---|
| A | `factual_govt` | `"govt_service"` | `chunk_type == "govt_service"` |
| B | `factual_wiki` | `"wiki"` | `chunk_type == "wiki"` |
| C | `factual_mixed` | `None` | No filter |

---

## Phase 4: Deterministic Pipeline (`cogops/agents/pipeline.py`)

### Step 11: Assertion check

```python
assert router_result.intent in ("factual_govt", "factual_wiki", "factual_mixed")
```

### Step 12: Normalizer (`cogops/pipeline/normalize.py`)

```python
sub_queries = normalize_sub_queries(router_result.sub_queries_bengali)
```

- Applies `_SYNONYM_RES`: `প্লেন` → `বিমান`, `ট্রেন` → `রেল`, `টিকেট` → `টিকিট`
- Strips fillers: `আচ্ছা`, `ভাই`, `দেখেন`, `শুনুন`
- Collapses whitespace

**Example A:** `["এনআইডি কার্ড স্ট্যাটাস চেক"]` (no changes needed)

### Step 13: LLM Expander (`cogops/pipeline/query_expand.py`)

```python
sub_queries = await expand_sub_queries_llm(
    sub_queries, secondary_client, secondary_model,
)
```

1. **Hardcoded fast-path:** Regex match against `_DOC_TYPE_EXPANSIONS`
   - Example A: `"এনআইডি"` matches → returns `"এনআইডি কার্ড স্ট্যাটাস চেক | জাতীয় পরিচয়পত্র স্ট্যাটাস | স্মার্ট কার্ড চেক"`
2. **Cache hit:** Checks `_EXPANDER_CACHE` dict (module-level LRU)
3. **LLM fallback:** If no hardcoded match, calls secondary LLM with compact JSON prompt
4. **Fail-open:** On error/timeout, returns original query unchanged

**Example A output:**
```python
[
    "এনআইডি কার্ড স্ট্যাটাস চেক | জাতীয় পরিচয়পত্র স্ট্যাটাস | স্মার্ট কার্ড চেক"
]
```

---

## Phase 5: Jiggasha Retrieval (`jiggasha/service.py`)

### Step 14: `_call_jiggasha()` builds POST payload

```python
payload = {
    "sub_queries": [
        "এনআইডি কার্ড স্ট্যাটাস চেক | জাতীয় পরিচয়পত্র স্ট্যাটাস | স্মার্ট কার্ড চেক"
    ],
    "top_k_per_sub": 20,
    "rerank": True,
    "candidate_cap_global": 30,
    "keep_cap": 24,
    "weak_per_sub_cap": 3,
    "fallback_cosine_min": 0.50,
    "chunk_type": "govt_service",   # ← NEW: from PipelineConfig
}
```

### Step 15: Jiggasha receives request at `/search`

```python
@app.post("/search")
async def search(req: SearchRequest):
    if req.sub_queries:
        return await _search_multi(req)
```

### Step 16: Embed each sub-query

```python
vec = await asyncio.to_thread(embedder.embed, query)
```

- Calls `Embedder.embed()` → POST to `http://172.22.8.106:5001/v1/embeddings`
- Model: `qwen3-embed`
- Returns: 4096-dimensional float vector

### Step 17: Qdrant query with `chunk_type` filter

```python
query_filter = models.Filter(
    must=[models.FieldCondition(
        key="chunk_type",
        match=models.MatchValue(value="govt_service"),
    )]
) if chunk_type else None

hits = qdrant_client.query_points(
    collection_name="bnwiki_chunks",   # ← was "jiggasha_data"
    query=vec,
    limit=20,
    query_filter=query_filter,
).points
```

**Example A:** Only points with `chunk_type="govt_service"` are searched.

**Example B:** Only points with `chunk_type="wiki"` are searched.

**Example C:** No filter — searches all 932,788 points.

### Step 18: Hit → Passage mapping (schema translation)

```python
def _hit_to_passage(hit):
    payload = hit.payload or {}
    # UUID fallback for passage_id
    pid_raw = payload.get("passage_id")
    if pid_raw is not None:
        passage_id = int(pid_raw)
    else:
        id_str = str(hit.id)
        try:
            passage_id = int(id_str)
        except ValueError:
            hex_part = id_str.replace("-", "")[-8:]
            passage_id = int(hex_part, 16)

    # Unified schema → legacy schema
    category = payload.get("category") or payload.get("page_title", "")
    sub_category = payload.get("sub_category") or payload.get("section", "")
    service = payload.get("service") or payload.get("subsection", "")
    topic = payload.get("topic") or payload.get("page_title", "")

    return {
        "passage_id": passage_id,
        "text": payload.get("text", ""),
        "category": category,
        "sub_category": sub_category,
        "service": service,
        "topic": topic,
        "chunk_type": payload.get("chunk_type", ""),
        "score": float(hit.score),
    }
```

### Step 19: Merge candidates across sub-queries

```python
candidates = _merge_candidates(per_sub, global_cap=30)
```

- Deduplicates by `passage_id`
- Keeps highest cosine score
- Tracks which sub-queries produced each passage (`sub_indices`)

### Step 20: LLM Reranker (`jiggasha/rerank.py`)

```python
result = await run_rerank(
    sub_queries=sub_queries,
    candidates=candidates,
    secondary_client=secondary_client,
    secondary_model="qwen36",
    ...
)
```

**One batched LLM call** classifies each candidate as:
- `0` = `yes` (directly answers the sub-query)
- `1` = `weak` (tangentially related)

**Example A rerank result:**
```json
{
  "1": [[42, 0], [47, 0], [51, 1]],   // sub-query 1: pid 42=yes, 47=yes, 51=weak
}
```

### Step 21: Policy application (`_apply_policy`)

- Keep ALL `yes` passages
- Backfill with `weak` passages up to `weak_per_sub_cap=3` per sub-query
- Global cap: `keep_cap=24`

### Step 22: Response returned to pipeline

```json
{
  "sub_queries": ["এনআইডি কার্ড স্ট্যাটাস চেক | জাতীয় পরিচয়পত্র স্ট্যাটাস | স্মার্ট কার্ড চেক"],
  "passages": [
    {
      "passage_id": 42,
      "text": "স্মার্ট কার্ড ও জাতীয়পরিচয়পত্র > স্মার্ট কার্ড > অনলাইনে স্মার্ট কার্ড চেক > ...",
      "category": "স্মার্ট কার্ড ও জাতীয়পরিচয়পত্র",
      "sub_category": "স্মার্ট কার্ড",
      "service": "অনলাইনে স্মার্ট কার্ড চেক",
      "topic": "স্মার্ট কার্ড তৈরি হয়েছে কি না অনলাইনে চেক করবে কীভাবে?",
      "chunk_type": "govt_service",
      "score": 0.8048
    },
    ...
  ],
  "rerank": {"1": [[42, 0], [47, 0], [51, 1]]},
  "degraded": false,
  "elapsed_ms": 2450
}
```

---

## Phase 6: Pipeline Post-Retrieval

### Step 23: Source map allocation

```python
source_map = _allocate_source_map_from_rerank(passages, rerank, sub_queries)
```

Maps each passage to `[S1], [S2], ...` tags with verdict metadata:

```python
"S1": {
    "passage_id": 42,
    "text": "...",
    "category": "স্মার্ট কার্ড ও জাতীয়পরিচয়পত্র",
    "sub_category": "স্মার্ট কার্ড",
    "service": "অনলাইনে স্মার্ট কার্ড চেক",
    "topic": "...",
    "chunk_type": "govt_service",
    "score": 0.8048,
    "verdict": "yes",
    "sub_indices": [0],
    "tool": "jiggasha",
}
```

### Step 24: Document-type guard (`cogops/pipeline/query_expand.py`)

```python
if not check_document_type_match(raw_query, source_map):
    yield refusal_text_bn
    return
```

- Extracts document type from raw query (e.g., `"বিবাহ সনদ"`, `"পাসপোর্ট"`)
- Checks if ANY retrieved passage metadata matches that document type
- If user explicitly asked for a doc type NOT in corpus → refusal

**Example A:** No explicit doc type → passes.

### Step 25: Disambiguation detection

```python
disambiguate, disambig_candidates = _detect_disambiguation(
    source_map, raw_query, sub_queries, cfg,
    intent=router_result.intent,
)
```

**Fires ONLY when:**
1. Single sub-query
2. Short query (≤6 tokens)
3. ≥2 distinct `(category, sub_category)` among yes/weak passages
4. **AND** intent is `factual_govt` (wiki/mixed skip disambiguation)

**Example A:** Single sub-query, but query is 5 tokens. If passages span ≥2 distinct `(category, sub_category)` pairs → disambiguation fires.

**Example B:** `factual_wiki` → **skipped** (returns `False, []`)

**Example C:** `factual_mixed` → **skipped** (returns `False, []`)

### Step 26: Composer context building

```python
ctx_parts = ["<context>"]
for tag, meta in source_map.items():
    hb = []
    hb.append(f"বিভাগ: {meta['category']}")
    hb.append(f"উপ-বিভাগ: {meta['sub_category']}")
    hb.append(f"সেবা: {meta['service']}")
    hb.append(f"বিষয়: {meta['topic']}")
    # NEW: source type label
    if meta['chunk_type'] == "wiki":
        hb.append("উৎস: উইকিপিডিয়া")
    elif meta['chunk_type'] == "govt_service":
        hb.append("উৎস: সরকারি সেবা")
    header = " | ".join(hb)
    ctx_parts.append(f"[{tag}] ({header})")
    ctx_parts.append(meta["text"])
ctx_parts.append("</context>")
```

**Example A composer context snippet:**
```
<context>
[S1] (বিভাগ: স্মার্ট কার্ড ও জাতীয়পরিচয়পত্র | উপ-বিভাগ: স্মার্ট কার্ড | সেবা: অনলাইনে স্মার্ট কার্ড চেক | বিষয়: স্মার্ট কার্ড তৈরি হয়েছে কি না অনলাইনে চেক করবে কীভাবে? | উৎস: সরকারি সেবা)
স্মার্ট কার্ড ও জাতীয়পরিচয়পত্র > স্মার্ট কার্ড > অনলাইনে স্মার্ট কার্ড চেক > ...
...
</context>

<user_query>
এনআইডি কার্ডের স্ট্যাটাস কিভাবে জানব?
</user_query>
```

**Example B composer context snippet:**
```
[S1] (বিভাগ: বাংলাদেশের স্বাধীনতা যুদ্ধ | উপ-বিভাগ: স্বাধীনতা যুদ্ধের সময়কাল | বিষয়: বাংলাদেশের স্বাধীনতা যুদ্ধ | উৎস: উইকিপিডিয়া)
১৯৭১ সালে বাংলাদেশ স্বাধীনতা যুদ্ধ শুরু হয়...
...
</context>
```

---

## Phase 7: Primary LLM Streaming (Composer)

### Step 27: System prompt

```python
COMPOSER_SYSTEM_PROMPT = """\
You are **আশা**, a Bengali assistant for Bangladesh government services.
Your user-facing language is Formal Bengali (প্রমিত বাংলা).

ROLE
You will receive context passages tagged [S1], [S2], … inside a <context> block,
and the user's raw question inside a <user_query> block. Compose a short,
accurate, cited Bengali answer using ONLY the context.

CITATION RULES
- Cite passages with [S1], [S2], etc. inline.
- Every factual claim MUST have a citation.
- Do NOT invent facts not in the context.
- Do NOT add a Sources block; the system appends it automatically.
"""
```

### Step 28: Streaming response

```python
async for chunk in primary_client.chat.completions.create(
    model="qwen36",
    messages=[system, ...history, user_msg],
    stream=True,
    temperature=0.1,
    max_tokens=2048,
):
    text = chunk.choices[0].delta.content or ""
    yield {"type": "answer_chunk", "channel": "answer", "content": text}
```

**Example A streamed answer:**
```
এনআইডি কার্ডের স্ট্যাটাস জানার জন্য নিচের পদ্ধতিগুলো অনুসরণ করতে পারেন:

**১. অনলাইনে চেক** [S1]
services.nidw.gov.bd ওয়েবসাইটে গিয়ে...

**২. এসএমএস-এর মাধ্যমে** [S2]
১০৫ নম্বরে মেসেজ পাঠিয়ে...
```

### Step 29: `<thinking>` block stripping

`ThinkingParser` strips `<thinking>…</thinking>` blocks from the LLM output before showing to the user.

---

## Phase 8: Post-Flight (`_post_flight`)

### Step 30: Strip composer-emitted Sources block

The composer sometimes emits its own `--- **সূত্র**` block. `_post_flight` strips it to prevent duplication.

### Step 31: Strip unknown `[S#]` tags

If the composer hallucinated a citation tag not in `source_map`, it's removed.

### Step 32: NLI Verification (`cogops/verifier/nli.py`) — optional

```python
if cfg.verifier_enabled:
    verify_claims(answer_text, source_map)
```

- One batched secondary-LLM call checks if each claim is entailed by its cited passage
- Misaligned claims are redacted / refused based on `verifier_policy`

**Example A:** Claims match passages → all entailed.

### Step 33: Append canonical Sources block

```python
sources_block = build_sources_block(source_map, used_tags)
```

**Example A Sources block:**
```
---
**সূত্র (Sources)**
- [S1] স্মার্ট কার্ড ও জাতীয়পরিচয়পত্র — স্মার্ট কার্ড তৈরি হয়েছে কি না অনলাইনে চেক করবে কীভাবে? · passage_id 42 (jiggasha · সরকারি সেবা)
- [S2] স্মার্ট কার্ড ও জাতীয়পরিচয়পত্র — এসএমএস-এর মাধ্যমে স্মার্ট কার্ড চেক করতে হয় কীভাবে? · passage_id 47 (jiggasha · সরকারি সেবা)
```

**Example B Sources block:**
```
---
**সূত্র (Sources)**
- [S1] বাংলাদেশের স্বাধীনতা যুদ্ধ — স্বাধীনতা যুদ্ধের সময়কাল · passage_id 1048576 (jiggasha · উইকিপিডিয়া)
- [S2] কামালপুর যুদ্ধ — তথ্যসারণী · passage_id 1048592 (jiggasha · উইকিপিডিয়া)
```

---

## Phase 9: Event Yielding & Persistence

### Step 34: `final_answer` event

```python
yield {
    "type": "final_answer",
    "channel": "both",
    "content": final_answer + sources_block,
    "source_map": source_map,
    "turn_id": turn_id,
}
```

### Step 35: `answer_complete` event

```python
yield {"type": "answer_complete", "channel": "both", "turn_id": turn_id}
```

### Step 36: Redis persistence

```python
self._persist(user_id, turn_id, original_query, final_answer_text)
```

Appends to Redis list keyed by `session_id`.

---

## Phase 10: HTTP Response Streaming

### Step 37: FastAPI streams SSE/JSON events

```json
{"type": "answer_chunk", "content": "এনআইডি কার্ডের স্ট্যাটাস জানার জন্য..."}
{"type": "answer_chunk", "content": "**১. অনলাইনে চেক** [S1]..."}
...
{"type": "final_answer", "content": "...\n---\n**সূত্র (Sources)**\n- [S1] ... · সরকারি সেবa"}
{"type": "answer_complete"}
```

---

## Complete Decision Tree

```
User Query
    │
    ▼
┌─────────────────┐
│  Stage 0:       │
│  Sanitize       │
│  (NFC, bounds)  │
└────────┬────────┘
         │ invalid?
         ▼
    ┌─────────┐
    │ Refusal │
    └─────────┘
         │ valid
         ▼
┌─────────────────┐
│  Stage 1:       │
│  Router         │
│  (LLM + rules)  │
└────────┬────────┘
         │
    ┌────┴────┬─────────────┬──────────────┬─────────────────┐
    ▼         ▼             ▼              ▼                 ▼
political  personal_law  chitchat     factual_govt      factual_wiki
refuse     refuse         greeting      │                 │
    │         │             │           ▼                 ▼
    │         │             │      chunk_type="govt"   chunk_type="wiki"
    │         │             │           │                 │
    │         │             │           ▼                 ▼
    │         │             │    ┌──────────────┐   ┌──────────────┐
    │         │             │    │ Jiggasha     │   │ Jiggasha     │
    │         │             │    │ Qdrant filter│   │ Qdrant filter│
    │         │             │    │ chunk_type   │   │ chunk_type   │
    │         │             │    │ ="govt_service"│  │ ="wiki"      │
    │         │             │    └──────┬───────┘   └──────┬───────┘
    │         │             │           │                 │
    │         │             │           └────────┬────────┘
    │         │             │                    ▼
    │         │             │         ┌──────────────────┐
    │         │             │         │ Reranker (yes/weak)│
    │         │             │         └────────┬─────────┘
    │         │             │                    ▼
    │         │             │         ┌──────────────────┐
    │         │             │         │ Source Map [S1]… │
    │         │             │         │ Doc-type guard   │
    │         │             │         │ Disambiguation   │
    │         │             │         └────────┬─────────┘
    │         │             │                    ▼
    │         │             │         ┌──────────────────┐
    │         │             │         │ Composer (primary│
    │         │             │         │ LLM stream)      │
    │         │             │         └────────┬─────────┘
    │         │             │                    ▼
    │         │             │         ┌──────────────────┐
    │         │             │         │ Post-flight:     │
    │         │             │         │ strip, verify,   │
    │         │             │         │ append Sources   │
    │         │             │         └────────┬─────────┘
    │         │             │                    ▼
    └────┬────┴─────┬───────┴────────┬─────────┘
         │          │                │
         ▼          ▼                ▼
    ┌─────────────────────────────────────┐
    │  Streamed response to user          │
    │  (answer_chunks + final_answer)     │
    └─────────────────────────────────────┘
```

---

## Key Data Flows

### Jiggasha Request → Qdrant

```
HTTP POST localhost:10000/search
  {
    "sub_queries": ["এনআইডি কার্ড স্ট্যাটাস চেক"],
    "rerank": true,
    "chunk_type": "govt_service"   // or "wiki" or null
  }
         │
         ▼
  Embedder.embed() → vLLM @ :5001 → 4096-dim vector
         │
         ▼
  Qdrant.query_points(
    collection="bnwiki_chunks",
    query=vector,
    limit=20,
    query_filter=Filter(     // only if chunk_type set
      must=[FieldCondition(
        key="chunk_type",
        match=MatchValue(value="govt_service")
      )]
    )
  )
         │
         ▼
  Returns top-20 points (passage_id, text, page_title, section,
  subsection, chunk_type, score)
         │
         ▼
  Reranker LLM classifies each as yes(0) / weak(1)
         │
         ▼
  Policy: keep all yes + weak backfill up to cap
         │
         ▼
  Response JSON → pipeline
```

### Source Map → Composer Context

```
source_map["S1"] = {
  "text": "...",
  "category": "স্মার্ট কার্ড ও জাতীয়পরিচয়পত্র",   // from page_title
  "sub_category": "স্মার্ট কার্ড",                  // from section
  "service": "অনলাইনে স্মার্ট কার্ড চেক",            // from subsection
  "topic": "...",
  "chunk_type": "govt_service",
  "score": 0.8048,
  "verdict": "yes",
}
         │
         ▼
Composer sees:
  [S1] (বিভাগ: স্মার্ট কার্ড ও জাতীয়পরিচয়পত্র |
        উপ-বিভাগ: স্মার্ট কার্ড |
        সেবা: অনলাইনে স্মার্ট কার্ড চেক |
        বিষয়: ... |
        উৎস: সরকারি সেবা)
  <passage text>
```

---

*End of trace.*
