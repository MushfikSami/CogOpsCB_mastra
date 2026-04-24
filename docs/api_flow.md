# CogOpsCB — Complete API Flow Documentation

This document describes the end-to-end flow of how a user query is handled when the CogOpsCB GovOps AI Agent API is running. It covers the architecture, request pipeline, tools, event streaming, session management, debug mode, configuration, and key design decisions.

---

## 1. Architecture Overview

```
┌──────────────┐     ┌───────────────┐     ┌──────────────────┐
│   Client     │────▶│   FastAPI     │────▶│  Orchestrator    │
│ (Streamlit / │     │  api.py       │     │  orchestrator.py │
│  REST caller) │◀────│  POST /chat/  │◀────│                  │
│              │     │  stream       │     └────────┬─────────┘
└──────────────┘     └───────────────┘              │
                              │                      │
              ┌───────────────┼──────────────────────┤
              │               │                      │
              ▼               ▼                      ▼
        ┌───────────┐  ┌──────────┐          ┌──────────────┐
        │  Stream    │  │  Redis   │          │  Reasoning   │
        │  Events    │  │  Store   │          │  Loop        │
        │  (NDJSON)  │  │  (turns, │          │  reasoning   │
        │            │  │  summary, │          │  loop.py     │
        │            │  │  clarify) │          │              │
        └────────────┘  └──────────┘          └──────┬───────┘
                                                     │
                            ┌────────────────────────┼────────────────────────┐
                            │                        │                        │
                            ▼                        ▼                        ▼
                  ┌─────────────────┐    ┌───────────────────┐    ┌──────────────────┐
                  │  Neo4j Graph    │    │  Primary LLM      │    │  Secondary LLM   │
                  │  (Graphiti)     │    │  (qwen3)          │    │  (qwen3-secondary) │
                  │  bolt://localhost:7687 │ │  localhost:5000   │    │  localhost:5002   │
                  └─────────────────┘    └───────────────────┘    └──────────────────┘
                            │
                            ▼
                  ┌─────────────────┐
                  │  Triton Server  │
                  │  (Gemma embed)  │
                  │  localhost:6000 │
                  └─────────────────┘
                            │
                            ▼
                  ┌─────────────────┐
                  │  Reranker LLM   │
                  │  (binary cls)   │
                  │  localhost:5001 │
                  └─────────────────┘
```

### Three LLM Endpoints

| Endpoint | Role | Model | Port |
|---|---|---|---|
| **Primary LLM** | Agent reasoning + native thinking + tool calling | `qwen3-primary` | 5000 |
| **Reranker LLM** | Binary classification (passage relevance 0/1) for Graphiti search | `qwen3-reranker` | 5001 |
| **Secondary LLM** | General tasks: post-tool refine, extract from docs, delegate, sub-agents | `qwen3-secondary` | 5002 |

### Embedder

Triton Inference Server runs a Gemma embedding model (`gemma_embedding`) that converts text to 768-dimensional vectors for the Graphiti knowledge graph hybrid search.

---

## 2. Complete Request Flow

Here is the step-by-step journey of a single user query from HTTP arrival to response delivery.

### Step 1 — HTTP Request Arrives

**File:** [api.py:99-142](../api.py#L99-L142)

```
POST /chat/stream
Content-Type: application/json
X-Debug-Key: SuperDebugCoTCB  ← optional, for debug mode

Body:
{
    "user_id": "a1b2c3d4",
    "query": "পাসপোর্ট ফি কত?"
}
```

- **Validation:** Empty query returns HTTP 400 with `"Query cannot be empty."`
- **Query logging:** The query text and ISO-8601 timestamp (Bangladesh timezone) are appended to `data/query_log.jsonl`. Old entries (>10 days) are pruned on every write.
- **Session lookup:** `get_agent_session(user_id)` is called — returns an existing `Orchestrator` instance if one exists, or creates a new one.
- **Debug detection:** If `X-Debug-Key` header matches the `ADMIN_DEBUG_SECRET` env var, debug mode is enabled.

### Step 2 — Session Management

**File:** [api.py:47-80](../api.py#L47-L80)

- **In-memory store:** `active_sessions: Dict[str, Orchestrator]` maps `user_id` → `Orchestrator` instance. Protected by `asyncio.Lock`.
- **First request:** New `Orchestrator(config_path="configs/config.yml")` is instantiated. This loads config, initializes all LLM clients, Redis, tool registry, and system prompt.
- **Subsequent requests:** The existing Orchestrator is reused, preserving its in-memory `self.history` and `self.feedback_history`.
- **No automatic expiry:** Sessions persist for the lifetime of the process. Users can explicitly clear via `POST /session/clear`.

### Step 3 — Clarification & Follow-up Resolution

**File:** [orchestrator.py:176-205](../cogops/agents/orchestrator.py#L176-L205)

Before the main reasoning loop, the orchestrator checks for context:

1. **Pending clarification:** If `redis_store.get_clarification(user_id)` returns data (from a previous `ask_user` call), the question and options are replayed alongside the user's reply. The clarification is cleared from Redis.
2. **Short followup detection:** If the query is short (≤16 chars) or matches patterns like "3", "the second one", "tell me more", the last assistant reply is fetched from Redis and injected as context.
3. **Otherwise:** The raw user query is used as-is.

### Step 4 — Message Construction & Token Budgeting

**File:** [orchestrator.py:207-228](../cogops/agents/orchestrator.py#L207-L228)

```python
messages = [
    {"role": "system", "content": self.system_prompt + rolling_summary_delta},
    {"role": "user", "content": user_query},
]
# Truncate to token budget
max_ctx = 32000           # max_context_tokens from config
reservation = 3500        # system_prompt_reservation from config
budget = max_ctx - reservation  # = 28500 tokens
messages = truncate_messages_to_budget(messages, max_tokens=budget, ...)
```

- **System prompt:** Cached once at orchestrator init. Contains agent identity, tool descriptions, rules, and safety guidelines.
- **Rolling summary:** Fetched from Redis. Prepend text: `"Recent conversation summary:\n{summary}"`
- **Truncation:** Oldest non-system messages are dropped first. If still over budget, the last message is hard-truncated.

### Step 5 — Reasoning Loop

**File:** [reasoning_loop.py:109-495](../cogops/llm/reasoning_loop.py#L109-L495)

The core conversation engine runs in a `while turn_count < max_turns` loop (default max 10 turns).

**Each turn:**

1. **`turn_start` event** emitted (debug channel).
2. **Tool choice policy:**
   - Turn 1: `tool_choice="required"` — model MUST call a tool. This enforces the "never answer from parametric knowledge" rule.
   - Turn 2+: `tool_choice="auto"` — model can stop when it has enough context.
3. **LLM streaming call:** `client_llm.chat.completions.create(stream=True)` with the model, messages, and tool schemas.
4. **Streaming chunks processed:**
   - **Native thinking tokens** (`delta.reasoning`) → `reasoning_chunk` event (debug channel)
   - **Content tokens** (`delta.content`) → `answer_chunk` event (user channel)
   - **Tool call fragments** (`delta.tool_calls`) → accumulated into `tool_call_index_map`
5. **End of stream:**
   - **Usage tokens** (prompt_tokens, completion_tokens, total_tokens) → `usage` event (debug channel)
   - **No tool calls:** If the model produced text without calling any tool, and we're on turn 2+, it's the final answer → break loop.
   - **Tool calls:** Emit `tool_call` event, execute tools, feed results back as messages.

**Short-circuit path — `answer_directly`:** When the `answer_directly` meta-tool is called, its result contains a sentinel (`__ANSWER_DIRECTLY__::category::text`). The loop detects this, streams the text directly to the user, emits `direct_answer` event, and exits immediately without feeding the tool result back.

**Post-tool refinement:** When a tool result exceeds 600 tokens (configurable), the secondary LLM is invoked to extract only the parts relevant to the user's query. Debug stream shows both raw and refined content via `tool_result_refined` event.

### Step 6 — Tool Execution

**File:** [reasoning_loop.py:295-433](../cogops/llm/reasoning_loop.py#L295-L433)

For each tool call returned by the model:

1. **Parse arguments:** JSON is parsed from `tool_call["function"]["arguments"]`. Invalid JSON produces an error result.
2. **Dispatch:** The bound handler function is looked up in `available_tools`. Async functions are awaited; sync functions (wrapped by `functools.partial`) are run via `asyncio.to_thread`.
3. **Result capture:** The string result is stored with timing metadata.
4. **Error handling:** Exceptions are caught and returned as `"System Error executing tool: ..."` results. `ClarificationRequested` is re-raised to terminate the stream.
5. **Result fed back:** A `{"role": "tool", "tool_call_id": ..., "content": ...}` message is appended to the conversation for the next LLM call.

### Step 7 — Event Streaming to Client

**File:** [api.py:117-141](../api.py#L117-L141)

The API returns a `StreamingResponse` with `media_type="application/x-ndjson"`. Each event is a JSON object on its own line:

```json
{"type": "reasoning_chunk", "channel": "debug", "data": "Let me think about..."}
{"type": "tool_call", "channel": "debug", "tool_calls": [...], "turn": 1}
{"type": "tool_result", "channel": "debug", "call_id": "...", "content": "...", "status": "ok"}
{"type": "answer_chunk", "channel": "user", "content": "আসসালামু আলাইকুম"}
```

**Channel filtering** ([channels.py:10-23](../cogops/events/channels.py#L10-L23)):
- **Normal mode:** Only events with `channel="user"` or `channel="both"` pass through.
- **Debug mode:** Events with `channel="debug"` or `channel="both"` pass through.

Each yielded event has a tiny `await asyncio.sleep(0.001)` delay to prevent backpressure.

### Step 8 — Post-Answer Persistence

**File:** [orchestrator.py:302-336](../cogops/agents/orchestrator.py#L302-L336)

After the reasoning loop completes with a final answer:

1. **`answer_complete` event** emitted (both channels).
2. **Turn stored to Redis:** `{"turn_id": "...", "user": "...", "assistant": "..."}` appended to `session:{user_id}:turns` (Redis list, TTL 86400s).
3. **Last assistant metadata stored:** The final response and any enumerated options extracted from it are stored in `session:{user_id}:last_assistant` for short-followup resolution.
4. **Background summarizer fires:** `asyncio.create_task(run_summarizer_task(...))` — sends the last turn to the secondary LLM to update the rolling summary in Redis (max 300 tokens). Non-blocking.

### Step 9 — Error Handling

If any exception occurs during `process_query()`:
- An `error` event with `"channel": "user"` is yielded with the Bengali fallback message: `"একটি প্রযুক্তিগত ত্রুটির কারণে আমি এই মুহূর্তে সাহায্য করতে পারছি না।"`
- If the context length is exceeded, a `"Context limit reached. Please clear session."` error is shown.

---

## 3. Event Types Reference

Every event has `type`, `channel`, and data fields. Here is the complete list:

| Event Type | Channel | Key Fields | When Emitted |
|---|---|---|---|
| `turn_start` | debug | `turn_number` | Each reasoning loop iteration |
| `reasoning_chunk` | debug | `data` (string) | Native thinking tokens from primary LLM, streamed chunk-by-chunk |
| `tool_call` | debug | `tool_calls` (array), `turn` | After primary LLM emits tool calls |
| `tool_result` | debug | `call_id`, `content`, `duration_ms`, `status` ("ok"/"error") | After each tool execution |
| `tool_result_refined` | debug | `call_id`, `raw`, `refined` | When secondary LLM refines a large tool result |
| `answer_chunk` | user | `content` (string) | Text tokens streamed from primary LLM |
| `usage` | debug | `tokens` (object with prompt_tokens, completion_tokens, total_tokens) | End of each LLM streaming call |
| `clarification_needed` | both | `question`, `options` (array), `reason`, `turn_id` | When `ask_user` tool is called |
| `direct_answer` | debug | `category`, `call_id` | After `answer_directly` short-circuit |
| `answer_complete` | both | `turn_id` | After the final answer is fully assembled |
| `turn_end` | debug | `turn_number` | End of each reasoning loop iteration |
| `error` | both/user | `content` (string) | On system errors or context limit exceeded |

---

## 4. Complete Tool Reference

All 17 tools are registered in [registry.py](../cogops/tools/registry.py). Each tool has a JSON schema (visible to the LLM) and a handler function (executed by the orchestrator). Context-dependent tools have their server-side parameters bound at request time via `bind_tools()`.

### Graph Tools (10 tools)

All query the Neo4j Graphiti knowledge graph (`qwen34neo4j` database).

| Tool | Description | Parameters | Returns |
|---|---|---|---|
| `graph_search` | Hybrid search (BM25 + vector similarity + BFS) with RRF reranking across nodes, edges, and episodes | `query: string` | Formatted text with Node/Edge/Passage sections, each with scores |
| `entity_search` | Find entities by partial/fuzzy name match, ranked by match quality (exact / case-insensitive / partial) | `search_term: string`, `max_results: int` | Markdown table with Name, Summary, Match Rank |
| `entity_detail` | Get full details of a specific entity by exact name or UUID | `identifier: string` | UUID, Name, Summary, Group ID, Created timestamp |
| `node_explore` | Get all connections (incoming and outgoing) for an entity, grouped by relation type | `entity_name: string`, `max_results: int` | Grouped facts by relation type |
| `relation_browse` | List all available relation-name values with their edge counts | `filter_prefix: string?`, `top_n: int` | Markdown table of relation names and counts |
| `relation_filter` | Given a relation name, return all entity pairs connected by it | `relation_name: string`, `max_results: int` | Table with Source, Target, Fact columns |
| `similar_entities` | Find semantically similar entities via vector cosine similarity | `entity_name: string`, `max_results: int`, `min_score: float` | Table with Entity, Similarity Score, Summary |
| `path_find` | Find paths between two entities (1-N hops) | `start_entity: string`, `end_entity: string`, `max_hops: int`, `max_paths: int` | Entity chains with relation types |
| `episodic_search` | Search raw passage data in Episodic nodes by text content | `search_term: string`, `max_results: int` | Passage text with category, truncated to 300 chars |
| `graph_stats` | Get graph-level statistics: node counts, relation type distribution, edge counts | `detail_level: "basic"\|"detailed"` | Markdown tables with node counts, edge count, optionally top 20 relation types |

### Secondary LLM Tools (4 tools)

These call the secondary LLM endpoint or do text processing.

| Tool | Description | Parameters | Returns |
|---|---|---|---|
| `grep_passage` | Regex grep over a text passage, returns matched lines with context | `passage: string`, `pattern: string`, `context_lines: int` | Matched lines with line numbers and surrounding context |
| `extract_from_document` | Use secondary LLM to extract relevant information from a long document | `document: string`, `topic: string` | Concise structured list of relevant extractions |
| `delegate_task` | One-shot instruction task to secondary LLM | `instruction: string`, `context: string?` | LLM response to the instruction |
| `spawn_subagent` | Spawn a self-contained reasoning loop on the secondary LLM with a restricted tool set | `task: string`, `allowed_tools: string[]` | Final text answer from the sub-agent's reasoning loop |

### Interaction Tools (2 tools)

| Tool | Description | Parameters | Returns |
|---|---|---|---|
| `ask_user` | Ask the user for clarification when the query is ambiguous | `question: string`, `options: string[]?`, `reason: string?` | **Never returns normally.** Raises `ClarificationRequested` exception which terminates the stream. |
| `answer_directly` | Meta-tool for non-factual replies (chit-chat, identity, safety). Short-circuits the loop. | `category: "chitchat"\|"identity"\|"safety_deflect"\|"abuse"\|"illegal"`, `text: string` | Sentinel string `__ANSWER_DIRECTLY__::category::text` — detected by reasoning loop, text streamed directly to user |

### History Tool (1 tool)

| Tool | Description | Parameters | Returns |
|---|---|---|---|
| `history_query` | Query stored conversation history | `mode: "lookup"\|"summarize"\|"recent"\|"ask"`, `query: string?`, `n: int` | Lookup: matching Q/A pairs. Summarize: rolling summary. Recent: last N turns. Ask: secondary LLM answer from history. |

### Tool Selection Guide

The system prompt ([system.py](../cogops/prompts/system.py)) provides the model with this intent-to-tool mapping:

- Factual info about services → `graph_search` or `episodic_search`
- Named entity lookup → `entity_search`
- Full entity details → `entity_detail`
- All entity connections → `node_explore`
- Relation type listing → `relation_browse`
- Relation-specific pairs → `relation_filter`
- Similar concepts → `similar_entities`
- Path between entities → `path_find`
- Graph stats → `graph_stats`
- Grep a passage → `grep_passage`
- Extract from document → `extract_from_document`
- Multi-step subtask → `spawn_subagent`
- Ambiguous query → `ask_user`
- Short followup / reference → `history_query`
- Chit-chat / identity / safety → `answer_directly`

---

## 5. Debug Mode

Debug mode is controlled by the `ADMIN_DEBUG_SECRET` environment variable ([.env:31](../.env#L31)).

**How it works:**

1. Client sends `X-Debug-Key` header with the secret value.
2. `api.py:114` compares the header against `ADMIN_DEBUG_SECRET`.
3. If they match, `debug_mode = True`.
4. All events emitted by the reasoning loop are tagged with a channel (`"user"`, `"debug"`, or `"both"`).
5. The channel filter ([channels.py](../cogops/events/channels.py)) determines what the client receives:
   - **Debug mode:** receives `debug` + `both` events (reasoning chunks, tool calls, tool results, usage stats, etc.)
   - **Normal mode:** receives only `user` + `both` events (just the answer text)

**Debug information visible:**
- **Reasoning chunks:** The model's native thinking tokens, streamed in real-time
- **Tool calls:** Which tools were called, with arguments, on which turn
- **Tool results:** Full content returned by each tool, with execution duration and status
- **Token usage:** Prompt tokens, completion tokens, total tokens per LLM call
- **Refinement:** When a tool result is condensed by the secondary LLM, both raw and refined content are shown

**Streamlit frontend:** The `Admin Debug Secret` text input in the sidebar ([app.py:70](../app.py#L70)) passes the secret as `X-Debug-Key` header. When set, the expanders "Reasoning" and "Tool Logs" are shown expanded by default.

---

## 6. Session & History Management

### In-Memory Session Store

**File:** [api.py:47-49](../api.py#L47-L49)

```python
active_sessions: Dict[str, Orchestrator] = {}
session_lock = asyncio.Lock()
```

Maps `user_id` → `Orchestrator` instance. Thread-safe via `asyncio.Lock`. Sessions persist for the process lifetime.

### Redis Session Store

**File:** [redis_store.py](../cogops/session/redis_store.py)

Redis is the persistent session backend. If Redis is unavailable, the store gracefully falls back to disabled mode (no persistence, no summarization, no clarification resolution).

| Redis Key | Type | Content | TTL |
|---|---|---|---|
| `session:{user_id}:turns` | List (LPUSH) | JSON objects: `{"turn_id": "...", "user": "...", "assistant": "..."}` | 86400s (1 day) |
| `session:{user_id}:summary` | String | Rolling conversation summary (updated by background task) | 86400s (1 day) |
| `session:{user_id}:clarification` | String | JSON: `{"question": "...", "options": [...], "reason": "...", "turn_id": "..."}` | 86400s (1 day) |
| `session:{user_id}:last_assistant` | String | JSON: `{"assistant_text": "...", "options": [...], "turn_id": "..."}` | 86400s (1 day) |

**Key operations:**
- `store_turn(user_id, turn)` — Append a turn to the list
- `get_recent_turns(user_id, n=5)` — Get last N turns
- `get_summary(user_id)` / `set_summary(user_id, summary)` — Rolling summary
- `get_clarification(user_id)` / `set_clarification(user_id, q)` / `clear_clarification(user_id)` — Pending clarification
- `get_last_assistant_meta(user_id)` / `set_last_assistant_meta(user_id, meta)` — Last reply for short-followup resolution
- `clear_all(user_id)` — Delete all keys for a user (used by `/session/clear`)

### Local In-Memory History

**File:** [orchestrator.py:94-95](../cogops/agents/orchestrator.py#L94-L95)

```python
self.feedback_history: List[Dict[str, Any]] = []
self.history: List[Tuple[str, str]] = []
```

- `self.history`: List of (user_query, assistant_response) tuples. Only in memory. Cleared on `clear_session()`.
- `self.feedback_history`: List of feedback entries (max 5). Negative feedback (`bad`, `unhelpful`, `wrong`) surfaces to system context via `get_negative_feedback()`.

### Query Log

**File:** [query_log.py](../cogops/session/query_log.py)

Append-only JSONL file (`data/query_log.jsonl`) storing query text and ISO-8601 timestamp (Bangladesh timezone, UTC+6). Each write triggers a prune of entries older than 10 days. Accessible via `GET /query-log`.

### Clear Session

**File:** [api.py:160-170](../api.py#L160-L170)

```
POST /session/clear
Body: {"user_id": "..."}
```

Clears all Redis keys for the user, removes the Orchestrator from `active_sessions`, and resets `self.history` and `self.feedback_history`.

### Feedback

**File:** [api.py:144-158](../api.py#L144-L158)

```
POST /feedback
Body: {"user_id": "...", "turn_id": "...", "rating": "good"\|"bad"\|"unhelpful"\|"wrong", "comment": "..."}
```

Records feedback in `self.feedback_history`. Negative feedback is surfaced to the system prompt via `get_negative_feedback()`.

### Rolling Summarizer

**File:** [summarizer.py](../cogops/session/summarizer.py)

After each answer, a background task sends the last user/assistant turn to the secondary LLM to produce an updated rolling summary. The summary preserves: unresolved questions, entities mentioned, user preferences. It drops: resolved details, exact quotes. Max 300 tokens (configurable via `SUMMARIZER_MAX_TOKENS`).

---

## 7. Configuration & Environment Variables

### Environment Variables (.env)

| Variable | Controls | Default | File Location |
|---|---|---|---|
| `TRITON_URL` | Embedding server URL | `localhost:6000` | [orchestrator.py:77-78](../cogops/agents/orchestrator.py#L77-L78), [client.py:43](../cogops/graph/client.py#L43) |
| `TRITON_MODEL_NAME` | Gemma embedding model name | `gemma_embedding` | [client.py:44](../cogops/graph/client.py#L44) |
| `TRITON_TOKENIZER` | Embedding tokenizer model | `onnx-community/embeddinggemma-300m-ONNX` | [client.py:45](../cogops/graph/client.py#L45) |
| `LLM_BASE_URL` | Primary LLM endpoint | `http://localhost:5000/v1/` | [clients.py:59](../cogops/llm/clients.py#L59) |
| `LLM_API_KEY` | Primary LLM API key | — | [clients.py:59](../cogops/llm/clients.py#L59) |
| `LLM_MODEL_NAME` | Primary LLM model name | `qwen3-primary` | [clients.py:66](../cogops/llm/clients.py#L66) |
| `RERANKER_BASE_URL` | Reranker endpoint | `http://localhost:5001/v1/` | [clients.py:60](../cogops/llm/clients.py#L60) |
| `RERANKER_API_KEY` | Reranker API key | — | [clients.py:60](../cogops/llm/clients.py#L60) |
| `RERANKER_MODEL_NAME` | Reranker model name | `qwen3-reranker` | [clients.py:60](../cogops/llm/clients.py#L60) |
| `SECONDARY_BASE_URL` | Secondary LLM endpoint | `http://localhost:5002/v1/` | [clients.py:61](../cogops/llm/clients.py#L61) |
| `SECONDARY_API_KEY` | Secondary LLM API key | — | [clients.py:61](../cogops/llm/clients.py#L61) |
| `SECONDARY_MODEL_NAME` | Secondary LLM model name | `qwen3-secondary` | [clients.py:61](../cogops/llm/clients.py#L61) |
| `TOKENIZER_MODEL_NAME` | HuggingFace tokenizer model | `Qwen/Qwen2.5-32B-Instruct` | [orchestrator.py:100-106](../cogops/agents/orchestrator.py#L100-L106) |
| `NEO4J_URI` | Neo4j connection URI | `bolt+ssc://localhost:7687` | [client.py:62](../cogops/graph/client.py#L62) |
| `NEO4J_USER` | Neo4j username | `neo4j` | [client.py:63](../cogops/graph/client.py#L63) |
| `NEO4J_PASSWORD` | Neo4j password | — | [client.py:64](../cogops/graph/client.py#L64) |
| `NEO4J_DATABASE` | Neo4j database name | `qwen34neo4j` | [client.py:65](../cogops/graph/client.py#L65) |
| `ADMIN_DEBUG_SECRET` | Debug mode secret | `""` (empty = disabled) | [api.py:114](../api.py#L114) |
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379/0` | [redis_store.py:34](../cogops/session/redis_store.py#L34) |
| `REDIS_SESSION_TTL_SECONDS` | Session TTL | `86400` (1 day) | [redis_store.py:35](../cogops/session/redis_store.py#L35) |
| `SUMMARIZER_MAX_TOKENS` | Max tokens for rolling summary | `300` | [orchestrator.py:128-131](../cogops/agents/orchestrator.py#L128-L131) |

### Configuration YAML (configs/config.yml)

| Section | Parameter | Purpose | Default |
|---|---|---|---|
| `agent_name` | — | Agent name shown in system prompt | `"আশা"` |
| `agent_story` | — | Agent identity description | Bengali description of being a Bangladeshi government digital assistant |
| `llm.thinking` | `true/false` | Enable native thinking in primary LLM | `true` |
| `llm.max_context_tokens` | — | Max context window size | `32000` |
| `reranker.max_context_tokens` | — | Reranker context window | `32000` |
| `secondary.max_context_tokens` | — | Secondary LLM context window | `32000`
| `secondary.grep_passage.context_lines` | — | Context lines around grep matches | `2` |
| `secondary.extract_from_document.max_tokens` | — | Max output tokens for extract | `2048` |
| `secondary.delegate_task.max_tokens` | — | Max output tokens for delegate | `2048` |
| `secondary.spawn_subagent.max_turns` | — | Max reasoning turns for sub-agent | `5` |
| `reasoning.max_turns` | — | Max iterations in reasoning loop | `10` |
| `graphiti.search.limit` | — | Max results for graph_search | `5` |
| `graphiti.search.min_score` | — | Min relevance score threshold | `0.5` |
| `graphiti.entity_search.max_results` | — | Max entity search results | `10` |
| `graphiti.node_explore.max_results` | — | Max node explore results | `100` |
| `graphiti.relation_browse.top_n` | — | Max relation types shown | `100` |
| `graphiti.relation_filter.max_results` | — | Max relation filter results | `50` |
| `graphiti.similar_entities.max_results` | — | Max similar entity results | `10` |
| `graphiti.similar_entities.min_score` | — | Min cosine similarity threshold | `0.5` |
| `graphiti.path_find.max_hops` | — | Max path hops | `3` |
| `graphiti.path_find.max_paths` | — | Max paths to return | `5` |
| `graphiti.episodic_search.max_results` | — | Max episodic search results | `10` |
| `graphiti.graph_stats.detail_level` | — | Default detail level for stats | `"basic"` |
| `history_query.default_recent_n` | — | Default turns for recent mode | `3` |
| `summarizer.max_tokens_env` | — | Env var name for summarizer token limit | `"SUMMARIZER_MAX_TOKENS"` |
| `post_tool_refine.enabled` | — | Enable post-tool refinement | `true` |
| `post_tool_refine.threshold_tokens` | — | Token threshold for triggering refine | `600` |
| `token_management.system_prompt_reservation` | — | Tokens reserved for system prompt | `3500` |
| `llm_call_parameters.thinking_general.*` | — | Temperature, top_p, top_k, etc. for thinking mode | temp=1.0, top_p=0.95 |
| `llm_call_parameters.instruct_general.*` | — | Generation params for normal instructions | temp=0.7, top_p=0.8 |
| `llm_call_parameters.instruct_reasoning.*` | — | Generation params for reasoning calls | temp=1.0, top_p=1.0 |
| `llm_call_parameters.max_tokens` | — | Max output tokens per LLM call | `2048` |
| `response_templates.error_fallback` | — | Bengali error message on failures | `"একটি প্রযুক্তিগত ত্রুটির কারণে..."` |

---

## 8. Docker Infrastructure

| Directory | Purpose | Service |
|---|---|---|
| `dockers/embGemmaTriton/` | Triton Inference Server serving Gemma embedding model | Provides 768-dim vectors for Graphiti hybrid search |
| `dockers/neo4j/` | Neo4j graph database with Graphiti integration | Knowledge graph storage (nodes, edges, episodes) |
| `dockers/qwen3/` | Qwen3 LLM serving via vLLM | Primary LLM endpoint for agent reasoning + tool calling |
| `dockers/redis/` | Redis session store | Persistent session data, turns, summaries, clarifications |

### Inter-Service Communication

```
Agent API (port 9000)
    ├──▶ Neo4j (bolt://localhost:7687) — Graphiti graph operations
    ├──▶ Triton (localhost:6000)        — Text embedding
    ├──▶ Primary LLM (localhost:5000)  — Chat completions, tool calls, native thinking
    ├──▶ Reranker (localhost:5001)     — Binary classification (0/1)
    ├──▶ Secondary LLM (localhost:5002)| — Sub-agents, extract, delegate, refine
    └──▶ Redis (localhost:6379)        — Session state
```

---

## 9. Additional Services

### Evaluation CLI

**File:** [evaluate.py](../evaluate.py)

Two modes for testing the orchestrator:

```bash
# Interactive mode — type queries, see full results, optionally save JSON
python evaluate.py --mode interactive

# Batch mode — run queries from CSV, save individual JSON reports + summary
python evaluate.py --mode batch --csv evaluation/query.csv --output reports/
```

Each evaluation captures all events: reasoning, tool calls, tool results, final response, clarifications, and timing.

### Streamlit Frontend

**File:** [app.py](../app.py)

Web UI providing:
- Chat interface with streaming responses
- Debug panels (expandable Reasoning and Tool Logs) when debug key is entered
- Session clear button
- Clarification option buttons for interactive disambiguation
- User ID tracking (random 8-char per browser session)

### Production Service

**File:** [govtchat.service](../govtchat.service)

systemd unit file for deploying the API as a persistent service.

---

## 10. Key Design Decisions & Flow Guarantees

| Decision | Enforcement | Rationale |
|---|---|---|
| **Tool-first rule** | `tool_choice="required"` on reasoning loop turn 1 | Never answer factual questions from parametric knowledge — always from the knowledge graph |
| **`answer_directly` meta-tool** | Satisfies the tool-first constraint for chit-chat/identity/safety | Prevents the model from being forced to call graph tools for non-factual interactions |
| **Never answers from training data** | System prompt: "Never answer a factual question from your own training data" | Bangladesh government data may be wrong or outdated in training data |
| **Post-tool refinement** | Secondary LLM condenses results >600 tokens | Keeps context window manageable, prevents token budget overflow |
| **Short followup resolution** | Detects short/numeric replies, injects last assistant context | Users reply "3" or "the second one" expecting the model to understand the reference |
| **Rolling summary** | Background task updates after each turn, prepended to system prompt | Maintains conversation context across turns without consuming the full token budget |
| **Fallback strategy** | System prompt instructs model to try different tools/keywords if first search fails | Bangla/English transliteration differences, varied entity naming |
| **Channel-based event filtering** | Events tagged `user`/`debug`/`both`, filtered at API layer | Clean user experience by default, full transparency with debug auth |
| **Graceful Redis fallback** | If Redis unavailable, operations continue without persistence | System degrades gracefully (no summarization, no clarification resolution, no history queries) |
| **Retry with exponential backoff** | LLM calls retry up to 3 times with exponential wait | Handles transient LLM endpoint failures |

---

## Appendix: API Endpoint Reference

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/chat/stream` | `X-Debug-Key` header (optional) | Main chat endpoint. Streams NDJSON events. |
| `POST` | `/session/clear` | — | Clears conversation history for a user. |
| `POST` | `/feedback` | — | Submit feedback for a specific turn. |
| `GET` | `/query-log` | — | Return stored queries (last 10 days). |
| `GET` | `/health` | — | System status and active session count. |

### Request/Response Examples

**Chat request:**
```json
POST /chat/stream
Body: {"user_id": "abc123", "query": "পাসপোর্ট ফি কত?"}

Streaming response (NDJSON):
{"type":"tool_call","channel":"debug","tool_calls":[...],"turn":1}
{"type":"tool_result","channel":"debug","call_id":"...","content":"...","status":"ok"}
{"type":"answer_chunk","channel":"user","content":"পাসপোর্ট"}
{"type":"answer_chunk","channel":"user","content":"ের ফি..."}
{"type":"answer_complete","channel":"both","turn_id":"..."}
```

**Clear session request:**
```json
POST /session/clear
Body: {"user_id": "abc123"}

Response: {"status": "success", "message": "Session cleared."}
```

**Feedback request:**
```json
POST /feedback
Body: {"user_id": "abc123", "turn_id": "xyz789", "rating": "bad", "comment": "উত্তর ভুল"}

Response: {"status": "success", "message": "Feedback recorded."}
```
