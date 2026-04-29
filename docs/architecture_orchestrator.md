# Orchestrator & Reasoning Loop — Architecture Reference

Complete technical reference for `cogops/agents/orchestrator.py` and `cogops/llm/reasoning_loop.py`.

---

## Orchestrator (`cogops/agents/orchestrator.py`)

The Orchestrator is the central entry point for every user query. It manages configuration, tool binding, message assembly, token budgeting, and post-processing.

### Class: `Orchestrator`

#### Attributes (set in `__init__`)

| Attribute | Type | Source | Default |
|---|---|---|---|
| `config` | `Dict` | YAML config file | — |
| `agent_name` | `str` | `config.agent_name` | `"Gov Assistant"` |
| `agent_story` | `str` | `config.agent_story` | `""` |
| `llm_service` | `AsyncLLMService` | 3 LLM endpoints (primary + secondary) | — |
| `redis_store` | `RedisSessionStore` | Redis connection (per-user sessions) | — |
| `tools_schema` | `List[Dict]` | Tool JSON schemas from registry | — |
| `raw_tool_map` | `Dict[str, Callable]` | Raw tool callbacks (unbound) | — |
| `tools_desc_str` | `str` | JSON-serialized tool schemas | — |
| `system_prompt` | `str` | Cached system prompt (single instance) | — |
| `history` | `List[Tuple[str,str]]` | In-memory Q/A pairs (session-level) | `[]` |
| `max_turns` | `int` | `config.reasoning.max_turns` | `10` |
| `max_concurrent_query` | `int` | `config.reasoning.max_concurrent_query` | `2` |
| `max_input_chars` | `int` | `config.reasoning.max_input_chars` | `1000` |
| `large_input_error` | `str` | `config.reasoning.large_input_error` | Bengali rejection message |
| `summarizer_max_tokens` | `int` | env var or `config.summarizer.max_tokens_default` | `300` |
| `_tokenizer_model_name` | `str` | `config.token_management.tokenizer_model_default` or env | — |
| `system_prompt_reservation` | `int` | `config.token_management.system_prompt_reservation` | `3500` |

#### `__init__(config_path)`

1. Loads YAML config
2. Extracts agent identity (`agent_name`, `agent_story`)
3. Initializes `AsyncLLMService` with primary + secondary LLM clients
4. Creates `RedisSessionStore` from config/env
5. Builds tool registry: `build_tool_registry()` → `(tools_schema, raw_tool_map)`
6. Serializes tool schemas to JSON string for system prompt injection
7. Creates cached `system_prompt` via `get_system_prompt()` — **cached once, shared across all sessions**
8. Initializes tokenizer (used for message truncation)
9. Sets token budget reservation

#### `_initialize_llm()`

Creates `AsyncLLMService` with two endpoints:
- **Primary LLM** — main agent model (Qwen3 with ReAct reasoning)
- **Secondary LLM** — used by `history_query` (ask mode) and background summarizer

Clients are `AsyncOpenAI` instances configured from `EndpointConfig` objects loaded from `config.llm` and `config.secondary` sections.

#### `_build_tool_context(user_id)` → `ToolContext`

Creates per-request context injected into context-dependent tools:

| Field | Source | Purpose |
|---|---|---|
| `user_id` | parameter | Redis key prefix |
| `store` | `self.redis_store` | Session turns/summary access |
| `secondary_client` | `llm_service.client_secondary` | history_query "ask" mode, summarizer |
| `secondary_model` | `llm_service.llm_config.model` | Secondary LLM model name |
| `tool_map` | `self.raw_tool_map` | Full callback map (unused in current tools) |
| `tools_schema` | `self.tools_schema` | Full schema list (unused in current tools) |

#### `process_query(user_query, debug_mode, user_id)` → `AsyncGenerator[Dict, None]`

The main request pipeline:

```
process_query
  │
  ├── 1. Generate turn_id (UUID[:8])
  │
  ├── 2. Build messages
  │      system: self.system_prompt
  │      user:   date_line + rolling_summary + "\n\n" + user_query
  │      (date_line + rolling_summary are part of user content, not system)
  │
  ├── 3. Token budget truncation
  │      budget = max_context_tokens(32000) - system_prompt_reservation(3500) = ~28500
  │      Drop oldest non-system messages to fit
  │      Hard-truncate last message if still over budget
  │
  ├── 4. Build extra_body
  │      { "max_tokens": 2048 }
  │      (No thinking toggle — reasoning handled via native thinking channel)
  │
  ├── 5. Bind tools with per-request context
  │      ctx = _build_tool_context(user_id)
  │      bound_tool_map = bind_tools(raw_tool_map, ctx)
  │
  ├── 6. Run reasoning loop
  │      stream_gen = stream_with_tool_calls(
  │          client_llm, model, messages, tools_schema,
  │          bound_tool_map, max_turns, extra_body,
  │      )
  │
  ├── 7. Process stream events
  │      - Yield all debug events (turn_start, reasoning_chunk, tool_call,
  │        tool_result, turn_end)
  │      - Accumulate answer_chunk events into full_answer_accumulator
  │      (answer_chunk is NOT yielded to the caller directly)
  │
  ├── 8. On completion
  │      - final_response = joined chunks, stripped
  │      - yield answer_complete event
  │      - persist turn to Redis (store_turn)
  │      - persist last assistant meta to Redis (set_last_assistant_meta)
  │      - fire background summarizer task (asyncio.create_task)
  │
  └── 9. Error handling
        - reasoning loop error → yield error event, continue
        - critical orchestrator error → yield hardcoded Bengali fallback, return
```

**Error handling layers:**

| Layer | Scope | Behavior |
|---|---|---|
| Inner try/except | Reasoning loop | Yields `{"type": "error", "content": str(e)}`, continues |
| Outer try/except | Full pipeline | Yields hardcoded Bengali error, returns |

#### `clear_session()`
Clears `self.history`. Does NOT clear Redis.

---

## Reasoning Loop (`cogops/llm/reasoning_loop.py`)

The `stream_with_tool_calls()` function is the ReAct conversation engine. It runs a while loop that alternates between calling the primary LLM and executing tool calls, streaming events to the caller.

### Function: `stream_with_tool_calls(...)` → `AsyncGenerator[Dict, None]`

#### Parameters

| Param | Type | Description |
|---|---|---|
| `client_llm` | `AsyncOpenAI` | Primary LLM client |
| `model` | `str` | Model name |
| `messages` | `List[Dict]` | Conversation history (system + user + assistant + tool messages) |
| `tools_schema` | `List[Dict]` | Tool JSON schemas |
| `available_tools` | `Dict[str, Callable]` | Bound tool callbacks (name → function) |
| `max_turns` | `int` | Maximum reasoning turns | `10` |
| `extra_body` | `Dict` | vLLM extra params (max_tokens) |
| `**kwargs` | — | Passed to LLM API call (e.g., repetition_penalty, top_p, top_k) |

#### Top-level structure

```
while turn_count < max_turns:
    turn_count += 1
    yield turn_start(debug)

    # 1. LLM call
    stream = client_llm.chat.completions.create(...)
    #   tool_choice = "auto" (always)
    #   tools = tools_schema (always, if non-empty)
    #   stream = True

    # 2. Parse streaming chunks
    #   - reasoning chunks  → reasoning_chunk(debug)
    #   - content chunks    → answer_chunk(both)
    #   - tool call fragments → accumulate in tool_call_index_map

    # 3. Build assistant message + append to transcript
    response_message = { role: "assistant", content?, tool_calls? }

    # 4a. No tool calls → is_last_turn = True → break
    # 4b. Tool calls exist:
    #     → yield tool_call(debug)
    #     → Execute each tool:
    #       - Parse JSON args
    #       - Call function (async/sync detection via functools.partial support)
    #       - Yield tool_result(debug)
    #       - Append tool message to transcript
    # 5. Yield turn_end(debug)

    # Catch blocks:
    #   - BadRequestError (context length) → raise ContextLengthExceededError
    #   - Other → yield error(both), raise

    # Exit → log "Reached max turns" if no final answer
```

#### Turn lifecycle

Each turn follows this sequence:

1. **`turn_start` event** (debug) — Turn N of max_turns

2. **LLM API call** (`chat.completions.create`)
   - `tool_choice = "auto"` — model freely decides when to call tools
   - `tools = tools_schema` — model sees all 3 tool schemas
   - `stream = True` — streaming response
   - `extra_body` includes `max_tokens`

3. **Stream parsing** — iterates over chunks:
   - **Reasoning chunk**: Native thinking from model. Yields `reasoning_chunk(debug)` immediately
   - **Content chunk**: User-facing text. Appends to `full_content_accumulator`, yields `answer_chunk(both)` immediately
   - **Tool call fragments**: Reassembled from deltas into `tool_call_index_map`

4. **After stream completes**:
   - Assistant message assembled with `content` and `tool_calls` (if any)
   - Assistant message appended to `messages` list

5. **Branch — No tool calls**:
   - `is_last_turn = True`
   - `break` — loop ends
   - `turn_end` is NOT yielded (loop exited)

6. **Branch — Tool calls present**:
   - `tool_call` event (debug) with call IDs, function names, arguments
   - Execute tools (see below)
   - `turn_end` event (debug) at end of turn

#### Tool execution

For each tool call in the turn:

1. **Parse arguments** — `json.loads(function.arguments)`. On failure: `"Error: Invalid JSON arguments generated by model."`

2. **Dispatch to handler** — `available_tools[function_name]`
   - Async function → `await fn(**args)`
   - Sync function (functools.partial around async) → detect via `getattr(fn, "func")` + `iscoroutinefunction`
   - Sync function (truly sync) → `asyncio.to_thread(fn, **args)`
   - Function not found → `"Error: Tool 'X' not defined in available_tools_map."`

3. **Yield `tool_result` event** (debug):
   - `content`: The tool's string result
   - `duration_ms`: Execution time
   - `status`: `"ok"` or `"error"` (if starts with "Error")
   - Tool message appended to `messages` list

**Note:** `answer_directly` is no longer a tool. It is a system-prompt protocol where the model writes text directly without calling any tool. Similarly, `ask_user` is a protocol — the model writes a clarifying question as text. The reasoning loop handles only the 3 actual tools: `search_knowledge`, `search_wiki`, `history_query`.

#### Error handling

| Exception | Trigger | Behavior |
|---|---|---|
| `BadRequestError` (context length) | Prompt too large for model | Yields `error(user): "Context limit reached. Please clear session."`, raises `ContextLengthExceededError` |
| `BadRequestError` (other) | Invalid request | Logs error, re-raises |
| Generic `Exception` | Any unexpected error | Yields `error(both): "An internal error occurred."`, re-raises |
| Retryable (ConnectionError, TimeoutError, RuntimeError) | Network failures | Caught by `@retry` decorator — exponential backoff, 3 attempts max |

#### Exit conditions

| Condition | Outcome |
|---|---|
| Model produces no tool calls | `is_last_turn = True`, `break` → loop ends cleanly |
| `max_turns` reached | Log warning, loop ends |
| Context length exceeded | Raises `ContextLengthExceededError` (propagates to orchestrator) |

---

## Event Flow

Events are tagged with a channel:

| Channel | Who sees it | Events |
|---|---|---|
| `debug` | Debug mode only | `turn_start`, `reasoning_chunk`, `tool_call`, `tool_result`, `turn_end` |
| `both` | Everyone (user + debug) | `answer_chunk` |
| `user` | User mode only | `error` |

### Full event sequence for a typical turn (with tool calls):

```
turn_start(debug)
reasoning_chunk(debug)   ← model's internal thought
reasoning_chunk(debug)
... (more reasoning)
tool_call(debug)         ← model decided to call a tool
tool_result(debug)       ← tool execution result (raw content, duration, status)
turn_end(debug)          ← turn complete
```

For a final turn (model produces no tool call):

```
turn_start(debug)
reasoning_chunk(debug)
answer_chunk(both)       ← model's final answer text
answer_chunk(both)
...
→ break (loop ends)
→ no turn_end yielded
```

**System-prompt protocol turns (no tool call):**
The model writes text directly when it matches a Direct Reply or Ask User protocol. These look identical to a final turn in the event stream — just `turn_start`, reasoning, answer_chunks, then break.

---

## Data Flow Between Orchestrator and Reasoning Loop

```
Orchestrator                              Reasoning Loop
─────────────                              ──────────────
process_query()
  │
  ├─ Build messages:
  │  system: system_prompt
  │  user:   date + summary + query
  │
  ├─ Truncate to token budget
  │
  ├─ Bind tools (per-request context)
  │
  ├─ stream_with_tool_calls(...)
  │    ← messages (grows each turn)
  │    ← tools_schema (3 tools)
  │    ← bound_tool_map
  │    └─ yield events
  │
  │    ← stream_gen (AsyncGenerator)
  │       ├─ yield debug events
  │       └─ accumulate answer_chunk
  │
  ├─ final_response = joined chunks
  │
  ├─ persist to Redis
  │
  └─ fire summarizer task (async)
```

Key invariant: **messages list grows across turns**. Each turn appends:
1. Assistant message (content + tool_calls)
2. For each tool call: tool message (role="tool", content=result)

The system prompt and date_line/rolling_summary are in the initial system message at index 0.

---

## Configuration Mapping

| Orchestrator attribute | Config path | Default |
|---|---|---|
| `max_turns` | `reasoning.max_turns` | `10` |
| `max_concurrent_query` | `reasoning.max_concurrent_query` | `2` |
| `max_input_chars` | `reasoning.max_input_chars` | `1000` |
| `large_input_error` | `reasoning.large_input_error` | Bengali rejection |
| `summarizer_max_tokens` | `summarizer.max_tokens_default` | `300` |
| `system_prompt_reservation` | `token_management.system_prompt_reservation` | `3500` |
| LLM max_tokens | `llm_call_parameters.max_tokens` | `2048` |
| Max context tokens | `llm.max_context_tokens` | `32000` |
| Secondary LLM model | `secondary.model_name_env` | env var |
| Secondary LLM base_url | `secondary.base_url_env` | env var |
