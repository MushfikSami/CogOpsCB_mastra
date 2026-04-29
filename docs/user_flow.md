# GovOps Agent — User Request Flow

Complete diagram of a user request from submission to response.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER SENDS QUERY                             │
│                         POST /chat/stream                            │
│                         { user_id, query }                           │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  API LAYER  (api.py: /chat/stream)                                  │
│                                                                      │
│  1. Check: query empty? → HTTP 400                                  │
│  2. Log query → query_log.jsonl                                     │
│  3. Get/create agent session → Orchestrator                         │
│  4. PRE-FILTER: len(query) > max_input_chars?                       │
│     → YES → yield error event ("প্রশ্নটি খুব বড়...")                 │
│     → return immediately (model never sees input)                   │
│  5. NO → proceed to orchestrator                                    │
│                                                                      │
│  Configurable from configs/config.yml:                              │
│    reasoning.max_input_chars: 1000                                  │
│    reasoning.large_input_error: "প্রশ্নটি..."                         │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  ORCHESTRATOR  (cogops/agents/orchestrator.py)                       │
│                                                                      │
│  6. SHORT FOLLOW-UP RESOLUTION:                                     │
│     _is_short_followup(original_query, max_chars=16)?                │
│     → YES → inject previous assistant reply as context              │
│     → Extract enumerated options from previous reply                  │
│     → Prepend context before sending to model                       │
│     → NO → pass original query unchanged                           │
│                                                                      │
│  7. BUILD MESSAGES:                                                 │
│     [system_prompt + date_line + rolling_summary, user_query]       │
│                                                                      │
│  8. TOKEN BUDGET TRUNCATION:                                        │
│     max_ctx = 32000 - system_prompt_reservation (3500)              │
│     Drop oldest messages to fit budget                              │
│                                                                      │
│  9. BUILD TOOL CONTEXT:                                             │
│     ToolContext(user_id, store=RedisSessionStore)                   │
│     bind_tools(raw_tool_map, ctx) → bound_tool_map                  │
│                                                                      │
│  10. SET UP REACT REASONING LOOP:                                   │
│      extra_body = thinking params + max_tokens                      │
│      thinking = enabled based on config                             │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  REACT REASONING LOOP                                                │
│  (cogops/llm/reasoning_loop.py: stream_with_tool_calls)             │
│                                                                      │
│  REPEAT (up to max_turns=10):                                       │
│                                                                      │
│    ┌──────────────────────────────────────────────────────────┐     │
│    │ TURN N:                                                 │     │
│    │                                                         │     │
│    │  tool_choice = "required" (turn 1)                      │     │
│    │  tool_choice = "auto"     (turn 2+)                     │     │
│    │                                                         │     │
│    │  LLM API CALL (streaming):                              │     │
│    │    client_llm.chat.completions.create(                  │     │
│    │      model, messages, tools, tool_choice,               │     │
│    │      stream=True, extra_body, ...                        │     │
│    │    )                                                    │     │
│    │                                                         │     │
│    │  STREAM CHUNKS:                                        │     │
│    │    - reasoning_chunk      → debug channel              │     │
│    │    - answer_chunk         → both channel (live)        │     │
│    │    - usage                → debug channel (final)      │     │
│    │                                                         │     │
│    │  ASSEMBLE:                                             │     │
│    │    - If tool_calls exist:                               │     │
│    │      → yield tool_call event (debug)                   │     │
│    │      → EXECUTE tools (see below)                       │     │
│    │    - If NO tool_calls:                                 │     │
│    │      → turn 1 error (shouldn't happen)                 │     │
│    │      → turn 2+: IS LAST TURN, break loop               │     │
│    │                                                         │     │
│    │  APPEND assistant message to transcript                 │     │
│    └──────────────────────────────────────────────────────────┘     │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────┐     │
│  │ TOOL EXECUTION (per tool call):                          │     │
│  │                                                          │     │
│  │  1. Parse function name + JSON arguments                 │     │
│  │  2. Execute via asyncio (async or sync)                  │     │
│  │  3. Capture raw result                                   │     │
│  │                                                          │     │
│  │  4. CHECK: answer_directly sentinel?                     │     │
│  │     → YES → yield tool_result (debug)                   │     │
│  │          → stream text to user in chunks                 │     │
│  │          → IS LAST TURN, return                         │     │
│  │                                                          │     │
│  │  5. NO → yield tool_result event (debug)                │     │
│  │       → APPEND tool result message to transcript         │     │
│  │                                                          │     │
│  │  [Post-tool refine removed — no secondary LLM]          │     │
│  └──────────────────────────────────────────────────────────┘     │
│                                                                      │
│  ERROR HANDLING:                                                    │
│    - BadRequestError (context length) → ContextLengthExceededError  │
│    - Other → yield error event, raise                              │
│    - Retry: exponential backoff (3 attempts) for network errors    │
│                                                                      │
│  EXIT:                                                            │
│    - Reached max turns → log warning                              │
│    - Model stopped tool-calling → normal exit                     │
│    - answer_directly short-circuit → early exit                   │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  POST-PROCESSING  (orchestrator.py)                                  │
│                                                                      │
│  11. ACCUMULATE answer chunks → final_response                      │
│  12. yield answer_complete event                                    │
│                                                                      │
│  13. PERSIST TO REDIS (if user_id):                                 │
│      store_turn(user_id, turn_id, user_query, final_response)       │
│      set_last_assistant_meta(user_id, assistant_text, options)      │
│                                                                      │
│  [Background summarizer removed — no secondary LLM]                │
│                                                                      │
│  14. YIELD next chunk / end stream                                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Event Types (NDJSON Stream)

Each event is a JSON line with `type`, `channel` (user/debug/both), and event-specific fields.

| Event Type | Channel | Description |
|---|---|---|
| `turn_start` | debug | Turn N of max_turns |
| `reasoning_chunk` | debug | Part of model's internal thought |
| `answer_chunk` | both | Live streaming of user-visible text |
| `tool_call` | debug | Tool name, arguments, turn number |
| `tool_result` | debug | Tool output, duration, status |
| `direct_answer` | debug | Category of answer_directly call |
| `usage` | debug | Token counts (prompt, completion, total) |
| `turn_end` | debug | Turn completed |
| `answer_complete` | both | Full response finished |
| `error` | user | Error message |

---

## Debug Mode

When the `X-Debug-Key` header matches `ADMIN_DEBUG_SECRET`:
- Includes `debug` and `both` channel events
- Excludes `user`-only events

Without debug key:
- Includes `both` channel events only (user-visible)
- Excludes `debug`-only events (internal reasoning, tool metadata)

---

## Configuration Points

| Setting | Config Path | Default | Where Applied |
|---|---|---|---|
| Max input chars | `reasoning.max_input_chars` | 1000 | API pre-filter |
| Large input error | `reasoning.large_input_error` | "প্রশ্নটি খুব বড়..." | API pre-filter |
| Max turns | `reasoning.max_turns` | 10 | Reasoning loop |
| Short follow-up max chars | `reasoning.short_followup_max_chars` | 16 | Orchestrator |
| Max concurrent queries | `reasoning.max_concurrent_query` | 2 | System prompt |
| Context tokens | `llm.max_context_tokens` | 32000 | Message truncation |
| System prompt reservation | `token_management.system_prompt_reservation` | 3500 | Truncation budget |
| Redis TTL | `session.ttl_default` | 86400 | Session storage |
