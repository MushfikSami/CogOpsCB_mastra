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
│  6. BUILD MESSAGES:                                                 │
│     [system_prompt (system),                                         │
│      date_line + rolling_summary + user_query (user)]                │
│                                                                      │
│  7. TOKEN BUDGET TRUNCATION:                                        │
│     max_ctx = 32000 - system_prompt_reservation (3500)              │
│     Drop oldest messages to fit budget                              │
│                                                                      │
│  8. BUILD TOOL CONTEXT:                                             │
│     ToolContext(user_id, store=RedisSessionStore)                   │
│     bind_tools(raw_tool_map, ctx) → bound_tool_map                  │
│                                                                      │
│  9. SET UP REACT REASONING LOOP:                                    │
│      extra_body = { max_tokens: 2048 } (no thinking toggle)         │
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
│    │  tool_choice = "auto" (always, every turn)              │     │
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
│    │                                                         │     │
│    │  ASSEMBLE:                                             │     │
│    │    - If tool_calls exist:                               │     │
│    │      → yield tool_call event (debug)                   │     │
│    │      → EXECUTE tools (see below)                       │     │
│    │    - If NO tool_calls:                                 │     │
│    │      → IS LAST TURN, break loop                        │     │
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
│  │  4. Yield tool_result event (debug)                      │     │
│  │  5. APPEND tool result message to transcript             │     │
│  │                                                          │     │
│  │  No answer_directly sentinel — it is a system-prompt     │     │
│  │  protocol, not a tool. The model writes text directly    │     │
│  │  when it matches greeting/identity/safety categories.    │     │
│  │  No ask_user tool — the model asks clarifying questions  │     │
│  │  directly as text.                                       │     │
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
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  POST-PROCESSING  (orchestrator.py)                                  │
│                                                                      │
│  10. ACCUMULATE answer chunks → final_response                      │
│  11. yield answer_complete event                                    │
│                                                                      │
│  12. PERSIST TO REDIS (if user_id):                                 │
│      store_turn(user_id, turn_id, user_query, final_response)       │
│      set_last_assistant_meta(user_id, assistant_text, turn_id)      │
│                                                                      │
│  13. BACKGROUND ROLLING SUMMARY (asyncio.create_task):              │
│      Fires secondary LLM to update rolling summary in Redis          │
│                                                                      │
│  14. YIELD answer_complete event → end                              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Event Types (NDJSON Stream)

Each event is a JSON line with `type`, `channel` (user/debug/both), and event-specific fields.

| Event Type | Channel | Description |
|---|---|---|
| `turn_start` | debug | Turn N of max_turns |
| `reasoning_chunk` | debug | Part of model's internal thought |
| `answer_chunk` | both | Model's text output (accumulated for final response) |
| `tool_call` | debug | Tool name, arguments, turn number |
| `tool_result` | debug | Tool output, duration, status |
| `turn_end` | debug | Turn completed |
| `answer_complete` | both | Full response finished |
| `error` | user | Error message |

---

## System-Prompt Protocols (Not Tools)

These are behaviors instructed in the system prompt, NOT tool calls:

| Protocol | When Model Uses It | What Happens |
|---|---|---|
| **Direct Reply** | Greeting, identity, safety/deflect, abuse, illegal, no_info_found | Model writes text as-is. No tool call. The reasoning loop delivers it naturally. |
| **Ask User** | Ambiguous query where the model needs clarification | Model writes a question as text. The orchestrator collects the user's reply and re-invokes the loop. |

The remaining 3 tools are actual tool calls:
- `search_knowledge` — Jiggasha government services
- `search_wiki` — Wikipedia general knowledge
- `history_query` — Conversation history lookup

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
| Max concurrent queries | `reasoning.max_concurrent_query` | 2 | System prompt |
| Context tokens | `llm.max_context_tokens` | 32000 | Message truncation |
| System prompt reservation | `token_management.system_prompt_reservation` | 3500 | Truncation budget |
| Redis TTL | `session.ttl_default` | 86400 | Session storage |
