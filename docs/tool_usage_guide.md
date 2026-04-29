# GovOps Agent (আশা) — Tool Usage Guide

## Architecture Overview

The agent is a **ReAct (Thought-Action-Observation)** agent powered by Qwen3 with native thinking. The reasoning loop runs for up to 10 turns (configurable):

1. Every turn uses `tool_choice="auto"` — the model freely decides when to call tools
2. Executes called tools and feeds results back
3. The model goes through **THOUGHT → ACTION → OBSERVATION** cycles

The system prompt contains the complete decision logic. Below is the structured extraction.

---

## System-Prompt Protocols (Not Tools)

`answer_directly` and `ask_user` are **NOT tools** anymore. They are system-prompt protocols where the model simply writes text — no function call.

### Direct Reply Protocol

When the model's intent matches these categories, it writes the answer directly as text:

| Situation | Behavior |
|---|---|
| Greeting, small talk | Friendly greeting in Bengali |
| "Who are you?" / capabilities (about yourself) | Identity reply in Bengali |
| Political/religious/controversial | Acknowledge AI role, decline, offer services |
| Abusive/insulting input | Ask politely for civil language |
| Illegal/dangerous request | Refuse clearly |
| Both searches returned nothing | Polite Bengali reply: no info available |

The model writes the text — the reasoning loop delivers it naturally.

### Ask User Protocol

When the model's intent is genuinely ambiguous and needs clarification, it writes a question directly as text (optionally with numbered options). No function call.

---

## Tool Inventory

| Tool | Category | Requires Redis |
|---|---|---|
| `search_knowledge` | Search — Government Services (Jiggasha) | No |
| `search_wiki` | Search — General Knowledge (Wikipedia) | No |
| `history_query` | Conversation History | Yes |

**3 tools total.** No `answer_directly`, no `ask_user`.

---

## Tool Breakdown

### `search_knowledge(formal_query, keyword_string)`

**When to call:**
- Any Bangladesh government service inquiry (procedures, fees, document requirements, offices, boards, departments, regulations)
- Covers 30+ services: education, passports, NID, birth/death registration, land, trade licenses, vehicles, utilities, pensions, disaster management, social safety, law & security, health

**Parameters:**
- `formal_query`: Exact question in formal Bengali (বাংলা) — as on an official government form
- `keyword_string`: Space-separated Bengali keywords (3-8 words) — key terms that appear in the database

**What the agent sees:** `combined_context` — ranked passages from the government database
**Debug logs show:** node paths, full text, relevance scores per result

**Fallback rule:** If this returns no results, call `search_wiki` next with the **same** parameters.

---

### `search_wiki(formal_query, keyword_string)`

**When to call:**
- General knowledge about Bangladesh, world events, history, public figures
- Fallback when `search_knowledge` returns nothing for a government query
- Non-government information requests

**Parameters:**
- `formal_query`: Same as search_knowledge
- `keyword_string`: Same as search_knowledge

**What the agent sees:** `combined_context` — Wikipedia article excerpts
**Debug logs show:** article titles, URLs, published_at timestamps

---

### `history_query(mode, query?, n?)`

**When to call:**
- Query is ambiguous and could refer to something discussed previously
- User asks about earlier turns
- Short/numeric messages that might reference prior context (e.g., "3", "second one", "tell me more")

**Modes:**

| Mode | When to Use | Parameters |
|---|---|---|
| `recent` | Resolving ambiguous follow-ups | `n` = turns to return (default 3) |
| `lookup` | Searching for a specific term | `query` = search term |
| `summarize` | Conversation overview | — |
| `ask` | Complex contextual questions | `query` = the question |

**Requires:** `user_id`, Redis store available.

---

## Complete Decision Flow

```
API receives query
        │
        ▼
┌─────────────────────────┐
│ 1. Length check         │ ── Yes ──→ error event + return
│     (configurable max)  │              (model never sees input)
└─────────────────────────┘
        │ No
        ▼
┌─────────────────────────┐
│ 2. ReAct THOUGHT:       │
│    Model classifies:    │
│                         │
│    Matches Direct Reply │
│    (greeting/identity/  │
│    safety/abuse/illegal/│
│    no_info)?            │
│    → YES → write text   │
│    → NO → continue      │
│                         │
│    Ambiguous ref?       │
│    → history_query      │
│                         │
│    Gov service?         │
│    → search_knowledge   │
│    → if empty           │
│    → search_wiki        │
│                         │
│    General knowledge?   │
│    → search_wiki        │
│                         │
│    Ambiguous?           │
│    → ask_user (text)    │
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│ 3. ReAct OBSERVATION:   │
│    Results sufficient?  │
│    → Yes → final answer │
│    → No  → THOUGHT again│
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│ 4. Final answer         │
│    → Redis persistence  │
│    → answer_complete    │
│    → background summary │
└─────────────────────────┘
```

---

## ReAct Paradigm

### Every Turn (tool_choice="auto")

The model freely decides:

| Situation | Action |
|---|---|
| Matches Direct Reply Protocol | Write text directly (no tool call) |
| Ambiguous — needs clarification | Ask user directly (no tool call) |
| Ambiguous — needs history | `history_query(mode='recent', n=3)` |
| Government service inquiry | `search_knowledge(formal_query, keyword_string)` |
| `search_knowledge` returned nothing | `search_wiki(formal_query, keyword_string)` |
| Both returned nothing | Direct Reply Protocol (write "no info" text) |
| General knowledge | `search_wiki(formal_query, keyword_string)` |

---

## Debug Mode

When the `X-Debug-Key` header matches the server secret:

**Tool call events show:**
- Which tool was called
- Parameters passed (formal_query, keyword_string, etc.)
- Call duration

**Tool result events show:**
- `combined_context` — what the model sees as observation
- Full tool output including results metadata (node paths, titles, URLs, scores)

**Reasoning events show:**
- `reasoning_chunk` — the model's internal THOUGHT process
- `answer_chunk` — streaming of user-visible text

---

## Session Management

### Redis Storage
- **Turns**: Redis list (LPUSH, most recent first), max 20 turns fetched
- **Rolling summary**: Redis string key
- **Last assistant meta**: Last reply + turn_id
- **TTL**: 86400 seconds (24 hours) default

### Message Truncation
- System prompt reserves ~3500 tokens
- Remaining budget = `max_context_tokens` (32000) - reservation
- Oldest non-system messages dropped first
- If still over budget, hard truncation on last message

---

## Key Constraints & Rules

1. **Never answer from parametric knowledge on factual queries** — always call a tool first
2. **Parallel calls preferred** — call multiple tools in a single turn, not sequentially
3. **Direct Reply and Ask User are protocols, not tools** — the model writes text, no function call
4. **Formal Bengali** for all user-facing text; search queries may use Bengali or English
5. **Never expose tool names, arguments, or internal reasoning** to the user
6. **Never reference information sources** ("according to the database") — state answers naturally
7. **Maximum 2 concurrent different questions** per response (configurable)
8. **Time awareness** — Bangladesh office hours: Sunday-Thursday 9am-5pm, Friday-Saturday closed
9. **Search chain**: Jiggasha first for gov services → Wikipedia fallback if empty → write "no info" reply if both empty
