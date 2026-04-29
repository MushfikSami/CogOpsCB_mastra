# GovOps Agent (আশা) — ReAct Tool Usage Guide

## Complete Decision Matrix: When to Call Each Tool

This document describes every tool in the **আশা** (Asha) Bangladesh Government Service Agent, the conditions under which it should be called, and the complete decision flow.

---

## Architecture Overview

The agent is a **ReAct (Thought-Action-Observation)** agent powered by Qwen3 with native thinking. The reasoning loop runs for up to 10 turns (configurable):

1. On **Turn 1**, forces `tool_choice="required"` — the model MUST call some tool
2. On **Turn 2+**, uses `tool_choice="auto"` — the model may stop calling tools
3. Executes called tools and feeds results back
4. The model goes through **THOUGHT → ACTION → OBSERVATION** cycles

The system prompt contains the complete decision logic. Below is the structured extraction.

---

## Tool Inventory

| Tool | Category | Requires Redis |
|---|---|---|
| `search_knowledge` | Search — Government Services (Jiggasha) | No |
| `search_wiki` | Search — General Knowledge (Wikipedia) | No |
| `history_query` | Conversation History | Yes |
| `ask_user` | Interaction — Clarification | No |
| `answer_directly` | Interaction — Direct Response | No |

---

## Tool Breakdown

### Search Tools

#### `search_knowledge(formal_query, keyword_string)`

**When to call:**
- Any Bangladesh government service inquiry (procedures, fees, document requirements, offices, boards, departments, regulations)
- Covers 30+ services: education, passports, NID, birth/death registration, land, trade licenses, vehicles, utilities, pensions, disaster management, social safety, law & security, health

**Parameters:**
- `formal_query`: Exact question in formal Bengali (বাংলা) — as on an official government form
- `keyword_string`: Space-separated Bengali keywords (3-8 words) — key terms that appear in the database

**What the agent sees:** `combined_context` — ranked passages from the government database
**Debug logs show:** node paths, full text, relevance scores per result

**Fallback rule:** If this returns no results, call `search_wiki` next with the **same** parameters.

**Key prompt text:**
> "ALWAYS call search_knowledge first for Bangladesh government service queries."

---

#### `search_wiki(formal_query, keyword_string)`

**When to call:**
- General knowledge about Bangladesh, world events, history, public figures
- Fallback when `search_knowledge` returns nothing for a government query
- Non-government information requests

**Parameters:**
- `formal_query`: Same as search_knowledge
- `keyword_string`: Same as search_knowledge

**What the agent sees:** `combined_context` — Wikipedia article excerpts
**Debug logs show:** article titles, URLs, published_at timestamps

**Key prompt text:**
> "If search_knowledge returns no results, call search_wiki with the same formal_query and keyword_string."

---

### Interaction Tools

#### `answer_directly(category, text)`

**Purpose:** Meta-tool satisfying the "must call a tool" constraint for non-factual replies. The reasoning loop detects a sentinel string (`__ANSWER_DIRECTLY__::`) and streams the reply directly, then ends.

**Categories:**

| Category | When to Use |
|---|---|
| `chitchat` | Greetings, small talk, pleasantries |
| `identity` | Questions about YOUR OWN identity ("who are you?") — NOT third parties |
| `safety_deflect` | Political, religious, controversial opinions |
| `abuse` | Abusive/insulting user messages |
| `illegal` | Weapons, violence, tax evasion, hacking |
| `no_info_found` | Both search_knowledge and search_wiki returned nothing |

**When to call:**
- Non-factual queries (greetings, identity, safety)
- Gibberish/nonsense input (model decides this)
- Both search tools returned no results

**When NOT to call:**
- **Never as a first tool on a factual query** — search first, answer later

**Mechanics:**
- Returns sentinel `__ANSWER_DIRECTLY__::category::text`
- Reasoning loop detects sentinel and streams text to user (12 chars, 15ms apart)
- Immediately ends — no further turns

---

#### `ask_user(question, options?, reason?)`

**When to call:**
- Query is genuinely ambiguous between multiple services
- You recognize a category exists but the user hasn't specified which one

**Examples:**
| Ambiguous Input | Multiple Options |
|---|---|
| "লাইসেন্স নবায়ন করতে চাই" | trade, vehicle, professional license |
| "সার্টিফিকেট লাগবে" | birth, death, income, residence |
| "কর দিতে চাই" | income tax, land tax, VAT |
| "পাসপোর্ট" | new, renewal, emergency, cancellation |

**How to use:** Provide 2-4 concrete options in Bangla.

**When NOT to call:** Before attempting a search — only use after search narrows to candidates.

---

### History Tool

#### `history_query(mode, query?, n?)`

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

**Key prompt text:**
> "Use when the query is ambiguous and could refer to something discussed previously — call history_query(mode='recent', n=3) first to check if the user is referencing a prior turn, then decide."

**Orchestrator-level follow-up resolution:**
- Before the model sees the query, the orchestrator checks if it's a short follow-up
- If so, it injects the previous assistant reply as context
- The model can then use `history_query` if it needs more context

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
│ 2. Short follow-up?     │ ── Yes ──→ Orchestrator injects
│     (≤16 chars)          │    previous reply as context
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│ 3. ReAct THOUGHT:       │
│    Model classifies:    │
│                         │
│    Non-factual?         │
│    → answer_directly    │
│                         │
│    Gibberish?           │
│    → answer_directly    │
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
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│ 4. ReAct OBSERVATION:   │
│    Results sufficient?  │
│    → Yes → final answer │
│    → No  → THOUGHT again│
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│ 5. Final answer         │
│    → Redis persistence  │
│    → answer_complete    │
└─────────────────────────┘
```

---

## ReAct Paradigm

### Turn 1 (tool_choice="required")

The model MUST call at least one tool. Classification:
- Non-factual → `answer_directly(category, text)`
- Ambiguous/follow-up → `history_query(mode='recent', n=3)`
- Gov service → `search_knowledge(formal_query, keyword_string)`
- General knowledge → `search_wiki(formal_query, keyword_string)`

### Turn 2+ (tool_choice="auto")

The model may stop calling tools when it has enough data. If not:
- If `search_knowledge` returned nothing → `search_wiki`
- If both returned nothing → `answer_directly(no_info_found)`
- If data incomplete → more tool calls (prefer parallel in single turn)

### Answer Directly Short-Circuit

1. Model calls `answer_directly(category, text)`
2. Reasoning loop detects sentinel → streams text to user
3. Immediately ends — no further turns

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
- **Last assistant meta**: Last reply + enumerated options for follow-up resolution
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
3. **Never call `answer_directly` first on a factual query** — search first, answer later
4. **Formal Bengali** for all user-facing text; search queries may use Bengali or English
5. **Never expose tool names, arguments, or internal reasoning** to the user
6. **Never reference information sources** ("according to the database") — state answers naturally
7. **Maximum 2 concurrent different questions** per response (configurable)
8. **Time awareness** — Bangladesh office hours: Sunday-Thursday 9am-5pm, Friday-Saturday closed
9. **Search chain**: Jiggasha first for gov services → Wikipedia fallback if empty → answer_directly(no_info_found) if both empty
