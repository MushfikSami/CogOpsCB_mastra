# GovOps Agent — Tool Usage Guide

## Complete Decision Matrix: When to Call Each Tool

This document describes every tool in the **আশা** (Asha) Bangladesh Government Service Agent, the conditions under which it should be called, and the complete decision flow.

---

## Architecture Overview

The agent uses a **reasoning loop** that runs for up to 10 turns (configurable). Each turn:

1. Sends messages + tool schemas to the primary LLM (Qwen3)
2. On **Turn 1**, forces `tool_choice="required"` — the model MUST call some tool
3. On **Turn 2+**, uses `tool_choice="auto"` — the model may stop calling tools
4. Executes called tools, optionally refines large results via a secondary LLM
5. Feeds results back and loops

The system prompt contains the complete decision logic. Below is the structured extraction.

---

## Tool Inventory

| Tool | Category | Tier | Requires Secondary LLM | Requires Redis | Requires GraphDB |
|---|---|---|---|---|---|
| `search_knowledge` | Knowledge Search | 1 | No | No | No |
| `history_query` | Conversation History | 1.5 | No (except `ask` mode) | Yes | No |
| `ask_user` | Interaction | 2 | No | No | No |
| `answer_directly` | Interaction (Meta-tool) | 2 | No | No | No |
| `tree_explorer` | Graph Exploration | Tier 3 | No | No | Yes (Neo4j/Graphiti) |
| `get_by_uuid` | Graph Lookup | Tier 3 | No | No | Yes (Neo4j/Graphiti) |
| `grep_passage` | Text Processing | Tier 3 | No | No | No |
| `extract_from_document` | Text Extraction | Tier 3 | Yes | No | No |
| `delegate_task` | Text Delegation | Tier 3 | Yes | No | No |

---

## Tier-by-Tier Breakdown

### Tier 1 — Call First, Always (for factual queries)

#### `search_knowledge(query, top_k=10)`

**When to call:**
- Any factual/informational query about government services, procedures, fees, regulations, offices, entities, or documents
- Any query where the model has partial knowledge — always verify with tool results
- Covers 30+ government services: birth/death registration, education, passport, land, trade license, vehicle, utility, pension, disaster management, social safety, law/security, health, etc.

**When NOT to call:**
- Chit-chat, greetings, small talk
- Identity questions ("who are you?", "what can you do?")
- Political, religious, or controversial topics
- Abusive/insulting input (use `answer_directly` with `abuse` category)
- Illegal/dangerous requests (use `answer_directly` with `illegal` category)
- Gibberish or nonsense input
- Input exceeding `max_input_chars` (default 1000)
- When the message is short/numeric and refers to previous context (use `history_query` first)

**How to formulate the query:**
- Use the most **formal Bengali** possible while maintaining user intent
- The API handles colloquial-to-formal reformulation internally
- The query should be broad and descriptive, capturing the full user intent
- English proper nouns may be included where appropriate

**What it returns:**
- Reformulated query (colloquial → formal Bengali)
- Ranked passages with node names, text content, and relevance scores
- Uses the Jiggasha Knowledge API endpoint

**Key prompt text:**
> "You MUST call `search_knowledge(query)` for ALL factual queries and whenever you are uncertain about the answer."

---

### Tier 1.5 — Check History Before Searching

#### `history_query(mode, query?, n?)`

**When to call (MUST be called BEFORE any information tool):**
- User message is **short** (<=16 characters)
- User message is **numeric** (e.g., "3", "2nd")
- User message **refers to something previously discussed** (e.g., "the second one", "tell me more", "what about that?")
- User explicitly asks about earlier turns
- Any ambiguous follow-up

**Modes:**

| Mode | When to Use | Parameters | Description |
|---|---|---|---|
| `recent` | Resolving ambiguous follow-ups; seeing what was just discussed | `n` = number of recent turns (default 3) | Returns the last N turns verbatim with user/AI text |
| `lookup` | Searching for a specific term mentioned earlier | `query` = search term | Substring/regex search across all stored Q/A pairs |
| `summarize` | Getting an overview of the conversation | — | Returns the rolling summary maintained in Redis |
| `ask` | Complex contextual questions about history ("what did we discuss about X?") | `query` = the question | Passes history + question to secondary LLM for inference |

**Configuration:**
- `default_recent_n`: 3
- `ask_turns_limit`: 10 (how many turns the `ask` mode sees)
- `default_max_turns`: 20 (max turns fetched from Redis)

**What it requires:**
- `user_id` (always required)
- Redis store must be available
- `ask` mode additionally needs secondary LLM client + model

**Key prompt text:**
> "Call `history_query(mode='recent', n=3)` FIRST to get context, then proceed with the appropriate tool."

**Short follow-up resolution (Orchestrator-level):**
- The orchestrator intercepts short follow-ups BEFORE sending to the LLM
- It injects the previous assistant reply + enumerated options into the user message
- Example: User says "3" → orchestrator appends the full previous AI response as context
- The LLM then uses `history_query(mode='recent', n=2)` if it needs more context

---

### Tier 2 — Context-Dependent Interaction Tools

#### `ask_user(question, options?, reason?)`

**When to call:**
- User's query is **genuinely ambiguous** between multiple possible services
- A search returned too many unrelated matches
- You recognize a category exists but the user hasn't specified which one

**Concrete examples from the prompt:**
| Ambiguous Input | Multiple Options |
|---|---|
| "লাইসেন্স নবায়ন করতে চাই" | trade license, vehicle license, professional license |
| "সার্টিফিকেট লাগবে" | birth, death, income, residence certificates |
| "কর দিতে চাই" | income tax, land tax, VAT |
| "পাসপোর্ট" | new, renewal, emergency, cancellation |

**How to use:**
- Provide **2-4 concrete options** for the user to choose from
- Include a `reason` explaining why clarification is needed
- Write the question in Bangla

**When NOT to call:**
- Before attempting a `search_knowledge` call — always search first
- For clearly unambiguous queries
- When the ambiguity could be resolved with more information

**Key prompt text:**
> "Only call `ask_user` after a search attempt has genuinely narrowed things down to several distinct candidates."

---

#### `answer_directly(category, text)`

**Purpose:** Meta-tool that satisfies the "must call a tool" constraint for non-factual turns. The reasoning loop detects a sentinel string (`__ANSWER_DIRECTLY__::`) and short-circuits to stream the reply directly to the user.

**Categories and when to use:**

| Category | When to Use | Response Pattern |
|---|---|---|
| `chitchat` | Greetings, small talk, pleasantries | Friendly, formal Bengali response |
| `identity` | Questions about **your own** identity or capabilities ("who are you?", "what can you do?") | State you are the Bangladesh government digital assistant |
| `safety_deflect` | Political, religious, or controversial opinion questions | Acknowledge you are an AI government assistant, decline to give opinions, offer to help with service-related topics |
| `abuse` | Abusive or insulting user messages | Ask politely for civil language, reaffirm helpfulness |
| `illegal` | Weapons, violence, tax evasion, hacking, dangerous requests | Refuse clearly; do not suggest alternatives |
| `no_info_found` | Search was performed but no relevant information found | Politely state no official information is available (in Bengali) |

**When to call:**
- The user message is a greeting, small talk, or chit-chat
- The user asks about your identity or capabilities
- The user asks political/religious/controversial questions
- The user sends abusive/insulting messages
- The user makes illegal/dangerous requests
- `search_knowledge` returned no relevant results (use `no_info_found`)
- The user message is gibberish, nonsense, or exceeds max input length

**When NOT to call:**
- **Never as a first tool on a factual query** — always search first
- Never for questions about third parties or public figures (that's `identity` only for yourself)

**Important mechanics:**
- When called, the tool returns a sentinel string `__ANSWER_DIRECTLY__::category::text`
- The reasoning loop detects this and **immediately streams the text** to the user
- No further turns are made — the conversation for this query ends
- The text is streamed in small chunks (12 chars every 15ms) for a live-streaming feel

**Key prompt text:**
> "Never call `answer_directly` as a first tool on a factual query."

---

### Tier 3 — Process Retrieved Data (Require Prior Tool Results)

These tools take **already-retrieved text** and transform it. They require a `passage` parameter with content from Tier 1/2 results.

#### `grep_passage(passage, pattern, context_lines=2)`

**When to call:**
- You have a long passage from `search_knowledge` or `tree_explorer`
- You need to find specific information within it using regex patterns
- You want to verify or locate specific facts (e.g., fee amounts, deadlines, required documents)

**Example:**
- Passage mentions multiple procedures → grep for specific fee amounts: `"fee|ফি|amount|টাক"`
- Passage is long → grep for the specific section about document requirements

**How it works:**
- Pure regex search — no LLM involved, fast and deterministic
- Returns matched lines with configurable surrounding context

**Configuration:**
- `context_lines`: default 2 (configurable under `secondary.grep_passage.context_lines`)

---

#### `extract_from_document(document, topic, secondary_client, secondary_model)`

**When to call:**
- You have a very long document/passages from search results
- You need to extract specific facts or information from it using LLM reasoning
- The document is too long to process efficiently with simple regex

**Example:**
- Search returned a multi-page procedure document → extract the step-by-step process
- Search returned a regulation document → extract the eligibility criteria

**How it works:**
- Passes document (up to 8000 chars) + topic to the secondary LLM
- Returns a concise, structured list of relevant information
- Uses `EXTRACT_PROMPT`: "Extract everything relevant from the document below about the following topic."

**Configuration:**
- `max_doc_chars`: 8000 (configurable under `secondary.extract_from_document`)
- `max_tokens`: 2048

**Requires:**
- Secondary LLM client and model must be configured
- If unavailable, returns "Secondary LLM not configured. Cannot extract."

---

#### `delegate_task(instruction, context?, secondary_client, secondary_model)`

**When to call:**
- You need to delegate pure text processing to the secondary LLM
- Summarizing already-retrieved text
- Compacting dense information into a readable format
- Extracting specific facts from retrieved text in a structured way

**Examples:**
- "Summarize the above in 5 bullet points"
- "Extract all deadlines mentioned in this text"
- "Convert this procedure into a step-by-step numbered list"

**How it works:**
- Uses `DELEGATE_PROMPT`: `{instruction}\n\nContext: {context}`
- Returns the secondary LLM's result
- More general-purpose than `extract_from_document`

**Configuration:**
- `max_tokens`: 2048

**Requires:**
- Secondary LLM client and model must be configured

---

### Graph Tools (Neo4j / Graphiti Knowledge Graph)

These tools interact with the Bangladesh Government Knowledge Graph stored in Neo4j via Graphiti.

#### `tree_explorer(query)`

**When to call:**
- The user asks about any government service, procedure, fee, document requirement, or process
- You want to explore the hierarchical knowledge structure around an entity
- You need a query-aware view of related entities, their relationships, and episodes

**What it does:**
1. Uses Graphiti hybrid search for broad retrieval
2. Applies Qwen deep semantic reranking to prune irrelevant branches
3. Builds a hierarchical tree of entities, edges, and episode summaries
4. Returns structured Markdown with tables

**CRITICAL query formulation rules:**

| User Intent | Query Strategy |
|---|---|
| Action/How-to | Include process/procedure terms (e.g., "[Document] update process") |
| Definition/What is | Specify intent (e.g., "[Entity] definition" or "purpose of [Entity]") |
| Provider/Who issues | Target the provider (e.g., "authority to issue [Document]") |
| Broad/Exploratory | ONLY if user asks intentionally vague questions, use broad "[Entity]" name |

**Configuration:**
- `min_score`: 0.8 — Qwen cross-encoder threshold (nodes + edges)
- `keep_top_n`: 15 — initial candidates passed to reranker
- `max_edges_per_entity`: 30

**Note:** The `tree_explorer` tool is **NOT registered** in the main tool registry (`build_tool_registry` does not include it). It is available for external use (notebooks, ingestion) but not called by the reasoning loop. The reasoning loop uses `search_knowledge` (Jiggasha API) instead.

---

#### `get_by_uuid(uuid, entity_type)`

**When to call:**
- The user provides a UUID to inspect
- A `tree_explorer` result contains a UUID the user wants to explore
- You need full details of a specific graph entity, episode, or edge

**Entity types:**
| Type | Description | Returns |
|---|---|---|
| `entity` | Entity node | Name, summary, all properties, relations |
| `episode` | Episodic node | Parsed JSON content with category, topic, service, full text |
| `edge` | RELATES_TO edge | Relation type, fact, source/target entities, episodes |

**Configuration:**
- `max_relations_displayed`: 10
- `max_fact_display_chars`: 150
- `max_prop_display_chars`: 200
- `max_episode_display_chars`: 500
- `max_snippet_chars`: 60

**Note:** Like `tree_explorer`, `get_by_uuid` is NOT registered in the main tool registry. It is a utility tool for external graph queries.

---

## Complete Decision Flow

```
User Message Received
        │
        ▼
┌─────────────────────────┐
│ 1. Is input too long?   │ ── Yes ──→ answer_directly(category=?, text="প্রশ্নটি খুব বড়...")
│     (max_input_chars)   │
└─────────────────────────┘
        │ No
        ▼
┌─────────────────────────┐
│ 2. Is it short/numeric? │ ── Yes ──→ Inject previous assistant reply as context
│     (≤16 chars)          │    Orchestrator-level follow-up resolution
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│ 3. Is it gibberish?     │ ── Yes ──→ answer_directly (appropriate category)
└─────────────────────────┘
        │ No
        ▼
┌──────────────────────────────────────────┐
│ 4. CLASSIFY THE USER INTENT:             │
│                                           │
│  ┌─ Non-factual? ────────────────────────┐│
│  │  Greeting/small talk                  ││
│  │  → answer_directly(category="chitchat")││
│  │                                       ││
│  │  Identity question (about yourself)   ││
│  │  → answer_directly(category="identity")││
│  │                                       ││
│  │  Political/religious/controversial    ││
│  │  → answer_directly(category="safety") ││
│  │                                       ││
│  │  Abusive input                        ││
│  │  → answer_directly(category="abuse")  ││
│  │                                       ││
│  │  Illegal/dangerous request            ││
│  │  → answer_directly(category="illegal")││
│  └───────────────────────────────────────┘│
│                                           │
│  ┌─ Factual/informational? ──────────────┐│
│  │  Ambiguous between multiple services? ││
│  │  → First: search_knowledge(query)     ││
│  │  → If results still ambiguous:        ││
│  │     ask_user(question, options)       ││
│  │                                       ││
│  │  Clear factual query?                 ││
│  │  → search_knowledge(query)            ││
│  │  → Read results                       ││
│  │  → If results sufficient: produce     ││
│  │     answer (no tool call, turn 2+)    ││
│  │  → If results incomplete:             ││
│  │     grep_passage / extract_from_doc   ││
│  │     → delegate_task for formatting    ││
│  │                                       ││
│  │  No results from search?              ││
│  │  → Try different keywords             ││
│  │  → If still nothing:                  ││
│  │     answer_directly(category="no_info")││
│  └───────────────────────────────────────┘│
└──────────────────────────────────────────┘
```

---

## Turn-by-Turn Flow

### Turn 1 (tool_choice="required")

1. The model MUST call at least one tool
2. For factual queries → `search_knowledge(query)`
3. For non-factual → `answer_directly(category, text)`
4. For short/numeric follow-ups → `history_query(mode='recent', n=3)` (if context was already injected by orchestrator, use that first)

### Turn 2+ (tool_choice="auto")

1. The model may produce a final answer without calling tools
2. If data is insufficient, call additional tools (but prefer **parallel calls** in a single turn)
3. Confidence target: by turn 3, the model should be confident it has enough data
4. If `search_knowledge` results already suffice, do NOT call more tools

### Answer Directly Short-Circuit

1. Model calls `answer_directly(category, text)`
2. Reasoning loop detects the `__ANSWER_DIRECTLY__::` sentinel
3. Streams the text directly to the user in chunks (12 chars, 15ms apart)
4. **Immediately ends** — no further turns, no feeding result back to LLM

---

## Post-Tool Refinement

When a tool result exceeds the **refine threshold** (600 tokens by default):

1. The reasoning loop invokes a secondary LLM call
2. Prompt: "Extract only the parts of the tool output below that are relevant to the user's query..."
3. If the secondary LLM returns relevant data → replaces the raw result
4. If it returns "NO_RELEVANT_DATA" → keeps the raw result
5. Debug stream shows both raw and refined content
6. User stream sees only the refined content

This is controlled by `post_tool_refine.enabled` (default true) and `post_tool_refine.threshold_tokens` (default 600).

---

## Background Processes

### Session Summarizer

After each answer is complete:
1. Fires in the background via `asyncio.create_task`
2. Passes current summary + new turn to secondary LLM
3. Produces an updated rolling summary (under 300 tokens)
4. Stores it in Redis for future context
5. Preserves: unresolved questions, entities mentioned, user preferences
6. Drops: resolved details, exact quotes

---

## Session Management

### Redis Storage
- **Turns**: Stored as a Redis list (LPUSh, most recent first), max 20 turns fetched
- **Rolling summary**: Stored as a Redis string key, updated after each turn
- **Last assistant meta**: Stores the last assistant reply + enumerated options for follow-up resolution
- **TTL**: 86400 seconds (24 hours) by default

### Message Truncation
- System prompt reserves ~3500 tokens
- Remaining budget = `max_context_tokens` (32000) - reservation
- Oldest non-system messages are dropped first
- If still over budget, hard truncation on the last message

---

## Tool Registry (What's Actually Available)

The `build_tool_registry()` function registers these tools in order:

1. **Knowledge**: `search_knowledge`
2. **Secondary**: `grep_passage`, `extract_from_document`, `delegate_task`
3. **Interaction**: `ask_user`, `answer_directly`
4. **History**: `history_query`

**Not registered** (external use only):
- `tree_explorer` — graph tree builder
- `get_by_uuid` — graph entity/episode/edge lookup

---

## Key Constraints & Rules

1. **Never answer from parametric knowledge on factual queries** — always call a tool first
2. **Parallel calls preferred** — call multiple tools in a single turn, not sequentially
3. **Never call `answer_directly` first on a factual query** — search first, answer later
4. **Formal Bengali** for all user-facing text; search queries may use Bengali or English
5. **Never expose tool names or internal reasoning** to the user
6. **Never reference information sources** ("according to the database", etc.) — state answers naturally
7. **Maximum 2 concurrent different questions** per response (configurable)
8. **Time awareness** — Bangladesh office hours: Sunday-Thursday 9am-5pm, Friday-Saturday closed
9. **Gibberish/oversized input** → reject with `answer_directly`, don't waste tokens searching
