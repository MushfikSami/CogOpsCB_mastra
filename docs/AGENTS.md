# Agents (`cogops/agents/`)

The **orchestrator agent pipeline** is the production query path.  Each layer is an independent agent with its own system prompt, LLM call pattern, and failure mode.  The orchestrator wires them sequentially and yields typed events to user and debug channels.

---

## `orchestrator.py`

**Role:** Per-request façade.  The only entry point callers use.

**Why it exists:**
- Holds references to all 6 agents, both LLM clients, and the Redis session store.
- Runs `process_query()` which executes the pipeline layer by layer.
- Handles fast-path branches (date/time queries bypass retrieval; chitchat bypasses everything; guard-rail refusals return immediately).
- Loads conversation history from Redis, appends each turn, persists the full trace.
- Catches unhandled exceptions and emits a static fallback answer so the user never sees a raw stack trace.
- Injects a Bangladesh-time reminder into every LLM call.

**Key design choice:** The orchestrator never returns a flat refusal when retrieval is empty.  It always calls the composer, letting the composer decide whether to answer from related passages, direct the user to an office, or admit the topic is missing.

---

## `input_guard.py` — Layer 0

**Role:** Pure-code input validation with **zero LLM latency**.

**Why it exists:**
- Rejects garbage before any expensive LLM call is made.
- Checks: empty/whitespace, length caps, NUL bytes, control-character ratio, prompt-injection regex patterns, repetition spam, Shannon entropy, token-bomb detection.
- Returns either a cleaned NFC-normalized query or a rejection reason that triggers an immediate static refusal.

**Key design choice:** All checks are deterministic regex and arithmetic.  No ML model.  This guarantees sub-millisecond rejection of obviously bad input.

---

## `intent_classifier.py` — Layer 1

**Role:** Classifies user intent into one of: `factual`, `chitchat`, `ambiguous`, `harmful`, `system_probe`, `multi_question`.

**Why it exists:**
- The pipeline behaves differently for each intent.  Factual queries go to retrieval; chitchat gets a friendly reply; ambiguous queries ask for clarification; harmful queries get a refusal.
- Uses a **single secondary-LLM JSON call** for classification.  Fast, structured, cheap.
- Before the LLM call, zero-latency keyword banks catch self-harm, illegal activity, political manipulation, blasphemy, and system-probe attempts.
- Domain-vocabulary override: if the query contains Bangladesh government-service terms (পাসপোর্ট, এনআইডি, জন্ম সনদ, etc.), intent is forced to `factual` regardless of the LLM's opinion.
- Removed hard personal-law refusals; those now run the full pipeline.

**Key design choice:** Two-stage gate (deterministic keyword → LLM JSON) gives speed + nuance.  The domain override prevents the LLM from misclassifying clear service questions as chitchat.

---

## `query_processor.py` — Layer 2

**Role:** Transforms raw user queries into clean, standalone, formal search queries.

**Why it exists:**
- Users speak informally, use pronouns referring to previous turns, mix Banglish with Bengali, and add conversational filler.  Jiggasha needs formal, self-contained queries to retrieve good passages.
- Three sequential steps:
  1. **QueryDisambiguator** — resolves pronouns and references using conversation history (`এটা`, `ওটা`, `তারপর`, etc.).
  2. **QueryFormalizer** — converts casual spoken Bengali/Banglish into formal document-search language.
  3. **QueryFanOut** — normalizes synonyms, strips fillers, and caps the number of parallel sub-queries sent to Jiggasha.

**Key design choice:** Each sub-step is a separate secondary-LLM call.  This is slower than one mega-prompt but dramatically improves accuracy because each prompt has a single, narrow job.

---

## `retrieval_agent.py` — Layer 3

**Role:** Interfaces with the external Jiggasha search service.

**Why it exists:**
- Abstracts the Jiggasha HTTP API behind a clean async interface.
- Calls Jiggasha **in parallel** for all sub-queries from Layer 2.
- Merges results deduplicating by `passage_id` (keeps the highest score).
- Builds the `[S#] → passage_meta` **source_map** that downstream agents consume.  Without this map, citations are meaningless.
- Optional **ReAct loop**: a `RetrievalJudge` (secondary LLM) evaluates whether passages are sufficient and issues a refined query if not.  Max iterations are capped (default 2) to prevent runaway loops.

**Key design choice:** Parallel Jiggasha calls + merge-by-id means multi-aspect questions (e.g. "passport requirements and fees") retrieve comprehensive coverage without duplicate passages.  The ReAct loop is off by default (`max_react_iterations: 0`) because Jiggasha's instruction-based retrieval is already high-quality.

---

## `composer_agent.py` — Layer 4 + Layer 5

**Role:** Streams the primary LLM answer and verifies it post-generation.

**Why it exists:**
- **ComposerAgent** streams the primary LLM with inline `[S#]` citations.  Uses `ThinkingParser` to separate `<thinking>` blocks from user-facing text.
- **PostFlightVerifier** sanitizes the raw composer output before the user sees it:
  - Strips any composer-emitted "Sources" blocks (the verifier builds its own canonical block).
  - Strips "mode-mix" paragraphs (gap admission + generic advice) **unless** the tail has citations.
  - Strips hallucinated citation tags (tags not present in `source_map`).
  - Runs **NLI verification** on cited claims.
  - Applies policy (`redact` / `refuse` / `warn`) based on NLI verdicts.
  - Appends the canonical Bengali Sources block.

**Key design choice:** Streaming gives perceived speed (first token in ~200ms).  Post-flight verification happens on the full text, not token-by-token, because NLI needs complete sentences.  The verifier is fail-soft: on any error it lets the raw answer through rather than blocking the user.

---

## `pipeline.py`

**Role:** Self-contained deterministic implementation of Stages 2→4.

**Why it exists:**
- This was the original pipeline core before the orchestrator was introduced.
- It runs Jiggasha retrieval in parallel, merges/deduplicates, optionally runs a ReAct judge-and-refine loop, detects disambiguation needs, streams the composer, and executes post-flight stripping/verification — all in one module.
- Today it serves as a **reference implementation** and a deterministic fallback.  The orchestrator delegates to individual agents for production.

**Key design choice:** Monolithic vs. modular trade-off.  `pipeline.py` is easier to reason about end-to-end; `orchestrator.py` + `agents/` is easier to modify, test, and prompt-tune one layer at a time.
