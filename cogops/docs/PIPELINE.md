# Deterministic Pipeline (`cogops/pipeline/`)

The `pipeline/` package contains the **original reference implementation** of query routing, normalization, sanitization, and document-type guarding.  Most of its active logic has moved into the orchestrator agents, but these modules remain for the deterministic fallback path and for shared utilities.

---

## `router.py`

**Role:** Stage 1 of the deterministic pipeline — intent classification + query splitting + pronoun resolution in a single LLM call.

**Why it exists:**
- Replaces three separate LLM calls (intent classifier + query splitter + normalizer) with **one secondary-LLM JSON call**.
- Classifies intent into `factual_govt` / `factual_wiki` / `factual_mixed` / `chitchat` / `political_refuse`.
- Resolves pronouns via conversation history.
- Splits multi-part questions into up to 3 formal Bengali sub-queries.
- Fast-path shortcuts: Bengali single-question domain queries skip the LLM entirely; hard-refusal keywords short-circuit immediately.

**Key design choice:** The one-shot JSON approach minimizes latency (one round-trip vs. three) but gives the LLM more to do in a single context window.  The orchestrator chose the multi-agent approach for better per-layer prompt control.

---

## `normalize.py`

**Role:** Deterministic Bengali query normalizer.

**Why it exists:**
- Synonym replacement, filler stripping, and whitespace collapse before embedding retrieval.
- Pure regex and string manipulation — zero LLM cost.
- Normalizes common Banglish variants into formal Bengali (e.g. `_nid_` → `জাতীয় পরিচয়পত্র`).

**Key design choice:** Deterministic normalization lives outside the LLM because it is faster, cheaper, and more consistent than asking an LLM to normalize text.

---

## `sanitize.py`

**Role:** Legacy Stage 0 pure-code input validator.

**Why it exists:**
- Used by the deterministic pipeline before the router is called.
- Checks length, binary content, prompt injection, spam, and entropy — same concerns as `agents/input_guard.py` but in a standalone module.
- Today `agents/input_guard.py` is the production gate; `sanitize.py` remains for backward compatibility with the deterministic pipeline.

---

## `query_expand.py`

**Role:** Document-type guard for the deterministic pipeline.

**Why it exists:**
- When a user explicitly names a document type (e.g. `বিবাহ সনদ`, `এনআইডি`), it verifies that retrieved passages' metadata actually match that type.
- If none match, the pipeline refuses rather than offering irrelevant disambiguation options.
- Older query-expansion functions (`expand_sub_query`, `extract_document_type`) are now deprecated passthrough stubs; formalization moved to the router and Jiggasha.

**Key design choice:** The document-type guard prevents the embarrassing case where a user asks about a marriage certificate and the system retrieves passport passages because of overlapping vocabulary ("apply", "form", "fee").
