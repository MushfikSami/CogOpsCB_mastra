# Prompts (`cogops/prompts/`)

All LLM prompts live here so they can be versioned, reviewed, and tuned independently of agent logic.  No business logic — just strings and string builders.

---

## `composer.py`

**Role:** Composer system prompt.

**Why it exists:**
- The longest and most carefully tuned prompt in the system.  It instructs the primary LLM how to:
  - Answer in Bengali using retrieved passages.
  - Cite passages inline with `[S#]` tags.
  - Handle three scenarios: **DIRECT** (exact match → cite and answer), **PARTIAL** (related info → summarize with citations + guide to office), **EMPTY** (no passages → state domain not found + direct to likely office).
  - Avoid "no info found" flat refusals when passages exist.
  - Avoid mode-mixing (gap admission + generic advice in the same paragraph).
  - Never hallucinate facts not in passages.

**Key design choice:** The prompt is explicit about the three scenarios because Bengali government questions often have partial coverage in the corpus.  A vague prompt produces vague answers; an explicit prompt produces structured, useful answers.

---

## `system.py`

**Role:** ReAct agent system prompt.

**Why it exists:**
- Used by the deterministic pipeline's reasoning loop.
- Enforces tool-first protocol, citation format, refusal rules, and anti-hallucination guidelines.
- Less frequently tuned than `composer.py` because the ReAct path is off by default.

---

## `time_reminder.py`

**Role:** Builds per-turn Bangladesh Standard Time (UTC+6) reminder.

**Why it exists:**
- Injected into every LLM call as an `assistant` message (not concatenated into the system prompt, because the provider rejects multiple system messages).
- Prevents time-sensitive answers from being wrong due to stale training data.  E.g. "পাসপোর্ট ফি কত" — the LLM needs to know it's 2026, not 2023.
- Format: "আজ [date] বাংলাদেশ সময় [time]।"

**Key design choice:** Injected as `role: "assistant"` rather than appending to `role: "system"` because the vLLM provider schema forbids multiple system messages.

---

## `messages.py`

**Role:** Centralized user-facing fallback strings.

**Why it exists:**
- Stores static strings for error fallbacks: technical-error message, server-load message, etc.
- Centralizing them ensures consistent tone across all failure paths.
