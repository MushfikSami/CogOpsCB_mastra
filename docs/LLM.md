# LLM Infrastructure (`cogops/llm/` + `cogops/utils/`)

This layer abstracts the vLLM-compatible endpoints and provides streaming utilities.

---

## `clients.py`

**Role:** Client factory for vLLM-compatible OpenAI-style endpoints.

**Why it exists:**
- Creates an `AsyncOpenAI` client from environment-driven config (`api_key`, `base_url`, `model_name`).
- Exposes `health_check()` for readiness probes.
- The orchestrator instantiates **two services**:
  - **Primary** — large model (gemma 4 31B-it 4bit awq, 256K ctx) for composer streaming.
  - **Secondary** — smaller/faster model (qwen3.6 35b-A3b 4bit awq, 128K ctx) for classifier, router, judge, instruction generator, and verifier.

**Key design choice:** Two-client architecture separates latency-critical small tasks (classifier, verifier) from quality-critical large tasks (composer).  If the primary is down, secondary can still classify intent and verify; if secondary is down, primary can still compose (though without verification).

---

## `reasoning_loop.py`

**Role:** ReAct tool-calling while loop with retry logic.

**Why it exists:**
- Implements the classic ReAct pattern: the LLM can choose to call a tool or emit a final answer.
- Non-streaming tool turns, then a streaming final-answer turn.
- Works around Qwen3.6 / vLLM streaming-tools bugs by using non-streaming for tool calls.
- Used by the deterministic pipeline and by the RetrievalAgent's optional ReAct judge loop.

**Key design choice:** The loop is capped at a maximum number of tool turns to prevent infinite loops.  Error handling is fail-soft: on any exception the loop exits and returns whatever answer was generated so far.

---

## `utils/thinking_parser.py`

**Role:** Streaming parser that separates `<thinking>` blocks from user-facing text.

**Why it exists:**
- Some models (Qwen3, DeepSeek-R1 style) emit reasoning inside `<thinking>` or ` ```thinking` blocks before the actual answer.
- Users should not see raw reasoning chains.  This parser splits the stream into two channels: "thinking" (for debug) and "answer" (for the user).
- Handles partial tokens across chunk boundaries.

**Key design choice:** Streaming parser operates on raw bytes/chunks, not on complete strings, so it works with the primary LLM's SSE stream without buffering the entire response.
