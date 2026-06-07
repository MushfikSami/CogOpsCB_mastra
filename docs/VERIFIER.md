# Verification Layer (`cogops/verifier/`)

The verifier is **Layer 5** of the orchestrator pipeline (and Stage 4 of the deterministic pipeline).  It runs after the composer has generated an answer.  Its job is to catch hallucinations, unsupported claims, and mode-mixed output before the user sees it.

---

## `nli.py`

**Role:** Natural-Language-Inference claim verifier.

**Why it exists:**
- The composer can hallucinate facts, misquote fees, or invent office names even when it cites passages.  NLI checks whether each claim is actually supported by its cited passage.
- Takes `(tag, sentence)` pairs extracted from the composed answer and verifies each claim against its cited passage in a **single batched secondary-LLM JSON call**.
- Verdicts: `entailed`, `partial`, `not_entailed`.
- Strict on numbers, dates, fees, URLs, and office names — any mismatch flags `not_entailed`.
- On timeout or error it degrades gracefully to all-`entailed` so the user is never blocked by a verifier failure.

**Key design choice:** Batched NLI (one LLM call for all claims) is ~5× faster than one call per claim.  The fail-soft design prevents the verifier from becoming a single point of failure.

---

## `citations.py`

**Role:** Regex-based citation extractor and canonical Sources block builder.

**Why it exists:**
- Extracts `[S#]` tags from composer output so the verifier knows which passage to check for each claim.
- Strips hallucinated citation tags (tags that appear in the answer but are not in `source_map`).
- Builds the canonical Bengali **Sources block** that is appended to the final answer.  Deduplicates by first citation occurrence.

**Key design choice:** Regex extraction is fast and reliable for the `[S#]` format.  The canonical Sources block replaces any Sources block the composer might have generated, ensuring consistent formatting.

---

## `policy.py`

**Role:** Pure-logic policy engine that turns NLI verdicts into actions.

**Why it exists:**
- Decouples "what the NLI says" from "what we do about it."
- Three policies:
  - **`redact`** (default): replaces unsupported sentences with `[তথ্যটি নিশ্চিত করা যায়নি]`.
  - **`refuse`**: returns a full refusal if any claim is unsupported.
  - **`warn`**: keeps the answer but adds a debug warning.
- Today the orchestrator uses `redact` so the user gets a partial but honest answer rather than a flat refusal.

**Key design choice:** Policy is configurable per-deployment.  A government-service chatbot should be helpful first; `redact` preserves helpfulness while marking uncertainty.
