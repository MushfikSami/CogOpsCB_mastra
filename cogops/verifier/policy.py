"""
cogops/verifier/policy.py

Decision policy that turns NLI verdicts into a final answer + event stream.

Policies:
  - "redact":  not_entailed sentences are replaced with [তথ্য যাচাইযোগ্য নয়];
               partial → keep + emit warning; entailed → keep.
               If >50% of cited sentences are not_entailed, escalate to refuse.
  - "refuse":  any single not_entailed → replace the whole answer with the
               static refusal template.
  - "warn":    keep the answer as-is, emit warnings for non-entailed. Used
               automatically when the verifier times out or errors.

This module is pure logic — no LLM calls, no I/O. Caller provides:
  - the raw answer text (already had hallucinated tags stripped upstream)
  - the (tag, sentence) pairs that were verified
  - the verdicts list (aligned to pairs)
  - the policy name
  - the refusal_text string
"""

import logging
from typing import Any, Dict, List, Literal, Tuple

logger = logging.getLogger(__name__)

PolicyName = Literal["redact", "refuse", "warn"]
UNSUPPORTED_REDACTION_BN = "[তথ্য যাচাইযোগ্য নয়]"
_REFUSE_ESCALATION_FRACTION = 0.5


def apply_policy(
    answer: str,
    pairs: List[Tuple[str, str]],
    verdicts: List[str],
    policy: PolicyName,
    refusal_text: str,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Apply the verifier policy. Returns (final_answer, events_to_emit).

    Events are dicts with `type` and `channel` set, ready to be yielded by the
    orchestrator. Event types used here: `unsupported_claim`, `verification_result`.

    The answer is mutated WITHOUT changing [S#] tags or the Sources block — the
    Sources block is appended AFTER policy runs, so this function only deals
    with the prose part of the answer.
    """
    if not pairs:
        return answer, [
            {"type": "verification_result", "channel": "debug",
             "verdicts": [], "policy": policy, "action": "noop_no_pairs"},
        ]

    events: List[Dict[str, Any]] = []

    not_entailed_count = sum(1 for v in verdicts if v == "not_entailed")
    partial_count = sum(1 for v in verdicts if v == "partial")
    entailed_count = sum(1 for v in verdicts if v == "entailed")

    summary_event = {
        "type": "verification_result",
        "channel": "debug",
        "policy": policy,
        "total": len(verdicts),
        "entailed": entailed_count,
        "partial": partial_count,
        "not_entailed": not_entailed_count,
    }

    # ----- WARN policy: keep answer, emit warnings only -----
    if policy == "warn":
        for (tag, sentence), v in zip(pairs, verdicts):
            if v != "entailed":
                events.append({
                    "type": "unsupported_claim",
                    "channel": "debug",
                    "tag": tag,
                    "sentence": sentence,
                    "verdict": v,
                    "action": "kept_warn_policy",
                })
        summary_event["action"] = "warn_only"
        events.append(summary_event)
        return answer, events

    # ----- REFUSE policy: any failure → full refusal -----
    if policy == "refuse":
        if not_entailed_count > 0:
            for (tag, sentence), v in zip(pairs, verdicts):
                if v == "not_entailed":
                    events.append({
                        "type": "unsupported_claim",
                        "channel": "debug",
                        "tag": tag,
                        "sentence": sentence,
                        "verdict": v,
                        "action": "triggered_refuse",
                    })
            summary_event["action"] = "full_refusal"
            events.append(summary_event)
            return refusal_text, events
        summary_event["action"] = "passed"
        events.append(summary_event)
        return answer, events

    # ----- REDACT policy (default) -----
    # Escalate to full refusal if too many sentences failed.
    if not_entailed_count and (not_entailed_count / max(len(verdicts), 1)) > _REFUSE_ESCALATION_FRACTION:
        for (tag, sentence), v in zip(pairs, verdicts):
            if v == "not_entailed":
                events.append({
                    "type": "unsupported_claim",
                    "channel": "debug",
                    "tag": tag,
                    "sentence": sentence,
                    "verdict": v,
                    "action": "triggered_escalation_to_refusal",
                })
        summary_event["action"] = "escalated_refusal"
        events.append(summary_event)
        return refusal_text, events

    # Sentence-level redaction: replace each not_entailed sentence with the
    # redaction marker. Use substring replace — sentences come straight from
    # extract_citations() which split on `।`/`.`/`!`/`?`, so they should be
    # unique substrings of the answer. If a sentence has been previously
    # redacted (duplicate not_entailed verdicts on the same sentence), the
    # second replace becomes a no-op.
    final = answer
    redacted_sentences = set()
    for (tag, sentence), v in zip(pairs, verdicts):
        if v == "not_entailed" and sentence not in redacted_sentences:
            if sentence and sentence in final:
                final = final.replace(sentence, UNSUPPORTED_REDACTION_BN)
                redacted_sentences.add(sentence)
            events.append({
                "type": "unsupported_claim",
                "channel": "debug",
                "tag": tag,
                "sentence": sentence,
                "verdict": v,
                "action": "redacted",
            })
        elif v == "partial":
            events.append({
                "type": "unsupported_claim",
                "channel": "debug",
                "tag": tag,
                "sentence": sentence,
                "verdict": v,
                "action": "kept_partial",
            })

    summary_event["action"] = (
        "redacted" if redacted_sentences else
        ("partial_only" if partial_count else "all_entailed")
    )
    events.append(summary_event)
    return final, events
