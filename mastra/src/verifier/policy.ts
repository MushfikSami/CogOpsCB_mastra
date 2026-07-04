/**
 * policy.ts — port of cogops/verifier/policy.py
 *
 * Turns NLI verdicts into a final answer + event list. Pure logic, no I/O.
 */

import { evt, type StreamEvent } from "../types.js";

export type PolicyName = "redact" | "refuse" | "warn";
export type Verdict = "entailed" | "partial" | "not_entailed";

export const UNSUPPORTED_REDACTION_BN = "[তথ্য যাচাইযোগ্য নয়]";
const REFUSE_ESCALATION_FRACTION = 0.5;

export function applyPolicy(
  answer: string,
  pairs: Array<[string, string]>,
  verdicts: Verdict[],
  policy: PolicyName,
  refusalText: string,
): [string, StreamEvent[]] {
  if (pairs.length === 0) {
    return [
      answer,
      [evt("verification_result", "debug", { verdicts: [], policy, action: "noop_no_pairs" })],
    ];
  }

  const events: StreamEvent[] = [];
  const notEntailed = verdicts.filter((v) => v === "not_entailed").length;
  const partial = verdicts.filter((v) => v === "partial").length;
  const entailed = verdicts.filter((v) => v === "entailed").length;

  const summary: Record<string, unknown> = {
    policy,
    total: verdicts.length,
    entailed,
    partial,
    not_entailed: notEntailed,
  };

  if (policy === "warn") {
    pairs.forEach(([tag, sentence], i) => {
      if (verdicts[i] !== "entailed") {
        events.push(
          evt("unsupported_claim", "debug", {
            tag,
            sentence,
            verdict: verdicts[i],
            action: "kept_warn_policy",
          }),
        );
      }
    });
    summary.action = "warn_only";
    events.push(evt("verification_result", "debug", summary));
    return [answer, events];
  }

  if (policy === "refuse") {
    if (notEntailed > 0) {
      pairs.forEach(([tag, sentence], i) => {
        if (verdicts[i] === "not_entailed") {
          events.push(
            evt("unsupported_claim", "debug", {
              tag,
              sentence,
              verdict: verdicts[i],
              action: "triggered_refuse",
            }),
          );
        }
      });
      summary.action = "full_refusal";
      events.push(evt("verification_result", "debug", summary));
      return [refusalText, events];
    }
    summary.action = "passed";
    events.push(evt("verification_result", "debug", summary));
    return [answer, events];
  }

  // --- REDACT (default) ---
  if (notEntailed && notEntailed / Math.max(verdicts.length, 1) > REFUSE_ESCALATION_FRACTION) {
    pairs.forEach(([tag, sentence], i) => {
      if (verdicts[i] === "not_entailed") {
        events.push(
          evt("unsupported_claim", "debug", {
            tag,
            sentence,
            verdict: verdicts[i],
            action: "triggered_escalation_to_refusal",
          }),
        );
      }
    });
    summary.action = "escalated_refusal";
    events.push(evt("verification_result", "debug", summary));
    return [refusalText, events];
  }

  let final = answer;
  const redactedSentences = new Set<string>();
  pairs.forEach(([tag, sentence], i) => {
    const v = verdicts[i];
    if (v === "not_entailed" && !redactedSentences.has(sentence)) {
      if (sentence && final.includes(sentence)) {
        final = final.split(sentence).join(UNSUPPORTED_REDACTION_BN);
        redactedSentences.add(sentence);
      }
      events.push(
        evt("unsupported_claim", "debug", { tag, sentence, verdict: v, action: "redacted" }),
      );
    } else if (v === "partial") {
      events.push(
        evt("unsupported_claim", "debug", { tag, sentence, verdict: v, action: "kept_partial" }),
      );
    }
  });

  summary.action =
    redactedSentences.size > 0 ? "redacted" : partial > 0 ? "partial_only" : "all_entailed";
  events.push(evt("verification_result", "debug", summary));
  return [final, events];
}
