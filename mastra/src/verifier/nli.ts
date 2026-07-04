/**
 * nli.ts — port of cogops/verifier/nli.py
 *
 * Batched NLI verifier. ONE secondary-LLM call per turn; on timeout/error we
 * degrade to all-"entailed" so the answer is never blocked by an infra failure.
 */

import { generateText } from "ai";
import { secondaryModel } from "../config.js";
import { buildTimeReminder } from "../time-reminder.js";
import { NLI_SYSTEM_PROMPT } from "../prompts.js";
import type { SourceMap } from "../types.js";
import type { Verdict } from "./policy.js";

const PAREN_RE = /\s*\([^)]*(?:অর্থাৎ|তার|অর্থা|যেমন|যেমনটি)[^)]*\)\s*/g;

function stripExplanatoryParentheticals(text: string): string {
  return text.replace(PAREN_RE, " ").trim();
}

function buildUserPrompt(pairs: Array<[string, string]>, sourceMap: SourceMap): string {
  const blocks: string[] = [];
  pairs.forEach(([tag, sentence], i) => {
    const evidence = sourceMap[tag]?.text || "(evidence missing)";
    const coreClaim = stripExplanatoryParentheticals(sentence);
    blocks.push(`### Pair ${i}\nClaim (cites [${tag}]): ${coreClaim}\nEvidence [${tag}]: ${evidence}\n`);
  });
  return (
    "Verify each claim against its cited evidence.\n\n" +
    blocks.join("\n") +
    "\nReturn a single JSON object as specified."
  );
}

function extractJson(raw: string): any {
  const trimmed = raw.trim();
  try {
    return JSON.parse(trimmed);
  } catch {
    const m = trimmed.match(/\{[\s\S]*\}/);
    if (m) return JSON.parse(m[0]);
    throw new Error("no JSON object in NLI response");
  }
}

export interface NliUsage {
  prompt: number;
  completion: number;
}

export async function verifyClaims(
  pairs: Array<[string, string]>,
  sourceMap: SourceMap,
  timeoutMs = 6000,
): Promise<[Verdict[], NliUsage | null]> {
  if (pairs.length === 0) return [[], null];

  const fastpath = new Map<number, Verdict>();
  const validIndices: number[] = [];
  const validPairs: Array<[string, string]> = [];
  pairs.forEach(([tag, sentence], i) => {
    if (!(tag in sourceMap)) fastpath.set(i, "not_entailed");
    else {
      validIndices.push(i);
      validPairs.push([tag, sentence]);
    }
  });

  if (validPairs.length === 0) {
    return [pairs.map((_, i) => fastpath.get(i) ?? "entailed"), null];
  }

  const verdictsByIdx = new Map<number, Verdict>(fastpath);
  let usage: NliUsage | null = null;

  try {
    const controller = new AbortController();
    const t = setTimeout(() => controller.abort(), timeoutMs);
    let res;
    try {
      res = await generateText({
        model: secondaryModel,
        temperature: 0.0,
        maxTokens: 512,
        abortSignal: controller.signal,
        messages: [
          { role: "system", content: NLI_SYSTEM_PROMPT },
          { role: "assistant", content: buildTimeReminder() },
          { role: "user", content: buildUserPrompt(validPairs, sourceMap) },
        ],
      });
    } finally {
      clearTimeout(t);
    }
    if (res.usage) {
      usage = {
        prompt: res.usage.promptTokens ?? 0,
        completion: res.usage.completionTokens ?? 0,
      };
    }
    const data = extractJson(res.text);
    const items: Array<{ i?: number; v?: string }> = data.verdicts ?? [];
    for (const item of items) {
      const iLocal = item.i;
      const v = (item.v ?? "").toLowerCase();
      if (typeof iLocal !== "number" || iLocal < 0 || iLocal >= validPairs.length) continue;
      if (v !== "entailed" && v !== "partial" && v !== "not_entailed") continue;
      verdictsByIdx.set(validIndices[iLocal], v as Verdict);
    }
  } catch (e) {
    // Degrade to all-entailed for the valid pairs.
    for (const i of validIndices) {
      if (!verdictsByIdx.has(i)) verdictsByIdx.set(i, "entailed");
    }
  }

  return [pairs.map((_, i) => verdictsByIdx.get(i) ?? "entailed"), usage];
}
