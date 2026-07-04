/**
 * citations.ts — port of cogops/verifier/citations.py
 *
 * Citation extraction + Sources block builder. This module is the single
 * authority for the canonical Bengali সূত্র (Sources) block.
 */

import type { SourceMap } from "../types.js";

const CITE_RE = /\[S(\d+)\]/g;
// Bengali danda (।) + latin terminators.
const SENTENCE_SPLIT_RE = /(?<=[।.!?])\s+/;

export function extractCitationTags(answer: string): string[] {
  const out: string[] = [];
  if (!answer) return out;
  for (const m of answer.matchAll(CITE_RE)) out.push(`S${m[1]}`);
  return out;
}

function splitSentences(text: string): string[] {
  if (!text) return [];
  return text
    .trim()
    .split(SENTENCE_SPLIT_RE)
    .filter((p) => p.trim());
}

export function extractCitations(answer: string): Array<[string, string]> {
  if (!answer) return [];
  const pairs: Array<[string, string]> = [];
  for (const sentence of splitSentences(answer)) {
    for (const m of sentence.matchAll(CITE_RE)) {
      pairs.push([`S${m[1]}`, sentence.trim()]);
    }
  }
  return pairs;
}

export function stripUnknownTags(answer: string, sourceMap: SourceMap): [string, string[]] {
  if (!answer) return ["", []];
  const dropped: string[] = [];
  const cleaned = answer.replace(CITE_RE, (full, num) => {
    const tag = `S${num}`;
    if (!(tag in sourceMap)) {
      dropped.push(tag);
      return "";
    }
    return full;
  });
  return [cleaned, dropped];
}

export function buildSourcesBlock(sourceMap: SourceMap, usedTags: string[]): string {
  const seen: string[] = [];
  for (const tag of usedTags) {
    if (tag in sourceMap && !seen.includes(tag)) seen.push(tag);
  }
  if (seen.length === 0) return "";

  const lines = ["", "---", "**সূত্র (Sources)**"];
  for (const tag of seen) {
    const meta = sourceMap[tag];
    const category = meta.category ?? "";
    const topic = meta.topic ?? "";
    const passageId = meta.passage_id;
    const tool = meta.tool ?? "";
    const chunkType = meta.chunk_type ?? "";
    const descriptorBits = [category, topic].filter(Boolean);
    const descriptor = descriptorBits.length ? descriptorBits.join(" — ") : "(no metadata)";
    const suffix = passageId ? ` · passage_id ${passageId}` : "";
    const toolSuffix = tool ? ` (${tool})` : "";
    let sourceLabel = "";
    if (chunkType === "wiki") sourceLabel = " · উইকিপিডিয়া";
    else if (chunkType === "govt_service") sourceLabel = " · সরকারি সেবা";
    lines.push(`- [${tag}] ${descriptor}${suffix}${toolSuffix}${sourceLabel}`);
  }
  return lines.join("\n");
}
