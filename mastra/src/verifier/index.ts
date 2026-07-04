/**
 * verifier/index.ts — port of PostFlightVerifier (Layer 5).
 *
 * Strip composer-emitted Sources blocks / mode-mix paragraphs / unknown tags,
 * run batched NLI, apply the redact/refuse/warn policy, append the canonical
 * Sources block. Never throws; degrades gracefully.
 */

import { VERIFIER } from "../config.js";
import { evt, type SourceMap, type StreamEvent } from "../types.js";
import {
  buildSourcesBlock,
  extractCitations,
  extractCitationTags,
  stripUnknownTags,
} from "./citations.js";
import { applyPolicy } from "./policy.js";
import { verifyClaims } from "./nli.js";

const CITE_TAG_RE = /\[S\d+\]/;
const CITE_TAG_RE_G = /\[S\d+\]/g;

const SOURCES_PATTERNS: RegExp[] = [
  /\n+---\s*\n+\*\*\s*(?:সূত্র|উৎস|Sources)\b/i,
  /(?:^|\n+)---\s*\n+\*\*\s*(?:সূত্র|উৎস|Sources)\b/i,
  /\n+\*\*\s*(?:সূত্র|উৎস|Sources)\s*\(?(?:Sources)?\)?\s*\*\*/i,
  /(?:^|\n+)\*\*\s*(?:সূত্র|উৎস|Sources)\s*\(?(?:Sources)?\)?\s*\*\*/i,
  /\n+(?:সূত্র|উৎস|Sources)\s*[:：]/i,
  /(?:^|\n+)(?:সূত্র|উৎস|Sources)\s*[:：]/i,
  /\n+(?:\s*[-*]\s*\[S\d+\][^\n]*\n*)+$/,
];

const PARTIAL_GAP_RE =
  /((?:নির্দিষ্ট\s+[^।\n]{0,40}\s+)?উল্লেখ\s+নেই|উল্লিখিত\s+নয়|(?:সঠিক|নির্দিষ্ট)\s+তথ্য\s+পাওয়া\s+যায়নি|(?:তথ্য\s+)?প্রসঙ্গে\s+(?:নেই|উল্লেখ\s+নেই)|নির্দিষ্ট\s+তথ্য\s+নেই)/i;

const MODE_MIX_PARAGRAPH_RE =
  /(?:তবে|তথাপি|তবু|যদিও)[^।\n]*(?:সাধারণ(?:ভাবে|ত)?|সাধারণ\s+পদ্ধতি|সাধারণ\s+আইনানুগ|নিচে\s+দেওয়া\s+হলো|নিচে\s+দেয়া\s+হলো)/i;

const B_HEADER_RE = /\n\s*এই\s+নির্দিষ্ট\s+বিষয়ে\s+সঠিক\s+তথ্য\s+পাওয়া\s+যায়নি/;

function stripComposerSourcesBlock(text: string): string {
  if (!text) return text;
  for (const pat of SOURCES_PATTERNS) {
    const m = text.match(pat);
    if (!m || m.index === undefined) continue;
    let body = text.slice(0, m.index).replace(/\s+$/, "");
    const trailing = text.slice(m.index);
    if (!CITE_TAG_RE.test(body)) {
      const trailingTags = [...new Set(trailing.match(CITE_TAG_RE_G) ?? [])];
      if (trailingTags.length) {
        const tagSuffix = " " + trailingTags.join(" ");
        const lines = body.replace(/\s+$/, "").split("\n");
        for (let i = lines.length - 1; i >= 0; i--) {
          if (lines[i].trim()) {
            lines[i] = lines[i].replace(/\s+$/, "") + tagSuffix;
            break;
          }
        }
        body = lines.join("\n");
      }
    }
    text = body;
  }
  return text;
}

function stripModeMixParagraph(text: string): [string, boolean] {
  if (!text) return [text, false];
  const gap = text.match(PARTIAL_GAP_RE);
  if (!gap || gap.index === undefined) return [text, false];
  const gapEnd = gap.index + gap[0].length;
  const tail = text.slice(gapEnd);
  const mix = tail.match(MODE_MIX_PARAGRAPH_RE);
  if (!mix || mix.index === undefined) return [text, false];
  const bridgeStart = gapEnd + mix.index;
  const restAfterBridge = text.slice(bridgeStart);
  if (CITE_TAG_RE.test(restAfterBridge)) return [text, false];
  const bMatch = restAfterBridge.match(B_HEADER_RE);
  let cleaned: string;
  if (bMatch && bMatch.index !== undefined) {
    cleaned = (
      text.slice(0, bridgeStart).replace(/\s+$/, "") +
      "\n\n" +
      restAfterBridge.slice(bMatch.index).replace(/^\s+/, "")
    ).trim();
  } else {
    cleaned = text.slice(0, bridgeStart).replace(/\s+$/, "");
  }
  return [cleaned, true];
}

export interface VerifyResult {
  finalAnswer: string;
  events: StreamEvent[];
  sourcesBlock: string;
}

export async function verify(rawAnswer: string, sourceMap: SourceMap): Promise<VerifyResult> {
  const events: StreamEvent[] = [];
  const refusalText = "দুঃখিত, এই প্রশ্নের জন্য নির্ভরযোগ্য সরকারি তথ্য পাওয়া যায়নি।";

  let cleaned = stripComposerSourcesBlock(rawAnswer);

  const [modeCleaned, modeStripped] = stripModeMixParagraph(cleaned);
  cleaned = modeCleaned;
  if (modeStripped) {
    events.push(
      evt("mode_mix_stripped", "debug", {
        note: "removed 'general procedure' paragraph after partial-gap caveat",
      }),
    );
  }

  const [tagCleaned, dropped] = stripUnknownTags(cleaned, sourceMap);
  cleaned = tagCleaned;
  for (const tag of dropped) {
    events.push(
      evt("unsupported_claim", "debug", {
        tag,
        verdict: "tag_not_in_source_map",
        action: "stripped",
      }),
    );
  }

  const usedTags = extractCitationTags(cleaned);
  if (usedTags.length === 0 && Object.keys(sourceMap).length > 0) {
    return { finalAnswer: refusalText, events, sourcesBlock: "" };
  }

  cleaned = stripComposerSourcesBlock(cleaned).replace(/\s+$/, "");

  if (VERIFIER.enabled) {
    let pairs = extractCitations(cleaned);
    pairs = pairs.filter(([t]) => t in sourceMap);
    if (pairs.length) {
      events.push(evt("verification_start", "debug", { count: pairs.length }));
      try {
        const [verdicts, nliUsage] = await verifyClaims(pairs, sourceMap, VERIFIER.timeoutMs);
        const [policed, policyEvents] = applyPolicy(
          cleaned,
          pairs,
          verdicts,
          VERIFIER.policy,
          refusalText,
        );
        cleaned = policed;
        events.push(
          evt("verification_result", "debug", {
            pairs: pairs.length,
            verdicts,
            token_usage: nliUsage,
          }),
        );
        for (const pe of policyEvents) events.push(pe);
      } catch (e) {
        events.push(
          evt("verification_result", "debug", { action: "degraded_failed", error: String(e) }),
        );
      }
    }
  }

  if (cleaned.trim() === refusalText.trim()) {
    return { finalAnswer: refusalText, events, sourcesBlock: "" };
  }

  const usedTagsAfter = extractCitationTags(cleaned);
  const sourcesBlock = buildSourcesBlock(sourceMap, usedTagsAfter);
  const final = cleaned.replace(/\s+$/, "") + (sourcesBlock ? "\n\n" + sourcesBlock : "");
  return { finalAnswer: final, events, sourcesBlock };
}
