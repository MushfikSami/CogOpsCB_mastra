/**
 * query-processor.ts — port of cogops/agents/query_processor.py (Layer 2).
 *
 * disambiguate (history-aware) → formalize (casual→formal Bengali) → fan-out
 * (normalize synonyms/fillers + cap). LLM steps fail-open to the input.
 */

import { generateText } from "ai";
import { secondaryModel } from "./config.js";
import { DISAMBIG_SYSTEM_PROMPT, FORMALIZER_SYSTEM_PROMPT } from "./prompts.js";
import { buildTimeReminder } from "./time-reminder.js";
import type { ChatHistoryMsg } from "./types.js";

export interface ProcessedQuery {
  queries: string[];
  overflow: string[];
  disambiguated: string;
  formalized: string;
}

async function llmCall(system: string, user: string, temperature: number, timeoutMs: number): Promise<string | null> {
  try {
    const controller = new AbortController();
    const t = setTimeout(() => controller.abort(), timeoutMs);
    let res;
    try {
      res = await generateText({
        model: secondaryModel,
        temperature,
        maxTokens: 256,
        abortSignal: controller.signal,
        messages: [
          { role: "system", content: system },
          { role: "assistant", content: buildTimeReminder() },
          { role: "user", content: user },
        ],
      });
    } finally {
      clearTimeout(t);
    }
    let out = res.text.trim();
    out = out.replace(/^["']|["']$/g, "").trim();
    return out || null;
  } catch {
    return null;
  }
}

async function disambiguate(query: string, history: ChatHistoryMsg[]): Promise<string> {
  if (history.length === 0) return query;
  const block = history
    .slice(-6)
    .filter((m) => (m.content ?? "").trim())
    .map((m) => `${m.role === "user" ? "User" : "Assistant"}: ${m.content.trim()}`)
    .join("\n");
  const user =
    `Conversation history:\n${block}\n\nCurrent user message: ${query}\n\n` +
    "Rewrite the current message as a standalone query (no pronouns, no ambiguous references):";
  const out = await llmCall(DISAMBIG_SYSTEM_PROMPT, user, 0.0, 4000);
  return out ?? query;
}

async function formalize(query: string): Promise<string> {
  const out = await llmCall(
    FORMALIZER_SYSTEM_PROMPT,
    `Casual query: ${query}\n\nFormalized query:`,
    0.1,
    4000,
  );
  return out ?? query;
}

// --- Fan-out normalization (synonyms + fillers) ---
const SYNONYM_ROOTS: Array<[string, string]> = [
  ["প্লেন", "বিমান"],
  ["ট্রেন", "রেল"],
  ["টিকেট", "টিকিট"],
];
const BN_SUFFIXES = ["", "ের", "ে", "ি", "া", "ো", "ী", "ু", "ূ", "ৃ", "ং", "ঃ", "়", "ঁ"];
const BN_CHAR = "[\\u0980-\\u09FF]";

const SYNONYM_RES: Array<[RegExp, string]> = [];
for (const [informal, formal] of SYNONYM_ROOTS) {
  for (const suffix of BN_SUFFIXES) {
    const oldForm = informal + suffix;
    const newForm = formal + suffix;
    const pattern = `(?<!${BN_CHAR})${oldForm.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}(?!${BN_CHAR})`;
    SYNONYM_RES.push([new RegExp(pattern, "g"), newForm]);
  }
}

const FILLER_RE =
  /(?<![ঀ-৿])(আচ্ছা|ভাই|দেখেন|শুনুন|বলুন\s+তো|বলো\s+তো|জানাবেন|জানাব|একটু|কিন্তু|তাই|তাহলে|তো)(?![ঀ-৿])/gi;

function normalize(text: string): string {
  if (!text) return "";
  let out = text.normalize("NFC").trim();
  for (const [re, rep] of SYNONYM_RES) out = out.replace(re, rep);
  out = out.replace(FILLER_RE, "");
  out = out.replace(/[ \t]+/g, " ");
  out = out.replace(/\n{3,}/g, "\n\n");
  out = out.replace(/^[.,!;:\-]+|[.,!;:\-]+$/g, "");
  return out.trim();
}

function fanOut(queries: string[], maxConcurrent: number): [string[], string[]] {
  let normalized = queries.map(normalize).filter(Boolean);
  if (normalized.length === 0) normalized = queries.map((q) => q.trim()).filter(Boolean);
  return [normalized.slice(0, maxConcurrent), normalized.slice(maxConcurrent)];
}

export async function processQuery(
  rawQuery: string,
  subQueries: string[],
  history: ChatHistoryMsg[],
  maxConcurrent = 3,
): Promise<ProcessedQuery> {
  const disambiguated = await disambiguate(rawQuery, history);

  let formalizedQueries: string[];
  let formalized: string;
  if (subQueries.length > 1) {
    formalizedQueries = [];
    for (const sq of subQueries) formalizedQueries.push(await formalize(sq));
    formalized = formalizedQueries[0] ?? disambiguated;
  } else {
    formalized = await formalize(disambiguated);
    formalizedQueries = [formalized];
  }

  const [accepted, overflow] = fanOut(formalizedQueries, maxConcurrent);
  return { queries: accepted, overflow, disambiguated, formalized };
}
