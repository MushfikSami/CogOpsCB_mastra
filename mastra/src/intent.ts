/**
 * intent.ts — port of cogops/agents/intent_classifier.py (Layer 1).
 *
 * Hard-refusal keyword banks run BEFORE the LLM (zero-latency safety). Then a
 * single secondary-LLM JSON call classifies intent + splits sub-queries. A
 * domain-vocabulary override never flips AWAY from factual.
 */

import { generateText } from "ai";
import { secondaryModel } from "./config.js";
import { INTENT_SYSTEM_PROMPT } from "./prompts.js";
import { buildTimeReminder } from "./time-reminder.js";
import type { ChatHistoryMsg } from "./types.js";

export type Intent =
  | "factual"
  | "chitchat"
  | "ambiguous"
  | "harmful"
  | "system_probe"
  | "multi_question";

export interface IntentResult {
  intent: Intent;
  guardRailTriggered: boolean;
  guardRailCategory: string | null;
  subQueries: string[];
  needsClarification: boolean;
  clarificationPromptBn: string | null;
  confidence: number;
  notes: string[];
  usage: { prompt: number; completion: number } | null;
}

export function isFactual(r: IntentResult): boolean {
  return r.intent === "factual" || r.intent === "multi_question";
}

export function shouldRefuse(r: IntentResult): boolean {
  return r.guardRailTriggered || r.intent === "harmful" || r.intent === "system_probe";
}

const HARD_SELF_HARM = [
  "আত্মহত্যা করতে চাই",
  "আত্মহত্যা করব",
  "মরতে চাই",
  "জীবন শেষ করতে চাই",
  "নিজেকে মারতে চাই",
  "বাঁচতে চাই না",
];

const HARD_ILLEGAL = [
  "বোমা তৈরি",
  "বোমা বানানো",
  "মাদক চোরাচালান",
  "মাদক ব্যবসা",
  "অস্ত্র কিনতে",
  "অস্ত্র বিক্রি",
  "চুরি করতে",
  "ডাকাতি করতে",
  "খুন করতে",
  "মানি লন্ডারিং",
  "ফেক এনআইডি",
];

const HARD_SYSTEM_PROBE = [
  "তোমার সিস্টেম প্রম্পট",
  "তোমার ইনস্ট্রাকশন",
  "তোমার নির্দেশনা",
  "system prompt",
  "your instructions",
  "your system prompt",
  "ignore all previous",
  "disregard all previous",
  "prompt injection",
  "jailbreak",
  "দয়া করে তোমার সিস্টেম",
  "তুমি কীভাবে কাজ কর",
  "তোমার মডেল কোনটি",
  "তোমার আলগোরিদম কী",
];

const DOMAIN_VOCAB = [
  "পাসপোর্ট", "passport", "এনআইডি", "NID", "জাতীয় পরিচয়", "পরিচয়পত্র",
  "ভোটার", "চারিত্রিক সনদ", "নাগরিকত্ব", "সনদ", "জন্ম নিবন্ধন",
  "মৃত্যু নিবন্ধন", "বিবাহ নিবন্ধন", "তালাক", "ওয়ারিশ", "উত্তরাধিকার", "প্রতিবন্ধী",
  "ফি", "চার্জ", "মূল্য", "tax", "ট্যাক্স", "কর", "ভ্যাট", "VAT", "সঞ্চয়পত্র", "মূসক",
  "লাইসেন্স", "license", "ড্রাইভিং", "BRTA", "BRTC",
  "জমি", "ভূমি", "খতিয়ান", "দলিল", "সিএস", "বিএস",
  "বিদ্যুৎ", "গ্যাস", "পানি", "DESCO", "WASA", "নেসকো", "ডিপিডিসি",
  "প্রি-পেইড", "পোস্ট-পেইড", "মিটার",
  "মেট্রো", "MRT", "বিমান", "এয়ারলাইন্স", "চেক-ইন", "টিকেট",
  "এসএসসি", "এইচএসসি", "বোর্ড", "শিক্ষাবোর্ড", "সার্টিফিকেট", "সনদপত্র",
  "সরকার", "সরকারি", "মন্ত্রণালয়", "অধিদপ্তর", "দপ্তর", "ministry",
  "৯৯৯", "পুলিশ ক্লিয়ারেন্স", "মানহানি", "সাইবার", "cyber security",
  "টিসিবি", "ভাতা", "জুলাই যোদ্ধা",
];

const MAX_SUB_QUERIES = 3;

function hardMatch(text: string, keywords: string[]): boolean {
  const lower = text.toLowerCase();
  return keywords.some((kw) => text.includes(kw) || lower.includes(kw.toLowerCase()));
}

function domainHit(text: string): boolean {
  const lower = text.toLowerCase();
  return DOMAIN_VOCAB.some((v) => lower.includes(v.toLowerCase()));
}

function fastGuardCheck(text: string): IntentResult | null {
  const base: IntentResult = {
    intent: "harmful",
    guardRailTriggered: true,
    guardRailCategory: null,
    subQueries: [],
    needsClarification: false,
    clarificationPromptBn: null,
    confidence: 0,
    notes: [],
    usage: null,
  };
  if (hardMatch(text, HARD_SELF_HARM))
    return { ...base, guardRailCategory: "self_harm", notes: ["hard_self_harm_match"] };
  if (hardMatch(text, HARD_ILLEGAL))
    return { ...base, guardRailCategory: "illegal", notes: ["hard_illegal_match"] };
  if (hardMatch(text, HARD_SYSTEM_PROBE))
    return {
      ...base,
      intent: "system_probe",
      guardRailCategory: "system_probe",
      notes: ["hard_system_probe_match"],
    };
  return null;
}

function formatHistory(history: ChatHistoryMsg[]): string {
  return history
    .filter((m) => (m.content ?? "").trim())
    .map((m) => `${m.role === "user" ? "User" : "Assistant"}: ${m.content.trim()}`)
    .join("\n");
}

function extractJson(raw: string): any {
  try {
    return JSON.parse(raw.trim());
  } catch {
    const m = raw.match(/\{[\s\S]*\}/);
    if (m) return JSON.parse(m[0]);
    throw new Error("no JSON in intent response");
  }
}

const VALID_INTENTS: Intent[] = [
  "factual",
  "chitchat",
  "ambiguous",
  "harmful",
  "system_probe",
  "multi_question",
];

export async function classifyIntent(
  query: string,
  history: ChatHistoryMsg[] = [],
): Promise<IntentResult> {
  const text = (query ?? "").trim();
  const notes: string[] = [];

  const empty: IntentResult = {
    intent: "chitchat",
    guardRailTriggered: false,
    guardRailCategory: null,
    subQueries: [],
    needsClarification: false,
    clarificationPromptBn: null,
    confidence: 0,
    notes: ["empty_query"],
    usage: null,
  };
  if (!text) return empty;

  const fast = fastGuardCheck(text);
  if (fast) return fast;

  let intent: Intent = "factual";
  let subQueries: string[] = [text];
  let guardTriggered = false;
  let guardCategory: string | null = null;
  let needsClarification = false;
  let clarificationPrompt: string | null = null;
  let confidence = 0.5;
  let usage: { prompt: number; completion: number } | null = null;

  try {
    const historyBlock = formatHistory(history);
    const userContent = historyBlock ? `${historyBlock}\n\nCurrent message: ${text}` : text;
    if (historyBlock) notes.push("history_included");

    const controller = new AbortController();
    const t = setTimeout(() => controller.abort(), 5000);
    let res;
    try {
      res = await generateText({
        model: secondaryModel,
        temperature: 0.0,
        maxTokens: 512,
        abortSignal: controller.signal,
        messages: [
          { role: "system", content: INTENT_SYSTEM_PROMPT },
          { role: "assistant", content: buildTimeReminder() },
          { role: "user", content: userContent },
        ],
      });
    } finally {
      clearTimeout(t);
    }
    if (res.usage)
      usage = { prompt: res.usage.promptTokens ?? 0, completion: res.usage.completionTokens ?? 0 };
    const data = extractJson(res.text);

    const cand = String(data.intent ?? "").toLowerCase().trim();
    if (VALID_INTENTS.includes(cand as Intent)) intent = cand as Intent;
    else notes.push(`unknown_intent=${cand}; default factual`);

    guardTriggered = Boolean(data.guard_rail_triggered);
    guardCategory = typeof data.guard_rail_category === "string" ? data.guard_rail_category : null;
    needsClarification = Boolean(data.needs_clarification);
    clarificationPrompt =
      typeof data.clarification_prompt_bn === "string" ? data.clarification_prompt_bn : null;
    confidence = Number(data.confidence ?? 0.5);

    const rawSubs = data.sub_queries;
    if (!Array.isArray(rawSubs)) {
      notes.push("sub_queries not a list; using raw query");
    } else {
      const cleaned = rawSubs
        .slice(0, MAX_SUB_QUERIES)
        .filter((s: unknown) => typeof s === "string")
        .map((s: string) => s.normalize("NFC").trim())
        .filter(Boolean);
      subQueries = intent === "factual" || intent === "multi_question" ? (cleaned.length ? cleaned : [text]) : [];
      if (rawSubs.length > MAX_SUB_QUERIES) notes.push(`truncated_sub_queries_to_${MAX_SUB_QUERIES}`);
    }
  } catch (e) {
    notes.push(`classifier_error: ${String(e)}`);
  }

  // Domain-vocab override — never override AWAY from factual.
  if (intent !== "factual" && intent !== "multi_question" && domainHit(text)) {
    notes.push(`domain_override: ${intent}→factual`);
    intent = "factual";
    if (subQueries.length === 0) subQueries = [text];
    guardTriggered = false;
    guardCategory = null;
  }

  return {
    intent,
    guardRailTriggered: guardTriggered,
    guardRailCategory: guardCategory,
    subQueries: intent === "factual" || intent === "multi_question" ? subQueries : [],
    needsClarification,
    clarificationPromptBn: clarificationPrompt,
    confidence,
    notes,
    usage,
  };
}
