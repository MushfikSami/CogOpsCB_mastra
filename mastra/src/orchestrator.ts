/**
 * orchestrator.ts — port of Orchestrator.process_query.
 *
 * Runs one user turn through the 6 layers and yields events matching the Python
 * NDJSON contract. Conversational history + working memory are owned by Mastra
 * (LibSQL); short-circuit branches (guard/chitchat/ambiguous/date) return static
 * templates before retrieval.
 */

import { randomUUID } from "node:crypto";
import { inputGuardCheck } from "./input-guard.js";
import { classifyIntent, isFactual, shouldRefuse, type IntentResult } from "./intent.js";
import { processQuery } from "./query-processor.js";
import { retrieve } from "./tools/jiggasha.js";
import { compose } from "./composer.js";
import { verify } from "./verifier/index.js";
import { memory } from "./memory.js";
import { RESPONSES, guardRailResponse } from "./responses.js";
import { buildTimeReminder } from "./time-reminder.js";
import { evt, type ChatHistoryMsg, type StreamEvent } from "./types.js";

const DATE_QUERY_RE =
  /(\bdate\b|\btoday\b|\bhijri\b|\barabic\s+calendar\b|আজকের\s+তারিখ|আজ\s+কত\s+তারিখ|আরবি\s+তারিখ|হিজরি\s+তারিখ|আজ\s+কি\s+দিন|আজ\s+কোন\s+দিন|সময়\s+কত|বাংলাদেশ\s+সময়)/iu;

function buildDateAnswer(query: string): string {
  const lower = query.toLowerCase();
  const reminder = buildTimeReminder();
  let dateLine = "";
  let weekdayLine = "";
  let timeLine = "";
  for (const line of reminder.split("\n")) {
    if (line.startsWith("- Date:")) dateLine = line.replace("- Date:", "").trim().replace(/\s+/g, " ");
    else if (line.startsWith("- Weekday:"))
      weekdayLine = line.replace("- Weekday:", "").trim().replace(/\s+/g, " ");
    else if (line.startsWith("- Time:")) timeLine = line.replace("- Time:", "").trim().replace(/\s+/g, " ");
  }
  if (lower.includes("hijri") || lower.includes("arabic") || query.includes("আরবি") || query.includes("হিজরি")) {
    return `আমি বর্তমানে হিজরি/আরবি ক্যালেন্ডারের তারিখ দিতে পারছি না। বাংলাদেশ সময় অনুযায়ী আজ ${weekdayLine}। আজকের তারিখ ${dateLine}।`;
  }
  return `আজ ${weekdayLine}। আজকের তারিখ ${dateLine} এবং বর্তমান সময় ${timeLine}।`;
}

/** Read recent turns from Mastra memory for history-aware intent/query steps. */
async function loadHistory(resourceId: string, threadId: string): Promise<ChatHistoryMsg[]> {
  try {
    const res: any = await memory.query({
      threadId,
      resourceId,
      selectBy: { last: 8 },
    });
    const msgs = (res?.uiMessages ?? res?.messages ?? []) as any[];
    const out: ChatHistoryMsg[] = [];
    for (const m of msgs) {
      const role = m.role === "user" ? "user" : m.role === "assistant" ? "assistant" : null;
      if (!role) continue;
      const content =
        typeof m.content === "string"
          ? m.content
          : Array.isArray(m.content)
            ? m.content.map((c: any) => c.text ?? "").join("")
            : (m.text ?? "");
      if (content) out.push({ role, content });
    }
    return out;
  } catch {
    return [];
  }
}

/** Persist the finished turn so Mastra memory has the raw exchange. */
async function persistTurn(
  resourceId: string,
  threadId: string,
  userText: string,
  assistantText: string,
): Promise<void> {
  try {
    await memory.createThread({ threadId, resourceId }).catch(() => {});
    await memory.saveMessages({
      messages: [
        {
          id: randomUUID(),
          threadId,
          resourceId,
          role: "user",
          content: userText,
          createdAt: new Date(),
          type: "text",
        },
        {
          id: randomUUID(),
          threadId,
          resourceId,
          role: "assistant",
          content: assistantText,
          createdAt: new Date(),
          type: "text",
        },
      ] as any,
    });
  } catch {
    // memory persistence is best-effort
  }
}

export async function* processQueryStream(
  userQuery: string,
  resourceId: string,
  threadId: string,
): AsyncGenerator<StreamEvent> {
  const turnId = randomUUID().slice(0, 8);
  const originalQuery = userQuery ?? "";

  try {
    // ----- Layer 0: InputGuard -----
    const [cleanQuery, refusalReason] = inputGuardCheck(originalQuery);
    if (refusalReason !== null) {
      yield evt("sanitize_verdict", "debug", { reason: refusalReason, turn_id: turnId });
      yield evt("answer_chunk", "both", { content: RESPONSES.input_invalid_refusal_bn });
      yield evt("final_answer", "both", {
        content: RESPONSES.input_invalid_refusal_bn,
        turn_id: turnId,
        source_map: {},
        reason: `sanitize_${refusalReason}`,
      });
      yield evt("answer_complete", "both", { turn_id: turnId });
      await persistTurn(resourceId, threadId, originalQuery, RESPONSES.input_invalid_refusal_bn);
      return;
    }

    // ----- History (from Mastra memory) -----
    const history = await loadHistory(resourceId, threadId);
    yield evt("history_loaded", "debug", { turns: history.length, turn_id: turnId });

    // ----- Layer 1: IntentClassifier -----
    let intent: IntentResult;
    try {
      intent = await classifyIntent(cleanQuery, history);
    } catch (e) {
      intent = {
        intent: "factual",
        guardRailTriggered: false,
        guardRailCategory: null,
        subQueries: [cleanQuery],
        needsClarification: false,
        clarificationPromptBn: null,
        confidence: 0.5,
        notes: [`classifier_exception: ${String(e)}`],
        usage: null,
      };
    }

    yield evt("intent_classified", "debug", {
      intent: intent.intent,
      guard_rail_triggered: intent.guardRailTriggered,
      guard_rail_category: intent.guardRailCategory,
      sub_queries: intent.subQueries,
      needs_clarification: intent.needsClarification,
      confidence: intent.confidence,
      notes: intent.notes,
      token_usage: intent.usage,
      turn_id: turnId,
    });

    // ----- Guard-rail branch -----
    if (shouldRefuse(intent)) {
      const text = guardRailResponse(intent.guardRailCategory);
      yield evt("answer_chunk", "both", { content: text });
      yield evt("final_answer", "both", {
        content: text,
        turn_id: turnId,
        source_map: {},
        reason: `guard_rail_${intent.guardRailCategory}`,
      });
      yield evt("answer_complete", "both", { turn_id: turnId });
      await persistTurn(resourceId, threadId, originalQuery, text);
      return;
    }

    // ----- Chitchat branch -----
    if (intent.intent === "chitchat") {
      yield evt("answer_chunk", "both", { content: RESPONSES.chitchat_greeting_bn });
      yield evt("final_answer", "both", {
        content: RESPONSES.chitchat_greeting_bn,
        turn_id: turnId,
        source_map: {},
        reason: "chitchat",
      });
      yield evt("answer_complete", "both", { turn_id: turnId });
      await persistTurn(resourceId, threadId, originalQuery, RESPONSES.chitchat_greeting_bn);
      return;
    }

    // ----- Ambiguous branch -----
    if (intent.intent === "ambiguous" && intent.needsClarification) {
      const clarification =
        intent.clarificationPromptBn ||
        "আপনার প্রশ্নটি একাধিক বিষয় জড়িত বোধ হচ্ছে। অনুগ্রহ করে আরও স্পষ্ট করে জানান।";
      yield evt("answer_chunk", "both", { content: clarification });
      yield evt("final_answer", "both", {
        content: clarification,
        turn_id: turnId,
        source_map: {},
        reason: "ambiguous_clarification",
      });
      yield evt("answer_complete", "both", { turn_id: turnId });
      await persistTurn(resourceId, threadId, originalQuery, clarification);
      return;
    }

    // ----- Date / Time branch -----
    if (DATE_QUERY_RE.test(cleanQuery)) {
      const dateAnswer = buildDateAnswer(cleanQuery);
      yield evt("answer_chunk", "both", { content: dateAnswer });
      yield evt("final_answer", "both", {
        content: dateAnswer,
        turn_id: turnId,
        source_map: {},
        reason: "date_time_direct",
      });
      yield evt("answer_complete", "both", { turn_id: turnId });
      await persistTurn(resourceId, threadId, originalQuery, dateAnswer);
      return;
    }

    // ----- Factual / Multi-question branch -----
    if (!isFactual(intent)) {
      intent.intent = "factual";
      if (intent.subQueries.length === 0) intent.subQueries = [cleanQuery];
    }

    // Layer 2: QueryProcessor
    const processed = await processQuery(cleanQuery, intent.subQueries, history);
    yield evt("queries_processed", "debug", {
      queries: processed.queries,
      overflow: processed.overflow,
      disambiguated: processed.disambiguated,
      formalized: processed.formalized,
      turn_id: turnId,
    });

    // Layer 3: RetrievalAgent
    const retrieval = await retrieve(processed.queries);
    yield evt("retrieval_done", "debug", {
      passages_returned: retrieval.passages.length,
      instructions: retrieval.instructions,
      elapsed_ms: retrieval.elapsedMs,
      errors: retrieval.errors,
      turn_id: turnId,
    });
    yield evt("source_map_allocated", "debug", {
      n_sources: Object.keys(retrieval.sourceMap).length,
      tags: Object.keys(retrieval.sourceMap),
      turn_id: turnId,
    });

    // Layer 4: ComposerAgent (streaming, memory-backed)
    let rawAnswer = "";
    const gen = compose(cleanQuery, retrieval.sourceMap, resourceId, threadId);
    let next = await gen.next();
    while (!next.done) {
      yield next.value;
      next = await gen.next();
    }
    rawAnswer = next.value.rawAnswer;

    if (!rawAnswer) {
      yield evt("answer_chunk", "both", { content: RESPONSES.refusal_text_bn });
      yield evt("final_answer", "both", {
        content: RESPONSES.refusal_text_bn,
        turn_id: turnId,
        source_map: {},
        reason: "composer_empty",
      });
      yield evt("answer_complete", "both", { turn_id: turnId });
      await persistTurn(resourceId, threadId, originalQuery, RESPONSES.refusal_text_bn);
      return;
    }

    // Layer 5: PostFlightVerifier
    const { finalAnswer, events: postEvents, sourcesBlock } = await verify(
      rawAnswer,
      retrieval.sourceMap,
    );
    for (const ev of postEvents) yield ev;

    if (sourcesBlock) {
      yield evt("answer_chunk", "both", { content: "\n\n" + sourcesBlock });
    }

    // source_map without text (mirrors Python final_answer payload)
    const sourceMapNoText: Record<string, unknown> = {};
    for (const [tag, meta] of Object.entries(retrieval.sourceMap)) {
      const { text, ...rest } = meta;
      sourceMapNoText[tag] = rest;
    }

    yield evt("final_answer", "both", {
      content: finalAnswer,
      source_map: sourceMapNoText,
      turn_id: turnId,
    });
    yield evt("answer_complete", "both", { turn_id: turnId });

    await persistTurn(resourceId, threadId, originalQuery, finalAnswer);
  } catch (e) {
    const fallback =
      "দুঃখিত, এই মুহূর্তে সার্ভারে অতিরিক্ত চাপ রয়েছে। অনুগ্রহ করে কিছুক্ষণ পরে আবার চেষ্টা করুন।";
    yield evt("answer_chunk", "both", { content: fallback, error: String(e) });
    yield evt("answer_complete", "both", { turn_id: turnId });
  }
}
