/**
 * composer.ts — port of ComposerAgent (Layer 4).
 *
 * Streams the primary-LLM answer via the memory-carrying Mastra agent. Thread +
 * working memory are supplied through { resource, thread }, so raw history is NOT
 * manually injected — Mastra's memory does that (compressed).
 *
 * Yields composer_start / answer_chunk / reasoning_chunk / composer_done events,
 * matching the Python event contract.
 */

import { mastra } from "./mastra.js";
import { buildTimeReminder } from "./time-reminder.js";
import { COMPOSER } from "./config.js";
import { ThinkingParser } from "./thinking-parser.js";
import { evt, type SourceMap, type StreamEvent } from "./types.js";

/** Render the composer's user message: <context> + <user_query>. */
export function buildComposerUserMessage(rawUserQuery: string, sourceMap: SourceMap): string {
  const ctx: string[] = ["<context>"];
  const tags = Object.keys(sourceMap);
  if (tags.length === 0) {
    ctx.push("(no passages retrieved)");
  } else {
    for (const tag of tags) {
      const meta = sourceMap[tag];
      const hb: string[] = [];
      const fields: Array<[keyof SourceMap[string], string]> = [
        ["category", "বিভাগ"],
        ["sub_category", "উপ-বিভাগ"],
        ["service", "সেবা"],
        ["topic", "বিষয়"],
      ];
      for (const [key, label] of fields) {
        const v = (meta[key] as string) || "";
        if (v) hb.push(`${label}: ${v}`);
      }
      if (meta.chunk_type === "wiki") hb.push("উৎস: উইকিপিডিয়া");
      else if (meta.chunk_type === "govt_service") hb.push("উৎস: সরকারি সেবা");
      const header = hb.length ? hb.join(" | ") : "—";
      ctx.push("");
      ctx.push(`[${tag}] (${header})`);
      ctx.push(meta.text ?? "");
    }
  }
  ctx.push("</context>");
  return `${ctx.join("\n")}\n\n<user_query>\n${rawUserQuery}\n</user_query>`;
}

export interface ComposeResult {
  rawAnswer: string;
}

export async function* compose(
  rawQuery: string,
  sourceMap: SourceMap,
  resourceId: string,
  threadId: string,
): AsyncGenerator<StreamEvent, ComposeResult> {
  const agent = mastra.getAgent("composer");
  const userMessage = buildComposerUserMessage(rawQuery, sourceMap);

  yield evt("composer_start", "debug", { n_sources: Object.keys(sourceMap).length });

  const parser = new ThinkingParser();
  const answerAcc: string[] = [];

  const stream = await agent.stream(
    [
      { role: "assistant", content: buildTimeReminder() },
      { role: "user", content: userMessage },
    ],
    {
      memory: { resource: resourceId, thread: threadId },
      temperature: COMPOSER.temperature,
      topP: COMPOSER.topP,
      maxTokens: COMPOSER.maxTokens,
    },
  );

  for await (const text of stream.textStream) {
    if (!text) continue;
    for (const [channel, piece] of parser.feed(text)) {
      if (channel === "answer") {
        answerAcc.push(piece);
        yield evt("answer_chunk", "both", { content: piece });
      } else {
        yield evt("reasoning_chunk", "debug", { content: piece });
      }
    }
  }
  for (const [channel, piece] of parser.flush()) {
    if (channel === "answer") {
      answerAcc.push(piece);
      yield evt("answer_chunk", "both", { content: piece });
    } else {
      yield evt("reasoning_chunk", "debug", { content: piece });
    }
  }

  const rawAnswer = answerAcc.join("").trim();
  yield evt("composer_done", "debug", { chars: rawAnswer.length });
  return { rawAnswer };
}
