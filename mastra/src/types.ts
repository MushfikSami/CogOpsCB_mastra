/**
 * types.ts — event + domain shapes shared across the orchestrator.
 *
 * The event shape mirrors cogops/events/types.py exactly: every event is a flat
 * object with `type` and `channel` ("user" | "debug" | "both") plus payload.
 */

export type Channel = "user" | "debug" | "both";

export interface StreamEvent {
  type: string;
  channel: Channel;
  [key: string]: unknown;
}

export function evt(type: string, channel: Channel, data: Record<string, unknown> = {}): StreamEvent {
  return { type, channel, ...data };
}

export interface ChatHistoryMsg {
  role: "user" | "assistant";
  content: string;
}

/** A retrieved passage as returned by Jiggasha /search. */
export interface Passage {
  passage_id?: number;
  text?: string;
  category?: string;
  sub_category?: string;
  service?: string;
  topic?: string;
  chunk_type?: string;
  score?: number;
  rerank_score?: number | null;
}

/** Value in the [S#] → meta source map (mirrors build_source_map). */
export interface SourceMeta {
  passage_id: number;
  text: string;
  category: string;
  sub_category: string;
  service: string;
  topic: string;
  chunk_type: string;
  score: number;
  rerank_score: number | null;
  verdict: "yes";
  tool: "jiggasha";
}

export type SourceMap = Record<string, SourceMeta>;
