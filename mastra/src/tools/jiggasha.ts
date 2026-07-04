/**
 * tools/jiggasha.ts — port of cogops/agents/retrieval_agent.py (Layer 3).
 *
 * - `jiggashaSearch` : a Mastra tool (Zod-validated) that POSTs one query to the
 *   Python Jiggasha /search service (Qdrant-backed, port 10000). Jiggasha itself
 *   is untouched.
 * - `retrieve()`     : parallel per-sub-query search, dedupe-by-passage_id merge,
 *   optional ReAct sufficiency loop, and [S#] source_map construction.
 */

import { createTool } from "@mastra/core/tools";
import { generateText } from "ai";
import { z } from "zod";
import { JIGGASHA_ENDPOINT, JIGGASHA_TIMEOUT_MS, RETRIEVAL, secondaryModel } from "../config.js";
import { JUDGE_SYSTEM_PROMPT } from "../prompts.js";
import { buildTimeReminder } from "../time-reminder.js";
import type { Passage, SourceMap } from "../types.js";

async function postSearch(query: string): Promise<any> {
  const payload = {
    query,
    top_k: RETRIEVAL.topKFetch,
    use_instruction: RETRIEVAL.useInstruction,
    cosine_threshold: RETRIEVAL.cosineThreshold,
    token_budget: RETRIEVAL.tokenBudget,
  };
  let lastErr: unknown = null;
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const controller = new AbortController();
      const t = setTimeout(() => controller.abort(), JIGGASHA_TIMEOUT_MS);
      let resp: Response;
      try {
        resp = await fetch(JIGGASHA_ENDPOINT, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
          signal: controller.signal,
        });
      } finally {
        clearTimeout(t);
      }
      if (resp.status >= 500 && attempt < 2) {
        await new Promise((r) => setTimeout(r, 250 * (attempt + 1)));
        continue;
      }
      if (!resp.ok) throw new Error(`jiggasha HTTP ${resp.status}`);
      return await resp.json();
    } catch (e) {
      lastErr = e;
      if (attempt < 2) {
        await new Promise((r) => setTimeout(r, 250 * (attempt + 1)));
        continue;
      }
      throw e;
    }
  }
  throw lastErr ?? new Error("jiggasha call failed without exception");
}

/** Mastra tool wrapper — registered in the harness for agentic dispatch/tests. */
export const jiggashaSearch = createTool({
  id: "jiggashaSearch",
  description:
    "Search the Bangladesh government-services Bengali corpus (Qdrant via Jiggasha). Returns ranked passages.",
  inputSchema: z.object({
    query: z.string().describe("Formal Bengali search query"),
    top_k: z.number().int().optional(),
    use_instruction: z.boolean().optional(),
    cosine_threshold: z.number().optional(),
    token_budget: z.number().int().optional(),
  }),
  outputSchema: z.object({
    results: z.array(z.any()),
    hits_total: z.number(),
  }),
  execute: async ({ context }) => {
    const raw = await postSearch(context.query);
    const results = (raw.results ?? []) as Passage[];
    return { results, hits_total: results.length };
  },
});

interface MergeResult {
  results: Passage[];
  instructions: string[];
  errors: string[] | null;
  elapsed_ms: number;
}

/** Parallel per-query search, dedupe by passage_id, keep best score. */
async function searchMulti(queries: string[]): Promise<MergeResult> {
  if (queries.length === 0) return { results: [], instructions: [], errors: null, elapsed_ms: 0 };

  const settled = await Promise.allSettled(queries.map((q) => postSearch(q)));
  const merged = new Map<number, Passage>();
  const instructions: string[] = [];
  const errors: string[] = [];
  let maxElapsed = 0;
  let firstErr: unknown = null;

  for (const s of settled) {
    if (s.status === "rejected") {
      errors.push(String(s.reason));
      if (firstErr === null) firstErr = s.reason;
      continue;
    }
    const res = s.value;
    maxElapsed = Math.max(maxElapsed, res.elapsed_ms ?? 0);
    if (res.instruction) instructions.push(res.instruction);
    for (const p of (res.results ?? []) as Passage[]) {
      const pid = p.passage_id;
      if (pid === undefined || pid === null) continue;
      const existing = merged.get(pid);
      if (!existing) {
        merged.set(pid, { ...p });
      } else {
        const ns = p.rerank_score;
        const os = existing.rerank_score;
        if (ns != null && os != null) {
          if (ns > os) merged.set(pid, { ...p });
        } else if (ns != null) {
          merged.set(pid, { ...p });
        } else if ((p.score ?? 0) > (existing.score ?? 0)) {
          merged.set(pid, { ...p });
        }
      }
    }
  }

  if (merged.size === 0 && firstErr !== null) throw firstErr;

  const sorted = [...merged.values()].sort((a, b) => {
    const ar = a.rerank_score ?? 0;
    const br = b.rerank_score ?? 0;
    if (br !== ar) return br - ar;
    return (b.score ?? 0) - (a.score ?? 0);
  });

  return {
    results: sorted,
    instructions,
    errors: errors.length ? errors : null,
    elapsed_ms: maxElapsed,
  };
}

/** RetrievalJudge — sufficiency verdict + optional refined query (fail-open). */
async function judge(query: string, passages: Passage[]): Promise<[string, string | null]> {
  if (passages.length === 0) return ["insufficient", query];
  const summary = passages
    .slice(0, 5)
    .map((p, i) => `[${i + 1}] ${(p.text ?? "").slice(0, 300)}`)
    .join("\n\n");
  try {
    const controller = new AbortController();
    const t = setTimeout(() => controller.abort(), 5000);
    let res;
    try {
      res = await generateText({
        model: secondaryModel,
        temperature: 0.0,
        maxTokens: 128,
        abortSignal: controller.signal,
        messages: [
          { role: "system", content: JUDGE_SYSTEM_PROMPT },
          { role: "assistant", content: buildTimeReminder() },
          {
            role: "user",
            content: `Query: ${query}\n\nRetrieved passages:\n${summary}\n\nJudge sufficiency and provide refined query if needed.`,
          },
        ],
      });
    } finally {
      clearTimeout(t);
    }
    const m = res.text.match(/\{[\s\S]*\}/);
    const data = JSON.parse(m ? m[0] : res.text);
    const sufficiency = String(data.sufficiency ?? "sufficient").toLowerCase();
    let refined = data.refined_query;
    refined = typeof refined === "string" && refined.trim() ? refined.trim() : null;
    return [sufficiency, refined];
  } catch {
    return ["sufficient", null];
  }
}

export function buildSourceMap(passages: Passage[]): SourceMap {
  const sourceMap: SourceMap = {};
  passages.forEach((p, idx) => {
    const pid = Number(p.passage_id ?? 0);
    if (!Number.isFinite(pid) || pid <= 0) return;
    const tag = `S${idx + 1}`;
    sourceMap[tag] = {
      passage_id: pid,
      text: p.text ?? "",
      category: p.category ?? "",
      sub_category: p.sub_category ?? "",
      service: p.service ?? "",
      topic: p.topic ?? "",
      chunk_type: p.chunk_type ?? "",
      score: Number(p.score ?? 0),
      rerank_score: p.rerank_score != null ? Number(p.rerank_score) : null,
      verdict: "yes",
      tool: "jiggasha",
    };
  });
  return sourceMap;
}

export interface RetrievalResult {
  passages: Passage[];
  sourceMap: SourceMap;
  instructions: string[];
  elapsedMs: number;
  errors: string[] | null;
}

/** Full Layer-3 retrieval: search, ReAct-refine, cap, source_map. */
export async function retrieve(queries: string[]): Promise<RetrievalResult> {
  const jres = await searchMulti(queries);
  let passages = jres.results;

  if (RETRIEVAL.maxReactIterations > 0) {
    for (let iter = 0; iter < RETRIEVAL.maxReactIterations; iter++) {
      const primaryQuery = queries[0] ?? "";
      const [sufficiency, refined] = await judge(primaryQuery, passages);
      if (sufficiency === "sufficient" || !refined) break;
      let refinedRes: MergeResult;
      try {
        refinedRes = await searchMulti([refined]);
      } catch {
        break;
      }
      const existingIds = new Set(passages.map((p) => p.passage_id).filter((x) => x != null));
      let added = 0;
      for (const p of refinedRes.results) {
        if (p.passage_id != null && !existingIds.has(p.passage_id)) {
          passages.push(p);
          existingIds.add(p.passage_id);
          added++;
        }
      }
      if (added === 0) break;
    }
  }

  if (passages.length > RETRIEVAL.mergeGlobalCap) {
    passages = passages.slice(0, RETRIEVAL.mergeGlobalCap);
  }

  return {
    passages,
    sourceMap: buildSourceMap(passages),
    instructions: jres.instructions,
    elapsedMs: jres.elapsed_ms,
    errors: jres.errors,
  };
}
