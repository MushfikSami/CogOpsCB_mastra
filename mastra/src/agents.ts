/**
 * agents.ts — Mastra Agent definitions.
 *
 * The Composer is the memory-carrying agent: it holds the LibSQL-backed thread
 * + resource-scoped working memory and the jiggashaSearch tool (registered for
 * agentic dispatch / tests; the live pipeline calls retrieval directly).
 *
 * The deterministic LLM layers (intent, query-processing, judge, NLI) are
 * implemented as code steps using the ai-sdk directly — see their own modules.
 */

import { Agent } from "@mastra/core/agent";
import { AGENT_NAME, primaryModel } from "./config.js";
import { getComposerPrompt } from "./prompts.js";
import { memory } from "./memory.js";
import { jiggashaSearch } from "./tools/jiggasha.js";

export const composerAgent = new Agent({
  name: "composer",
  instructions: getComposerPrompt(AGENT_NAME),
  model: primaryModel,
  memory,
  tools: { jiggashaSearch },
});
