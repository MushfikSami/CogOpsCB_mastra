/**
 * mastra.ts — the central Mastra harness (singleton orchestrator).
 *
 * Binds the memory-carrying agent(s) and LibSQL storage into one typed
 * interface. Pull agents via `mastra.getAgent('composer')`.
 */

import { Mastra } from "@mastra/core";
import { composerAgent } from "./agents.js";
import { storage } from "./memory.js";

export const mastra = new Mastra({
  agents: { composer: composerAgent },
  // Cast: see note in memory.ts — LibSQLStore supports flag skew only.
  storage: storage as any,
});
