/**
 * memory.ts — LibSQL-backed Mastra memory.
 *
 * Implements the two memory behaviors from the integration plan:
 *
 *  1. Observational / compressed thread memory — instead of feeding raw history,
 *     we cap recent messages (`lastMessages`) and let a TokenLimiter processor
 *     keep the active window small and cacheable, preventing context bloat.
 *
 *  2. Working memory (cross-thread) — a resource-scoped structured template
 *     (scope: "resource") acts as a persistent per-user scratchpad. Mastra's
 *     updateWorkingMemory tool overwrites colliding facts (collision erasure),
 *     so the newest value replaces the old one.
 */

import { Memory } from "@mastra/memory";
import { LibSQLStore } from "@mastra/libsql";
import { TokenLimiter } from "@mastra/memory/processors";
import { MASTRA_DB_URL } from "./config.js";

export const storage = new LibSQLStore({ url: MASTRA_DB_URL });

// Structured, resource-scoped working memory. New info that collides with an
// existing field overwrites it — the old value is functionally erased.
const WORKING_MEMORY_TEMPLATE = `# User Scratchpad (resource-scoped, cross-thread)

## Identity & Locale
- Preferred language:
- Location / District:

## Active Task
- Current service / document in focus:
- Specific goal (fee / procedure / eligibility / contact):

## Known Entities (overwrite on collision)
- NID / Passport / other document type mentioned:
- Names / offices referenced:

## Constraints & Parameters
- Deadlines or dates mentioned:
- Other user-supplied parameters:
`;

export const memory = new Memory({
  // Cast: LibSQLStore@0.10.3 predates core's `resourceWorkingMemory` supports
  // flag; the store implements it at runtime. Drop the cast when versions align.
  storage: storage as any,
  options: {
    // Observational: keep the active window small; compress out obsolete turns.
    lastMessages: 6,
    workingMemory: {
      enabled: true,
      scope: "resource",
      template: WORKING_MEMORY_TEMPLATE,
    },
  },
  processors: [
    // Anti-overfitting: hard cap on tokens fed back into the context window.
    new TokenLimiter(24000),
  ],
});
