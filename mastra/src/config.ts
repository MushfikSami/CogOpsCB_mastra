/**
 * config.ts
 *
 * Central runtime configuration for the Mastra sidecar. Mirrors the seams from
 * the Python service's configs/config.yml + .env so the two stacks stay aligned:
 *   - Primary LLM   (composer, streaming)   ← LLM_* env
 *   - Secondary LLM (intent/query/judge/NLI) ← SECONDARY_* env
 *   - Jiggasha retrieval endpoint            ← JIGGASHA_ENDPOINT
 *
 * vLLM speaks the OpenAI protocol, so both providers are OpenAI-compatible
 * clients pointed at the existing self-hosted base URLs.
 */

import { createOpenAI } from "@ai-sdk/openai";
import * as dotenv from "dotenv";

dotenv.config();

function env(name: string, fallback: string): string {
  const v = process.env[name];
  return v === undefined || v === "" ? fallback : v;
}

// --- Model endpoints (map vLLM endpoints; identical to the Python service) ---
export const PRIMARY_BASE_URL = env("LLM_BASE_URL", "http://localhost:5000/v1/");
export const PRIMARY_API_KEY = env("LLM_API_KEY", "sk-noop");
export const PRIMARY_MODEL = env("LLM_MODEL_NAME", "qwen36");

export const SECONDARY_BASE_URL = env("SECONDARY_BASE_URL", "http://localhost:5000/v1/");
export const SECONDARY_API_KEY = env("SECONDARY_API_KEY", "sk-noop");
export const SECONDARY_MODEL = env("SECONDARY_MODEL_NAME", "qwen36");

// --- Jiggasha retrieval (unchanged Python microservice on :10000) ---
export const JIGGASHA_ENDPOINT = env("JIGGASHA_ENDPOINT", "http://localhost:10000/search");
export const JIGGASHA_TIMEOUT_MS = Number(env("JIGGASHA_TIMEOUT", "45")) * 1000;

// --- LibSQL storage (threads + working/observational memory) ---
export const MASTRA_DB_URL = env("MASTRA_DB_URL", "file:./mastra.db");

// --- Service ---
export const MASTRA_PORT = Number(env("MASTRA_PORT", "9100"));

// --- Retrieval knobs (mirror configs/config.yml tools.jiggasha) ---
export const RETRIEVAL = {
  topKFetch: 100,
  useInstruction: true,
  cosineThreshold: 0.5,
  tokenBudget: 28000,
  mergeGlobalCap: 50,
  maxReactIterations: 2,
};

// --- Composer generation knobs (mirror pipeline.composer) ---
export const COMPOSER = {
  temperature: 0.1,
  topP: 0.95,
  maxTokens: 2048,
};

export const AGENT_NAME = "আশা";

// --- Verifier ---
export const VERIFIER = {
  enabled: true,
  policy: "redact" as const,
  timeoutMs: 6000,
};

// OpenAI-compatible providers. `compatibility: "compatible"` relaxes strict
// OpenAI-only assumptions for self-hosted vLLM.
const primaryProvider = createOpenAI({
  baseURL: PRIMARY_BASE_URL,
  apiKey: PRIMARY_API_KEY,
  compatibility: "compatible",
});

const secondaryProvider = createOpenAI({
  baseURL: SECONDARY_BASE_URL,
  apiKey: SECONDARY_API_KEY,
  compatibility: "compatible",
});

export const primaryModel = primaryProvider(PRIMARY_MODEL);
export const secondaryModel = secondaryProvider(SECONDARY_MODEL);
