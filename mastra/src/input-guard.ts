/**
 * input-guard.ts — port of cogops/agents/input_guard.py (Layer 0).
 *
 * Pure-code validation, zero LLM latency. Returns [cleanQuery, null] on success
 * or ["", reasonCode] on rejection.
 */

export const REASON_EMPTY = "empty";
export const REASON_TOO_LONG = "too_long";
export const REASON_BINARY_OR_CONTROL = "binary_or_control";
export const REASON_INJECTION = "injection_attempt";
export const REASON_SPAM = "spam";
export const REASON_LOW_ENTROPY = "low_entropy";
export const REASON_TOKEN_BOMB = "token_bomb";

export interface GuardConfig {
  maxChars: number;
  controlCharThreshold: number;
  entropyThreshold: number;
  entropyMinLength: number;
  tokenBombMinWords: number;
  tokenBombMaxAvgLen: number;
}

const DEFAULT_CONFIG: GuardConfig = {
  maxChars: 4096,
  controlCharThreshold: 0.1,
  entropyThreshold: 1.5,
  entropyMinLength: 50,
  tokenBombMinWords: 12,
  tokenBombMaxAvgLen: 2.0,
};

const INJECTION_PATTERNS: RegExp[] = [
  /ignore (?:all |previous |above |prior )?(?:instructions|prompts|rules|directives)/i,
  /disregard (?:all |previous |the )?(?:instructions|prompts|rules|system)/i,
  /(?:^|\s)system\s*:\s*/i,
  /<\/?(?:context|system|user|assistant|im_start|im_end)\s*>/i,
  /\{\{[^}]*system[^}]*\}\}/,
  /<\|[^|]{0,40}\|>/,
  /ager (?:shob |sob )?kotha (?:vule|bhule) jao/i,
  /jailbreak|DAN mode|you are now|new instructions/i,
];

// Match any character repeated 100+ times (equiv. to Python (.)\1{99,}).
const REPETITION_RE = /(.)\1{99,}/s;

function isControlChar(cp: number): boolean {
  // Unicode category Cc: C0 (0x00-0x1F) and C1 (0x7F-0x9F).
  return (cp >= 0x00 && cp <= 0x1f) || (cp >= 0x7f && cp <= 0x9f);
}

function controlCharFraction(text: string): number {
  if (!text) return 0;
  let bad = 0;
  for (const ch of text) {
    if (ch === "\n" || ch === "\t" || ch === "\r") continue;
    const cp = ch.codePointAt(0)!;
    if (isControlChar(cp)) bad += 1;
  }
  return bad / text.length;
}

function shannonEntropy(text: string): number {
  if (!text) return 0;
  const freq = new Map<string, number>();
  for (const ch of text) freq.set(ch, (freq.get(ch) ?? 0) + 1);
  let entropy = 0;
  const length = text.length;
  for (const count of freq.values()) {
    const p = count / length;
    entropy -= p * Math.log2(p);
  }
  return entropy;
}

function isTokenBomb(text: string, cfg: GuardConfig): boolean {
  const words = text.split(/\s+/).filter((w) => w.trim());
  if (words.length < cfg.tokenBombMinWords) return false;
  const avgLen = words.reduce((s, w) => s + w.length, 0) / words.length;
  return avgLen < cfg.tokenBombMaxAvgLen;
}

export function inputGuardCheck(
  query: string | null | undefined,
  cfg: GuardConfig = DEFAULT_CONFIG,
): [string, string | null] {
  if (query === null || query === undefined) return ["", REASON_EMPTY];
  if (typeof query !== "string") return ["", REASON_BINARY_OR_CONTROL];
  if (!query.trim()) return ["", REASON_EMPTY];
  if (query.length > cfg.maxChars) return ["", REASON_TOO_LONG];
  if (query.includes("\x00")) return ["", REASON_BINARY_OR_CONTROL];
  if (controlCharFraction(query) > cfg.controlCharThreshold) return ["", REASON_BINARY_OR_CONTROL];
  for (const pat of INJECTION_PATTERNS) {
    if (pat.test(query)) return ["", REASON_INJECTION];
  }
  if (REPETITION_RE.test(query)) return ["", REASON_SPAM];
  if (query.length >= cfg.entropyMinLength && shannonEntropy(query) < cfg.entropyThreshold) {
    return ["", REASON_LOW_ENTROPY];
  }
  if (isTokenBomb(query, cfg)) return ["", REASON_TOKEN_BOMB];

  // NFC-normalize, trim, collapse whitespace.
  let clean = query.normalize("NFC").trim();
  clean = clean.replace(/[ \t]+/g, " ");
  clean = clean.replace(/\n{3,}/g, "\n\n");
  return [clean, null];
}
