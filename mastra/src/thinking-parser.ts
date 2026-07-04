/**
 * thinking-parser.ts — port of cogops/utils/thinking_parser.py
 *
 * Splits a streaming LLM response into ("answer" | "thinking", piece) pairs.
 * Handles <thinking>…</thinking> and ```thinking … ``` blocks, unclosed tags,
 * and tags spanning chunk boundaries (holds back a margin).
 */

const OPEN_TAG = "<thinking>";
const OPEN_BLOCK = "```thinking";
const OPEN_MARKERS = [OPEN_TAG, OPEN_BLOCK].sort((a, b) => b.length - a.length);
const OPEN_MAX = Math.max(...OPEN_MARKERS.map((m) => m.length));
const HOLDBACK = OPEN_MAX + 4;

export type ParsedPiece = ["answer" | "thinking", string];

export class ThinkingParser {
  private buffer = "";
  private inThinking = false;
  private openStyle: "tag" | "block" | null = null;

  private channel(): "answer" | "thinking" {
    return this.inThinking ? "thinking" : "answer";
  }

  private findOpen(buf: string): { style: "tag" | "block"; tag: string; idx: number } | null {
    let best: { style: "tag" | "block"; tag: string; idx: number } | null = null;
    for (const tag of OPEN_MARKERS) {
      const idx = buf.indexOf(tag);
      if (idx >= 0 && (best === null || idx < best.idx)) {
        best = { style: tag === OPEN_TAG ? "tag" : "block", tag, idx };
      }
    }
    return best;
  }

  private findClose(buf: string): { tag: string; idx: number } | null {
    const closeTag = this.openStyle === "tag" ? "</thinking>" : "```";
    const idx = buf.indexOf(closeTag);
    return idx >= 0 ? { tag: closeTag, idx } : null;
  }

  private *emitSafeTail(): Generator<ParsedPiece> {
    if (!this.buffer) return;
    if (this.buffer.length <= HOLDBACK) return; // could be a partial tag
    const safe = this.buffer.slice(0, -HOLDBACK);
    if (safe) yield [this.channel(), safe];
    this.buffer = this.buffer.slice(-HOLDBACK);
  }

  *feed(text: string): Generator<ParsedPiece> {
    if (!text) return;
    this.buffer += text;
    for (;;) {
      if (this.inThinking) {
        const result = this.findClose(this.buffer);
        if (result === null) {
          yield* this.emitSafeTail();
          return;
        }
        const before = this.buffer.slice(0, result.idx);
        if (before) yield [this.channel(), before];
        this.buffer = this.buffer.slice(result.idx + result.tag.length);
        this.inThinking = false;
      } else {
        const result = this.findOpen(this.buffer);
        if (result === null) {
          yield* this.emitSafeTail();
          return;
        }
        const before = this.buffer.slice(0, result.idx);
        if (before) yield [this.channel(), before];
        this.buffer = this.buffer.slice(result.idx + result.tag.length);
        this.openStyle = result.style;
        this.inThinking = true;
      }
    }
  }

  *flush(): Generator<ParsedPiece> {
    if (!this.buffer) return;
    yield [this.inThinking ? "thinking" : "answer", this.buffer];
    this.buffer = "";
    this.openStyle = null;
  }
}
