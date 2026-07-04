/**
 * server.ts — HTTP boundary for the Mastra sidecar.
 *
 * Endpoints (consumed by the Python FastAPI proxy in api.py):
 *   POST /chat/stream  — NDJSON stream of events (ALL channels; the Python proxy
 *                        applies filter_for_user / filter_for_debug + X-Debug-Key).
 *   GET  /health       — liveness + LibSQL/Jiggasha reachability hint.
 *
 * Events are emitted verbatim as {type, channel, ...} — the same contract as
 * cogops/events/types.py — so the existing UI works unchanged.
 */

import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { MASTRA_PORT } from "./config.js";
import { processQueryStream } from "./orchestrator.js";

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on("data", (c) => chunks.push(c as Buffer));
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf-8")));
    req.on("error", reject);
  });
}

async function handleChatStream(req: IncomingMessage, res: ServerResponse): Promise<void> {
  let body: { user_id?: string; query?: string; thread_id?: string };
  try {
    body = JSON.parse((await readBody(req)) || "{}");
  } catch {
    res.writeHead(400, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ error: "invalid JSON" }));
    return;
  }

  const userId = body.user_id?.trim();
  const query = body.query ?? "";
  if (!userId) {
    res.writeHead(400, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ error: "user_id required" }));
    return;
  }
  // One active thread per user mirrors the current per-user session model.
  const threadId = body.thread_id?.trim() || userId;

  res.writeHead(200, {
    "Content-Type": "application/x-ndjson; charset=utf-8",
    "Cache-Control": "no-cache",
    Connection: "keep-alive",
  });

  try {
    for await (const event of processQueryStream(query, userId, threadId)) {
      res.write(JSON.stringify(event) + "\n");
    }
  } catch (e) {
    res.write(JSON.stringify({ type: "error", channel: "both", error: String(e) }) + "\n");
  } finally {
    res.end();
  }
}

const server = createServer((req, res) => {
  const url = req.url ?? "";
  if (req.method === "POST" && url === "/chat/stream") {
    handleChatStream(req, res).catch((e) => {
      if (!res.headersSent) res.writeHead(500);
      res.end(JSON.stringify({ error: String(e) }));
    });
    return;
  }
  if (req.method === "GET" && url === "/health") {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ status: "ok", service: "cogops-mastra" }));
    return;
  }
  res.writeHead(404, { "Content-Type": "application/json" });
  res.end(JSON.stringify({ error: "not found" }));
});

server.listen(MASTRA_PORT, () => {
  // eslint-disable-next-line no-console
  console.log(`[cogops-mastra] listening on :${MASTRA_PORT}`);
});
