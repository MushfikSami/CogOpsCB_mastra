# CogOpsCB — Mastra Sidecar

TypeScript/Node service that owns the **6-agent orchestrator** and **memory**
for CogOpsCB. The Python FastAPI app (`../api.py`) proxies `POST /chat/stream`
to this service and relays its NDJSON stream unchanged. The Bengali retrieval
microservice **Jiggasha** (Python, `:10000`) is untouched and called here as a
tool.

## Architecture

```
FastAPI api.py  ──HTTP──▶  Mastra sidecar (this, :9100)
                              ├ Mastra harness (mastra.ts) — singleton, getAgent('composer')
                              ├ Layer 0 InputGuard      (input-guard.ts, code)
                              ├ Layer 1 IntentClassifier (intent.ts, secondary LLM + fast-path guards)
                              ├ Layer 2 QueryProcessor   (query-processor.ts)
                              ├ Layer 3 RetrievalAgent   (tools/jiggasha.ts) ──▶ Jiggasha :10000 ──▶ Qdrant
                              ├ Layer 4 ComposerAgent    (composer.ts) — Mastra Agent w/ memory (primary LLM, streaming)
                              ├ Layer 5 PostFlightVerifier (verifier/*) — strip + NLI + policy + Sources
                              └ Memory (memory.ts) — @mastra/libsql
                                   ├ Observational: lastMessages cap + TokenLimiter (anti-overfitting)
                                   └ Working memory: resource-scoped template, collision-erasure via updateWorkingMemory
```

Events are emitted verbatim as `{ type, channel, ... }` (channel = `user` |
`debug` | `both`), matching `cogops/events/types.py`, so the existing UI and the
`X-Debug-Key` gate keep working. The proxy does the channel filtering.

## Config

All via env (shared with the Python `.env`; see `../.env.example`):

| Var | Meaning | Default |
|-----|---------|---------|
| `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL_NAME` | Primary (composer) vLLM endpoint | `http://localhost:5000/v1/` |
| `SECONDARY_*` | Secondary (intent/query/judge/NLI) vLLM endpoint | same |
| `JIGGASHA_ENDPOINT` | Jiggasha `/search` URL | `http://localhost:10000/search` |
| `MASTRA_DB_URL` | LibSQL/Turso URL for memory | `file:./mastra.db` |
| `MASTRA_PORT` | Listen port | `9100` |

vLLM speaks the OpenAI protocol, so both models are OpenAI-compatible providers
pointed at the existing base URLs — no endpoint changes vs. the Python service.

## Run

```bash
cd mastra
npm install
npm run dev          # tsx watch (development)
# or
npm run build && npm start   # dist/server.js (production)
```

Then start the Python proxy as before (`uvicorn api:app --port 9000`) with
`MASTRA_URL=http://localhost:9100`, and keep Jiggasha running on `:10000`.

## Notes

- The deterministic LLM layers (intent, query-processing, retrieval judge, NLI)
  call the ai-sdk directly for strict JSON / temperature-0 behavior. The
  **Composer** is the memory-carrying `Mastra.Agent` registered in the harness.
- `LibSQLStore` is cast to `any` at two binding sites because `@mastra/libsql`
  pinned here predates core's `resourceWorkingMemory` supports flag (runtime is
  fine). Remove the casts when the package versions align.
