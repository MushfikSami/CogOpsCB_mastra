# CogOpsCB — Bangladesh Government Services Chatbot

A bilingual (Bengali / English) chatbot for Bangladesh government services.
Citation-grounded answers from a fixed government-services corpus. Refuses
when the corpus doesn't cover the question; never hallucinates citations.

## Architecture

Deterministic 5-stage pipeline (no ReAct loop, no LLM-side tool dispatch):

```
sanitize → router → Jiggasha multi-query rerank → composer → NLI verify
 (regex)   (sLLM)   (HTTP: embed+Qdrant+sLLM)    (pLLM,    (sLLM,
                                                  stream)   post-flight)
```

- **sanitize** — `cogops/pipeline/sanitize.py`. Length cap, control-char
  strip, prompt-injection regex.
- **router** — `cogops/pipeline/router.py`. ONE secondary-LLM call returns
  `{intent, sub_queries_bengali}`. Fast-path skips the LLM for pure-Bengali
  domain queries.
- **Jiggasha multi-query rerank** — `jiggasha/service.py` +
  `jiggasha/rerank.py`. Parallel embed → Qdrant top-K per sub → merge →
  ONE batched secondary-LLM call returns per-query class-index verdicts
  (`{"1":[[pid,0],...]}` where 0=yes, 1=weak). Cosine safety-net on LLM
  failure.
- **composer** — `cogops/agents/pipeline.py`. ONE streaming primary-LLM call
  with `<context>` + optional `<disambiguate>` + `<user_query>` blocks. The
  composer cites with `[S#]` tags allocated by the chatbot.
- **NLI verify** — `cogops/verifier/nli.py`. Post-flight, non-blocking,
  fail-soft. Strips unsupported `[S#]` claims under the configured policy.

**Disambiguation**: when the normalized intent is short (≤6 tokens) and the
kept passages span ≥2 distinct `(category, sub_category)` tuples, the bot
asks the user to clarify instead of guessing.

## Services

Two systemd units split the work:

- **`jiggasha-search.service`** — Bengali passage retrieval (embed +
  Qdrant + LLM rerank). Listens on `:10000`.
- **`govtchat.service`** — Chatbot API (sanitize + router + composer +
  verify) + the UI. Listens on `:9000`. `Requires=jiggasha-search.service`.

### Install / refresh

```sh
sudo cp jiggasha-search.service /etc/systemd/system/
sudo cp govtchat.service        /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now jiggasha-search.service
sudo systemctl enable --now govtchat.service
```

### Inspect

```sh
systemctl status jiggasha-search.service govtchat.service
journalctl -u jiggasha-search -f
journalctl -u govtchat -f
```

### Restart after a code change

```sh
sudo systemctl restart jiggasha-search.service   # picks up jiggasha/*
sudo systemctl restart govtchat.service          # picks up cogops/*, api.py
```

## UI

Open `http://<host>:9000/ui` in a browser.

- **User mode** (default): only the streamed assistant answer + sources block.
- **Debug mode**: paste the admin debug key (env `ADMIN_DEBUG_SECRET` on the
  server) into the field at the bottom of the page. A right-hand pane lights
  up showing every pipeline stage live — sanitize / router / history /
  retrieval / disambiguation / composer / verification / final_answer.
  Each stage shows token counts (`in=… out=…`) for the LLM call it covers.
  The right pane has the **full retrieved passages JSON** and the rerank
  verdicts behind expandable `<details>` blocks.

No build step; the page is a single self-contained `cogops/ui/index.html`
served by FastAPI. Tailwind is loaded from a CDN; everything else is
vanilla JS.

## Tests

### Unit tests

```sh
PYTHONPATH=. python3 -m pytest \\
  tests/test_core.py tests/test_grounding.py tests/test_input_handling.py \\
  tests/test_pipeline.py jiggasha/tests/test_rerank.py -v
```

### End-to-end harness (concurrent sessions)

`tests/run_tests.py` simulates N simultaneous users:

```sh
ADMIN_DEBUG_SECRET=<the key from .env> \\
PYTHONPATH=. python3 tests/run_tests.py \\
    --api-url http://localhost:9000 \\
    --concurrent 8 \\
    --seed 42
```

How it works:

1. Loads `user_query_examples.md` (117 queries) and the hand-tagged
   `tests/query_categories.yml`.
2. Shuffles with the seed (reproducible).
3. Distributes round-robin into N sessions. Each session has its own
   `user_id` so its conversation history accumulates server-side.
4. Sessions run **in parallel** via `asyncio.gather`; queries **within** a
   session run sequentially.
5. For each query, all NDJSON events from `/chat/stream` are collected and
   written as a single JSON file:

   ```
   debug_results/<run_ts>/
     session_01/
       q01_<safehash>.json
       q02_<safehash>.json
       …
       _session_summary.json
     session_02/
       …
     _run_summary.md
   ```

   The per-query JSON contains:
   - `query`, `category`, `original_corpus_index`, `session_index`,
     `query_index_in_session`, `user_id`
   - `outcome`: `{pass, verdict, detail}` from `tests/_classifier.py`
   - `summary`: intent, sub_queries, passages_returned, rerank_yes/weak,
     disambiguate_fired, final_answer, citation_tags, per-stage
     `token_usage` (`router` / `rerank` / `composer` / `nli`)
   - `events`: the full NDJSON event log

6. Exit code is `0` iff no CRITICAL-category failures
   (`gov_factual` / `political_refuse` / `chitchat`); else `1`.

CLI flags:

| flag           | default | meaning                                  |
|----------------|---------|------------------------------------------|
| `--api-url`    | `http://localhost:9000`                                       | API host
| `--concurrent` | `8`     | simultaneous sessions (~simulated users) |
| `--seed`       | `42`    | shuffle seed                             |
| `--limit N`    | -       | run only the first N queries after shuffle |
| `--timeout`    | `180`   | per-query timeout in seconds             |

## Local dev (no systemd)

```sh
# Jiggasha
cd jiggasha && python3 service.py     # listens on :10000

# API + UI (in another shell)
cd .. && python3 -m uvicorn api:app --host 0.0.0.0 --port 9000

# Open http://localhost:9000/ui
```

`jiggasha/.env` holds embedder / Qdrant / secondary-LLM credentials.
`.env` at the repo root holds the chatbot's primary/secondary LLM
credentials and `ADMIN_DEBUG_SECRET`.

## Config knobs

`configs/config.yml`:

- `tools.jiggasha.*` — multi-query rerank knobs (`top_k_per_sub`,
  `keep_cap`, `weak_per_sub_cap`, `fallback_cosine_min`, `timeout_seconds`).
- `pipeline.composer.*` — composer temperature / top_p / max_tokens.
- `pipeline.disambiguation.*` — `min_distinct_services`,
  `short_query_token_cap`, `candidate_cap`.
- `verifier.*` — NLI policy (`redact` / `refuse` / `warn`) + timeout.

`jiggasha/config.yml`:

- `qdrant.*`, `embedder.*` — connection details.
- `rerank.*` — defaults for the in-service LLM rerank (overridable per
  request).

## Status

- Pipeline: shipped, 117/117 e2e gate green on last full run.
- LLM rerank: lives inside the search service (Jiggasha). One HTTP call
  per turn returns vetted, per-sub bucketed passages.
- Disambiguation: triggers on short intents with spread services.
- Verifier: enabled with `policy: redact`.
- PII redaction: deferred (out of scope).
