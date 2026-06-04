# Session & Logging (`cogops/session/`)

This package handles conversation persistence, query audit trails, and session traces.

---

## `logger.py`

**Role:** Unified query and session trace logger.

**Why it exists:**
- Writes query logs and full interaction traces to PostgreSQL if `DATABASE_URL` is available and `psycopg2` is installed; otherwise falls back to rolling JSONL files.
- Stores raw queries with timestamps and complete session traces (events, tool calls, reasoning chunks, answer chunks, durations).
- The orchestrator calls this after every turn to persist the full trace.

**Key design choice:** Graceful degradation (Postgres → JSONL) means the system works on a laptop without Postgres installed.

---

## `query_log.py`

**Role:** Append-only JSONL query log.

**Why it exists:**
- Simple, durable, no-schema logging for quick debugging.
- Automatic 10-day retention pruning so the file does not grow unbounded.
- Each line is a JSON object with query text, timestamp, user_id, and response summary.

---

## `session_logger.py`

**Role:** JSONL-only audit logger for full streaming interaction traces.

**Why it exists:**
- Collects every event emitted during a session: user messages, assistant chunks, debug events, tool calls, errors.
- Written as one JSON object per session to a timestamped JSONL file.
- Used for post-hoc debugging and evaluation of pipeline behavior.

---

## `redis_store.py`

**Role:** Redis-backed conversation turn store.

**Why it exists:**
- Persists conversation history across requests so the disambiguator and classifier have context.
- Seamless in-memory fallback when Redis is unavailable.  The orchestrator does not crash if Redis is down.
- Stores turns as a simple list of `{role, content}` dicts.

**Key design choice:** The in-memory fallback is a `defaultdict(list)` keyed by user_id.  This is not persistent across restarts, but it keeps the system functional during Redis outages.
