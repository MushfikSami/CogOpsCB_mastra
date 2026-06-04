# Tools (`cogops/tools/`)

The tools package bridges the agent layer to external services.  Today the only external tool is Jiggasha (passage retrieval).

---

## `jiggasha.py`

**Role:** Retrieval tool for the ReAct agent.

**Why it exists:**
- Posts queries to the external Jiggasha service (Qdrant vector search with dynamic instruction-based retrieval).
- Filters results by a minimum score threshold.
- Allocates monotonic `[S#]` tags via `ToolContext` so passages are consistently referenced across the pipeline.
- Returns Bengali-formatted passage blocks for the LLM plus structured telemetry sources.
- Returns a `NO_RELEVANT_RESULTS` sentinel when nothing clears the threshold so the composer knows the context is empty.

**Key design choice:** Monotonic tag allocation (`S1`, `S2`, `S3`...) ensures citations are stable across ReAct iterations.  Without this, a passage could be `S1` in turn 1 and `S3` in turn 2, breaking citations.

---

## `registry.py`

**Role:** Plugin tool registry.

**Why it exists:**
- Discovers tool modules by config slug (e.g. `jiggasha`).
- Builds OpenAI-style function schemas from tool module metadata.
- Binds server-side parameters (like `ToolContext`) that the LLM does not know about but the tool implementation needs.
- Designed to be extensible: adding a new tool means adding a module and a config entry, no orchestrator changes.

**Key design choice:** Schema auto-generation from Python type hints keeps tool definitions in one place (the module) instead of duplicating them in config files.
