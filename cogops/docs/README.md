# CogOps Source Documentation

This folder documents every module and file in `cogops/` by **architectural purpose**, not by code.  For implementation details, read the source files and their docstrings.

---

## What CogOps Is

CogOps is a 6-layer agent pipeline that answers Bengali government-service questions.  It receives a user query, classifies intent, formalizes the query, retrieves passages from a Qdrant vector store (via Jiggasha), composes a cited answer with a primary LLM, and verifies the answer with an NLI-based post-flight checker.

Two implementations coexist:

1. **Orchestrator Agent Pipeline** (`agents/`) — the production path.  6 sequential layers, each an independent agent with its own system prompt and LLM call pattern.
2. **Deterministic Pipeline** (`pipeline/`) — the original reference implementation.  A single self-contained module that runs Stages 2→4 (retrieve → compose → verify) in one shot.

---

## Architecture Map

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 0   InputGuard        ── pure-code validation (no LLM)   │
├─────────────────────────────────────────────────────────────────┤
│  Layer 1   IntentClassifier  ── secondary-LLM JSON intent       │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2   QueryProcessor    ── disambiguate → formalize → fan  │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3   RetrievalAgent    ── Jiggasha calls + ReAct judge    │
├─────────────────────────────────────────────────────────────────┤
│  Layer 4   ComposerAgent     ── primary-LLM streaming answer    │
├─────────────────────────────────────────────────────────────────┤
│  Layer 5   PostFlightVerifier ── strip, NLI-check, Sources      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Directory Guide

| Doc file | Covers |
|----------|--------|
| [`AGENTS.md`](AGENTS.md) | Orchestrator + 6 agents (Layers 0–5) |
| [`PIPELINE.md`](PIPELINE.md) | Deterministic pipeline (router, normalize, sanitize, query-expand) |
| [`VERIFIER.md`](VERIFIER.md) | NLI verifier, citation parser, policy engine |
| [`LLM.md`](LLM.md) | LLM client factory, reasoning loop, thinking parser |
| [`PROMPTS.md`](PROMPTS.md) | System prompts, composer prompt, time reminder, message strings |
| [`SESSION.md`](SESSION.md) | Query logs, session traces, Redis store |
| [`TOOLS.md`](TOOLS.md) | Jiggasha tool, tool registry |
| [`EVENTS.md`](EVENTS.md) | Event types and channel filters |
| [`CONFIG.md`](CONFIG.md) | Config loader, endpoint credentials |
