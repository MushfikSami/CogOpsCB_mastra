"""Deterministic 5-stage pipeline for the GovOps chatbot.

Stages:
  0. sanitize    — pure-code input validation (length, encoding, injection regex)
  1. router      — single secondary-LLM call: intent + split + Bengali normalize
  2. retrieve    — parallel Jiggasha calls + dedupe + adaptive K
  3. compose     — primary LLM streams cited Bengali answer (no tools)
  4. post-flight — strip unknown [S#], NLI verify, build Sources block
                   (NEVER blocks the user-visible stream)
"""
