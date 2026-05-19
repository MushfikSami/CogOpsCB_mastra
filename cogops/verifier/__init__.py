"""
cogops/verifier/

Post-generation grounding pipeline:

  - citations.py — regex-based [S#] extraction, hallucinated-tag stripping,
                   Sources block builder.
  - intent.py    — (step 4) secondary-LLM intent classifier.
  - nli.py       — (step 5) batched NLI verifier.
  - policy.py    — (step 5) redact / refuse / warn decision policy.
"""
