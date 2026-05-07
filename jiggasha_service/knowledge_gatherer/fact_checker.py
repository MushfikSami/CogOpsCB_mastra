"""Fact-checking system.

Two-phase pipeline running AFTER answer synthesis:
  Phase 1: Extract verifiable claims from the answer (LLM + sentence-split fallback)
  Phase 2: Verify each claim against retrieved evidence (LLM, 3-way classification)

Overall status mapping:
  no_data    -> no relevant passages after reranking
  verified   -> all claims confirmed
  partial    -> mix of confirmed + unverified (no hallucinated)
  unverified -> any hallucinated claims or all unverified
"""

from __future__ import annotations

import json
import logging
import re

from openai import AsyncOpenAI

from .config import Config, get_config

logger = logging.getLogger(__name__)

CLAIM_EXTRACTION_SYSTEM = "You extract factual, verifiable claims from the following answer. Return a JSON array of strings. Exclude opinions, greetings, and meta-statements."

VERIFICATION_SYSTEM = (
    "You verify individual claims against retrieved evidence.\n"
    "Reply with a JSON object:\n"
    '{"status": "confirmed" | "unverified" | "hallucinated", '
    '"evidence_excerpt": "the exact text supporting the claim, or null", '
    '"reasoning": "brief explanation"}\n\n'
    "Rules:\n"
    '- "confirmed" = evidence text directly supports the claim (exact excerpt cited)\n'
    '- "unverified" = claim might be true but no direct evidence in passages\n'
    '- "hallucinated" = claim contradicts evidence or introduces unsupported details\n'
)

VERIFICATION_USER = (
    "Claim: {claim}\n\n"
    "Evidence:\n{evidence}\n\n"
    "Is this claim supported by the evidence?"
)


class FactChecker:
    def __init__(self, config: Config | None = None) -> None:
        self._config = config or get_config()
        self._llm = AsyncOpenAI(
            api_key=self._config.openai_api_key,
            base_url=self._config.openai_base_url,
        )
        self._model = self._config.llm_model

    async def check(self, answer: str, relevant_passages: list[tuple]) -> dict:
        if not relevant_passages:
            return {
                "overall_status": "no_data",
                "details": [],
                "status_message": "No confirmed information exists in database",
            }

        claims = await self._extract_claims(answer)
        if not claims:
            return {
                "overall_status": "verified",
                "details": [],
                "status_message": None,
            }

        # Build combined evidence
        evidence_parts = []
        for passage, score, _ in relevant_passages:
            text = (passage.get("text", "") or "")[:800]
            node = passage.get("web_node", "") or ""
            evidence_parts.append(f"[{node}]\n{text}")
        combined_evidence = "\n---\n".join(evidence_parts)

        # Verify each claim
        details: list[dict] = []
        confirmed = unverified = hallucinated = 0

        for claim in claims:
            result = await self._verify_claim(claim, combined_evidence)
            details.append(result)
            if result["status"] == "confirmed":
                confirmed += 1
            elif result["status"] == "hallucinated":
                hallucinated += 1
            else:
                unverified += 1

        total = len(details)
        if total == 0:
            overall = "verified"
        elif hallucinated > 0:
            overall = "unverified"
        elif confirmed == total:
            overall = "verified"
        elif confirmed > 0:
            overall = "partial"
        else:
            overall = "unverified"

        return {
            "overall_status": overall,
            "details": details,
            "status_message": None,
        }

    async def _extract_claims(self, answer: str) -> list[str]:
        try:
            resp = await self._llm.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": CLAIM_EXTRACTION_SYSTEM},
                    {"role": "user", "content": f"Answer:\n{answer}"},
                ],
                max_tokens=1000,
                temperature=0.0,
            )
            content = resp.choices[0].message.content or "[]"
            content = content.strip()
            if content.startswith("```"):
                content = "\n".join(
                    l for l in content.split("\n") if not l.startswith("```")
                ).strip()
            return json.loads(content)
        except (json.JSONDecodeError, Exception) as e:
            logger.error("Claim extraction failed: %s", e)
            return self._fallback_extract(answer)

    @staticmethod
    def _fallback_extract(answer: str) -> list[str]:
        sentences = re.split(r"(?<=[.।!])\s+", answer)
        return [s.strip() for s in sentences if len(s.strip()) > 20]

    async def _verify_claim(self, claim: str, evidence: str) -> dict:
        try:
            resp = await self._llm.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": VERIFICATION_SYSTEM},
                    {"role": "user", "content": VERIFICATION_USER.format(
                        claim=claim, evidence=evidence[:3000],
                    )},
                ],
                max_tokens=500,
                temperature=0.0,
            )
            content = resp.choices[0].message.content or "{}"
            content = content.strip()
            if content.startswith("```"):
                content = "\n".join(
                    l for l in content.split("\n") if not l.startswith("```")
                ).strip()
            result = json.loads(content)
            return {
                "claim": claim,
                "status": result.get("status", "unverified"),
                "evidence": result.get("evidence_excerpt"),
                "reasoning": result.get("reasoning", ""),
            }
        except (json.JSONDecodeError, Exception) as e:
            logger.error("Verification failed for '%s': %s", claim[:50], e)
            return {
                "claim": claim,
                "status": "unverified",
                "evidence": None,
                "reasoning": f"Verification error: {e}",
            }
