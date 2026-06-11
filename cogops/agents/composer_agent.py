"""
cogops/agents/composer_agent.py

Layer 4 — ComposerAgent + Layer 5 — PostFlightVerifier.

Composer:
  - Streaming primary-LLM call
  - Emits answer_chunk / reasoning_chunk events
  - System prompt: composer persona + time reminder + citation rules

PostFlightVerifier:
  - Strip composer-emitted sources blocks
  - Strip unknown citation tags
  - NLI batched verification
  - Policy application (redact / refuse / warn)
  - Append canonical Sources block
"""

from __future__ import annotations

import logging
import re
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from openai import AsyncOpenAI

from cogops.prompts.composer import get_composer_prompt
from cogops.prompts.time_reminder import build_time_reminder
from cogops.utils.thinking_parser import ThinkingParser
from cogops.verifier.citations import (
    build_sources_block,
    extract_citation_tags,
    extract_citations,
    strip_unknown_tags,
)
from cogops.verifier.nli import verify_claims
from cogops.verifier.policy import apply_policy

logger = logging.getLogger(__name__)


_CITE_TAG_RE = re.compile(r"\[S\d+\]")

# Composer-emitted Sources block patterns
_SOURCES_PATTERNS = [
    r"\n+---\s*\n+\*\*\s*(?:সূত্র|উৎস|Sources)\b",
    r"(?:^|\n+)---\s*\n+\*\*\s*(?:সূত্র|উৎস|Sources)\b",
    r"\n+\*\*\s*(?:সূত্র|উৎস|Sources)\s*\(?(?:Sources)?\)?\s*\*\*",
    r"(?:^|\n+)\*\*\s*(?:সূত্র|উৎস|Sources)\s*\(?(?:Sources)?\)?\s*\*\*",
    r"\n+(?:সূত্র|উৎস|Sources)\s*[:：]",
    r"(?:^|\n+)(?:সূত্র|উৎস|Sources)\s*[:：]",
    r"\n+(?:\s*[-*]\s*\[S\d+\][^\n]*\n*)+$",
]

# Mode-mix detection
_PARTIAL_GAP_RE = re.compile(
    r"("
    r"(?:নির্দিষ্ট\s+[^।\n]{0,40}\s+)?উল্লেখ\s+নেই"
    r"|উল্লিখিত\s+নয়"
    r"|(?:সঠিক|নির্দিষ্ট)\s+তথ্য\s+পাওয়া\s+যায়নি"
    r"|(?:তথ্য\s+)?প্রসঙ্গে\s+(?:নেই|উল্লেখ\s+নেই)"
    r"|নির্দিষ্ট\s+তথ্য\s+নেই"
    r")",
    flags=re.IGNORECASE,
)

_MODE_MIX_PARAGRAPH_RE = re.compile(
    r"(?:তবে|তথাপি|তবু|যদিও)[^।\n]*"
    r"(?:সাধারণ(?:ভাবে|ত)?|সাধারণ\s+পদ্ধতি|সাধারণ\s+আইনানুগ|"
    r"নিচে\s+দেওয়া\s+হলো|নিচে\s+দেয়া\s+হলো)",
    flags=re.IGNORECASE,
)

_B_HEADER_RE = re.compile(
    r"\n\s*এই\s+নির্দিষ্ট\s+বিষয়ে\s+সঠিক\s+তথ্য\s+পাওয়া\s+যায়নি"
)


def _strip_composer_sources_block(text: str) -> str:
    """Defensively cut any composer-emitted Sources/সূত্র block."""
    if not text:
        return text
    for pat in _SOURCES_PATTERNS:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if not m:
            continue
        body = text[: m.start()].rstrip()
        trailing = text[m.start():]
        if not _CITE_TAG_RE.search(body):
            trailing_tags = list(dict.fromkeys(_CITE_TAG_RE.findall(trailing)))
            if trailing_tags:
                tag_suffix = " " + " ".join(trailing_tags)
                lines = body.rstrip().split("\n")
                for i in range(len(lines) - 1, -1, -1):
                    if lines[i].strip():
                        lines[i] = lines[i].rstrip() + tag_suffix
                        break
                body = "\n".join(lines)
        text = body
    return text


def _strip_mode_mix_paragraph(text: str) -> Tuple[str, bool]:
    """Strip 'but here's the general procedure' after a gap admission.

    ONLY strips if the tail has no citations — if the tail contains [S#]
    tags, it is based on corpus passages and should be kept.
    """
    if not text:
        return text, False
    gap_match = _PARTIAL_GAP_RE.search(text)
    if not gap_match:
        return text, False
    tail = text[gap_match.end():]
    mix_match = _MODE_MIX_PARAGRAPH_RE.search(tail)
    if not mix_match:
        return text, False
    bridge_start = gap_match.end() + mix_match.start()
    rest_after_bridge = text[bridge_start:]
    # If the tail has citations, it's corpus-based related info — keep it.
    if _CITE_TAG_RE.search(rest_after_bridge):
        return text, False
    b_match = _B_HEADER_RE.search(rest_after_bridge)
    if b_match:
        end_rel = b_match.start()
        cleaned = (
            text[:bridge_start].rstrip()
            + "\n\n"
            + rest_after_bridge[end_rel:].lstrip()
        ).strip()
    else:
        cleaned = text[:bridge_start].rstrip()
    return cleaned, True


def _sanitize_history(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for msg in history or []:
        role = msg.get("role", "")
        content = msg.get("content", "") or ""
        if role == "assistant":
            content = re.sub(r"<thinking>.*?</thinking>\s*", "", content, flags=re.DOTALL)
        out.append({"role": role, "content": content})
    return out


def _build_composer_user_message(
    raw_user_query: str,
    source_map: Dict[str, Dict[str, Any]],
) -> str:
    """Render the composer's user message: <context> + <user_query>."""
    ctx_parts: List[str] = ["<context>"]
    if not source_map:
        ctx_parts.append("(no passages retrieved)")
    else:
        for tag, meta in source_map.items():
            hb: List[str] = []
            for key, label in (
                ("category", "বিভাগ"),
                ("sub_category", "উপ-বিভাগ"),
                ("service", "সেবা"),
                ("topic", "বিষয়"),
            ):
                v = meta.get(key, "") or ""
                if v:
                    hb.append(f"{label}: {v}")
            chunk_type = meta.get("chunk_type", "")
            if chunk_type == "wiki":
                hb.append("উৎস: উইকিপিডিয়া")
            elif chunk_type == "govt_service":
                hb.append("উৎস: সরকারি সেবা")
            header = " | ".join(hb) if hb else "—"
            ctx_parts.append("")
            ctx_parts.append(f"[{tag}] ({header})")
            ctx_parts.append(meta.get("text", ""))
    ctx_parts.append("</context>")

    sections = ["\n".join(ctx_parts)]
    sections.append("<user_query>\n" + raw_user_query + "\n</user_query>")
    return "\n\n".join(sections)


def _evt(type_: str, channel: str = "debug", **payload: Any) -> Dict[str, Any]:
    return {"type": type_, "channel": channel, **payload}


# ------------------------------------------------------------------
# ComposerAgent
# ------------------------------------------------------------------

class ComposerAgent:
    """Layer 4 — streaming composer (primary LLM)."""

    def __init__(
        self,
        primary_client: AsyncOpenAI,
        primary_model: str,
        agent_name: str = "আশা",
        temperature: float = 0.1,
        top_p: float = 0.95,
        max_tokens: int = 2048,
    ):
        self.client = primary_client
        self.model = primary_model
        self.agent_name = agent_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

    async def compose(
        self,
        raw_query: str,
        source_map: Dict[str, Dict[str, Any]],
        history: List[Dict[str, str]],
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream the composer's answer.

        Yields answer_chunk, reasoning_chunk, composer_done events.
        """
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": get_composer_prompt(agent_name=self.agent_name)}
        ]
        messages.extend(_sanitize_history(history))
        messages.append({"role": "assistant", "content": build_time_reminder()})
        messages.append({
            "role": "user",
            "content": _build_composer_user_message(raw_query, source_map),
        })

        yield _evt(
            "composer_start",
            model=self.model,
            prompt_chars=sum(len(m.get("content", "")) for m in messages),
            n_messages=len(messages),
        )

        answer_acc: List[str] = []
        composer_usage: Optional[Dict[str, int]] = None
        parser = ThinkingParser()

        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                stream_options={"include_usage": True},
            )
            async for chunk in stream:
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage is not None:
                    try:
                        composer_usage = {
                            "prompt": int(getattr(chunk_usage, "prompt_tokens", 0) or 0),
                            "completion": int(getattr(chunk_usage, "completion_tokens", 0) or 0),
                        }
                    except Exception:
                        pass
                try:
                    delta = chunk.choices[0].delta
                    text = (getattr(delta, "content", None) or "")
                except (IndexError, AttributeError):
                    continue
                if not text:
                    continue
                for channel, piece in parser.feed(text):
                    if channel == "answer":
                        answer_acc.append(piece)
                        yield _evt("answer_chunk", channel="both", content=piece)
                    else:
                        yield _evt("reasoning_chunk", content=piece)
            for channel, piece in parser.flush():
                if channel == "answer":
                    answer_acc.append(piece)
                    yield _evt("answer_chunk", channel="both", content=piece)
                else:
                    yield _evt("reasoning_chunk", content=piece)
        except Exception as e:
            logger.error("composer stream failed: %s", e, exc_info=True)
            raise

        raw_answer = "".join(answer_acc).strip()
        yield _evt(
            "composer_done",
            chars=len(raw_answer),
            token_usage=composer_usage,
            raw_answer=raw_answer,
        )

        if not raw_answer:
            raise RuntimeError("composer_empty")


# ------------------------------------------------------------------
# PostFlightVerifier
# ------------------------------------------------------------------

class PostFlightVerifier:
    """Layer 5 — strip, verify, apply policy, append Sources block."""

    def __init__(
        self,
        secondary_client: Optional[AsyncOpenAI],
        secondary_model: str,
        enabled: bool = True,
        timeout: float = 6.0,
        policy: str = "redact",
        refusal_text_bn: str = "দুঃখিত, এই প্রশ্নের জন্য নির্ভরযোগ্য সরকারি তথ্য পাওয়া যায়নি।",
    ):
        self.client = secondary_client
        self.model = secondary_model
        self.enabled = enabled
        self.timeout = timeout
        self.policy = policy
        self.refusal_text_bn = refusal_text_bn

    async def verify(
        self,
        raw_answer: str,
        source_map: Dict[str, Dict[str, Any]],
        skip_verify: bool = False,
    ) -> Tuple[str, List[Dict[str, Any]], str]:
        """Post-flight verification.

        Returns (final_answer, events, sources_block).
        Never raises; degrades gracefully.
        """
        events: List[Dict[str, Any]] = []

        # 1. Strip composer sources block
        cleaned = _strip_composer_sources_block(raw_answer)

        # 2. Strip mode-mix paragraphs
        cleaned, mode_mix_stripped = _strip_mode_mix_paragraph(cleaned)
        if mode_mix_stripped:
            events.append(_evt(
                "mode_mix_stripped",
                note="removed 'general procedure' paragraph after partial-gap caveat",
            ))

        # 3. Strip unknown citation tags
        cleaned, dropped = strip_unknown_tags(cleaned, source_map)
        if dropped:
            for tag in dropped:
                events.append(_evt(
                    "unsupported_claim",
                    tag=tag, verdict="tag_not_in_source_map", action="stripped",
                ))

        # 4. Check for citations
        used_tags = extract_citation_tags(cleaned)
        if not used_tags:
            # If source_map is empty, the composer had no passages to cite.
            # Allow its fallback answer (e.g. "go to X office") through.
            if source_map:
                return self.refusal_text_bn, events, ""

        # Final safety net: strip any remaining sources-like block
        cleaned = _strip_composer_sources_block(cleaned).rstrip()

        # 5. NLI verification
        if not skip_verify and self.enabled and self.client is not None:
            pairs = extract_citations(cleaned)
            pairs = [(t, s) for (t, s) in pairs if t in source_map]
            if pairs:
                events.append(_evt("verification_start", count=len(pairs)))
                try:
                    verdicts, nli_usage = await verify_claims(
                        pairs=pairs,
                        source_map=source_map,
                        secondary_client=self.client,
                        secondary_model=self.model,
                        timeout=self.timeout,
                    )
                    cleaned, policy_events = apply_policy(
                        answer=cleaned,
                        pairs=pairs,
                        verdicts=list(verdicts),
                        policy=self.policy,
                        refusal_text=self.refusal_text_bn,
                    )
                    events.append(_evt(
                        "verification_result",
                        pairs=len(pairs),
                        verdicts=list(verdicts),
                        token_usage=nli_usage,
                    ))
                    for pe in policy_events:
                        if "channel" not in pe:
                            pe["channel"] = "debug"
                        events.append(pe)
                except Exception as e:
                    logger.warning("Verifier pipeline failed (%s); keeping unverified.", e)
                    events.append(_evt(
                        "verification_result",
                        action="degraded_failed", error=str(e),
                    ))

        if cleaned.strip() == self.refusal_text_bn.strip():
            return self.refusal_text_bn, events, ""

        # 6. Build canonical Sources block
        used_tags_after = extract_citation_tags(cleaned)
        sources_block = build_sources_block(source_map, used_tags_after)
        final = cleaned.rstrip() + (("\n\n" + sources_block) if sources_block else "")
        return final, events, sources_block
