"""Pydantic models for API request/response."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RetrievalMode(str, Enum):
    web = "web"
    local = "local"


class FactCheckStatus(str, Enum):
    verified = "verified"
    partial = "partial"
    unverified = "unverified"
    no_data = "no_data"


class Source(BaseModel):
    id: str = ""
    text: str = ""
    collection: str = "web"
    relevance_score: float = 0.0


class FactCheckDetail(BaseModel):
    claim: str = ""
    status: str = "unverified"
    evidence: str | None = None


class KnowledgeGatherRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    mode: RetrievalMode = RetrievalMode.web
    top_k: int | None = None
    max_tao_steps: int | None = None


class KnowledgeGatherResponse(BaseModel):
    answer: str
    sources: list[Source] = []
    confidence: float = 0.0
    reasoning: str = ""
    fact_check: FactCheckDetail = Field(default_factory=FactCheckDetail)
    fact_check_status: FactCheckStatus = FactCheckStatus.unverified
    tao_steps: list[dict] = []
    mode: RetrievalMode = RetrievalMode.web
    status_message: str | None = None
