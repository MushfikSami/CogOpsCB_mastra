"""FastAPI server for Knowledge Gatherer Agent."""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import get_config
from .agent import TAOAgent
from .embedder import EmbedderClient
from .fact_checker import FactChecker
from .models import (
    FactCheckDetail,
    FactCheckStatus,
    KnowledgeGatherRequest,
    KnowledgeGatherResponse,
    RetrievalMode,
    Source,
)
from .retriever import Retriever
from .reranker import RerankerClient


def _build_app() -> FastAPI:
    config = get_config()
    embedder = EmbedderClient(config)
    retriever = Retriever(config, embedder)
    reranker = RerankerClient(config)
    agent = TAOAgent(config, embedder, retriever, reranker)
    fact_checker = FactChecker(config)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield

    app = FastAPI(
        title="Knowledge Gatherer Agent",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.post("/query", response_model=KnowledgeGatherResponse)
    async def query_endpoint(req: KnowledgeGatherRequest) -> KnowledgeGatherResponse:
        mode_str = req.mode.value
        top_k = req.top_k or config.top_k
        max_steps = req.max_tao_steps or config.max_tao_steps

        try:
            result = await agent.run(
                query=req.query,
                mode=mode_str,
                top_k=top_k,
                max_steps=max_steps,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

        # Fact-check
        relevant = result.get("relevant_passages", [])
        fact_check_result = await fact_checker.check(result["answer"], relevant)

        # Build sources
        sources = [
            Source(
                id=str(s.get("id", "")),
                text=s.get("text", ""),
                collection=s.get("source_collection", mode_str),
                relevance_score=s.get("relevance_score", 0.0),
            )
            for s in result.get("sources", [])
        ]

        # Fact check status
        fc_status = FactCheckStatus(
            fact_check_result.get("overall_status", "unverified")
        )

        fc_details = [
            FactCheckDetail(
                claim=d["claim"],
                status=d["status"],
                evidence=d.get("evidence"),
            )
            for d in fact_check_result.get("details", [])
        ]

        # Confidence from top reranker score
        rerank_results = result.get("rerank_results", [])
        confidence = round(float(rerank_results[0][1]), 4) if rerank_results else 0.0

        return KnowledgeGatherResponse(
            answer=result["answer"],
            sources=sources,
            confidence=confidence,
            reasoning=result["reasoning"],
            fact_check=(fc_details[0] if fc_details else FactCheckDetail()),
            fact_check_status=fc_status,
            tao_steps=result.get("tao_steps", []),
            mode=req.mode,
            status_message=fact_check_result.get("status_message"),
        )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        status: dict[str, str] = {}

        # LLM
        try:
            await agent._llm.chat.completions.create(
                model=agent._llm_model,
                messages=[{"role": "user", "content": "ok"}],
                max_tokens=1,
            )
            status["llm"] = "ok"
        except Exception:
            status["llm"] = "error"

        # Embedder
        try:
            await embedder.embed_query("ping")
            status["embedder"] = "ok"
        except Exception:
            status["embedder"] = "error"

        # Qdrant
        try:
            retriever._client.get_collection(config.qdrant_collection_web)
            status["qdrant"] = "ok"
        except Exception:
            status["qdrant"] = "error"

        return status

    @app.get("/tools")
    async def list_tools() -> dict[str, Any]:
        return {
            "tools": [
                "retrieve_from_qdrant (single search tool for TAO-ReAct loop)",
            ],
            "mode": "web" if config.qdrant_collection_web else "unknown",
        }

    return app


app = _build_app()
