import asyncio
import os
import json
import logging
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional
from fastapi import Header
from dotenv import load_dotenv
from cogops.agents.orchestrator import Orchestrator

from cogops.events.channels import filter_for_user, filter_for_debug
from cogops.session.query_log import QueryLog
from cogops.session.session_logger import SessionLogger

_query_log = QueryLog()
_session_logger = SessionLogger()

load_dotenv()

# --- Configuration ---
API_PORT = 9000
API_HOST = "0.0.0.0"
AGENT_CONFIG_PATH = "configs/config.yml"

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("api")

# --- API Application ---
app = FastAPI(
    title="GovOps API",
    description="Government Services AI Agent.",
    version="2.0.0"
)

# --- CORS (Allow all for development) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Session Store ---
active_sessions: Dict[str, Orchestrator] = {}
session_lock = asyncio.Lock()

# --- Request Models ---
class ChatRequest(BaseModel):
    user_id: str
    query: str

class SessionRequest(BaseModel):
    user_id: str

# --- Helper Functions ---

async def get_agent_session(user_id: str) -> Orchestrator:
    """Thread-safe retrieval or creation of an agent session."""
    async with session_lock:
        if user_id not in active_sessions:
            logger.info(f"Creating new session for User: {user_id}")
            try:
                agent = Orchestrator(config_path=AGENT_CONFIG_PATH)
                active_sessions[user_id] = agent
            except Exception as e:
                logger.error(f"Failed to create agent session: {e}")
                raise HTTPException(status_code=500, detail="Agent initialization failed.")
        else:
            logger.info(f"Resuming session for User: {user_id}")
        return active_sessions[user_id]

# --- Endpoints ---

@app.get("/query-log", tags=["System"])
async def query_log_endpoint():
    """Return the stored queries with timestamps (last 10 days, nothing else)."""
    return _query_log.entries


@app.get("/health", tags=["System"])
async def health_check():
    """System status and active session count."""
    return {
        "status": "online",
        "service": "GovOps",
        "active_sessions": len(active_sessions)
    }

@app.post("/chat/stream", tags=["Chat"])
async def stream_chat(request: ChatRequest, x_debug_key: Optional[str] = Header(None, alias="X-Debug-Key")):
    """
    Main Chat Endpoint.
    Streams the agent's response as Newline Delimited JSON (NDJSON).
    If X-Debug-Key header matches ADMIN_DEBUG_SECRET, debug events are included.
    Otherwise, only user-visible events (answer_chunk) are streamed.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    agent = await get_agent_session(request.user_id)
    max_chars = agent.max_input_chars
    large_error = agent.large_input_error

    # Log the incoming query (text + timestamp only, nothing else).
    _query_log.append(request.query)
    server_debug_secret = os.getenv("ADMIN_DEBUG_SECRET")
    debug_mode = (server_debug_secret is not None) and (x_debug_key == server_debug_secret)

    session_id = _session_logger.start_session(request.user_id, request.query)

    async def event_generator():
        # Length guard inside the generator so this function stays a plain
        # async def (not an async generator), allowing `return StreamingResponse`.
        if len(request.query) > max_chars:
            error_evt = {"type": "error", "content": large_error, "channel": "user"}
            _session_logger.ingest_event(error_evt)
            complete_evt = {"type": "answer_complete", "channel": "both"}
            _session_logger.ingest_event(complete_evt)
            _session_logger.finalize_session(request.user_id)
            yield json.dumps(error_evt) + "\n"
            yield json.dumps(complete_evt) + "\n"
            return

        try:
            async for event in agent.process_query(request.query, user_id=request.user_id):
                # Ingest ALL events (regardless of channel) for the audit log
                _session_logger.ingest_event(event)
                # Filter events based on debug mode
                if debug_mode:
                    filtered = filter_for_debug([event])
                else:
                    filtered = filter_for_user([event])

                for evt in filtered:
                    json_str = json.dumps(evt, ensure_ascii=True)
                    yield f"{json_str}\n"

        except Exception as e:
            logger.error(f"Stream error for user {request.user_id}: {e}")
            error_evt = {
                "type": "error",
                "content": "Server-side streaming error.",
                "channel": "user"
            }
            _session_logger.ingest_event(error_evt)
            _session_logger.finalize_session(request.user_id)
            err_msg = json.dumps(error_evt)
            yield f"{err_msg}\n"
            return

        # Finalize the session trace after all events are consumed
        _session_logger.finalize_session(request.user_id)

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")

@app.get("/session/audit", tags=["System"])
async def audit_endpoint(limit: int = 20):
    """Return the latest session traces with tool calls, reasoning, and answers."""
    traces = _session_logger.get_traces(limit=limit)
    # Strip raw event arrays for the API response to keep it reasonable
    summary = []
    for t in traces:
        summary.append({
            "user_id": t.get("user_id"),
            "query": t.get("query"),
            "session_id": t.get("session_id"),
            "start_time": t.get("start_time"),
            "end_time": t.get("end_time"),
            "event_count": t.get("event_count"),
            "tool_call_count": t.get("tool_call_count"),
            "tool_results": t.get("tool_results", []),
            "reasoning_chunks": t.get("reasoning_chunks", []),
            "total_answer": t.get("total_answer", ""),
            "total_reasoning": t.get("total_reasoning", ""),
        })
    return summary


@app.get("/session/audit/raw", tags=["System"])
async def audit_raw_endpoint(limit: int = 20):
    """Return full session traces including raw event arrays."""
    traces = _session_logger.get_traces(limit=limit)
    return traces


@app.post("/session/clear", tags=["Session"])
async def clear_session(request: SessionRequest):
    """Wipes the conversation history for a specific user."""
    async with session_lock:
        if request.user_id in active_sessions:
            active_sessions[request.user_id].clear_session()
            del active_sessions[request.user_id]
            logger.info(f"Cleared session for User: {request.user_id}")
            return {"status": "success", "message": "Session cleared."}
        else:
            return {"status": "ignored", "message": "No active session found."}

if __name__ == "__main__":
    print(f"Starting GovOps API on http://{API_HOST}:{API_PORT}")
    uvicorn.run(app, host=API_HOST, port=API_PORT)
