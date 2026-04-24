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

_query_log = QueryLog()

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
    title="GovOps Graphiti API",
    description="Government Services AI Agent backed by Neo4j Knowledge Graph.",
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

class FeedbackRequest(BaseModel):
    user_id: str
    turn_id: str
    rating: str  # "good", "bad", "unhelpful", "wrong"
    comment: Optional[str] = ""

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
        "service": "GovOps Graphiti",
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

    # Log the incoming query (text + timestamp only, nothing else).
    _query_log.append(request.query)

    agent = await get_agent_session(request.user_id)
    server_debug_secret = os.getenv("ADMIN_DEBUG_SECRET")
    debug_mode = (server_debug_secret is not None) and (x_debug_key == server_debug_secret)

    async def event_generator():
        try:
            async for event in agent.process_query(request.query, debug_mode=debug_mode, user_id=request.user_id):
                # Filter events based on debug mode
                if debug_mode:
                    # Include debug + both events (exclude user-only)
                    filtered = filter_for_debug([event])
                else:
                    # Include user + both events (exclude debug-only)
                    filtered = filter_for_user([event])

                for evt in filtered:
                    json_str = json.dumps(evt, ensure_ascii=True)
                    yield f"{json_str}\n"
                    await asyncio.sleep(0.001)

        except Exception as e:
            logger.error(f"Stream error for user {request.user_id}: {e}")
            err_msg = json.dumps({
                "type": "error",
                "content": "Server-side streaming error.",
                "channel": "user"
            })
            yield f"{err_msg}\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")

@app.post("/feedback", tags=["Feedback"])
async def submit_feedback(request: FeedbackRequest):
    """Store user feedback for a specific turn. Negative feedback surfaces to system context."""
    async with session_lock:
        agent = active_sessions.get(request.user_id)
        if agent:
            agent.add_feedback(
                user_id=request.user_id,
                turn_id=request.turn_id,
                rating=request.rating,
                comment=request.comment or "",
            )
            return {"status": "success", "message": "Feedback recorded."}
        else:
            return {"status": "ignored", "message": "No active session found."}

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
