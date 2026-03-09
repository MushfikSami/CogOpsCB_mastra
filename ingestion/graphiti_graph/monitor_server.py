"""
Real-time monitoring dashboard for data ingestion.
Runs on port 3456 and provides live progress updates.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.websockets import WebSocket

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IngestionMonitor:
    """Global state manager for ingestion monitoring."""

    def __init__(self):
        self.is_ingesting = False
        self.total_rows = 0
        self.processed_rows = 0
        self.current_row = 0
        self.current_episode_name = ""
        self.batch_number = 0
        self.errors: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.api_calls: list[dict[str, Any]] = []
        self.websocket_clients: list[WebSocket] = []

    def start_ingestion(self, total_rows: int):
        self.is_ingesting = True
        self.total_rows = total_rows
        self.processed_rows = 0
        self.current_row = 0
        self.errors = []
        self.events = []
        self._log_event("ingestion_started", f"Starting ingestion of {total_rows} rows")

    def update_progress(self, row_index: int, episode_name: str, success: bool, error: str | None = None):
        self.current_row = row_index
        self.processed_rows = row_index + 1
        self.current_episode_name = episode_name
        if success:
            self._log_event("row_processed", f"Row {row_index} ({episode_name}) - Success")
        else:
            self.errors.append({
                "row": row_index,
                "episode": episode_name,
                "error": error or "Unknown error",
                "timestamp": datetime.now().isoformat()
            })
            self._log_event("row_error", f"Row {row_index} ({episode_name}) - Failed: {error}")

    def batch_completed(self, batch_num: int):
        self.batch_number = batch_num
        self._log_event("batch_completed", f"Batch {batch_num} completed")

    def log_api_call(self, api_type: str, details: dict):
        self.api_calls.append({
            "type": api_type,
            "details": details,
            "timestamp": datetime.now().isoformat()
        })
        self._log_event("api_call", f"{api_type}: {json.dumps(details)}")

    def stop_ingestion(self, success: bool, message: str):
        self.is_ingesting = False
        self._log_event("ingestion_ended", message)
        self.broadcast_status()

    def _log_event(self, event_type: str, message: str):
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "message": message
        }
        self.events.insert(0, event)
        if len(self.events) > 100:
            self.events = self.events[:100]
        self.broadcast_event(event)

    def broadcast_event(self, event: dict):
        for client in self.websocket_clients[:]:
            try:
                asyncio.create_task(client.send_json(event))
            except Exception:
                self.websocket_clients.remove(client)

    def broadcast_status(self):
        status = self.get_status()
        for client in self.websocket_clients[:]:
            try:
                asyncio.create_task(client.send_json({"type": "status", "data": status}))
            except Exception:
                self.websocket_clients.remove(client)

    def get_status(self) -> dict:
        return {
            "is_ingesting": self.is_ingesting,
            "total_rows": self.total_rows,
            "processed_rows": self.processed_rows,
            "progress_percent": round((self.processed_rows / self.total_rows * 100) if self.total_rows > 0 else 0, 2),
            "current_row": self.current_row,
            "current_episode": self.current_episode_name,
            "batch_number": self.batch_number,
            "error_count": len(self.errors),
            "recent_events": self.events[:10],
            "recent_api_calls": self.api_calls[-5:]
        }


# Global monitor instance
monitor = IngestionMonitor()

# FastAPI app
app = FastAPI(title="Ingestion Monitor", version="1.0.0")

# HTML dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Ingestion Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #1a1a2e; color: #eee; padding: 20px; }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 { color: #00d4ff; margin-bottom: 20px; }
        .dashboard { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 20px; }
        .card { background: #16213e; border-radius: 10px; padding: 20px; border: 1px solid #0f3460; }
        .card h2 { color: #00d4ff; font-size: 14px; margin-bottom: 10px; text-transform: uppercase; }
        .card .value { font-size: 36px; font-weight: bold; color: #fff; }
        .card .unit { font-size: 14px; color: #888; }
        .progress-bar { background: #0f3460; height: 20px; border-radius: 10px; overflow: hidden; margin-top: 10px; }
        .progress-fill { height: 100%; background: linear-gradient(90deg, #00d4ff, #00ff88); transition: width 0.3s; }
        .section { background: #16213e; border-radius: 10px; padding: 20px; margin-bottom: 20px; border: 1px solid #0f3460; }
        .section h2 { color: #00d4ff; margin-bottom: 15px; font-size: 18px; }
        .event-list, .api-list { list-style: none; max-height: 300px; overflow-y: auto; }
        .event-item, .api-item { background: #0f3460; padding: 10px; margin: 5px 0; border-radius: 5px; font-size: 13px; }
        .event-item.error { border-left: 3px solid #ff4757; }
        .event-item.success { border-left: 3px solid #00ff88; }
        .event-time { color: #888; font-size: 11px; }
        .live-indicator { display: inline-block; width: 10px; height: 10px; background: #00ff88; border-radius: 50%; margin-right: 8px; animation: pulse 1s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        .disconnected { animation: blink 1s infinite; background: #ff4757 !important; }
        @keyframes blink { 50% { opacity: 0.5; } }
        .stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
        .stat { text-align: center; }
        .stat .value { font-size: 24px; }
        .refresh-btn { background: #00d4ff; color: #000; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; font-weight: bold; margin-top: 10px; }
        .refresh-btn:hover { background: #00ff88; }
    </style>
</head>
<body>
    <div class="container">
        <h1><span class="live-indicator" id="liveIndicator"></span>Ingestion Monitor Dashboard</h1>

        <div class="dashboard">
            <div class="card">
                <h2>Progress</h2>
                <div class="value" id="progressPercent">0<span class="unit">%</span></div>
                <div class="progress-bar">
                    <div class="progress-fill" id="progressFill" style="width: 0%"></div>
                </div>
            </div>
            <div class="card">
                <h2>Processed</h2>
                <div class="value" id="processedRows">0</div>
                <div class="unit">of <span id="totalRows">0</span> rows</div>
            </div>
            <div class="card">
                <h2>Current Row</h2>
                <div class="value" id="currentRow">-</div>
                <div class="unit" id="currentEpisode">No data</div>
            </div>
        </div>

        <div class="stats-grid" style="margin-bottom: 20px;">
            <div class="card stat">
                <h2>Batch</h2>
                <div class="value" id="batchNum">-</div>
            </div>
            <div class="card stat">
                <h2>Errors</h2>
                <div class="value" id="errorCount" style="color: #ff4757;">0</div>
            </div>
            <div class="card stat">
                <h2>Events</h2>
                <div class="value" id="eventCount">0</div>
            </div>
            <div class="card stat">
                <h2>Status</h2>
                <div class="value" id="statusText" style="font-size: 18px;">Idle</div>
            </div>
        </div>

        <div class="section">
            <h2>Event Log</h2>
            <ul class="event-list" id="eventList"></ul>
        </div>

        <div class="section">
            <h2>Recent API Calls</h2>
            <ul class="api-list" id="apiList"></ul>
        </div>

        <button class="refresh-btn" onclick="location.reload()">Refresh Page</button>
    </div>

    <script>
        const ws = new WebSocket(`ws://${window.location.host}/ws`);
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            if (data.type === 'status') {
                updateDashboard(data.data);
            } else if (data.type === 'event') {
                addEvent(data.data);
            }
        };
        ws.onclose = function() {
            document.getElementById('liveIndicator').classList.add('disconnected');
            document.getElementById('statusText').textContent = 'Disconnected';
        };
        ws.onopen = function() {
            document.getElementById('liveIndicator').classList.remove('disconnected');
            document.getElementById('statusText').textContent = 'Connected';
        };

        function updateDashboard(data) {
            document.getElementById('progressPercent').innerHTML = (data.progress_percent || 0) + '<span class="unit">%</span>';
            document.getElementById('progressFill').style.width = (data.progress_percent || 0) + '%';
            document.getElementById('processedRows').textContent = data.processed_rows || 0;
            document.getElementById('totalRows').textContent = data.total_rows || 0;
            document.getElementById('currentRow').textContent = data.current_row >= 0 ? data.current_row : '-';
            document.getElementById('currentEpisode').textContent = data.current_episode || 'No data';
            document.getElementById('batchNum').textContent = data.batch_number || '-';
            document.getElementById('errorCount').textContent = data.error_count || 0;
            document.getElementById('eventCount').textContent = data.recent_events?.length || 0;
            document.getElementById('statusText').textContent = data.is_ingesting ? 'Ingesting...' : 'Idle';
        }

        function addEvent(event) {
            const list = document.getElementById('eventList');
            const item = document.createElement('li');
            item.className = 'event-item ' + (event.message.includes('Error') || event.message.includes('Failed') ? 'error' : 'success');
            const time = new Date(event.timestamp).toLocaleTimeString();
            item.innerHTML = '<div class="event-time">' + time + '</div><div>' + event.message + '</div>';
            list.insertBefore(item, list.firstChild);
            if (list.children.length > 50) list.removeChild(list.lastChild);
        }
    </script>
</body>
</html>
"""


@app.get("/")
async def dashboard():
    return HTMLResponse(content=DASHBOARD_HTML)


@app.get("/api/status")
async def get_status():
    return JSONResponse(content=monitor.get_status())


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    monitor.websocket_clients.append(websocket)
    monitor.broadcast_status()
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        monitor.websocket_clients.remove(websocket)


@app.get("/api/logs")
async def get_logs():
    return JSONResponse(content={
        "events": monitor.events,
        "api_calls": monitor.api_calls,
        "errors": monitor.errors
    })


# Logging hook functions for ingestion scripts
def setup_ingestion_logging():
    """Setup custom handlers for ingestion logging."""

    class IngestionHandler(logging.Handler):
        def emit(self, record):
            msg = self.format(record)
            try:
                data = json.loads(msg)
                if "api_type" in data:
                    monitor.log_api_call(data["api_type"], data.get("details", {}))
            except json.JSONDecodeError:
                pass

    handler = IngestionHandler()
    handler.setFormatter(logging.Formatter('{"message": "%(message)s"}'))
    logging.getLogger().addHandler(handler)
    return handler


__all__ = ["app", "monitor", "setup_ingestion_logging"]
