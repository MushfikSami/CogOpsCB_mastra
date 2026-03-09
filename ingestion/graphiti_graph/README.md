# Graphiti Data Ingestion with Real-Time Monitoring

## Overview

This setup provides real-time monitoring of data ingestion processes with a web-based dashboard running on port 3456.

## Quick Start

### 1. Start Ingestion

```bash
# Using the helper script
./start_ingestion.sh /path/to/your/csv.csv

# Or run manually
# Terminal 1:
uvicorn monitor_server:app --host 0.0.0.0 --port 3456

# Terminal 2:
python ingest.py /path/to/your/csv.csv
```

### 2. Open Dashboard

Visit **http://localhost:3456** in your web browser to see real-time progress.

### 3. Stop Ingestion

```bash
./start_ingestion.sh stop
```

## File Structure

```
graphiti_graph/
├── monitor_server.py       # FastAPI monitoring server (port 3456)
├── ingest.py               # Main ingestion script
├── re_ingest.py            # Re-ingest failed rows
├── start_ingestion.sh      # Helper script to start processes
├── stop_ingestion.sh       # Helper script to stop processes
├── .env                    # Environment variables
├── .env.example            # Template for environment variables
└── .ingestion_pids/        # PID and log files (auto-created)
    ├── monitor.pid         # Monitor server PID
    ├── ingestion.pid       # Ingestion process PID
    ├── monitor.log         # Monitor server logs
    └── ingestion.log       # Ingestion logs
```

## Monitoring Dashboard Features

- **Live Progress Bar**: Visual progress indicator with percentage
- **Row Processing Status**: Current row being processed, episode name, batch number
- **Event Log**: Real-time stream of successes and errors
- **API Call Tracker**: Track LLM, embedding, and reranker calls
- **WebSocket Streaming**: Updates without page refresh
- **Status Indicator**: Pulsing green dot for active connection

## Usage Instructions

### Starting Ingestion

#### Option 1: Using the Helper Script (Recommended)

```bash
cd /home/vpa/CogOpsCB/ingestion/graphiti_graph
chmod +x start_ingestion.sh
./start_ingestion.sh /path/to/your/csv.csv
```

#### Option 2: Manual Start

```bash
# Terminal 1: Start Monitor Server
uvicorn monitor_server:app --host 0.0.0.0 --port 3456

# Terminal 2: Start Ingestion
python ingest.py /path/to/your/csv.csv

# Terminal 3: Open browser to http://localhost:3456
```

### Stopping Ingestion

#### Option 1: Using the Helper Script

```bash
./start_ingestion.sh stop
```

#### Option 2: Manual Stop

```bash
# Find PIDs
cat .ingestion_pids/monitor.pid
cat .ingestion_pids/ingestion.pid

# Kill processes
kill $(cat .ingestion_pids/monitor.pid)
kill $(cat .ingestion_pids/ingestion.pid)
```

### Monitoring Logs

```bash
# View all logs
tail -f .ingestion_pids/*.log

# View only ingestion logs
tail -f .ingestion_pids/ingestion.log

# View only monitor logs
tail -f .ingestion_pids/monitor.log
```

## Re-Ingesting Failed Rows

If some rows failed during the initial ingestion:

```bash
# Check the log for failed rows
grep "Failed" .ingestion_pids/ingestion.log

# Start monitor server
uvicorn monitor_server:app --host 0.0.0.0 --port 3456

# Re-ingest failed rows
python re_ingest.py /path/to/your/csv.csv .ingestion_pids/ingestion.log
```

## Environment Variables

Configure in `.env`:

```bash
# Concurrency Control
SEMAPHORE_LIMIT=2  # Number of concurrent rows to process

# Neo4j Database
NEO4J_URI="bolt+ssc://localhost:7687"
NEO4J_USER="neo4j"
NEO4J_PASSWORD="your_password"
NEO4J_DATABASE="qwen34neo4j"  # Target database

# LLM (vLLM / Qwen)
VLLM_BASE_URL="http://localhost:5000/v1/"
VLLM_API_KEY="api_key"
VLLM_MODEL_NAME="qwen35"

# Embedder (Triton)
TRITON_URL="localhost:6000"
TRITON_MODEL_NAME="gemma_embedding"
TRITON_TOKENIZER="onnx-community/embeddinggemma-300m-ONNX"
```

## Troubleshooting

### Monitor Server Not Starting

```bash
# Check logs
cat .ingestion_pids/monitor.log

# Check if port is in use
lsof -i :3456

# Kill existing process
kill $(lsof -t -i:3456)
```

### Ingestion Not Updating Dashboard

- Ensure monitor server is running
- Check WebSocket connection in browser console (F12)
- Verify `.env` has correct Neo4j settings

### Process Stuck

```bash
# Force kill all ingestion processes
pkill -f "python ingest.py"
pkill -f "monitor_server:app"
```

## API Endpoints

The monitor server provides these endpoints:

- **GET /** - Dashboard UI
- **GET /api/status** - Current status JSON
- **GET /api/logs** - All logs (events, API calls, errors)
- **WS /ws** - WebSocket for real-time updates

### Example: Check Status via API

```bash
curl http://localhost:3456/api/status
```

## Architecture

```
+---------------------+      WebSocket      +------------------+
|   ingest.py         | <-----------------> |  monitor_server  |
|   re_ingest.py      |                     |  (port 3456)     |
+---------------------+                     +--------+---------+
         |                                             |
         v                                             v
+---------------------+                     +------------------+
| Graphiti Core       |                     |   Browser        |
| - LLM Client        |                     |   Dashboard      |
| - Embedder          |                     |   (real-time)    |
| - Reranker          |                     +------------------+
| - Neo4j Driver      |
+---------------------+
```

## Notes

- Ingestion works even if monitor server is not running (graceful fallback)
- Log files are automatically created in `.ingestion_pids/`
- PID files allow precise process management
- Dashboard shows last 50 events and last 5 API calls for performance
