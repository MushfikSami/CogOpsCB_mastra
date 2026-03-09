#!/bin/bash

# Ingestion Script Runner
# Starts monitor server and ingestion in background

# Configuration
MONITOR_PORT=3456
CSV_FILE="${1:-/home/vpa/Documents/data_20260309/data_processed.csv}"
PID_DIR="${PWD}/.ingestion_pids"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}  Data Ingestion Starting...${NC}"
echo -e "${GREEN}================================${NC}"

# Check if CSV file exists
if [ ! -f "$CSV_FILE" ]; then
    echo -e "${RED}Error: CSV file not found: $CSV_FILE${NC}"
    echo "Usage: $0 <csv_file>"
    exit 1
fi

# Create PID directory
mkdir -p "$PID_DIR"

# Start Monitor Server
echo -e "${YELLOW}Starting Monitor Server on port $MONITOR_PORT...${NC}"
cd "$(dirname "$0")"
nohup uvicorn monitor_server:app --host 0.0.0.0 --port $MONITOR_PORT > "$PID_DIR/monitor.log" 2>&1 &
MONITOR_PID=$!
echo "$MONITOR_PID" > "$PID_DIR/monitor.pid"

# Wait for monitor to start
sleep 2

# Check if monitor started successfully
if pgrep -f "monitor_server:app" > /dev/null; then
    echo -e "${GREEN}Monitor Server started (PID: $MONITOR_PID)${NC}"
    echo -e "${GREEN}Dashboard available at: http://localhost:$MONITOR_PORT${NC}"
else
    echo -e "${RED}Warning: Monitor Server may have failed to start. Check $PID_DIR/monitor.log${NC}"
fi

# Start Ingestion
echo -e "${YELLOW}Starting ingestion of $CSV_FILE...${NC}"
nohup python ingest.py "$CSV_FILE" > "$PID_DIR/ingestion.log" 2>&1 &
INGEST_PID=$!
echo "$INGEST_PID" > "$PID_DIR/ingestion.pid"

# Wait a moment for ingestion to start
sleep 1

# Check if ingestion started
if pgrep -f "python ingest.py" > /dev/null; then
    echo -e "${GREEN}Ingestion started (PID: $INGEST_PID)${NC}"
else
    echo -e "${RED}Error: Ingestion failed to start. Check $PID_DIR/ingestion.log${NC}"
    exit 1
fi

# Display status
echo ""
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}  Ingestion Running in Background${NC}"
echo -e "${GREEN}================================${NC}"
echo ""
echo -e "Dashboard:  http://localhost:$MONITOR_PORT"
echo -e "Monitor PID:  $MONITOR_PID"
echo -e "Ingestion PID: $INGEST_PID"
echo -e "Logs:"
echo -e "  Monitor:  $PID_DIR/monitor.log"
echo -e "  Ingestion: $PID_DIR/ingestion.log"
echo ""
echo -e "${YELLOW}To stop both processes, run:${NC}"
echo -e "  $0 stop"
echo -e ""
echo -e "${YELLOW}To monitor logs in real-time, run:${NC}"
echo -e "  tail -f $PID_DIR/*.log"
echo -e ""
