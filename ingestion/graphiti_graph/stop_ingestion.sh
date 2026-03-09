#!/bin/bash

# Stop Ingestion Script
# Stops monitor server and ingestion processes

PID_DIR="${PWD}/.ingestion_pids"

echo "Stopping ingestion processes..."
echo ""

# Stop Monitor Server
if [ -f "$PID_DIR/monitor.pid" ]; then
    MONITOR_PID=$(cat "$PID_DIR/monitor.pid")
    if pgrep -p "$MONITOR_PID" > /dev/null; then
        echo "Stopping Monitor Server (PID: $MONITOR_PID)..."
        kill "$MONITOR_PID" 2>/dev/null
        sleep 1
        if pgrep -p "$MONITOR_PID" > /dev/null; then
            echo "Force killing monitor server..."
            kill -9 "$MONITOR_PID" 2>/dev/null
        fi
    fi
    rm -f "$PID_DIR/monitor.pid"
fi

# Stop Ingestion
if [ -f "$PID_DIR/ingestion.pid" ]; then
    INGEST_PID=$(cat "$PID_DIR/ingestion.pid")
    if pgrep -p "$INGEST_PID" > /dev/null; then
        echo "Stopping Ingestion (PID: $INGEST_PID)..."
        kill "$INGEST_PID" 2>/dev/null
        sleep 1
        if pgrep -p "$INGEST_PID" > /dev/null; then
            echo "Force killing ingestion..."
            kill -9 "$INGEST_PID" 2>/dev/null
        fi
    fi
    rm -f "$PID_DIR/ingestion.pid"
fi

# Also kill any remaining processes by name
echo "Cleaning up any remaining processes..."
pkill -f "monitor_server:app" 2>/dev/null
pkill -f "python ingest.py" 2>/dev/null

echo ""
echo "All processes stopped."
