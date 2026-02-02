#!/bin/bash

# Port to run Streamlit on
PORT=8501

echo "🚀 Starting Endee AI Research Assistant..."

# 1. Start Endee Server in background if not running
if ! lsof -i:8080 > /dev/null; then
    echo "📦 Starting Endee Vector Database..."
    export NDD_DATA_DIR=$(pwd)/data
    mkdir -p data
    ./server/build/ndd-neon-darwin > endee_server.log 2>&1 &
    sleep 2
else
    echo "✅ Endee Server is already running."
fi

# 2. Start Streamlit App
echo "💻 Launching UI..."
source project/.venv/bin/activate
streamlit run project/app.py --server.port $PORT
