#!/usr/bin/env bash
# Startup script for Multimodal Road Accident & Damage AI System

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure venv exists and is activated
if [ -d "$SCRIPT_DIR/venv" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
fi

# Ensure benchmark samples are created
python3 "$SCRIPT_DIR/sample_data/generate_sample_media.py"

echo "=========================================================="
echo " Starting Multimodal Road Accident & Damage AI System... "
echo "=========================================================="
streamlit run "$SCRIPT_DIR/app.py" --server.port 8501 --server.headless true
