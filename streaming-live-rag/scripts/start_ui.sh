#!/usr/bin/env bash
set -e
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$DIR"
export PYTHONPATH=.
echo "========================================================"
echo "  Launching Streaming Live RAG - Demo UI & Pipeline"
echo "========================================================"
echo "Starting FastAPI server at http://localhost:8000 ..."
(sleep 2 && (which xdg-open > /dev/null && xdg-open http://localhost:8000/demo || which open > /dev/null && open http://localhost:8000/demo)) &
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
