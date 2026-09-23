#!/usr/bin/env bash
set -e
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$DIR"
export PYTHONPATH=.
echo "========================================================"
echo "  Launching Streaming Live RAG - Demo UI & Pipeline"
echo "========================================================"
echo "Starting FastAPI server at http://localhost:8000 ..."
if which xdg-open > /dev/null; then
  xdg-open http://localhost:8000/demo &
elif which open > /dev/null; then
  open http://localhost:8000/demo &
fi
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
