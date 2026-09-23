@echo off
setlocal
cd /d "%~dp0\.."
set PYTHONPATH=.
echo ========================================================
echo   Launching Streaming Live RAG - Demo UI & Pipeline
echo ========================================================
echo Starting FastAPI server at http://localhost:8000 ...
start "" http://localhost:8000/demo
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
