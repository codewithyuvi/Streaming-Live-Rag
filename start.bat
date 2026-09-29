@echo off
cd /d "%~dp0streaming-live-rag"
echo Starting Streaming Live RAG server (Pure Python)...
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
pause
