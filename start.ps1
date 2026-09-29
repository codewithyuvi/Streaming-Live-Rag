Set-Location -Path "$PSScriptRoot\streaming-live-rag"
Write-Host "Starting Streaming Live RAG server (Pure Python)..." -ForegroundColor Cyan
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
