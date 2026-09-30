FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Install curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition and install
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download and cache FastEmbed models at build time (Gate G1 requirement)
RUN python -c "from fastembed import TextEmbedding, SparseTextEmbedding; from fastembed.rerank.cross_encoder import TextCrossEncoder; TextEmbedding('BAAI/bge-small-en-v1.5'); SparseTextEmbedding('Qdrant/bm25'); TextCrossEncoder('Xenova/ms-marco-MiniLM-L-6-v2')"

# Copy source tree
COPY . .
RUN pip install --no-cache-dir --no-deps .

EXPOSE 8000

# Run FastAPI via uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
