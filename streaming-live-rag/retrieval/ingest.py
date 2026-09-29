import os
import sys
import glob
import re
import logging
from typing import List, Dict, Any, Optional, Set

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from qdrant_client.models import Distance, VectorParams, PointStruct, SparseVectorParams, SparseVector, Modifier

from retrieval.hybrid_search import (
    get_qdrant_client,
    get_embedding_model,
    get_sparse_embedding_model,
    _collection_name,
)

logger = logging.getLogger(__name__)

COLLECTION_NAME = _collection_name()


def ensure_collection_exists(client=None, collection_name: str = None, recreate: bool = False):
    """Ensures Qdrant collection exists with both dense and sparse configurations."""
    client = client or get_qdrant_client()
    collection_name = collection_name or _collection_name()

    if client.collection_exists(collection_name):
        if recreate:
            logger.info("Recreating collection '%s'...", collection_name)
            client.delete_collection(collection_name)
        else:
            return

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": VectorParams(size=384, distance=Distance.COSINE)
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(modifier=Modifier.IDF)
        }
    )
    logger.info("Created collection '%s'.", collection_name)


import threading
import uuid
import hashlib

_ingest_lock = threading.RLock()


def get_next_doc_id() -> str:
    """Finds the next sequential Doc_XX identifier by inspecting corpus and collection exhaustively."""
    with _ingest_lock:
        max_id = 0
        # Check data/dev_corpus directory
        corpus_dir = os.path.join(os.path.dirname(__file__), "../data/dev_corpus")
        if os.path.exists(corpus_dir):
            for f in os.listdir(corpus_dir):
                filepath = os.path.join(corpus_dir, f)
                if os.path.isfile(filepath):
                    try:
                        with open(filepath, "r", encoding="utf-8", errors="ignore") as fh:
                            content = fh.read(2048)
                            for m in re.finditer(r"Doc_(\d+)", content):
                                max_id = max(max_id, int(m.group(1)))
                    except Exception:
                        pass

        # Check Qdrant collection points with exhaustive pagination
        try:
            client = get_qdrant_client()
            col = _collection_name()
            if client.collection_exists(col):
                offset = None
                while True:
                    points, offset = client.scroll(col, limit=250, offset=offset, with_payload=True)
                    for p in points:
                        doc_str = (p.payload or {}).get("doc_id", "")
                        if (m := re.search(r"Doc_(\d+)", doc_str)):
                            max_id = max(max_id, int(m.group(1)))
                    if offset is None:
                        break
        except Exception:
            pass

        return f"Doc_{max_id + 1:02d}"


def get_corpus_summary() -> Dict[str, Any]:
    """Returns metadata about all indexed documents and sections in the vector store."""
    col = _collection_name()
    try:
        client = get_qdrant_client()
        if not client.collection_exists(col):
            return {"collection": col, "total_chunks": 0, "documents": []}
        count = client.count(col).count
        points, _ = client.scroll(col, limit=500, with_payload=True)
        docs_map: dict[str, dict] = {}
        for p in points:
            payload = p.payload or {}
            doc_id = payload.get("doc_id", "Unknown")
            sec = payload.get("section", "1")
            tag = payload.get("tag", f"{doc_id} §{sec}")
            filename = payload.get("filename", "")
            if doc_id not in docs_map:
                docs_map[doc_id] = {
                    "doc_id": doc_id,
                    "filename": filename,
                    "tags": [],
                    "section_count": 0,
                }
            docs_map[doc_id]["tags"].append(tag)
            docs_map[doc_id]["section_count"] += 1
            if filename and not docs_map[doc_id]["filename"]:
                docs_map[doc_id]["filename"] = filename

        return {
            "collection": col,
            "total_chunks": count,
            "documents": list(docs_map.values()),
        }
    except Exception as e:
        logger.warning("Failed to get corpus summary (Qdrant may be offline): %s", e)
        return {"collection": col, "total_chunks": 0, "documents": [], "status": "offline", "note": str(e)}


def clear_corpus() -> bool:
    """Deletes and recreates the collection to reset the corpus under _ingest_lock."""
    with _ingest_lock:
        client = get_qdrant_client()
        col = _collection_name()
        ensure_collection_exists(client, col, recreate=True)
        return True


def ingest_sections(sections: List[Dict[str, Any]], reset: bool = False) -> Dict[str, Any]:
    """
    Computes dense and sparse embeddings for section dicts and upserts to Qdrant.
    Uses deterministic UUID5 point IDs formatted as uuid5(NAMESPACE_DNS, f"{doc_id}|{section}|{content_hash}").
    Guaranteed atomic and collision-free under _ingest_lock.
    """
    if not sections:
        return {"status": "empty", "chunks_indexed": 0, "tags": []}

    with _ingest_lock:
        client = get_qdrant_client()
        col = _collection_name()
        ensure_collection_exists(client, col, recreate=reset)

        embedding_model = get_embedding_model()
        sparse_embedding_model = get_sparse_embedding_model()

        texts = [s["text"] for s in sections]
        dense_embeddings = list(embedding_model.embed(texts))
        sparse_embeddings = list(sparse_embedding_model.embed(texts))

        points = []
        tags = []
        for i, sec in enumerate(sections):
            c_hash = hashlib.sha256(sec["text"].encode("utf-8")).hexdigest()[:16]
            # Deterministic UUID5 with | delimiter avoids ID collision and silent overwrites
            pid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{sec['doc_id']}|{sec['section']}|{c_hash}"))
            tags.append(sec["tag"])
            points.append(
                PointStruct(
                    id=pid,
                    vector={
                        "dense": dense_embeddings[i].tolist(),
                        "sparse": SparseVector(
                            indices=sparse_embeddings[i].indices.tolist(),
                            values=sparse_embeddings[i].values.tolist(),
                        ),
                    },
                    payload={
                        "doc_id": sec["doc_id"],
                        "section": sec["section"],
                        "tag": sec["tag"],
                        "text": sec["text"],
                        "filename": sec.get("filename", ""),
                        "file_hash": sec.get("file_hash", ""),
                        "content_hash": c_hash,
                    },
                )
            )

        client.upsert(collection_name=col, points=points)
        logger.info("Indexed %d sections into collection '%s'.", len(points), col)

        return {
            "status": "success",
            "chunks_indexed": len(points),
            "tags": tags,
            "doc_id": sections[0]["doc_id"] if sections else "",
            "collection": col,
        }


def ingest_file_or_text(
    filename: str,
    content_bytes: bytes | None = None,
    text_override: str | None = None,
    reset: bool = False
) -> Dict[str, Any]:
    """
    Synchronous ingestion pipeline executing under _ingest_lock in a single worker thread.
    Includes SHA-256 content deduplication, automatic doc_id allocation, and safe local persistence.
    """
    from retrieval.parsers import extract_sections

    with _ingest_lock:
        if content_bytes is None or len(content_bytes) == 0:
            if text_override:
                content_bytes = text_override.encode("utf-8")
                filename = filename or "direct_input.txt"
            else:
                raise ValueError("No file content or text provided.")

        content_hash = hashlib.sha256(content_bytes).hexdigest()

        # Check for existing document with identical content (deduplication)
        if not reset:
            try:
                client = get_qdrant_client()
                col = _collection_name()
                if client.collection_exists(col):
                    offset = None
                    while True:
                        points, offset = client.scroll(col, limit=200, offset=offset, with_payload=True)
                        for p in points:
                            payload = p.payload or {}
                            if payload.get("file_hash") == content_hash:
                                existing_doc_id = payload.get("doc_id", "Doc_01")
                                return {
                                    "status": "already_indexed",
                                    "message": f"Document with matching content already indexed as {existing_doc_id}.",
                                    "doc_id": existing_doc_id,
                                    "chunks_indexed": 0,
                                    "tags": [payload.get("tag", "")],
                                }
                        if offset is None:
                            break
            except Exception as e:
                logger.debug("Deduplication scroll check error: %s", e)

        # R1: Extract sections BEFORE resetting the corpus (protect existing data if parse fails)
        provisional_doc_id = "Doc_01" if reset else get_next_doc_id()
        sections = extract_sections(filename, content_bytes, provisional_doc_id)
        if not sections:
            return {"status": "empty", "chunks_indexed": 0, "tags": [], "doc_id": provisional_doc_id}

        corpus_dir = os.path.join(os.path.dirname(__file__), "../data/dev_corpus")
        if reset:
            # Parse succeeded; now safe to clear live collection
            clear_corpus()
            # R6: Clean up dynamically uploaded copies from dev_corpus on reset
            try:
                if os.path.exists(corpus_dir):
                    for old_f in glob.glob(os.path.join(corpus_dir, "Doc_*_*")):
                        try:
                            os.remove(old_f)
                        except OSError:
                            pass
            except Exception as e:
                logger.warning("Failed to clean up uploaded files on reset: %s", e)

            doc_id = "Doc_01"
            for sec in sections:
                sec["doc_id"] = doc_id
                sec["tag"] = f"{doc_id} §{sec['section']}"
        else:
            doc_id = provisional_doc_id

        for sec in sections:
            sec["file_hash"] = content_hash

        result = ingest_sections(sections, reset=False)
        result["doc_id"] = doc_id

        # Persist a local copy to data/uploads with doc_id namespacing
        try:
            uploads_dir = os.path.join(os.path.dirname(__file__), "../data/uploads")
            os.makedirs(uploads_dir, exist_ok=True)
            safe_name = f"{doc_id}_{re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)}"
            with open(os.path.join(uploads_dir, safe_name), "wb") as f:
                f.write(content_bytes)
        except Exception as e:
            logger.warning("Failed to save copy of uploaded file to uploads: %s", e)

        return result


def parse_corpus(corpus_dir: str, allowed_docs: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
    """
    Parses all supported files in the given directory.
    Supports .txt, .pdf, .docx, .pptx, .xlsx, .csv, .json, .yaml, .html, .md.
    If allowed_docs is provided (e.g. {'Doc_01', 'Doc_02'}), only chunks belonging to those IDs are returned.
    """
    from retrieval.parsers import extract_sections

    chunks = []
    all_files = sorted(glob.glob(os.path.join(corpus_dir, "*.*")))
    # Prioritize benchmark text files first so Doc_01 and Doc_02 are stable
    txt_files = [f for f in all_files if f.lower().endswith((".txt", ".md"))]
    other_files = [f for f in all_files if not f.lower().endswith((".txt", ".md", ".py", ".pyc"))]
    ordered_files = txt_files + other_files

    seen_hashes = set()
    used_ids = set()

    for file_path in ordered_files:
        filename = os.path.basename(file_path)
        ext = os.path.splitext(filename.lower())[1]
        if ext in (".py", ".pyc"):
            continue
        try:
            with open(file_path, "rb") as f:
                content = f.read()
            if not content.strip():
                continue

            content_hash = hashlib.sha256(content).hexdigest()
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)

            # Assign stable doc_id
            assigned_id = None
            if ext in (".txt", ".md"):
                try:
                    txt = content.decode("utf-8", errors="ignore")
                    m = re.search(r"Doc_(\d+)", txt)
                    if m:
                        candidate = f"Doc_{int(m.group(1)):02d}"
                        if candidate not in used_ids:
                            assigned_id = candidate
                except Exception:
                    pass

            if not assigned_id:
                m = re.search(r"Doc_(\d+)", filename)
                if m:
                    candidate = f"Doc_{int(m.group(1)):02d}"
                    if candidate not in used_ids:
                        assigned_id = candidate

            if not assigned_id or assigned_id in used_ids:
                idx = 1
                while f"Doc_{idx:02d}" in used_ids:
                    idx += 1
                assigned_id = f"Doc_{idx:02d}"

            used_ids.add(assigned_id)

            if allowed_docs and assigned_id not in allowed_docs:
                continue

            file_sections = extract_sections(filename, content, assigned_id)
            for s in file_sections:
                s["file_hash"] = content_hash
            chunks.extend(file_sections)
        except Exception as e:
            logger.warning("Failed to parse %s: %s", file_path, e)

    return chunks


def ingest(allowed_docs: Optional[Set[str]] = None):
    """Default batch ingestion entrypoint: indexes strictly Doc 1 and Doc 2."""
    if allowed_docs is None:
        allowed_docs = {"Doc_01", "Doc_02"}
    corpus_dir = os.path.join(os.path.dirname(__file__), "../data/dev_corpus")
    print(f"Parsing corpus from {corpus_dir} (strictly {sorted(list(allowed_docs))})...")
    chunks = parse_corpus(corpus_dir, allowed_docs=allowed_docs)
    print(f"Found {len(chunks)} chunks.")

    if not chunks:
        print("No chunks found. Exiting.")
        return

    res = ingest_sections(chunks, reset=True)
    print(f"Ingestion complete: {res['chunks_indexed']} chunks indexed in collection '{res['collection']}'.")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    ingest()
