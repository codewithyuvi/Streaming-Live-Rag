"""
Phase 4 — Result Merger & Deduplicator

Merges retrieval results from multiple parallel sub-query searches.
Deduplicates by Qdrant point ID, keeping the highest rerank score
when the same chunk appears in multiple sub-query result sets.

Design decisions:
- Dedup by point ID (not text hash) since Qdrant guarantees unique IDs.
- When a chunk appears in multiple sub-query results, we keep the highest
  score — this chunk is broadly relevant and should rank higher.
- Each merged result is tagged with which sub-queries it was relevant to,
  enabling the synthesis prompt to attribute citations per sub-intent.
"""

from dataclasses import dataclass, field


@dataclass
class MergedChunk:
    """A deduplicated chunk with provenance tracking."""
    point_id: int | str
    text: str
    tag: str  # e.g. "Doc_01 §1"
    doc_id: str
    section: str
    best_score: float
    source_sub_queries: list[str] = field(default_factory=list)


def merge_and_dedup(
    per_subquery_results: list[dict],
    top_k: int = 5
) -> list[MergedChunk]:
    """
    Merges results from multiple sub-query retrievals.

    Args:
        per_subquery_results: List of dicts, each with:
            - "sub_query": str
            - "scored_hits": list of (score: float, point) tuples
              where point has .id and .payload
        top_k: Max number of merged results to return.

    Returns:
        List of MergedChunk, sorted by best_score descending, capped at top_k.
    """
    seen: dict[int | str, MergedChunk] = {}

    for result in per_subquery_results:
        sub_query = result["sub_query"]
        scored_hits = result["scored_hits"]

        for score, point in scored_hits:
            pid = point.id

            if pid in seen:
                # Already seen — promote score if higher, add sub_query source
                existing = seen[pid]
                if score > existing.best_score:
                    existing.best_score = score
                if sub_query not in existing.source_sub_queries:
                    existing.source_sub_queries.append(sub_query)
            else:
                seen[pid] = MergedChunk(
                    point_id=pid,
                    text=point.payload.get("text", ""),
                    tag=point.payload.get("tag", ""),
                    doc_id=point.payload.get("doc_id", ""),
                    section=point.payload.get("section", ""),
                    best_score=score,
                    source_sub_queries=[sub_query]
                )

    # Sort by best_score descending, then cap
    merged = sorted(seen.values(), key=lambda c: c.best_score, reverse=True)
    return merged[:top_k]
