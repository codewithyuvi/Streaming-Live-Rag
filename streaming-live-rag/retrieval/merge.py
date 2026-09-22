"""
Phase 4 — Result Merger & Deduplicator with Quota Guarantee (H8 / Appendix B.2).
Prevents sub-intent starvation by ensuring each sub-query receives guaranteed evidence slots.
"""

from dataclasses import dataclass, field
from typing import Any


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


def merge_with_quota(
    per_sq: list[dict],
    total_k: int = 8,
    min_per_sq: int = 2
) -> list[dict]:
    """
    Guarantees each sub-intent its best min_per_sq chunks, then fills
    the remaining budget up to total_k by highest cross-encoder score.

    per_sq: [{'sub_query': str, 'scored_hits': [(score, point), ...]}]
    """
    chosen: dict[Any, dict] = {}
    order = []

    def take(sq: str, score: float, p: Any):
        pid = getattr(p, "id", None)
        if pid is None:
            pid = p.get("id") if isinstance(p, dict) else id(p)

        if pid not in chosen:
            chosen[pid] = {"point": p, "score": score, "subs": [sq]}
            order.append(pid)
        elif sq not in chosen[pid]["subs"]:
            chosen[pid]["subs"].append(sq)
            if score > chosen[pid]["score"]:
                chosen[pid]["score"] = score

    # 1) Every sub-intent is guaranteed min_per_sq evidence
    for r in per_sq:
        for score, p in r.get("scored_hits", [])[:min_per_sq]:
            take(r["sub_query"], score, p)

    # 2) Fill remaining budget by score
    rest = sorted(
        (
            (s, r["sub_query"], p)
            for r in per_sq
            for s, p in r.get("scored_hits", [])[min_per_sq:]
        ),
        key=lambda x: -x[0]
    )

    for score, sq, p in rest:
        if len(chosen) >= total_k:
            break
        take(sq, score, p)

    limit = max(total_k, len(per_sq) * min_per_sq) if per_sq else total_k
    return [chosen[i] for i in order][:limit]


def merge_and_dedup(
    per_subquery_results: list[dict],
    top_k: int = 8
) -> list[MergedChunk]:
    """
    Merges results from multiple sub-queries with per-intent quota guarantees.
    """
    if not per_subquery_results:
        return []

    merged_items = merge_with_quota(per_subquery_results, total_k=top_k, min_per_sq=2)
    chunks = []
    for item in merged_items:
        p = item["point"]
        payload = getattr(p, "payload", {}) if hasattr(p, "payload") else (p.get("payload", {}) if isinstance(p, dict) else {})
        pid = getattr(p, "id", None) if hasattr(p, "id") else (p.get("id") if isinstance(p, dict) else None)
        chunks.append(
            MergedChunk(
                point_id=pid,
                text=payload.get("text", "") if payload else "",
                tag=payload.get("tag", "") if payload else "",
                doc_id=payload.get("doc_id", "") if payload else "",
                section=payload.get("section", "") if payload else "",
                best_score=item["score"],
                source_sub_queries=item["subs"]
            )
        )
    return chunks
