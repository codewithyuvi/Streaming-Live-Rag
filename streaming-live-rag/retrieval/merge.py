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
        return pid

    # 1) Guaranteed sub-intents receive up to min_per_sq evidence slots
    for r in per_sq:
        if r.get("guaranteed", True):
            for score, p in r.get("scored_hits", [])[:min_per_sq]:
                take(r["sub_query"], score, p)

    # 1b) Provisional (early-retrieval) entries keep a minimum of 1 slot.
    # Without this, guaranteed delta quotas can exhaust the top_k budget and
    # silently drop the very evidence the system rushed to retrieve early.
    # The slot is *reserved*: prov_first ordering below protects it from the
    # final budget slice.
    prov_pids: list = []
    for r in per_sq:
        if not r.get("guaranteed", True):
            for score, p in r.get("scored_hits", [])[:1]:
                prov_pids.append(take(r["sub_query"], score, p))

    # 2) Fill remaining budget by score (non-guaranteed entries participate from hit 0)
    rest_items = []
    for r in per_sq:
        start_idx = min_per_sq if r.get("guaranteed", True) else 0
        for s, p in r.get("scored_hits", [])[start_idx:]:
            rest_items.append((s, r["sub_query"], p))

    rest = sorted(rest_items, key=lambda x: -x[0])

    for score, sq, p in rest:
        if len(chosen) >= total_k:
            break
        take(sq, score, p)

    # 3) Reserved provisional slots lead the final ordering, then the rest
    # in insertion order, then cut to budget. Provisional evidence is never
    # the entry dropped by the [:total_k] slice.
    prov_first = [pid for pid in dict.fromkeys(prov_pids) if pid in chosen]
    tail = [pid for pid in order if pid not in prov_first]
    return [chosen[i] for i in (prov_first + tail)][:total_k]


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
