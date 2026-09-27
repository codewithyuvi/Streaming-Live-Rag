"""
Gate 4 — Factual Grounding (G4).

Validates the ACTUAL synthesis output of the pipeline — never hand-written
strings. Each grounding_probes scenario in eval/labeled_set.yaml runs a real
turn through run_live_turn() with real retrieval (Qdrant dev corpus) and real
synthesis (Gemini). The produced answer is then:

  1. scored with retrieval.grounding.validate (claim-level support), and
  2. cross-checked independently: every citation tag in the answer must exist
     in the dev-corpus tag inventory (zero fabricated IDs).

SKIP (exit 2) when GEMINI_API_KEY is absent, or when Qdrant/the indexed
corpus is unreachable (grounding cannot be measured without retrieval).
Threshold: >= 85% mean citation support, zero fabricated IDs.
"""

import asyncio
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dotenv import load_dotenv
load_dotenv()

from eval.gates._common import (
    banner, footer, require_env, gate_main, GateSkipped, retrieval_preflight,
    STATUS_PASS, STATUS_FAIL,
)
from eval.dataset_loader import load_labeled_set
from streaming.engine import run_live_turn, EngineDeps
from streaming.live_stream import play_utterance
from retrieval.grounding import tags_in

THRESHOLD = 0.85
TAG_RE = re.compile(r"Doc_\d+\s*§[\d.]+")
ABSTAIN_PHRASE = "not available in the provided documents"


def _corpus_tag_inventory() -> set:
    """Every citation tag actually present in the dev corpus files."""
    corpus_dir = os.path.join(ROOT_DIR, "data", "dev_corpus")
    tags = set()
    if not os.path.isdir(corpus_dir):
        return tags
    for fn in os.listdir(corpus_dir):
        if not fn.endswith((".txt", ".md")):
            continue
        with open(os.path.join(corpus_dir, fn), "r", encoding="utf-8") as f:
            for m in TAG_RE.finditer(f.read()):
                tags.add(m.group(0).replace("  ", " ").strip())
    # normalize whitespace the same way grounding.canon does
    return {" ".join(t.split()) for t in tags}


def _retrieval_preflight() -> None:
    """Grounding is unmeasurable without a live retrieval backend."""
    retrieval_preflight()


def evaluate_g4():
    require_env("GEMINI_API_KEY")
    _retrieval_preflight()
    data = load_labeled_set()
    probes = data.get("grounding_probes", [])
    if not probes:
        raise GateSkipped("no grounding_probes in labeled_set.yaml")
    inventory = _corpus_tag_inventory()

    banner("GATE 4 — Factual Grounding on real synthesis output (G4)")
    print(f"Corpus tag inventory: {len(inventory)} tags")

    deps = EngineDeps.defaults()
    supports = []
    fabricated_total = 0
    details = []
    for p in probes:
        res = asyncio.run(run_live_turn(
            f"g4_{p['id']}",
            play_utterance(p["utterance"],
                           words_per_chunk=p.get("words_per_chunk", 3),
                           ms_per_chunk=p.get("ms_per_chunk", 50)),
            deps=deps,
            sink=lambda e: None,
        ))
        t = res.telemetry
        rep = t.grounding_report or {}
        cited = tags_in(res.answer)
        cited_norm = {" ".join(c.split()) for c in cited}
        fabricated_inv = sorted(cited_norm - inventory)
        fabricated_val = rep.get("fabricated_tags", []) or []
        fabricated_total += len(fabricated_inv) + len(fabricated_val)
        support = float(rep.get("score", 0.0))

        if p.get("expect_abstain"):
            abstained = (ABSTAIN_PHRASE in res.answer or bool(t.uncertainty)
                         or bool(rep.get("abstained")))
            ok = bool(abstained) and not fabricated_inv and not fabricated_val
            note = f"abstained={'yes' if abstained else 'no'}"
        else:
            supports.append(support)
            ok = support >= THRESHOLD and not fabricated_inv and not fabricated_val
            note = f"support={support:.2f}"

        details.append({"id": p["id"], "ok": ok, "note": note,
                        "fabricated": fabricated_inv + fabricated_val,
                        "answer": res.answer[:160]})

    mean_support = sum(supports) / len(supports) if supports else 1.0
    print(f"\nMean citation support: {mean_support:.1%} — target >= {THRESHOLD:.0%}")
    print(f"Fabricated IDs: {fabricated_total} — target 0")
    print("-" * 70)
    for d in details:
        icon = "OK  " if d["ok"] else "BAD "
        print(f"  [{icon}] {d['id']}: {d['note']}")
        if d["fabricated"]:
            print(f"          FABRICATED: {d['fabricated']}")
        print(f"          answer: {d['answer']}")

    ok = mean_support >= THRESHOLD and fabricated_total == 0
    footer(ok, "GATE 4")
    metric = f"{mean_support:.1%} support, {fabricated_total} fabricated"
    return (STATUS_PASS if ok else STATUS_FAIL), metric


if __name__ == "__main__":
    sys.exit(gate_main("G4", evaluate_g4))
