import os
import sys
import time
from fastapi.testclient import TestClient

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from retrieval.grounding import validate  # noqa: E402
from controller.heuristics import is_stable_enough  # noqa: E402
from api.main import _same_intent, app  # noqa: E402
from session.store import get_or_create_session, reset_store  # noqa: E402
from eval.dataset_loader import load_labeled_set  # noqa: E402
from retrieval.merge import merge_with_quota  # noqa: E402


def test_grounding_validator():
    rep = validate("The venue capacity is 30 [Doc_01 §1].", ["Doc_01 §1"])
    assert rep.ok is True
    assert rep.support == 1.0
    assert rep.fabricated == []
    assert rep.cited == ["Doc_01 §1"]

    # Fabricated tag detection
    rep_fab = validate("The venue capacity is 30 [Doc_99 §9].", ["Doc_01 §1"])
    assert rep_fab.ok is False
    assert "Doc_99 §9" in rep_fab.fabricated

    # Punctuation before bracket normalization
    rep_punct = validate("The maximum capacity is 30. [Doc_01 §1]", ["Doc_01 §1"])
    assert rep_punct.ok is True
    assert rep_punct.support == 1.0


def test_stability_heuristics():
    assert is_stable_enough("What is the capacity") is True
    assert is_stable_enough("What is the") is False
    assert is_stable_enough("Tell me about the policy for") is False
    assert is_stable_enough("Where is room 3B?") is True


def test_intent_deduplication():
    # Incomplete prefix must NOT suppress delta search
    assert _same_intent("what is the maximum capacity of the pune venue", "what is the maximum capacity of") is False
    # Identical queries must suppress
    assert _same_intent("what is the venue capacity", "what is the venue capacity") is True
    assert _same_intent("what is the venue capacity", None) is False


def test_session_lifecycle():
    reset_store()
    sess = get_or_create_session("pytest_session")

    # Turn 1: NEW_TOPIC -> version 1
    sess.commit("NEW_TOPIC", "q1", "ans1", ["Doc_01 §1"])
    assert sess.answer_version == 1
    assert sess.current_citations == ["Doc_01 §1"]

    # Turn 2: LATE_DETAIL -> version 2, union citations
    sess.commit("LATE_DETAIL", "q2", "ans2", ["Doc_02 §3"])
    assert sess.answer_version == 2
    assert "Doc_01 §1" in sess.current_citations
    assert "Doc_02 §3" in sess.current_citations

    # Turn 3: PRESENTATION_ONLY -> version unchanged (stays 2)
    sess.commit("PRESENTATION_ONLY", "format as bullets", "ans3", ["Doc_01 §1"])
    assert sess.answer_version == 2
    assert len(sess.turns) == 3


def test_dataset_integrity():
    data = load_labeled_set(os.path.join(ROOT_DIR, "eval", "labeled_set.yaml"))
    queries = data["queries"]
    assert len(queries) == 63
    ids = [q["id"] for q in queries]
    assert len(ids) == len(set(ids)), "Duplicate query IDs found in labeled_set.yaml"


def test_merge_quota_guarantee():
    sub1_hits = [(0.9, {"id": 1, "text": "chunk1"}), (0.8, {"id": 2, "text": "chunk2"})]
    sub2_hits = [(0.7, {"id": 3, "text": "chunk3"}), (0.6, {"id": 4, "text": "chunk4"})]
    per_sq = [
        {"sub_query": "sq1", "scored_hits": sub1_hits},
        {"sub_query": "sq2", "scored_hits": sub2_hits},
    ]
    merged = merge_with_quota(per_sq, total_k=4, min_per_sq=2)
    pids = [item["point"]["id"] for item in merged]
    assert 1 in pids and 2 in pids and 3 in pids and 4 in pids


def test_fastapi_endpoints():
    client = TestClient(app)
    # Health check
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"

    # Health check with providers
    res_live = client.get("/health?live=true")
    assert res_live.status_code == 200
    assert "providers" in res_live.json()

    # Demo UI serves HTML
    demo_res = client.get("/demo")
    assert demo_res.status_code == 200
    assert "text/html" in demo_res.headers.get("content-type", "")


def test_rate_limiter():
    from api.main import _check_rate_limit, _session_request_times
    _session_request_times["test_rate_session"] = [time.time()] * 60
    assert _check_rate_limit("test_rate_session") is False
    assert _check_rate_limit("fresh_session") is True
