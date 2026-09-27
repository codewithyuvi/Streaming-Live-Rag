"""
tests/test_thought_stream.py — Verification of honest thought-stream narration.

Tests:
1. Thought events fire in phase order 1→4 on a multi-intent utterance.
2. No fabricated citations: every [Doc_\\d+ §...] regex match in thought text
   is a subset of that turn's actual retrieval result doc IDs.
3. telemetry.thought_process is populated and equals the joined thought texts.
4. G6's subsequence order check still passes with interleaved thought events.
5. Thought emission adds no LLM calls (deps.synthesize_fn called exactly as before).
6. Phase-2 thought text contains the real continuation substring, and the
   provisional_fired_s value recorded in telemetry is the actual measured
   timestamp of the provisional retrieval (not a placeholder), matching the
   retrieval_started event's t_s from the same run.
"""

import asyncio
import os
import re
import sys

TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

from test_pipeline import make_fake_deps, run_turn  # noqa: E402


def test_thought_phases_order():
    """1. Thought events fire in phase order 1->4 on a multi-intent utterance."""
    utterance = (
        "I need to plan a customer workshop in Pune for 30 people "
        "and I need the cancellation policy."
    )
    res, events, _ = run_turn("ts_test_1", utterance, make_fake_deps(), wpc=2, mpc=30)
    thought_events = [e for e in events if e.get("type") == "thought"]
    phases = [e.get("phase") for e in thought_events]
    assert phases == [1, 2, 3, 4], f"Expected phases [1, 2, 3, 4], got {phases}"


def test_no_fabricated_citations():
    """2. No fabricated citations: every [Doc_\\d+ §...] regex match in thought text must be a subset of actual retrieval doc IDs."""
    utterance = (
        "I need to plan a customer workshop in Pune for 30 people "
        "and I need the cancellation policy."
    )
    deps = make_fake_deps()
    res, events, _ = run_turn("ts_test_2", utterance, deps, wpc=2, mpc=30)
    thought_events = [e for e in events if e.get("type") == "thought"]
    assert len(thought_events) > 0, "No thought events emitted"

    # Actual retrieval tags returned by make_fake_deps: Doc_01 §1, Doc_01 §2
    allowed_tags = {"Doc_01 §1", "Doc_01 §2"}

    # Extract all [Doc_XX §Y] matches from thought texts
    for te in thought_events:
        matches = re.findall(r"\[(Doc_\d+\s*§\s*[\w.\-]+)\]", te.get("text", ""))
        for match in matches:
            assert match in allowed_tags, f"Fabricated or un-retrieved citation '{match}' found in thought text: {te['text']}"


def test_thought_process_telemetry_populated():
    """3. telemetry.thought_process is populated and equals the joined thought texts."""
    utterance = (
        "I need to plan a customer workshop in Pune for 30 people "
        "and I need the cancellation policy."
    )
    res, events, _ = run_turn("ts_test_3", utterance, make_fake_deps(), wpc=2, mpc=30)
    thought_events = [e for e in events if e.get("type") == "thought"]
    expected_process = "\n\n".join(e.get("text", "") for e in thought_events)

    assert res.telemetry.thought_process == expected_process
    assert len(res.telemetry.thought_process) > 0


def test_g6_subsequence_order_with_interleaved_thoughts():
    """4. G6's subsequence order check still passes with interleaved thought events."""
    utterance = (
        "I need to plan a customer workshop in Pune for 30 people "
        "and I need the cancellation policy."
    )
    res, events, _ = run_turn("ts_test_4", utterance, make_fake_deps(), wpc=2, mpc=30)
    order = [e.get("type") for e in events]
    expected_seq = [
        "stream_started", "transcript", "controller",
        "retrieval_started", "utterance_end", "answer", "telemetry"
    ]
    pos = 0
    for want in expected_seq:
        assert want in order, f"Missing expected event {want}"
        pos = order.index(want, pos) + 1
    assert pos > 0


def test_thought_emission_adds_no_llm_calls():
    """5. Thought emission adds no LLM calls (deps.synthesize_fn called exactly as before)."""
    synth_calls = 0

    def _counted_synth(prompt):
        nonlocal synth_calls
        synth_calls += 1
        return {
            "text": (
                "The maximum capacity of the Pune workshop venue is 30 attendees "
                "[Doc_01 §1]. The cancellation policy requires 48 hours notice "
                "for a full refund [Doc_01 §2]."
            ),
            "input_tokens": 120,
            "output_tokens": 45,
        }

    deps = make_fake_deps()
    deps.synthesize_fn = _counted_synth

    utterance = (
        "I need to plan a customer workshop in Pune for 30 people "
        "and I need the cancellation policy."
    )
    res, events, _ = run_turn("ts_test_5", utterance, deps, wpc=2, mpc=30)
    # The normal pipeline calls synthesize_fn once for the answer (0 additional calls for thought)
    assert synth_calls == 1, f"Expected exactly 1 synthesize call, got {synth_calls}"


def test_phase2_continuation_and_provisional_fired_s_are_real():
    """
    6. Phase-2 thought text contains the real continuation substring, and the
    provisional_fired_s value recorded in telemetry is the actual measured
    timestamp of the provisional retrieval — not a placeholder or a value
    recomputed after the fact — and matches the t_s of the retrieval_started
    event emitted live during the same run.
    """
    utterance = (
        "I need to plan a customer workshop in Pune for 30 people "
        "and I need the cancellation policy."
    )
    res, events, _ = run_turn("ts_test_6", utterance, make_fake_deps(), wpc=2, mpc=30)

    # Real continuation substring: the engine's own phase-2 narration text
    # (streaming/engine.py), not just phase-ordering.
    thought_events = [e for e in events if e.get("type") == "thought"]
    phase2 = next(e for e in thought_events if e.get("phase") == 2)

    # The old boilerplate assertion ("Continuation absorbed without context
    # reset") is banned: it asserted continuity instead of demonstrating it.
    assert "Continuation absorbed without context reset" not in phase2["text"], (
        f"Phase-2 text still uses the canned assertion instead of real evidence: {phase2['text']!r}"
    )

    # Real continuation substring: the tail of the utterance that arrived
    # after the provisional trigger must be quoted verbatim in the narration.
    assert "cancellation policy" in phase2["text"], (
        f"Phase-2 thought text is missing the real continuation substring: {phase2['text']!r}"
    )

    # Real provisional_fired_s: cross-check telemetry against the live
    # retrieval_started event's measured t_s from this same run.
    retrieval_started_events = [e for e in events if e.get("type") == "retrieval_started"]
    assert len(retrieval_started_events) == 1, (
        f"Expected exactly one provisional retrieval_started event, got {len(retrieval_started_events)}"
    )
    fired_t_s = retrieval_started_events[0]["t_s"]

    assert res.telemetry.provisional_fired_s is not None, (
        "telemetry.provisional_fired_s was not recorded"
    )
    assert res.telemetry.provisional_fired_s == fired_t_s, (
        f"telemetry.provisional_fired_s ({res.telemetry.provisional_fired_s}) does not match the "
        f"actual measured retrieval_started t_s ({fired_t_s}) from this run — provisional_fired_s "
        f"must be the real engine-recorded timestamp, not a placeholder or recomputed value."
    )

    # The phase-2 narration must quote the real provisional fire time (the
    # measured t_s from this run, formatted to 2 decimals as the engine does),
    # so the reader can see WHEN the provisional search fired relative to the
    # continuation — demonstrated continuity, not an asserted one.
    assert f"{fired_t_s:.2f}s" in phase2["text"], (
        f"Phase-2 thought text does not quote the real provisional fire time "
        f"({fired_t_s:.2f}s): {phase2['text']!r}"
    )
