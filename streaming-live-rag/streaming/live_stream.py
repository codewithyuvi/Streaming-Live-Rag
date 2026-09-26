"""
streaming/live_stream.py — Honest real-time transcript chunk sources.

This module replaces the old arithmetic-only simulator with wall-clock pacing.
Every chunk carries a *measured* timestamp (time.monotonic() since stream start),
and chunks are actually emitted over real time via ``await asyncio.sleep``.

Two sources implement the same protocol (async iterable of StreamEvent):
  1. play_utterance() — replays a full utterance as a paced live stream.
     Used by POST /turn ("replay this utterance as if spoken live") and by
     the eval gates, so G2 early-retrieval timing is *measured*, not arithmetic.
  2. LiveQueueSource — fed by the /ws/stream WebSocket; the server timestamps
     each client chunk on arrival. Used for genuinely live input (mic / typing).

Voice may be simulated from transcripts per the organizer deck — but the
simulation must be temporally honest: chunks arrive over time, and the
controller only ever sees what has arrived *so far*.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional


@dataclass
class StreamEvent:
    """One event in a live transcript stream."""
    kind: str          # "chunk" | "utterance_end"
    text: str          # cumulative partial text ("chunk"), full utterance ("utterance_end")
    t_s: float         # wall-clock seconds since stream start (monotonic)


async def play_utterance(
    utterance: str,
    words_per_chunk: int = 2,
    ms_per_chunk: int = 300,
) -> AsyncIterator[StreamEvent]:
    """
    Replay ``utterance`` as a live ASR-style stream with real pacing.

    The first chunk is emitted immediately at t=0.0s; each subsequent chunk
    follows after ``ms_per_chunk`` of real elapsed time. Timestamps are measured
    with time.monotonic(), never computed arithmetically. A final
    ``utterance_end`` event marks the end of speech.

    Example timeline for a 6-word utterance at 2 words / 300 ms per chunk:
        t=0.0s  chunk "w1 w2"
        t=0.3s  chunk "w1 w2 w3 w4"
        t=0.6s  chunk "w1 w2 w3 w4 w5 w6"
        t=0.6s  utterance_end
    """
    words = (utterance or "").split()
    t0 = time.monotonic()
    current: list[str] = []

    if not words:
        yield StreamEvent(kind="utterance_end", text="", t_s=0.0)
        return

    for i in range(0, len(words), words_per_chunk):
        if i > 0:
            await asyncio.sleep(ms_per_chunk / 1000.0)
        current.extend(words[i:i + words_per_chunk])
        yield StreamEvent(
            kind="chunk",
            text=" ".join(current),
            t_s=round(time.monotonic() - t0, 3),
        )

    yield StreamEvent(
        kind="utterance_end",
        text=" ".join(words),
        t_s=round(time.monotonic() - t0, 3),
    )


class LiveQueueSource:
    """
    A chunk source fed by live client input (WebSocket).

    The server timestamps each chunk on arrival, so t_s reflects the true
    arrival time of the user's speech/typing. Call ``push_text`` as partials
    arrive and ``finish`` when the user stops speaking. Iteration ends after
    the ``utterance_end`` event.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[Optional[StreamEvent]] = asyncio.Queue()
        self._t0: Optional[float] = None
        self._last_text: str = ""
        self._finished: bool = False

    def _now(self) -> float:
        if self._t0 is None:
            self._t0 = time.monotonic()
        return round(time.monotonic() - self._t0, 3)

    async def push_text(self, partial_text: str) -> None:
        """Record one cumulative partial transcript chunk (server-timestamped)."""
        if self._finished:
            return
        self._last_text = partial_text or ""
        await self._queue.put(StreamEvent(kind="chunk", text=self._last_text, t_s=self._now()))

    async def finish(self) -> None:
        """Signal end of utterance; iteration will yield utterance_end then stop."""
        if self._finished:
            return
        self._finished = True
        await self._queue.put(StreamEvent(kind="utterance_end", text=self._last_text, t_s=self._now()))

    def __aiter__(self) -> AsyncIterator[StreamEvent]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[StreamEvent]:
        while True:
            ev = await self._queue.get()
            if ev is None:  # defensive: producer gone
                yield StreamEvent(kind="utterance_end", text=self._last_text, t_s=self._now())
                return
            yield ev
            if ev.kind == "utterance_end":
                return


# ---------------------------------------------------------------------------
# Deprecated shim: the old arithmetic-only generator.
# Kept for import compatibility only; the pipeline no longer uses it.
# ---------------------------------------------------------------------------
def simulate_stream(utterance: str, words_per_chunk: int = 3, ms_per_chunk: int = 300):
    """DEPRECATED: yields chunks instantly with fake arithmetic timestamps.

    Do not use for streaming decisions or timing measurements. Use
    :func:`play_utterance` for honest paced replay.
    """
    import warnings
    warnings.warn(
        "simulate_stream is deprecated: it emits chunks instantly with "
        "arithmetic timestamps. Use play_utterance for real-time replay.",
        DeprecationWarning,
        stacklevel=2,
    )
    words = (utterance or "").split()
    current_text = ""
    for i in range(0, len(words), words_per_chunk):
        chunk_words = words[i:i + words_per_chunk]
        if current_text:
            current_text += " "
        current_text += " ".join(chunk_words)
        t_offset = (i // words_per_chunk + 1) * (ms_per_chunk / 1000.0)
        # Local stand-in so this module has no import-time dependency on telemetry.
        yield {"t_offset_s": round(t_offset, 2), "partial_text": current_text}
