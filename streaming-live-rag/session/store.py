"""
session/store.py — Ephemeral Session Store with Clean Commit Semantics (C4 / C7 / M1 / Appendix B.3).

Tracks:
- Complete audit trail of turns
- Substantive query and answer lineage
- Incremental citations union on LATE_DETAIL
- Non-corrupting PRESENTATION_ONLY audit tracking
"""

import threading as _threading
import time as _time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TurnRecord:
    """Record of a single turn within a session."""
    turn_id: int
    utterance: str
    answer: str
    citations: list[str] = field(default_factory=list)
    sub_queries: list[str] = field(default_factory=list)
    refinement_type: str = "NEW_TOPIC"  # NEW_TOPIC | LATE_DETAIL | PRESENTATION_ONLY
    effective_query: str = ""

    def to_dict(self) -> dict:
        return {
            "turn_id": self.turn_id,
            "utterance": self.utterance,
            "answer": self.answer,
            "citations": self.citations,
            "sub_queries": self.sub_queries,
            "refinement_type": self.refinement_type,
            "effective_query": self.effective_query,
        }


@dataclass
class Session:
    """Ephemeral session state for one conversation."""
    session_id: str
    turns: list[TurnRecord] = field(default_factory=list)  # full audit trail (every turn, any kind)
    current_query: str = ""                                # the substantive question being refined
    current_answer: str = ""                               # last substantive answer -- never overwritten by chit-chat
    current_citations: list[str] = field(default_factory=list)
    answer_version: int = 0

    def commit(
        self,
        kind: str,
        utterance: str,
        answer: str,
        cited_tags: list[str],
        effective_query: str = "",
        sub_queries: list[str] | None = None
    ) -> TurnRecord:
        """
        Commits a turn with strict lifecycle semantics:
        - NEW_TOPIC: Starts new lineage, sets version = 1, replaces current answer and citations.
        - LATE_DETAIL: Updates answer/query, unions citations, increments version += 1.
        - PRESENTATION_ONLY: Records turn in audit trail only. Current state and version stay untouched.
        """
        turn_id = len(self.turns) + 1
        record = TurnRecord(
            turn_id=turn_id,
            utterance=utterance,
            answer=answer,
            citations=list(cited_tags),
            sub_queries=sub_queries or [],
            refinement_type=kind,
            effective_query=effective_query or utterance,
        )
        self.turns.append(record)
        # Cap audit trail to last 50 turns to prevent memory growth under extended use
        if len(self.turns) > 50:
            self.turns = self.turns[-50:]

        if kind == "NEW_TOPIC":
            self.current_query = effective_query or utterance
            self.current_answer = answer
            self.current_citations = list(dict.fromkeys(cited_tags))
            self.answer_version = 1
        elif kind == "LATE_DETAIL":
            self.current_query = effective_query or self.current_query
            self.current_answer = answer
            self.current_citations = list(dict.fromkeys(self.current_citations + list(cited_tags)))
            self.answer_version += 1
        # PRESENTATION_ONLY: audit trail only. current_* and answer_version stay untouched.

        return record

    def add_turn(self, turn: TurnRecord):
        """Backwards-compatible wrapper calling commit."""
        self.commit(
            kind=turn.refinement_type,
            utterance=turn.utterance,
            answer=turn.answer,
            cited_tags=turn.citations,
            effective_query=turn.effective_query,
            sub_queries=turn.sub_queries,
        )

    def get_last_turn(self) -> Optional[TurnRecord]:
        """Returns the most recent turn record, or None if no turns yet."""
        return self.turns[-1] if self.turns else None

    def get_history_context(self, max_turns: int = 5, max_chars_per_answer: int = 1500) -> str:
        """
        Returns a formatted string of recent substantive conversation history
        for injection into the synthesis prompt.
        Answers are truncated at a sentence/word boundary so one long turn
        cannot blow up the prompt or smuggle prompt-injection payloads.
        """
        recent = self.turns[-max_turns:]
        if not recent:
            return ""

        def _truncate(text: str, limit: int) -> str:
            if len(text) <= limit:
                return text
            cut = text[:limit]
            # Prefer a sentence boundary, else a word boundary.
            for sep in (". ", "? ", "! ", "\n"):
                idx = cut.rfind(sep)
                if idx > limit // 2:
                    return cut[: idx + 1].strip()
            space = cut.rfind(" ")
            return (cut[:space] if space > limit // 2 else cut).strip() + " …"

        lines = []
        for t in recent:
            lines.append(f"[Turn {t.turn_id}] User: {t.utterance[:500]}")
            # Do not truncate mid-sentence; include full answer or reasonable paragraph
            lines.append(f"[Turn {t.turn_id}] Assistant: {_truncate(t.answer, max_chars_per_answer)}")
            if t.citations:
                lines.append(f"[Turn {t.turn_id}] Citations: {', '.join(t.citations[:8])}")
        return "\n".join(lines)

    def get_all_prior_citations(self) -> list[str]:
        """Returns all citations used across prior turns."""
        all_cites = []
        for t in self.turns:
            all_cites.extend(t.citations)
        return list(dict.fromkeys(all_cites))


# ---------------------------------------------------------------------------
# Global ephemeral session store (thread-safe, TTL-evicted)
# ---------------------------------------------------------------------------

_sessions: dict[str, Session] = {}
_sessions_lock = _threading.Lock()
_sessions_last_access: dict[str, float] = {}
_SESSION_TTL_S = 3600.0  # evict idle sessions after 1h
_MAX_SESSIONS = 1000
# Per-session locks guard Session.commit / get_history_context so concurrent
# /turn requests for the same session_id cannot interleave turns, citations,
# or answer_version updates.
_session_locks: dict[str, _threading.Lock] = {}


def _evict_expired_locked(now: float | None = None) -> None:
    now = now if now is not None else _time.time()
    expired = [k for k, ts in _sessions_last_access.items() if now - ts > _SESSION_TTL_S]
    for k in expired:
        if k in _session_locks and _session_locks[k].locked():
            continue  # Never evict a session whose lock is currently acquired by an active turn
        _sessions.pop(k, None)
        _sessions_last_access.pop(k, None)
        _session_locks.pop(k, None)
    # Hard cap: drop oldest if over limit
    if len(_sessions) > _MAX_SESSIONS:
        oldest = sorted(_sessions_last_access.items(), key=lambda kv: kv[1])
        for k, _ in oldest:
            if len(_sessions) <= _MAX_SESSIONS:
                break
            if k in _session_locks and _session_locks[k].locked():
                continue
            _sessions.pop(k, None)
            _sessions_last_access.pop(k, None)
            _session_locks.pop(k, None)


def get_session_lock(session_id: str) -> "_threading.Lock":
    """Return the mutex serializing mutations for one session."""
    with _sessions_lock:
        lock = _session_locks.get(session_id)
        if lock is None:
            lock = _threading.Lock()
            _session_locks[session_id] = lock
        return lock


def get_or_create_session(session_id: str) -> Session:
    """Get existing session or create a new one."""
    with _sessions_lock:
        _evict_expired_locked()
        if session_id not in _sessions:
            _sessions[session_id] = Session(session_id=session_id)
        _sessions_last_access[session_id] = _time.time()
        return _sessions[session_id]


def delete_session(session_id: str) -> bool:
    """Delete a session. Returns True if it existed."""
    with _sessions_lock:
        _sessions_last_access.pop(session_id, None)
        _session_locks.pop(session_id, None)
        return _sessions.pop(session_id, None) is not None


def list_sessions() -> list[str]:
    """List all active session IDs."""
    with _sessions_lock:
        return list(_sessions.keys())


def reset_store():
    """Clears all sessions (useful for tests)."""
    with _sessions_lock:
        _sessions.clear()
        _sessions_last_access.clear()
        _session_locks.clear()
