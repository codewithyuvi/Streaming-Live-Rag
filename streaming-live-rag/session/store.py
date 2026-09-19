"""
Phase 5 — Ephemeral Session Store

In-memory session state keyed by session_id. Each session tracks:
- Conversation history (turns with queries, answers, citations)
- Current accumulated context for refinement detection
- Answer version counter for incremental updates

Design decisions:
- Pure dict-based in-memory store — no persistence across restarts.
- Session-bound: memory lives and dies with one conversation session.
- Thread-safe via a simple dict (FastAPI runs in a single event loop).
- No cross-session leakage (hard requirement from the hackathon rules).
"""

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


@dataclass
class Session:
    """Ephemeral session state for one conversation."""
    session_id: str
    turns: list[TurnRecord] = field(default_factory=list)
    answer_version: int = 0

    def add_turn(self, turn: TurnRecord):
        self.turns.append(turn)
        self.answer_version += 1

    def get_last_turn(self) -> Optional[TurnRecord]:
        return self.turns[-1] if self.turns else None

    def get_history_context(self, max_turns: int = 5) -> str:
        """
        Returns a formatted string of recent conversation history
        for injection into the synthesis prompt.
        """
        recent = self.turns[-max_turns:]
        if not recent:
            return ""

        lines = []
        for t in recent:
            lines.append(f"[Turn {t.turn_id}] User: {t.utterance}")
            lines.append(f"[Turn {t.turn_id}] Assistant: {t.answer[:200]}...")
            if t.citations:
                lines.append(f"[Turn {t.turn_id}] Citations: {', '.join(t.citations)}")
        return "\n".join(lines)

    def get_all_prior_citations(self) -> list[str]:
        """Returns all citations used across prior turns."""
        all_cites = []
        for t in self.turns:
            all_cites.extend(t.citations)
        return list(set(all_cites))


# ---------------------------------------------------------------------------
# Global session store (in-memory, ephemeral)
# ---------------------------------------------------------------------------

_sessions: dict[str, Session] = {}


def get_or_create_session(session_id: str) -> Session:
    """Get existing session or create a new one."""
    if session_id not in _sessions:
        _sessions[session_id] = Session(session_id=session_id)
    return _sessions[session_id]


def delete_session(session_id: str) -> bool:
    """Delete a session. Returns True if it existed."""
    return _sessions.pop(session_id, None) is not None


def list_sessions() -> list[str]:
    """List all active session IDs."""
    return list(_sessions.keys())
