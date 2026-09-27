from pydantic import BaseModel, Field

from typing import List, Optional

class StreamChunk(BaseModel):
    t_offset_s: float
    partial_text: str

class SessionState(BaseModel):
    session_id: str
    turns: list = Field(default_factory=list)
    current_answer_text: str = ""
    current_citations: List[str] = Field(default_factory=list)
    answer_version: int = 0

class ControllerDecision(BaseModel):
    trigger: str
    timestamp_s: float
    reason: str

class RetrievalEvent(BaseModel):
    timestamp_s: float
    query: str
    trigger: str

class LatenciesMs(BaseModel):
    retrieval: float = 0.0
    rerank: float = 0.0
    decompose: float = 0.0
    refinement: float = 0.0
    grounding: float = 0.0
    time_to_first_token: float = 0.0
    end_to_end: float = 0.0
    # Spec aliases from ARCHITECTURE_BRIEF §3 (kept optional for back-compat)
    controller: float = 0.0
    synthesis: float = 0.0
    retrieval_pipeline: float = 0.0

class TokenCost(BaseModel):
    input: int = 0
    output: int = 0
    usd_estimate: float = 0.0
    # Spec aliases from ARCHITECTURE_BRIEF §3
    fast_llm_tokens: int = 0
    synthesis_input_tokens: int = 0
    synthesis_output_tokens: int = 0

class TelemetryEvent(BaseModel):
    session_id: str
    turn_id: int
    controller_decisions: List[ControllerDecision] = Field(default_factory=list)
    controller_decision: Optional[ControllerDecision] = None  # backwards compatibility
    refinement_type: str = "NEW_TOPIC"  # NEW_TOPIC | LATE_DETAIL | PRESENTATION_ONLY
    retrieval_required: bool = True
    retrieval_skip_reason: str = ""
    retrieval_events: List[RetrievalEvent] = Field(default_factory=list)
    sub_queries: List[str] = Field(default_factory=list)
    answer: str = ""
    citations: List[str] = Field(default_factory=list)
    uncertainty: str = ""
    grounding_score: float = 0.0
    grounding_report: Optional[dict] = None
    answer_version: int = 0
    degraded: bool = False
    latencies_ms: LatenciesMs = Field(default_factory=LatenciesMs)
    token_cost: TokenCost = Field(default_factory=TokenCost)
    # Honest streaming anchors (wall-clock seconds since stream start).
    # utterance_end_s: when the user finished speaking; provisional_fired_s:
    # when the provisional retrieval task was actually created (None if never).
    # G2 "retrieval commenced prior to final transcript completion" is
    # provisional_fired_s < utterance_end_s — both measured, never arithmetic.
    utterance_end_s: float = 0.0
    provisional_fired_s: Optional[float] = None
    thought_process: str = ""
