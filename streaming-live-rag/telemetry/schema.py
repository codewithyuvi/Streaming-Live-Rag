try:
    from pydantic import BaseModel, Field
except ImportError:
    class BaseModel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

        def model_dump(self):
            def _to_dict(obj):
                if isinstance(obj, BaseModel):
                    return {k: _to_dict(v) for k, v in obj.__dict__.items()}
                elif isinstance(obj, list):
                    return [_to_dict(x) for x in obj]
                elif isinstance(obj, dict):
                    return {k: _to_dict(v) for k, v in obj.items()}
                return obj
            return _to_dict(self)

        def model_dump_json(self):
            import json
            return json.dumps(self.model_dump(), default=str)

    def Field(default=None, default_factory=None, **kwargs):
        if default_factory is not None:
            return default_factory()
        return default

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

class TokenCost(BaseModel):
    input: int = 0
    output: int = 0
    usd_estimate: float = 0.0

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
