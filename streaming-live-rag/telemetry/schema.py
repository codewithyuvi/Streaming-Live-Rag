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
    time_to_first_token: float = 0.0
    end_to_end: float = 0.0

class TokenCost(BaseModel):
    input: int = 0
    output: int = 0
    usd_estimate: float = 0.0

class TelemetryEvent(BaseModel):
    session_id: str
    turn_id: int
    controller_decision: Optional[ControllerDecision] = None
    retrieval_events: List[RetrievalEvent] = Field(default_factory=list)
    sub_queries: List[str] = Field(default_factory=list)
    answer: str = ""
    citations: List[str] = Field(default_factory=list)
    uncertainty: str = ""
    answer_version: int = 0
    latencies_ms: LatenciesMs = Field(default_factory=LatenciesMs)
    token_cost: TokenCost = Field(default_factory=TokenCost)
