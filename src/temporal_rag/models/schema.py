"""Esquemas de dados bitemporais para CRM e Benchmark."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class EventType(str, Enum):
    PLAN_CHANGE = "plan_change"
    TICKET = "ticket"
    EMAIL = "email"
    CALL = "call"
    INVOICE = "invoice"
    NPS = "nps"
    NOTE = "note"
    CONTACT_UPDATE = "contact_update"


class Customer(BaseModel):
    id: str
    name: str
    email: str
    phone: str
    company: str
    document_cpf: str
    created_at: datetime
    initial_plan: str
    current_plan: Optional[str] = None
    address: Optional[str] = None


class TimelineEvent(BaseModel):
    id: str
    customer_id: str
    event_type: EventType
    title: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Bitemporal fields
    # valid_from / valid_to: Real-world validity interval
    valid_from: datetime
    valid_to: Optional[datetime] = None  # None indicates indefinitely valid until superseded
    
    # recorded_at: When the system ingested or saved this record
    recorded_at: datetime

    def is_valid_as_of(self, target_date: datetime) -> bool:
        """Verifica se o evento era válido no mundo real na data informada."""
        if target_date < self.valid_from:
            return False
        if self.valid_to is not None and target_date >= self.valid_to:
            return False
        return True

    def was_recorded_by(self, target_date: datetime) -> bool:
        """Evita data leakage: verifica se o evento já estava gravado no CRM na data."""
        return self.recorded_at <= target_date


class BenchmarkCategory(str, Enum):
    AS_OF_STATE = "as_of_state"              # "Qual era o plano em 2024-03-10?"
    CHANGE_DETECTION = "change_detection"    # "Quando ocorreu a mudança de plano?"
    TEMPORAL_ORDERING = "temporal_ordering"  # "Qual foi o primeiro evento após X?"
    AGGREGATION = "aggregation"              # "Quantas faturas foram pagas em Q1?"
    UNANSWERABLE = "unanswerable"            # Perguntas fora do período ou sem dados
    CONTRADICTION = "contradiction"          # Fatos conflitantes ao longo do tempo


class BenchmarkSample(BaseModel):
    question_id: str
    customer_id: str
    category: BenchmarkCategory
    question: str
    as_of: datetime
    expected_answer: str
    expected_event_ids: List[str]
    description: Optional[str] = None


class RetrievalResult(BaseModel):
    event: TimelineEvent
    score: float
    temporal_penalty: float = 0.0
    final_score: float = 0.0


class AgentResponse(BaseModel):
    query: str
    customer_id: str
    as_of: Optional[datetime] = None
    answer: str
    confidence: float
    cited_event_ids: List[str] = Field(default_factory=list)
    retrieved_events: List[TimelineEvent] = Field(default_factory=list)
    abstained: bool = False
    reasoning_trace: List[str] = Field(default_factory=list)
