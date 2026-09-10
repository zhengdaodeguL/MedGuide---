from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict


Intent = Literal["disease", "drug", "exam", "department", "faq", "structured", "unknown"]
RiskLevel = Literal["low", "watch", "high"]


class PatientProfile(TypedDict, total=False):
    age: int
    sex: str
    duration: str
    symptoms: list[str]
    associated_symptoms: list[str]
    history: str
    chief_complaint: str


class WorkflowState(TypedDict, total=False):
    session_id: str
    # Private ownership marker.  It is never exposed by ``SessionStore.public``.
    owner_id: str
    user_message: str
    normalized_message: str
    profile: PatientProfile
    intent: Intent
    risk_level: RiskLevel
    risk_assessed: bool
    risk_flags: list[str]
    risk_history: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    structured_result: dict[str, Any] | None
    risk_advice: str
    rewritten_query: str
    response: str
    summary: str
    next_question: str | None
    events: list[dict[str, Any]]
    confidence: float
    turn_count: int
    latency_ms: float
    mode: str
    workflow_engine: str
    updated_at: str
    answer_source: str
    response_id: str
    response_ids: list[str]
    request_id: str
    request_results: dict[str, dict[str, Any]]
    request_fingerprints: dict[str, str]
    _request_fingerprint: str
    feedback_by_message: dict[str, dict[str, Any]]
    safety_blocked: bool
    safety_block_reason: str
    hallucination_label: bool
    _version: int


@dataclass(frozen=True)
class KnowledgeDocument:
    id: str
    title: str
    category: Intent
    source: str
    updated_at: str
    text: str
    source_url: str = ""
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class Citation:
    id: str
    title: str
    category: str
    source: str
    updated_at: str
    snippet: str
    score: float
    retrieval: str
    source_url: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "category": self.category,
            "source": self.source,
            "updated_at": self.updated_at,
            "snippet": self.snippet,
            "score": round(self.score, 4),
            "retrieval": self.retrieval,
            "source_url": self.source_url,
        }


@dataclass(frozen=True)
class RiskAssessment:
    level: RiskLevel
    flags: tuple[str, ...]
    advice: str


@dataclass(frozen=True)
class SQLQueryResult:
    intent: str
    sql: str
    params: tuple[Any, ...]
    columns: tuple[str, ...]
    rows: tuple[dict[str, Any], ...]
    blocked: bool = False
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "sql": self.sql,
            "columns": list(self.columns),
            "rows": [dict(row) for row in self.rows],
            "blocked": self.blocked,
            "reason": self.reason,
        }
