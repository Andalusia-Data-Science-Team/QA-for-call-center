from __future__ import annotations
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field

OverallAssessment = Literal["pass", "needs_review", "escalate", "error"]
FlagType = Literal["C2C", "C2B", "C2Com", "NC"]
Severity = Literal["critical", "moderate", "minor", "positive"]
AgentClassification = Literal["A", "B", "C", "D"]
ProfilingComment = Literal[
    "Poor Knowledge",
    "Poor System",
    "Poor Process",
    "Poor Report",
    "Poor Selling Skills",
    "Poor Behavior",
    "Poor Soft Skills",
]


class ComplianceFlag(BaseModel):
    type: FlagType
    severity: Severity
    description: str = Field(..., description="1–2 sentences describing the finding.")
    transcript_excerpt: str = Field(..., description="Verbatim excerpt from the transcript.")


class AgentPerformance(BaseModel):
    professionalism_score: float = Field(..., ge=0.0, le=1.0)
    accuracy_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    resolution_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    agent_classification: AgentClassification = Field(
        ...,
        alias="Agent Classification",
        description="A=No violations, B=<2 NC, C=1 Critical or >2 NC, D=>1 Critical",
    )
    profiling_comment: Optional[ProfilingComment] = Field(
        None,
        alias="Profiling Comment",
        description="Root-cause profiling label if performance issues exist. Optional.",
    )
    strengths: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class QAAnalysisResult(BaseModel):
    call_id: str
    agent_name: str
    overall_assessment: OverallAssessment
    assessment_reasoning: str = Field(..., description="2–4 sentences explaining the assessment.")
    compliance_flags: list[ComplianceFlag]
    agent_performance: AgentPerformance
    escalation_required: bool
    escalation_reason: Optional[str] = None
    conversation_link: Optional[str] = None
    business_unit: Optional[str] = None
    agent_email_address: Optional[str] = None
    supervisor_name: Optional[str] = None
    supervisor_email_address: Optional[str] = None
    coaching_status: Optional[str] = None
    escalated: bool = False
    qa_reviewed: bool = False
    qa_review_comment: Optional[str] = None
    # Expose the masked deterministic outcomes to API consumers alongside
    # compliance flags; no raw IBAN/account number is included. Bank and
    # location are independent checks (separate graph nodes) — each is None
    # when its own request type wasn't present in the call.
    bank_validation: Optional[dict[str, Any]] = None
    location_validation: Optional[dict[str, Any]] = None
    # Doctor validation is two independent checks (deterministic factual
    # info + separate LLM-based recommendation-suitability) — same pattern
    # as bank/location being independent of each other. Either is None
    # when its own applicability condition wasn't met for this call.
    doctor_validation: Optional[dict[str, Any]] = None
    doctor_scope_validation: Optional[dict[str, Any]] = None

    # monitoring only — not part of the public schema
    _latency_ms: Optional[float] = None
    _prompt_tokens: Optional[int] = None
    _completion_tokens: Optional[int] = None

    @classmethod
    def error_result(cls, call_id: str, reason: str, conversation_link: Optional[str] = None) -> "QAAnalysisResult":
        """Minimal safe result returned when analysis fails in a batch."""
        return cls(
            call_id=call_id,
            agent_name="Unknown",
            overall_assessment="error",
            assessment_reasoning=f"Analysis could not be completed: {reason}",
            compliance_flags=[],
            agent_performance=AgentPerformance(**{
                "Agent Classification": "D",
                "Profiling Comment": None,
                "professionalism_score": 0.0,
                "strengths": [],
                "improvements": [],
            }),
            escalation_required=False,
            escalation_reason=None,
            conversation_link=conversation_link,
            business_unit=None,
            agent_email_address=None,
            supervisor_name=None,
            supervisor_email_address=None,
            coaching_status=None,
            escalated=False,
            qa_reviewed=False,
            qa_review_comment=None,
            bank_validation=None,
            location_validation=None,
            doctor_validation=None,
            doctor_scope_validation=None,
        )


class BatchQAAnalysisResult(BaseModel):
    results: list[QAAnalysisResult]
    summary: dict
