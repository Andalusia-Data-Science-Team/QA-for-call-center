from __future__ import annotations
from typing import Literal, Optional
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

    # monitoring only — not part of the public schema
    _latency_ms: Optional[float] = None
    _prompt_tokens: Optional[int] = None
    _completion_tokens: Optional[int] = None

    @classmethod
    def error_result(cls, call_id: str, reason: str) -> "QAAnalysisResult":
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
        )


class BatchQAAnalysisResult(BaseModel):
    results: list[QAAnalysisResult]
    summary: dict
