from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field

OverallAssessment = Literal["pass", "needs_review", "escalate", "error"]
FlagType = Literal[
    "C2C",
    "C2B",
    "C2Com",
    "NC"
]
Severity = Literal["critical", "moderate", "minor", "positive"]

class ComplianceFlag(BaseModel):
    type: FlagType
    severity: Severity
    description: str = Field(..., description="1–2 sentences describing the finding.")
    transcript_excerpt: str = Field(..., description="Verbatim excerpt from the transcript.")


class AgentPerformance(BaseModel):
    professionalism_score: float = Field(..., ge=0.0, le=1.0)
    accuracy_score: float = Field(..., ge=0.0, le=1.0)
    resolution_score: float = Field(..., ge=0.0, le=1.0)
    strengths: list[str] = Field(..., min_length=1, max_length=3)
    improvements: list[str] = Field(..., min_length=1, max_length=3)

class QAAnalysisResult(BaseModel):
    call_id: str
    overall_assessment: OverallAssessment
    assessment_reasoning: str = Field(..., description="2–4 sentences explaining the assessment.")
    compliance_flags: list[ComplianceFlag]
    agent_performance: AgentPerformance
    escalation_required: bool
    escalation_reason: Optional[str] = None

    #not part of the spec but very useful for monitoring
    _latency_ms: Optional[float] = None
    _prompt_tokens: Optional[int] = None
    _completion_tokens: Optional[int] = None

    @classmethod
    def error_result(cls, call_id: str, reason: str) -> "QAAnalysisResult":
        """Minimal safe result returned when analysis fails in a batch."""
        return cls(
            call_id=call_id,
            overall_assessment="error",
            assessment_reasoning=f"Analysis could not be completed: {reason}",
            compliance_flags=[],
            agent_performance=AgentPerformance(
                professionalism_score=0.0,
                accuracy_score=0.0,
                resolution_score=0.0,
                strengths=["Unable to assess — analysis failed"],
                improvements=["Unable to assess — analysis failed"],
            ),
            escalation_required=False,
            escalation_reason=None,
        )


# Bonus Feature pidantic model

class BatchQAAnalysisResult(BaseModel):
    results: list[QAAnalysisResult]
    summary: dict
