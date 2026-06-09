from __future__ import annotations

import operator
from typing import Annotated, Any, Optional
from typing_extensions import TypedDict

from app.models.input import CallTranscript
from app.models.output import QAAnalysisResult


class AgentState(TypedDict, total=False):
    call: CallTranscript
    behavioral_criteria: str
    compliance_pillars: str
    script_templates: str
    scoring_weights: str
    system_prompt: str
    user_prompt: str
    raw_llm_text: str
    usage: dict[str, Any]
    parsed_data: dict[str, Any]
    result: Optional[QAAnalysisResult]
    error: Optional[str]
    error_node: Optional[str]
    node_trace: Annotated[list[str], operator.add]
