from __future__ import annotations

import operator
from typing import Annotated, Any, Optional
from typing_extensions import TypedDict

from app.models.input import CallTranscript
from app.models.output import QAAnalysisResult


class AgentState(TypedDict, total=False):
    # ── Input ─────────────────────────────────────────────────────────────
    call: CallTranscript

    # ── Criteria blocks (loaded in parallel) ──────────────────────────────
    behavioral_criteria: str
    compliance_pillars: str
    script_templates: str
    scoring_weights: str

    # ── Per-evaluation LLM sub-results (raw JSON dicts) ───────────────────
    # Each focused inference node stores its parsed output here so the
    # aggregate_results node can merge them into a single QAAnalysisResult.
    behavioral_eval: dict[str, Any]   # from infer_behavioral_evaluation
    compliance_eval: dict[str, Any]   # from infer_compliance_evaluation
    script_eval: dict[str, Any]       # from infer_script_matching
    scoring_eval: dict[str, Any]      # from infer_overall_scoring

    # ── Per-node usage tracking (list so all 4 LLM calls are preserved) ───
    usage_list: Annotated[list[dict[str, Any]], operator.add]

    # ── Final merged result ───────────────────────────────────────────────
    parsed_data: dict[str, Any]
    result: Optional[QAAnalysisResult]

    # ── Error handling ────────────────────────────────────────────────────
    error: Optional[str]
    error_node: Optional[str]

    # ── Execution trace ───────────────────────────────────────────────────
    node_trace: Annotated[list[str], operator.add]
