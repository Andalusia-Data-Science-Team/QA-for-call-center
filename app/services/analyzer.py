from __future__ import annotations

import json
import logging
import re

from pydantic import ValidationError

from app.models.input import CallTranscript
from app.models.output import QAAnalysisResult
from app.prompts.qa_prompt import SYSTEM_PROMPT, build_user_prompt
from app.services.llm_client import LLMClient, LLMError

logger = logging.getLogger(__name__)


class AnalysisError(Exception):
    """Raised when the LLM response cannot be parsed into a valid QAAnalysisResult."""


class CallAnalyzer:
    """
    CallAnalyzer — orchestrates the full analysis pipeline.

    Responsibilities:
    1. Build the prompt from the call transcript
    2. Call the LLM via the LLMClient abstraction
    3. Parse and validate the response with Pydantic
    4. Log observability data (latency, token usage, prompt, response)
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    async def analyze(self, call: CallTranscript) -> QAAnalysisResult:
        """
        Full analysis pipeline for a single call transcript.
        Returns a validated QAAnalysisResult.
        """
        user_prompt = build_user_prompt(call)

        # Log the outgoing prompt at DEBUG level (avoids log spam in production)
        logger.debug(
            "PROMPT | call_id=%s | system_len=%d user_len=%d",
            call.call_id,
            len(SYSTEM_PROMPT),
            len(user_prompt),
        )

        raw_text, usage = await self.llm.complete(SYSTEM_PROMPT, user_prompt)

        logger.debug(
            "RESPONSE | call_id=%s | latency=%.0fms tokens_in=%s tokens_out=%s | raw=%s",
            call.call_id,
            usage.get("latency_ms", 0),
            usage.get("input_tokens") or usage.get("prompt_tokens"),
            usage.get("output_tokens") or usage.get("completion_tokens"),
            raw_text[:500],  # truncate for log safety
        )

        result = self._parse_response(raw_text, call.call_id)
        return result

    def _parse_response(self, raw_text: str, call_id: str) -> QAAnalysisResult:
        """
        Parse raw LLM text to match QAAnalysisResult.
        """
        clean = self._strip_markdown_fences(raw_text)

        try:
            data: dict = json.loads(clean)
        except json.JSONDecodeError as exc:
            logger.error(
                "JSON parse failed | call_id=%s | raw_snippet=%s | error=%s",
                call_id,
                raw_text[:300],
                exc,
            )
            raise AnalysisError(f"LLM returned invalid JSON: {exc}") from exc

        # Always trust the call_id from the request, not from the LLM
        data["call_id"] = call_id

        try:
            result = QAAnalysisResult.model_validate(data)
        except ValidationError as exc:
            logger.error(
                "Pydantic validation failed | call_id=%s | errors=%s",
                call_id,
                exc.errors(),
            )
            raise AnalysisError(f"LLM response failed schema validation: {exc}") from exc

        # Integrity check: escalation_required must be consistent with overall_assessment
        if result.escalation_required and result.overall_assessment != "escalate":
            logger.warning(
                "Inconsistency: escalation_required=True but assessment=%s — correcting to 'escalate' | call_id=%s",
                result.overall_assessment,
                call_id,
            )
            result = result.model_copy(update={"overall_assessment": "escalate"})

        if result.overall_assessment == "escalate" and not result.escalation_required:
            logger.warning(
                "Inconsistency: assessment='escalate' but escalation_required=False — correcting | call_id=%s",
                call_id,
            )
            result = result.model_copy(update={"escalation_required": True})

        return result

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        """Remove ```json ... ``` wrappers that some LLMs insert despite instructions."""
        text = text.strip()
        # Match ```json\n...\n``` or just ```\n...\n```
        pattern = r"^```(?:json)?\s*\n?(.*?)\n?```$"
        match = re.match(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text
