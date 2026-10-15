"""Credit exhaustion must stop retries without discarding scheduled work."""

import asyncio

import pytest

from app.services import evaluation_service
from app.services.llm_client import LLMClient, LLMQuotaError
from app.models.input import CallTranscript
from app.models.output import QAAnalysisResult


def test_llm_credit_exhaustion_is_not_retried(monkeypatch):
    client = LLMClient("openrouter", "test-model")
    calls = 0

    async def exhausted(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise LLMQuotaError("OpenRouter insufficient credits (HTTP 402)")

    monkeypatch.setattr(client, "_call_with_mlflow_trace", exhausted)
    with pytest.raises(LLMQuotaError, match="insufficient credits"):
        asyncio.run(client.complete("system", "user"))
    assert calls == 1


def test_credit_exhaustion_is_a_recoverable_quota_failure():
    assert evaluation_service.classify_failure(
        RuntimeError("LLM call failed: OpenRouter insufficient credits (HTTP 402)")
    ) == "quota"


def test_pipeline_error_result_keeps_the_quota_classification():
    call = CallTranscript(
        call_id="conv-1", agent_name="A", Patient_Phone="123456789",
        call_date="2026-10-15", call_duration_seconds=60,
        department="Helpdesk", transcript="Agent: hello",
    )

    class ExhaustedAnalyzer:
        async def analyze(self, call, *, job_id=None):
            return QAAnalysisResult.error_result(
                call.call_id,
                "infer_service_evaluation: LLM call failed: "
                "OpenRouter insufficient credits (HTTP 402)",
            )

    outcome = asyncio.run(evaluation_service.evaluate_conversation(
        ExhaustedAnalyzer(), call, job_id="job-1",
    ))
    assert outcome.persisted is False
    assert outcome.failure_class == "quota"
