import pytest
from sqlalchemy.exc import ProgrammingError

from app.models.input import CallTranscript
from app.models.output import QAAnalysisResult
from app.services import sql_helpers


def test_call_level_params_maps_new_schema_fields():
    call = CallTranscript(
        call_id="call-001",
        agent_name="Agent One",
        agent_email="agent@example.com",
        Patient_Phone="966500000000",
        call_date="2026-07-15",
        call_duration_seconds=120,
        department="Scheduling",
        transcript="Hello world",
        business_unit="LCH",
    )
    result = QAAnalysisResult(
        call_id=call.call_id,
        agent_name=call.agent_name,
        overall_assessment="needs_review",
        assessment_reasoning="Needs review",
        compliance_flags=[],
        agent_performance={
            "professionalism_score": 0.8,
            "Agent Classification": "C",
            "Profiling Comment": "Poor Process",
            "strengths": ["Clear"],
            "improvements": ["Faster"],
        },
        escalation_required=True,
        escalation_reason="Critical issue",
        escalated=False,
        qa_reviewed=False,
        qa_review_comment=None,
        business_unit="LCH",
        agent_email_address="agent@example.com",
    )

    payload = sql_helpers._call_level_params(
        result,
        call,
        agent_id="42",
        channel_name="WhatsApp",
        weighted=0.75,
        created_at="2026-07-15 00:00:00",
    )

    assert payload["Agent_Classification"] == "C"
    assert payload["Profiling_Comment"] == "Poor Process"
    assert payload["BU"] == "LCH"
    assert payload["Agent_Email_Address"] == "agent@example.com"
    assert payload["Escalated"] == 0
    assert payload["QA_Reviewed"] == 0
    assert payload["QA_Review_Comment"] is None


def test_update_escalation_row_wraps_permission_errors(monkeypatch):
    class FakeResult:
        def __init__(self, row):
            self._row = row

        def fetchone(self):
            return self._row

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, statement, params=None):
            sql_text = str(statement)
            if "SELECT TOP 1" in sql_text:
                return FakeResult((244,))
            raise ProgrammingError(
                statement=sql_text,
                params=params,
                orig=Exception("[42000] [Microsoft][ODBC Driver 17 for SQL Server][SQL Server]The UPDATE permission was denied on the object 'Call_QA_Results'"),
            )

    class FakeEngine:
        def begin(self):
            return FakeConnection()

    monkeypatch.setattr(sql_helpers, "_build_engine", lambda: FakeEngine())

    with pytest.raises(sql_helpers.DatabaseWritePermissionError):
        sql_helpers.update_escalation_row("call-001", qa_reviewed=True, qa_review_comment="done")
