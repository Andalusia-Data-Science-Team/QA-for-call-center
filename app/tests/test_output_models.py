from models.output import QAAnalysisResult

def test_qaanalysisresult_preserves_conversation_link():
    result = QAAnalysisResult(
        call_id="call-001",
        agent_name="Agent One",
        overall_assessment="pass",
        assessment_reasoning="Looks good.",
        compliance_flags=[],
        agent_performance={
            "professionalism_score": 0.9,
            "Agent Classification": "A",
            "Profiling Comment": None,
            "strengths": [],
            "improvements": [],
        },
        escalation_required=False,
        conversation_link="https://example.com/conversation/001",
    )

    assert result.conversation_link == "https://example.com/conversation/001"


def test_error_result_accepts_conversation_link():
    result = QAAnalysisResult.error_result(
        "call-002",
        "boom",
        conversation_link="https://example.com/conversation/002",
    )

    assert result.conversation_link == "https://example.com/conversation/002"
