"""One regulation item should appear once per call, including stored results."""

from app.services import regulation_catalog


def _missing_name(**overrides):
    return {
        "type": "NC",
        "attribute": "Professional Personalization",
        "comment": "Didn’t ask for customer name",
        "description": "The agent requested an ID but not the customer's name.",
        "transcript_excerpt": "ممكن رقم الهويه ؟",
        **overrides,
    }


def test_duplicate_policy_item_keeps_one_finding_despite_different_evidence():
    flags = [_missing_name(), _missing_name(
        description="NC_001: The agent did not ask the name.",
        transcript_excerpt="تم الحجز باسم عادل",
    )]

    actual = regulation_catalog.deduplicate_regulation_flags(flags)

    assert len(actual) == 1
    assert actual[0]["transcript_excerpt"] == "ممكن رقم الهويه ؟"


def test_distinct_policy_items_keep_separate_findings_with_same_evidence():
    flags = [_missing_name(), {
        "type": "NC", "attribute": "Voice tone", "comment": "Unclear Voice Tone",
        "description": "The response was unclear.",
        "transcript_excerpt": "ممكن رقم الهويه ؟",
    }]

    actual = regulation_catalog.deduplicate_regulation_flags(flags)

    assert len(actual) == 2


def test_reviewed_duplicate_takes_precedence_for_stored_results():
    flags = [_missing_name(qa_reviewed=0), _missing_name(
        qa_reviewed=1,
        qa_review_comment="Confirmed by QA",
        transcript_excerpt="تم الحجز باسم عادل",
    )]

    actual = regulation_catalog.deduplicate_regulation_flags(flags)

    assert len(actual) == 1
    assert actual[0]["qa_reviewed"] == 1
    assert actual[0]["qa_review_comment"] == "Confirmed by QA"
