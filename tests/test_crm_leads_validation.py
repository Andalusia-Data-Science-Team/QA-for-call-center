import asyncio
import json
from types import SimpleNamespace

from app.agent import nodes as agent_nodes
from app.agent.agent import _build_token_usage_logger, _summarize_chat_usage
from app.prompts.qa_prompt import build_crm_lead_validation_prompt
from app.services.llm_client import LLMClient
from app.service_hub import crm_leads_validation as leads


def test_focused_llm_call_regenerates_invalid_json(monkeypatch):
    class FakeLLMClient:
        def __init__(self):
            self.prompts = []

        async def complete(self, system_prompt, user_prompt, max_tokens=None):
            self.prompts.append((user_prompt, max_tokens))
            if len(self.prompts) == 1:
                return (
                    "{\"assessment_reasoning\": \"unterminated",
                    {"prompt_tokens": 100, "completion_tokens": 2048, "cost_usd": 0.001, "provider": "openrouter", "model": "test-model", "finish_reason": "length"},
                )
            return (
                "{\"overall_assessment\": \"pass\"}",
                {"prompt_tokens": 50, "completion_tokens": 10, "cost_usd": 0.002, "provider": "openrouter", "model": "test-model", "finish_reason": "stop"},
            )

    monkeypatch.setattr(agent_nodes.settings, "LLM_JSON_PARSE_RETRIES", 1)
    client = FakeLLMClient()
    data, error = asyncio.run(
        agent_nodes._focused_llm_call(
            "infer_overall_scoring",
            "call-1",
            "return JSON",
            client,
            {},
            max_tokens=4096,
        )
    )

    assert error is None
    assert data["overall_assessment"] == "pass"
    assert data["_usage"]["requests"] == 2
    assert data["_usage"]["prompt_tokens"] == 150
    assert data["_usage"]["completion_tokens"] == 2058
    assert data["_usage"]["cost_usd"] == 0.003
    assert len(client.prompts) == 2
    assert client.prompts[0][1] == 4096
    assert "previous response was invalid or truncated" in client.prompts[1][0]


def test_llm_client_passes_per_call_output_limit(monkeypatch):
    client = LLMClient(provider="openrouter", model="test-model")
    captured = {}

    async def fake_call(system_prompt, user_prompt, max_tokens=None):
        captured["max_tokens"] = max_tokens
        return "{}", {}

    monkeypatch.setattr(client, "_call", fake_call)
    text, _usage = asyncio.run(
        client.complete("system", "user", max_tokens=4096)
    )

    assert text == "{}"
    assert captured["max_tokens"] == 4096


def test_chat_usage_summary_and_rotating_file(monkeypatch, tmp_path):
    entries = [
        {
            "node": "infer_behavioral_evaluation",
            "requests": 1,
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "cost_usd": 0.001,
            "cost_complete": True,
        },
        {
            "node": "infer_overall_scoring",
            "requests": 2,
            "prompt_tokens": 200,
            "completion_tokens": 40,
            "total_tokens": 240,
            "cost_usd": 0.002,
            "cost_complete": True,
        },
    ]
    summary = _summarize_chat_usage(entries)
    assert summary["request_count"] == 3
    assert summary["total_tokens"] == 360
    assert summary["cost_usd"] == 0.003
    assert summary["cost_complete"] is True

    log_path = tmp_path / "token_usage.log"
    monkeypatch.setattr(agent_nodes.settings, "TOKEN_USAGE_LOG_PATH", str(log_path))
    usage_logger, configured_path = _build_token_usage_logger()
    usage_logger.info(json.dumps({"call_id": "call-1", **summary}))
    for handler in usage_logger.handlers:
        handler.flush()

    record = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert configured_path == log_path
    assert record["call_id"] == "call-1"
    assert record["cost_usd"] == 0.003

    for handler in list(usage_logger.handlers):
        if getattr(handler, "baseFilename", None) == str(log_path):
            usage_logger.removeHandler(handler)
            handler.close()


def test_crm_lead_prompt_serializes_fetched_record():
    call = SimpleNamespace(
        call_id="call-1",
        agent_name="Agent Name",
        business_unit="MKR",
        call_date="2026-08-16",
        transcript="Patient requested an appointment.",
    )

    prompt = build_crm_lead_validation_prompt(
        call,
        {"record": {"leadid": "lead-1", "modifiedbyname": "Agent Name"}},
    )

    assert "\"leadid\": \"lead-1\"" in prompt
    assert "\"modifiedbyname\": \"Agent Name\"" in prompt


def test_lead_bu_codes_map_each_branch_to_one_code():
    expected = {
        "ADC": ("JDC", "MKR", "ADC", "BU-MKR", "bu-jdc", " adc "),
        "HJH": ("AHJ", "HJH", "BU-AHJ"),
        "ALW": ("CHT", "LCH", "ALW", "BU-LCH"),
        "AFW": ("SNB", "AFW", "BU-SNB"),
        "AKW": ("AKW", "BU-AKW"),
    }
    for code, aliases in expected.items():
        for alias in aliases:
            assert leads.lead_bu_code(alias) == code, alias
            assert leads.normalize_business_unit(alias) == code, alias
    assert leads.lead_bu_code("Andalusia Hospital") is None
    assert leads.lead_bu_code(None) is None


def test_crm_lead_prompt_compares_bu_codes_after_mapping():
    call = SimpleNamespace(
        call_id="call-1",
        agent_name="Agent Name",
        business_unit="AHJ",
        call_date="2026-08-16",
        transcript="Patient requested an appointment.",
    )
    record = {"leadid": "lead-1", "new_clinicbu": "BU-MKR"}

    prompt = build_crm_lead_validation_prompt(call, {"record": record})

    assert "Business unit: HJH" in prompt
    assert "\"new_clinicbu\": \"ADC\"" in prompt
    assert "ADC/JDC/MKR → ADC" in prompt and "AHJ/HJH → HJH" in prompt
    # The fetched record itself is left as CRM returned it.
    assert record["new_clinicbu"] == "BU-MKR"


def test_crm_lead_prompt_keeps_unmapped_clinic_bu_names():
    call = SimpleNamespace(
        call_id="call-1", agent_name="Agent Name", business_unit=None,
        call_date="2026-08-16", transcript="Patient requested an appointment.",
    )
    prompt = build_crm_lead_validation_prompt(
        call, {"record": {"new_clinicbu": "Andalusia Women's Hospital"}},
    )
    assert "Business unit: unknown" in prompt
    assert "\"new_clinicbu\": \"Andalusia Women's Hospital\"" in prompt


def _cm_row(sender_type, owner, content, ts):
    """One NEWFIX_CM_Chat_Search row, as in the transferred-chat sample."""
    import pandas as pd
    return {
        "UniqueId": "EE8BD2E7-01CB-F111-BF5B-000D3AA9DC2F",
        "AgentFullName": "Hesham Ibrahim",
        "Start_DateTime": pd.Timestamp("2026-10-18 17:40:38"),
        "Archive_DateTime": pd.Timestamp("2026-10-18 18:23:51"),
        "PatientPhoneNumber": 966506528417.0,
        "WebStoreName": "Call Center",
        "SenderType": sender_type, "OwnerName": owner, "Content": content,
        "CreationDateTime": pd.Timestamp(ts),
    }


def test_sanabel_booking_branch_overrides_incidental_hospital_mention():
    from app.models.input import CallTranscript

    for branch_name in ("فرع السنابل", "فرع سنابل"):
        call = CallTranscript.from_cm_rows([
            _cm_row("User", "Agent Name", "من خلال مستشفى اندلسية الكشف ب 140 ريال", "2026-10-18 17:41:04"),
            _cm_row("User", "Agent Name", f"تم تأكيد حجزك في {branch_name}", "2026-10-18 17:42:04"),
        ])

        assert call.business_unit == "SNB"
        prompt = build_crm_lead_validation_prompt(
            call, {"record": {"new_clinicbu": "SNB"}},
        )
        assert "Business unit: AFW" in prompt
        assert '"new_clinicbu": "AFW"' in prompt


def test_patient_bu_tag_overrides_agent_list_of_branch_locations():
    from app.models.input import CallTranscript

    call = CallTranscript.from_cm_rows([
        _cm_row("Relation", "Patient", "BU-ALW-Derma", "2026-10-18 17:41:04"),
        _cm_row(
            "User", "Agent Name",
            "جميع فروعنا بجدة: فرع السنابل، فرع المكرونة، فرع الأمير سلطان. باي فرع استفسارك؟",
            "2026-10-18 17:42:04",
        ),
    ])

    assert call.business_unit == "ALW"
    prompt = build_crm_lead_validation_prompt(
        call, {"record": {"new_clinicbu": "ALW"}},
    )
    assert "Business unit: ALW" in prompt
    assert '"new_clinicbu": "ALW"' in prompt


def test_dental_campaign_uses_configured_afw_branch_mapping():
    from app.models.input import CallTranscript

    call = CallTranscript.from_cm_rows([
        _cm_row(
            "Relation", "Patient",
            "BU-Clinics-Dental-GoogleSearch-برجاء اضغط ارسال للتحدث معنا مباشرة",
            "2026-10-18 17:41:04",
        ),
        _cm_row("User", "Agent Name", "حابه تشرفينا اي فرع بجدة", "2026-10-18 17:42:04"),
        _cm_row("Relation", "Patient", "اي فرع عادي انا حاليا فالباحه", "2026-10-18 17:43:04"),
    ])

    assert call.business_unit == "AFW"
    prompt = build_crm_lead_validation_prompt(
        call, {"record": {"new_clinicbu": "SNB"}},
    )
    assert "Business unit: AFW" in prompt
    assert '"new_clinicbu": "AFW"' in prompt


def test_transferred_chat_lists_every_agent_in_reply_order():
    from app.models.input import CallTranscript
    call = CallTranscript.from_cm_rows([
        _cm_row("Bot", "Bot", "أهلًا بك", "2026-10-18 17:40:52"),
        _cm_row("Relation", "الملكي", "ممكن اعرف مواعيد عيادة الدكتورة ولاء", "2026-10-18 17:40:53"),
        _cm_row("User", "Maha Gamal", "السلام عليكم مع حضرتك مها", "2026-10-18 17:41:04"),
        _cm_row("User", "Maha Gamal", "موعدك يوم الاربعاء", "2026-10-18 18:17:33"),
        _cm_row("User", "Hesham Ibrahim", "شكرًا لتواصلكم مع مجموعة أندلسية صحة", "2026-10-18 18:23:49"),
    ])
    assert call.agent_name == "Hesham Ibrahim"
    assert call.agent_names == ["Maha Gamal", "Hesham Ibrahim"]


def test_same_agent_name_ignores_case_spacing_and_extra_words():
    assert leads.same_agent_name(" maha  GAMAL ", "Maha Gamal")
    assert leads.same_agent_name("Maha Gamal Ahmed", "Maha Gamal")
    assert not leads.same_agent_name("Maha", "Maha Gamal")
    assert not leads.same_agent_name("Hesham Ibrahim", "Maha Gamal")
    assert not leads.same_agent_name(None, "Maha Gamal")


def test_fetch_crm_lead_selects_the_lead_naming_a_chat_agent(monkeypatch):
    monkeypatch.setattr(leads, "_crm_leads_is_configured", lambda: True)
    rows = [
        {"leadid": "other", "modifiedbyname": "Someone Else"},
        {"leadid": "maha", "modifiedbyname": "Maha Gamal"},
    ]
    monkeypatch.setattr(leads, "_run_crm_leads_query_with_retry", lambda query, params: rows)

    result = leads.fetch_crm_lead("506528417", "2026-10-18", agent_names=["Maha Gamal", "Hesham Ibrahim"])

    assert result["record"]["leadid"] == "maha"
    assert result["agent_name"] == "Maha Gamal"
    assert result["chat_agents"] == ["Maha Gamal", "Hesham Ibrahim"]


def test_fetch_crm_lead_matches_last_call_creator_and_falls_back_to_first(monkeypatch):
    monkeypatch.setattr(leads, "_crm_leads_is_configured", lambda: True)
    rows = [
        {"leadid": "first", "modifiedbyname": "Someone Else"},
        {"leadid": "hesham", "modifiedbyname": "Someone Else", "new_lastcallcreatedbyname": "hesham ibrahim"},
    ]
    monkeypatch.setattr(leads, "_run_crm_leads_query_with_retry", lambda query, params: rows)

    found = leads.fetch_crm_lead("506528417", "2026-10-18", agent_names=["Maha Gamal", "Hesham Ibrahim"])
    assert (found["record"]["leadid"], found["agent_name"]) == ("hesham", "Hesham Ibrahim")

    none = leads.fetch_crm_lead("506528417", "2026-10-18", agent_names=["Nobody Here"])
    assert (none["record"]["leadid"], none["agent_name"]) == ("first", None)


def test_crm_lead_prompt_evaluates_the_agent_the_lead_names():
    call = SimpleNamespace(
        call_id="call-1", agent_name="Hesham Ibrahim", business_unit="AHJ",
        call_date="2026-10-18", transcript="Agent: موعدك يوم الاربعاء",
    )
    lookup = {
        "record": {"leadid": "maha", "modifiedbyname": "Maha Gamal"},
        "agent_name": "Maha Gamal",
        "chat_agents": ["Maha Gamal", "Hesham Ibrahim"],
    }
    prompt = build_crm_lead_validation_prompt(call, lookup)
    assert "Agent: Maha Gamal" in prompt
    assert "Chat agents: Maha Gamal, Hesham Ibrahim" in prompt


EGYPT_PHONE = "201001234567"   # as the chat stores it: digits, no "+"


def test_notes_phone_pattern_is_plus_and_digits():
    for phone in ("201001234567", "+201001234567", "00201001234567", "+20 100 123 4567"):
        assert leads.notes_phone_pattern(phone) == "+201001234567", phone


def test_notes_contain_phone_needs_the_plus_number():
    for notes in ("رقم المريض +201001234567", "+201001234567 حجز عيون"):
        assert leads.notes_contain_phone(notes, EGYPT_PHONE), notes
    for notes in (None, "", "201001234567", "00201001234567", "+20 100 123 4567", "+201009999999"):
        assert not leads.notes_contain_phone(notes, EGYPT_PHONE), notes


def _non_saudi_fetch(monkeypatch, returned_rows, captured=None):
    monkeypatch.setattr(leads, "_crm_leads_is_configured", lambda: True)

    def fake_query(query, params):
        if captured is not None:
            captured.update(query=query, params=params)
        return list(returned_rows)

    monkeypatch.setattr(leads, "_run_crm_leads_query_with_retry", fake_query)
    return leads.fetch_crm_lead(EGYPT_PHONE, "2026-10-18", agent_names=["Maha Gamal"])


def test_non_saudi_number_is_looked_up_as_the_placeholder():
    values, non_saudi = leads._phone_values("+201001234567")
    assert non_saudi is True
    assert values == {
        "mobile_number": "555555550",
        "local_number": "0555555550",
        "intl_number": "966555555550",
        "plus_intl_number": "+966555555550",
    }


def test_non_saudi_query_filters_new_notes_with_like_plus_number(monkeypatch):
    captured = {}
    result = _non_saudi_fetch(monkeypatch, [
        {"leadid": "this-patient", "modifiedbyname": "Maha Gamal", "new_notes": "+201001234567"},
    ], captured)
    assert "AND new_notes LIKE :notes_phone" in captured["query"]
    assert captured["params"]["notes_phone"] == "%+201001234567%"
    assert captured["params"]["local_number"] == "0555555550"
    assert result["record"]["leadid"] == "this-patient"
    assert result["non_saudi"] is True and result["notes_phone_match"] is True


def test_saudi_query_has_no_new_notes_filter(monkeypatch):
    monkeypatch.setattr(leads, "_crm_leads_is_configured", lambda: True)
    captured = {}
    monkeypatch.setattr(
        leads, "_run_crm_leads_query_with_retry",
        lambda query, params: captured.update(query=query, params=params) or [{"leadid": "x"}],
    )
    leads.fetch_crm_lead("506528417", "2026-10-18")
    assert "new_notes LIKE" not in captured["query"]
    assert "notes_phone" not in captured["params"]


def test_non_saudi_without_the_plus_number_is_not_found(monkeypatch):
    # A path that ignored the filter returns a lead without "+<digits>".
    result = _non_saudi_fetch(monkeypatch, [
        {"leadid": "someone-elses", "modifiedbyname": "Maha Gamal", "new_notes": "201001234567"},
    ])
    assert result["status"] == "not_found"
    assert "non-Saudi" in result["message"]


def test_web_api_filters_new_notes_with_contains_plus_number(monkeypatch):
    captured = {}

    class Response:
        status_code = 200

        def json(self):
            return {"value": []}

    monkeypatch.setattr(leads, "_get_crm_leads_access_token", lambda force_refresh=False: "token")
    monkeypatch.setattr(leads, "_crm_leads_host", lambda: "org.crm.dynamics.com")
    monkeypatch.setattr(leads.httpx, "get", lambda url, **kw: captured.update(kw) or Response())
    leads._fetch_crm_leads_web_api_once(EGYPT_PHONE, "2026-10-18")
    assert "contains(new_notes,'+201001234567')" in captured["params"]["$filter"]


def test_notes_phone_check_flags_a_missing_number_and_passes_a_present_one():
    lookup = {"non_saudi": True, "record": {"new_notes": "حجز"}}
    flagged = leads.apply_notes_phone_check(
        {"crm_lead_status": "match", "crm_leads_flags": [], "field_checks": []}, lookup, EGYPT_PHONE,
    )
    assert flagged["crm_lead_status"] == "violation"
    assert "new_notes" in flagged["crm_leads_flags"][0]["description"]
    assert flagged["field_checks"][-1]["matches"] is False

    lookup["record"]["new_notes"] = "+201001234567"
    passed = leads.apply_notes_phone_check(
        {"crm_lead_status": "match", "crm_leads_flags": [], "field_checks": []}, lookup, EGYPT_PHONE,
    )
    assert passed["crm_lead_status"] == "match" and passed["crm_leads_flags"] == []
    assert passed["field_checks"][-1]["matches"] is True


def test_notes_phone_check_ignores_saudi_numbers():
    evaluation = {"crm_lead_status": "match", "crm_leads_flags": [], "field_checks": []}
    assert leads.apply_notes_phone_check(evaluation, {"non_saudi": False, "record": {}}, "506528417") == evaluation


def test_lead_clinic_bu_falls_back_to_source_bu_only_when_empty():
    assert leads.lead_clinic_bu({"new_clinicbu": "ALW", "new_sourcebusinessunitname": "AKW"}) == ("ALW", "new_clinicbu")
    for empty in (None, "", "   "):
        assert leads.lead_clinic_bu(
            {"new_clinicbu": empty, "new_sourcebusinessunitname": "BU-MKR"}
        ) == ("BU-MKR", "new_sourcebusinessunitname")
    assert leads.lead_clinic_bu({"new_clinicbu": None, "new_sourcebusinessunitname": None}) == (None, "new_clinicbu")


def _bu_prompt(record):
    call = SimpleNamespace(
        call_id="call-1", agent_name="Agent", business_unit="MKR",
        call_date="2026-10-19", transcript="Agent: تم الحجز",
    )
    return build_crm_lead_validation_prompt(call, {"record": record})


def test_crm_lead_prompt_uses_source_bu_when_clinic_bu_is_empty():
    prompt = _bu_prompt({"new_clinicbu": None, "new_sourcebusinessunitname": "BU-JDC"})
    assert "\"new_clinicbu\": \"ADC\"" in prompt
    assert "\"new_clinicbu_source\": \"new_sourcebusinessunitname\"" in prompt


def test_crm_lead_prompt_ignores_source_bu_when_clinic_bu_is_set():
    prompt = _bu_prompt({"new_clinicbu": "ALW", "new_sourcebusinessunitname": "BU-JDC"})
    assert "\"new_clinicbu\": \"ALW\"" in prompt
    assert "\"new_clinicbu_source\":" not in prompt


def test_fallback_reads_only_the_source_bu_name_column():
    record = leads._normalize_web_api_lead({
        "new_clinicbu": None,
        "new_sourcebusinessunitname": "BU-ALW",
        "_new_sourcebusinessunit_value@OData.Community.Display.V1.FormattedValue": "BU-AKW",
    })
    assert leads.lead_clinic_bu(record) == ("BU-ALW", "new_sourcebusinessunitname")

    without_name = leads._normalize_web_api_lead({
        "new_clinicbu": None,
        "_new_sourcebusinessunit_value@OData.Community.Display.V1.FormattedValue": "BU-AKW",
    })
    assert leads.lead_clinic_bu(without_name) == (None, "new_clinicbu")


def test_matched_attributes_report_the_fallback_bu():
    evaluation = {"field_checks": [{"field": "new_clinicbu", "matches": True, "expected": "ADC", "actual": "ADC"}]}
    record = {"new_clinicbu": None, "new_sourcebusinessunitname": "BU-MKR"}
    assert leads.matched_crm_lead_attributes(evaluation, record)[0]["crm_value"] == "BU-MKR"


def test_sql_compares_createdon_on_the_saudi_calendar_day():
    query = leads._load_query()
    assert "CAST(DATEADD(HOUR, 3, [createdon]) AS DATE) = :Report_Date" in query


def test_web_api_bounds_cover_the_saudi_calendar_day():
    # A lead created 2026-10-19 22:56Z is 2026-10-20 01:56 in Saudi Arabia.
    assert leads._saudi_day_utc_filter("2026-10-20") == (
        "createdon ge 2026-10-19T21:00:00Z and createdon lt 2026-10-20T21:00:00Z"
    )


def test_phone_values_cover_chat_and_dynamics_formats():
    values, non_saudi = leads._phone_values("+966512345678")
    assert non_saudi is False
    assert values == {
        "mobile_number": "512345678",
        "local_number": "0512345678",
        "intl_number": "966512345678",
        "plus_intl_number": "+966512345678",
    }
    assert leads._quoted_table_name("dbo.lead") == "[lead]"

    local_values, _ = leads._phone_values("0535569739")
    assert local_values == {
        "mobile_number": "535569739",
        "local_number": "0535569739",
        "intl_number": "966535569739",
        "plus_intl_number": "+966535569739",
    }


def test_crm_leads_server_defaults_to_dataverse_tds_port(monkeypatch):
    monkeypatch.setattr(leads.settings, "CRM_LEADS_SERVER", "org.crm.dynamics.com")
    assert leads._crm_leads_server() == "org.crm.dynamics.com,5558"

    monkeypatch.setattr(leads.settings, "CRM_LEADS_SERVER", "org.crm.dynamics.com,5558")
    assert leads._crm_leads_server() == "org.crm.dynamics.com,5558"


def test_fetch_crm_lead_uses_sql_query_and_newest_record(monkeypatch, caplog):
    captured = {}

    monkeypatch.setattr(leads.settings, "CRM_LEADS_SERVER", "example.crm.dynamics.com,5558")
    monkeypatch.setattr(leads, "_crm_leads_is_configured", lambda: True)

    def fake_query(query, params):
        captured.update(query=query, params=params)
        return [{"leadid": "newest"}]

    monkeypatch.setattr(leads, "_run_crm_leads_query_with_retry", fake_query)

    caplog.set_level("INFO", logger=leads.__name__)
    result = leads.fetch_crm_lead("512345678", "2026-09-02")

    assert result["status"] == "found"
    assert result["record"]["leadid"] == "newest"
    assert "SELECT" in captured["query"]
    assert captured["params"]["mobile_number"] == "512345678"
    assert captured["params"]["Report_Date"] == "2026-09-02"
    assert "phone=512345678 report_date=2026-09-02" in caplog.text


def test_fetch_crm_lead_falls_back_to_web_api_when_tds_fails(monkeypatch):
    monkeypatch.setattr(leads, "_crm_leads_is_configured", lambda: True)
    monkeypatch.setattr(
        leads,
        "_run_crm_leads_query_with_retry",
        lambda query, params: (_ for _ in ()).throw(
            RuntimeError("08S01 Communication link failure")
        ),
    )
    monkeypatch.setattr(
        leads,
        "_fetch_crm_leads_web_api_with_retry",
        lambda phone, report_date: [{"leadid": "from-web-api"}],
    )

    result = leads.fetch_crm_lead("512345678", "2026-09-02")

    assert result["status"] == "found"
    assert result["record"]["leadid"] == "from-web-api"


def test_matched_crm_lead_attributes_include_values_and_call_insights():
    matched = leads.matched_crm_lead_attributes(
        {
            "field_checks": [
                {
                    "field": "modifiedbyname",
                    "matches": True,
                    "expected": "Agent Name",
                    "actual": "fallback value",
                },
                {
                    "field": "new_doctor",
                    "matches": False,
                    "expected": "Expected Doctor",
                    "actual": "Other Doctor",
                },
            ]
        },
        {"modifiedbyname": "Agent Name"},
    )

    assert matched == [
        {
            "field": "modifiedbyname",
            "crm_value": "Agent Name",
            "call_insight": "Agent Name",
        }
    ]


def test_missing_lead_and_model_flags_are_always_c2b():
    missing = leads.missing_lead_evaluation("not found")
    assert missing["crm_leads_flags"][0]["type"] == "C2B"
    assert missing["crm_leads_flags"][0]["severity"] == "moderate"

    normalized = leads.normalize_crm_lead_evaluation(
        {
            "crm_leads_flags": [
                {
                    "type": "NC",
                    "severity": "critical",
                    "description": "wrong result",
                    "transcript_excerpt": "",
                }
            ]
        }
    )
    assert normalized["crm_lead_status"] == "violation"
    assert normalized["crm_leads_flags"] == [
        {
            "type": "C2B",
            "severity": "moderate",
            "description": "wrong result",
            "transcript_excerpt": "N/A",

        }
    ]
    synthesized = leads.normalize_crm_lead_evaluation(
        {
            "crm_lead_status": "violation",
            "crm_leads_flags": [],
            "field_checks": [
                {"field": "modifiedbyname", "matches": False, "reason": "wrong agent"}
            ],
        }
    )
    assert synthesized["crm_leads_flags"][0]["type"] == "C2B"
    assert "modifiedbyname" in synthesized["crm_leads_flags"][0]["description"]


def test_invalid_table_identifier_is_rejected():
    try:
        leads._quoted_table_name("dbo.lead; DROP TABLE lead")
    except ValueError:
        pass
    else:
        raise AssertionError("unsafe table identifier was accepted")


def test_crm_leads_reconnects_with_fresh_connections(monkeypatch):
    calls = []

    monkeypatch.setattr(leads.settings, "CRM_LEADS_MAX_RETRIES", 4)
    monkeypatch.setattr(leads.settings, "CRM_LEADS_RETRY_DELAY_SECONDS", 0)

    def flaky_query(query, params, force_token_refresh=False):
        calls.append(force_token_refresh)
        if len(calls) < 3:
            raise RuntimeError("08S01 Communication link failure")
        return [{"leadid": "reconnected"}]

    monkeypatch.setattr(leads, "_run_crm_leads_query_once", flaky_query)

    rows = leads._run_crm_leads_query_with_retry(
        "SELECT TOP 1 [leadid] FROM [lead]",
        params={},
    )

    assert rows == [{"leadid": "reconnected"}]
    assert calls == [False, True, True]


def test_web_api_normalizes_formatted_values(monkeypatch):
    class FakeResponse:
        status_code = 200

        @staticmethod
        def json():
            suffix = leads._FORMATTED_VALUE_SUFFIX
            return {
                "value": [
                    {
                        "leadid": "lead-1",
                        "leadsourcecode" + suffix: "WhatsApp",
                        "statuscode" + suffix: "Contacted",
                        "_modifiedby_value" + suffix: "Agent Name",
                    }
                ]
            }

    monkeypatch.setattr(leads.settings, "CRM_LEADS_SERVER", "org.crm.dynamics.com,5558")
    monkeypatch.setattr(leads, "_get_crm_leads_access_token", lambda force_refresh=False: "token")
    monkeypatch.setattr(leads.httpx, "get", lambda *args, **kwargs: FakeResponse())

    rows = leads._fetch_crm_leads_web_api_once("0512345678", "2026-09-02")

    assert rows[0]["leadid"] == "lead-1"
    assert rows[0]["modifiedbyname"] == "Agent Name"
    assert rows[0]["leadsourcecodename"] == "WhatsApp"


def test_web_api_does_not_retry_forbidden_response(monkeypatch):
    calls = []
    monkeypatch.setattr(leads.settings, "CRM_LEADS_MAX_RETRIES", 4)

    def forbidden(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("CRM Leads Web API authorization failed with HTTP 403")

    monkeypatch.setattr(leads, "_fetch_crm_leads_web_api_once", forbidden)

    try:
        leads._fetch_crm_leads_web_api_with_retry("0512345678", "2026-09-02")
    except RuntimeError as exc:
        assert "HTTP 403" in str(exc)
    else:
        raise AssertionError("HTTP 403 was unexpectedly swallowed")

    assert len(calls) == 1
