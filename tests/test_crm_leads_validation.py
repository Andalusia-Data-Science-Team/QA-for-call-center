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


def test_phone_values_cover_chat_and_dynamics_formats():
    values = leads._phone_values("+966512345678")
    assert values == {
        "mobile_number": "512345678",
        "local_number": "0512345678",
        "intl_number": "966512345678",
        "plus_intl_number": "+966512345678",
    }
    assert leads._quoted_table_name("dbo.lead") == "[lead]"

    local_values = leads._phone_values("0535569739")
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
