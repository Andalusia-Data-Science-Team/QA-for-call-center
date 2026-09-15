from app.service_hub import crm_leads_validation as leads


def test_phone_values_cover_chat_and_dynamics_formats():
    values = leads._phone_values("+966512345678")
    assert values == {
        "mobile_number": "512345678",
        "local_number": "0512345678",
        "intl_number": "966512345678",
        "plus_intl_number": "+966512345678",
    }
    assert leads._quoted_table_name("dbo.lead") == "[lead]"


def test_fetch_crm_lead_uses_sql_query_and_newest_record(monkeypatch):
    captured = {}

    monkeypatch.setattr(leads.settings, "CRM_LEADS_SERVER", "example.crm.dynamics.com,5558")
    monkeypatch.setattr(leads, "_crm_leads_is_configured", lambda: True)

    def fake_query(query, params):
        captured.update(query=query, params=params)
        return [{"leadid": "newest"}]

    monkeypatch.setattr(leads, "_run_crm_leads_query_with_retry", fake_query)

    result = leads.fetch_crm_lead("512345678", "2026-09-02")

    assert result["status"] == "found"
    assert result["record"]["leadid"] == "newest"
    assert "SELECT" in captured["query"]
    assert captured["params"]["mobile_number"] == "512345678"
    assert captured["params"]["Report_Date"] == "2026-09-02"


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

    def flaky_query(query, params, force_token_refresh):
        calls.append(force_token_refresh)
        if len(calls) < 3:
            raise RuntimeError("08S01 Communication link failure")
        return [{"leadid": "reconnected"}]

    monkeypatch.setattr(leads, "_run_query_with_retry", flaky_query)

    rows = leads._query_crm_leads_with_reconnect(
        "SELECT TOP 1 [leadid] FROM [lead]",
        params={},
        server="example.crm.dynamics.com,5558",
    )

    assert rows == [{"leadid": "reconnected"}]
    assert len(calls) == 3
    assert all(max_attempts == 1 for max_attempts, _server in calls)


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
