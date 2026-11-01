import sys
from types import SimpleNamespace

import pandas as pd
from fastapi.testclient import TestClient

from app import main


def test_conversation_id_lookup_uses_internal_limit_and_ignores_default_filters(monkeypatch):
    sys.path.insert(0, str(main.CM_DIR))
    import CM_DB_Handler
    import Chat_Retriever

    query_params = []

    class FakeHandler:
        def __init__(self, **kwargs):
            pass

        def connect(self):
            return True

        def execute_query_from_file(self, sql_file, params):
            query_params.append(params)
            return pd.DataFrame([{
                "UniqueId": "10248B91-A9CB-F111-BF5B-000D3AA9DC2F",
                "UserEmailAddress": "owner@example.com",
            }])

        def close(self):
            pass

    call = SimpleNamespace(call_id="10248B91-A9CB-F111-BF5B-000D3AA9DC2F", conversation_link=None, agent_email=None)
    monkeypatch.setattr(CM_DB_Handler, "CMDatabaseHandler", FakeHandler)
    monkeypatch.setattr(Chat_Retriever, "df_to_batch", lambda df: SimpleNamespace(calls=[call]))

    async def fail_analysis(call):
        raise RuntimeError("test analysis stopped")

    monkeypatch.setattr(main.analyzer, "analyze", fail_analysis)

    response = TestClient(main.app).post(
        "/retrieve-and-analyze",
        params={
            "conversation_id": call.call_id,
            "agent_email": "unrelated@example.com",
            "filter_date": "2020-01-01",
            "chat_limit": 10,
        },
    )

    assert response.status_code == 200
    assert query_params == [{
        "ConversationId": call.call_id.lower(),
        "AgentFullName": None,
        "AgentEmail": None,
        "FilterDate": None,
        "ChatLimit": 1000,
    }]
    assert call.agent_email == "owner@example.com"


def test_agent_date_search_keeps_requested_chat_limit(monkeypatch):
    sys.path.insert(0, str(main.CM_DIR))
    import CM_DB_Handler

    query_params = []

    class FakeHandler:
        def __init__(self, **kwargs):
            pass

        def connect(self):
            return True

        def execute_query_from_file(self, sql_file, params):
            query_params.append(params)
            return pd.DataFrame()

        def close(self):
            pass

    monkeypatch.setattr(CM_DB_Handler, "CMDatabaseHandler", FakeHandler)
    response = TestClient(main.app).post(
        "/retrieve-and-analyze",
        params={"agent_email": "agent@example.com", "filter_date": "2026-10-21", "chat_limit": 25},
    )

    assert response.status_code == 404
    assert query_params[0]["ChatLimit"] == 25
    assert query_params[0]["ConversationId"] is None


def test_invalid_conversation_id_is_rejected_before_query():
    response = TestClient(main.app).post(
        "/retrieve-and-analyze",
        params={"conversation_id": "not-a-uuid"},
    )
    assert response.status_code == 422
