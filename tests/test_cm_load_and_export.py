from datetime import datetime

import pandas as pd
import pytest

from app.CM import load_and_export


class FakeCursor:
    def __init__(self):
        self.calls = []
        self.closed = False

    def execute(self, sql, *params):
        self.calls.append((sql, params))

    def close(self):
        self.closed = True


class FakeConnection:
    def __init__(self):
        self.fake_cursor = FakeCursor()
        self.committed = False
        self.closed = False

    def cursor(self):
        return self.fake_cursor

    def commit(self):
        self.committed = True

    def close(self):
        self.closed = True


def cm_users():
    return pd.DataFrame([{
        "PersonID": "9f7d4cf0-a5a2-49d7-b739-c6d378d10d31",
        "FullName": "  Sara Ahmed  ",
        "PersonCreationDateTime": pd.Timestamp("2026-10-20 09:30:00"),
        "PersonEmailAddress": None,
        "UserEmailAddress": " Sara.Ahmed@example.com ",
    }])


def test_load_to_database_upserts_new_cm_people_fields_by_user_email(monkeypatch):
    connection = FakeConnection()
    monkeypatch.setattr(load_and_export, "load_credentials_from_json", lambda path, key: {
        "server": "test-server", "database": "DWH", "username": "test",
        "password": "test", "driver": "test-driver",
    })
    monkeypatch.setattr(load_and_export, "connect_to_database", lambda *args: connection)

    load_and_export.load_to_database(cm_users())

    assert len(connection.fake_cursor.calls) == 1
    sql, params = connection.fake_cursor.calls[0]
    assert "UPDATE [DWH].[AI].[Agents]" in sql
    assert "INSERT INTO [DWH].[AI].[Agents]" in sql
    assert "Agent_Email_Address = ?" in sql
    assert "Agent_id" not in sql
    assert params == (
        "Sara Ahmed", datetime(2026, 10, 20, 9, 30), "Sara.Ahmed@example.com",
        "sara.ahmed@example.com",
        "Sara Ahmed", datetime(2026, 10, 20, 9, 30), "Sara.Ahmed@example.com",
    )
    assert connection.committed and connection.closed and connection.fake_cursor.closed


def test_main_loads_the_fresh_query_result_after_export(monkeypatch):
    dataframe = cm_users()
    source = FakeConnection()
    calls = {}
    monkeypatch.setattr(load_and_export, "load_credentials_from_json", lambda path, key: {
        "server": "test-server", "database": "CM", "username": "test",
        "password": "test", "driver": "test-driver",
    })
    monkeypatch.setattr(load_and_export, "connect_to_database", lambda *args: source)
    monkeypatch.setattr(load_and_export, "execute_query", lambda connection, path: dataframe)
    monkeypatch.setattr(load_and_export, "export_to_excel", lambda data, path: calls.update(export=(data, path)))
    monkeypatch.setattr(load_and_export, "load_to_database", lambda data: calls.update(load=data))

    load_and_export.main()

    assert calls["export"] == (dataframe, load_and_export.OUTPUT_FILE)
    assert calls["load"] is dataframe
    assert source.closed


def test_load_to_database_rejects_old_account_columns_before_connecting(monkeypatch):
    monkeypatch.setattr(load_and_export, "connect_to_database", lambda *args: pytest.fail("connected before validation"))

    with pytest.raises(ValueError, match="PersonCreationDateTime"):
        load_and_export.load_to_database(pd.DataFrame([{
            "FullName": "Sara Ahmed",
            "AccountCreationDateTime": pd.Timestamp("2026-10-20"),
            "AccountEmailAddress": "sara@example.com",
        }]))
