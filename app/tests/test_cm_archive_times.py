from datetime import datetime

from app.services import cm_archive_times


def test_lookup_archive_datetimes_reads_local_cm_time_with_bound_ids(monkeypatch):
    class FakeRows:
        def mappings(self):
            return [
                {
                    "CallId": "52BDA5ED-7CCB-F111-BF5B-000D3AA9DC2F",
                    "ArchiveDateTime": datetime(2026, 10, 19, 8, 14, 37),
                }
            ]

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, query, params):
            assert "DATEADD(HOUR, 3, c.ArchiveDateTime)" in str(query)
            assert "c.UniqueId IN" in str(query)
            assert params == {"ids": ["52BDA5ED-7CCB-F111-BF5B-000D3AA9DC2F"]}
            return FakeRows()

    class FakeEngine:
        disposed = False

        def connect(self):
            return FakeConnection()

        def dispose(self):
            self.disposed = True

    engine = FakeEngine()
    monkeypatch.setattr(cm_archive_times, "_make_engine", lambda _creds: engine)

    found = cm_archive_times.lookup_archive_datetimes(
        ["52bda5ed-7ccb-f111-bf5b-000d3aa9dc2f"], {}
    )

    assert found == {
        "52BDA5ED-7CCB-F111-BF5B-000D3AA9DC2F": "2026-10-19T08:14:37"
    }
    assert engine.disposed
