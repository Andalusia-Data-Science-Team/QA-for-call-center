from datetime import datetime

import pandas as pd
import pytest

from app.models.input import CallTranscript


def _cm_row(archive_time, archive_field="ArchiveDateTime"):
    return {
        "UniqueId": "call-across-midnight",
        "AgentFullName": "Agent Name",
        "Start_DateTime": pd.Timestamp("2026-10-18 23:55:00"),
        archive_field: archive_time,
        "PatientPhoneNumber": 966501234567,
        "WebStoreName": "Call Center",
        "SenderType": "User",
        "OwnerName": "Agent Name",
        "Content": "Hello",
        "CreationDateTime": pd.Timestamp("2026-10-18 23:56:00"),
    }


@pytest.mark.parametrize("archive_field", ["ArchiveDateTime", "Archive_DateTime"])
def test_call_date_uses_archive_date_when_conversation_crosses_midnight(archive_field):
    call = CallTranscript.from_cm_rows([
        _cm_row(pd.Timestamp("2026-10-19 00:05:00"), archive_field)
    ])

    assert call.call_date == "2026-10-19"
    assert call.archive_datetime == datetime(2026, 10, 19, 0, 5)
    assert call.call_duration_seconds == 600


def test_call_date_falls_back_to_start_date_when_archive_is_missing():
    call = CallTranscript.from_cm_rows([_cm_row(pd.NaT)])

    assert call.call_date == "2026-10-18"
    assert call.archive_datetime is None
