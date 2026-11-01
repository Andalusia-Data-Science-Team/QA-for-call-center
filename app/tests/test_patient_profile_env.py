"""BU → MPI environment mapping used by the patient-profile stage."""
import asyncio
from types import SimpleNamespace

import pytest

from app.agent import patient_profile_nodes as pp


@pytest.mark.parametrize("bu, env", [
    ("LCH", "ALW"), ("CHT", "ALW"), ("ALW", "ALW"), ("BU-LCH", "ALW"),
    ("AHJ", "AHJ"), ("HJH", "AHJ"), ("BU-AHJ", "AHJ"),
    ("SNB", "SNB"), ("AFW", "SNB"), ("BU-SNB", "SNB"),
    ("MKR", "JDC"), ("ADC", "JDC"), ("JDC", "JDC"), ("BU-JDC", "JDC"),
    ("AKW", "AKW"), (" akw ", "AKW"),
    ("ERS", "ERS"), ("BU-ERS", "ERS"),
    (None, "AHJ"), ("", "AHJ"),
])
def test_mpi_env_for_bu(bu, env):
    assert pp.mpi_env_for_bu(bu) == env


def test_unknown_bu_has_no_mpi_env():
    assert pp.mpi_env_for_bu("XYZ") is None


def _state(bu):
    call = SimpleNamespace(call_id="call-1", business_unit=bu)
    return {"call": call, "iqama_number": "1138157407", "node_trace": []}


def test_fetch_uses_the_branch_env(monkeypatch):
    seen = {}
    monkeypatch.setattr(pp, "get_token_for_env", lambda env: "token")

    def fake_get_patient(env, iqama, token=None):
        seen["env"] = env
        return SimpleNamespace(status_code=200, text="null")

    monkeypatch.setattr(pp, "get_patient", fake_get_patient)
    result = asyncio.run(pp.fetch_patient_profile(_state("LCH")))
    assert seen["env"] == "ALW"
    assert result["patient_profile_result"]["env"] == "ALW"


def test_fetch_reports_an_env_without_mpi_host(monkeypatch):
    monkeypatch.delenv("MPI_API_URL_ERS", raising=False)
    monkeypatch.setattr(pp, "get_patient", lambda *a, **k: pytest.fail("must not call MPI"))
    result = asyncio.run(pp.fetch_patient_profile(_state("ERS")))
    profile = result["patient_profile_result"]
    assert profile["found"] is False and profile["env"] == "ERS"
    assert "No MPI host configured" in profile["error"]
