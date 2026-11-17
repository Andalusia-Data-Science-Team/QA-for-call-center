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


def test_expired_token_422_refreshes_with_ntlm_and_retries_once(monkeypatch):
    seen = []
    monkeypatch.setenv("MPI_NTLM_USERNAME", "ntlm-user")
    monkeypatch.setenv("MPI_NTLM_PASSWORD", "ntlm-password")
    monkeypatch.setattr(pp, "get_token_for_env", lambda env: "expired-token")

    def fake_login(env, username, password, ntlm_username=None, ntlm_password=None):
        assert (env, ntlm_username, ntlm_password) == ("SNB", "ntlm-user", "ntlm-password")
        return "fresh-token"

    def fake_get_patient(env, iqama, token=None):
        seen.append((env, token))
        if len(seen) == 1:
            return SimpleNamespace(status_code=422, text="You are unauthorized ! Access is denied due to Token Expire for AiModule")
        return SimpleNamespace(status_code=200, text="null")

    monkeypatch.setattr(pp, "login", fake_login)
    monkeypatch.setattr(pp, "get_patient", fake_get_patient)
    result = asyncio.run(pp.fetch_patient_profile(_state("SNB")))

    assert seen == [("SNB", "expired-token"), ("SNB", "fresh-token")]
    assert result["patient_profile_result"]["env"] == "SNB"
    assert result["patient_profile_result"]["error"] == "Patient not found in MPI"
    assert pp.os.environ["MPI_AUTH_TOKEN_SNB"] == "fresh-token"


def test_unrelated_422_does_not_retry(monkeypatch):
    seen = []
    monkeypatch.setattr(pp, "get_token_for_env", lambda env: "token")
    monkeypatch.setattr(pp, "login", lambda *a, **k: pytest.fail("must not refresh"))

    def fake_get_patient(env, iqama, token=None):
        seen.append(env)
        return SimpleNamespace(status_code=422, text="Unprocessable entity")

    monkeypatch.setattr(pp, "get_patient", fake_get_patient)
    result = asyncio.run(pp.fetch_patient_profile(_state("SNB")))

    assert seen == ["SNB"]
    assert result["patient_profile_result"]["error"] == "MPI returned status 422"


def test_fetch_reports_an_env_without_mpi_host(monkeypatch):
    monkeypatch.delenv("MPI_API_URL_ERS", raising=False)
    monkeypatch.setattr(pp, "get_patient", lambda *a, **k: pytest.fail("must not call MPI"))
    result = asyncio.run(pp.fetch_patient_profile(_state("ERS")))
    profile = result["patient_profile_result"]
    assert profile["found"] is False and profile["env"] == "ERS"
    assert "No MPI host configured" in profile["error"]
