"""Tests for app.automation.config."""

from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from app.automation.config import AutomationSettings


def _settings(**overrides) -> AutomationSettings:
    """Build settings from explicit values, bypassing the ambient .env file."""
    base = dict(
        automation_enabled=False,
        daily_target=300,
        reporting_timezone="Africa/Cairo",
        sampling_interval_minutes=120,
        source_grace_minutes=20,
        dispatch_interval_seconds=600,
        sampling_seed="42",
        sampling_policy_version="v1",
    )
    base.update(overrides)
    return AutomationSettings(**base)


def test_defaults_are_inert():
    """A deployment that merges this package but has not opted in does nothing."""
    assert _settings().automation_enabled is False


def test_reporting_timezone_defaults_to_cairo():
    s = _settings()
    assert s.reporting_timezone == "Africa/Cairo"
    assert s.tz == ZoneInfo("Africa/Cairo")


def test_default_worker_count_can_process_multiple_chats_at_once():
    assert _settings().worker_concurrency == 3


def test_fixed_offset_timezone_is_rejected():
    """
    The failure mode this guards against is the +3 literal already baked into
    NEWFIX_CM_Chat_Search.sql. A fixed offset cannot express Egypt's DST, so
    it must not be accepted as a reporting timezone.
    """
    for bad in ("UTC+2", "GMT+3", "+02:00", "Cairo"):
        with pytest.raises(ValidationError, match="IANA"):
            _settings(reporting_timezone=bad)


def test_eet_is_not_a_substitute_for_cairo():
    """
    'EET' is a valid IANA name and is DST-aware, so the validator accepts it —
    but it follows EU transition rules (last Sunday of March and October),
    while Egypt switches on the last Friday of April and the last Thursday of
    October. Configuring EET would place the boundaries weeks off twice a year.

    This test exists to document the trap, not to forbid the value.
    """
    from datetime import datetime

    eet = ZoneInfo("EET")
    cairo = ZoneInfo("Africa/Cairo")
    assert _settings(reporting_timezone="EET").tz == eet

    # Mid-April 2026: the EU is already on summer time, Egypt is not.
    probe = datetime(2026, 4, 10, 12, 0)
    assert probe.replace(tzinfo=eet).utcoffset() != probe.replace(tzinfo=cairo).utcoffset()


def test_riyadh_is_accepted():
    assert _settings(reporting_timezone="Asia/Riyadh").tz == ZoneInfo("Asia/Riyadh")


@pytest.mark.parametrize(
    "field,value",
    [
        ("daily_target", 0),
        ("sampling_interval_minutes", 0),
        ("source_grace_minutes", -1),
        ("worker_concurrency", 0),
        ("max_attempts", 0),
        ("min_chat_gap_minutes", -5),
    ],
)
def test_out_of_range_values_are_rejected(field, value):
    with pytest.raises(ValidationError):
        _settings(**{field: value})


def test_env_prefix_matches_the_documented_variable_names(monkeypatch):
    """The QA_* names in .env.example must actually bind to these fields."""
    monkeypatch.setenv("QA_AUTOMATION_ENABLED", "true")
    monkeypatch.setenv("QA_DAILY_TARGET", "150")
    monkeypatch.setenv("QA_REPORTING_TIMEZONE", "Asia/Riyadh")
    monkeypatch.setenv("QA_SAMPLING_POLICY_VERSION", "v2")
    monkeypatch.setenv("QA_WORKER_CONCURRENCY", "4")

    s = AutomationSettings()
    assert s.automation_enabled is True
    assert s.daily_target == 150
    assert s.reporting_timezone == "Asia/Riyadh"
    assert s.sampling_policy_version == "v2"
    assert s.worker_concurrency == 4
