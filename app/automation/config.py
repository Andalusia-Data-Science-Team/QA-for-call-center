"""
Configuration for the automated QA sampling layer.

Kept separate from app.config.Settings so that the automation process and
the API process can be configured independently, and so that importing the
API never pulls automation defaults into its environment surface.

Every setting is read from the QA_-prefixed environment variables listed in
docs/AUTOMATION_IMPLEMENTATION_PLAN.md.  Defaults are deliberately inert:
`enabled` is False, so a deployment that merges this package but has not yet
opted in does nothing.
"""

from __future__ import annotations

from functools import lru_cache
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AutomationSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="QA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Master switch ────────────────────────────────────────────────────────
    automation_enabled: bool = True

    # ── Budget ───────────────────────────────────────────────────────────────
    daily_target: int = Field(default=300, ge=1)

    # ── Reporting calendar ───────────────────────────────────────────────────
    # Africa/Cairo: UTC+2 (EET) in winter, UTC+3 (EEST) under DST, which Egypt
    # reinstated in 2023 — last Friday of April to last Thursday of October.
    # A reporting day is therefore 23, 24 or 25 hours long depending on the
    # date; app.automation.calendar derives every boundary from the real tz
    # database rather than from a fixed offset.
    #
    # Use 'Africa/Cairo', not 'EET'. Both are valid IANA zones and both observe
    # DST, but EET follows EU rules (last Sunday of March and October) while
    # Egypt switches on the last Friday of April and the last Thursday of
    # October — weeks apart, twice a year.
    reporting_timezone: str = "Africa/Cairo"
    sampling_interval_minutes: int = Field(default=120, ge=1)
    source_grace_minutes: int = Field(default=20, ge=0)

    # ── Scheduling ───────────────────────────────────────────────────────────
    dispatch_interval_seconds: int = Field(default=600, ge=1)

    # ── Sampling policy ──────────────────────────────────────────────────────
    sampling_seed: str = "42"
    sampling_policy_version: str = "v1"
    exclude_adjacent_chats: bool = True
    min_chat_gap_minutes: int = Field(default=0, ge=0)

    # ── Local ledger ─────────────────────────────────────────────────────────
    # We have no DDL rights on DWH, so every table the automation owns lives in
    # this SQLite file. DWH is read-and-insert only. The path must be on
    # durable storage and included in the host's backups: the frozen candidate
    # snapshots are the audit trail for past selections and cannot be
    # reconstructed once the source data has moved on.
    ledger_db_path: str = "data/qa_automation.sqlite3"

    # ── Worker ───────────────────────────────────────────────────────────────
    # LLM requests are network-bound; a few independent workers keep the
    # evaluation queue moving while each chat awaits several provider calls.
    worker_concurrency: int = Field(default=3, ge=1)
    max_attempts: int = Field(default=3, ge=1)
    lease_duration_minutes: int = Field(default=10, ge=1)

    @field_validator("reporting_timezone")
    @classmethod
    def validate_timezone(cls, v: str) -> str:
        """Fail at startup, not at the first midnight boundary."""
        try:
            ZoneInfo(v)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(
                f"Unknown reporting timezone {v!r}. This must be an IANA zone "
                f"name such as 'Africa/Cairo' or 'Asia/Riyadh' — a fixed offset "
                f"like 'UTC+2' cannot express daylight-saving transitions."
            ) from exc
        return v

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.reporting_timezone)


@lru_cache(maxsize=1)
def automation_settings() -> AutomationSettings:
    """
    Load the automation settings once, on first use.

    Deliberately lazy rather than a module-level instance. A bad value — an
    unparseable timezone, a negative quota — must fail where it is read, not
    at import. The same eager-instantiation pattern already bit this codebase
    once: see the note on `openrouter_api_key` in app/config.py, where a
    `None` default crashed the entire application at import time.
    """
    return AutomationSettings()
