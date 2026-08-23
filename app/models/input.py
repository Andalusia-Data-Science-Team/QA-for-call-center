from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Optional, Union

import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Constants (unchanged from previous version)
# ---------------------------------------------------------------------------

VALID_DEPARTMENTS = {
    "Scheduling", "Onboarding", "Helpdesk", "Follow-Ups", "Records",
}

TAG2_TO_DEPARTMENT: dict[str, str] = {
    "Offers": "Scheduling",
    "Appointment": "Scheduling",
    "Follow-Up": "Follow-Ups",
    "Records": "Records",
    "Support": "Helpdesk",
    "Onboarding": "Onboarding",
}

TAG1_TO_BU: dict[str, str] = {
    "AHJ": "AHJ", "MKR": "MKR", "LCH": "LCH",
    "LIVE": "LIVE", "SNB": "SNB", "ALW": "ALW", "AKW": "AKW",
}

BUSINESS_UNIT_KEYWORD_MAP: dict[str, str] = {
    "فرع سلطان": "LCH", "مكرونة": "MKR", "المكرونة": "MKR",
    "سلطان": "LCH", "جدة": "LIVE", "حي الجامعة": "LIVE",
    "سنابل": "SNB", "السنابل": "SNB", "صحة المرأة": "ALW",
    "المرأة": "ALW", "صحة الطفل": "AKW", "الطفل": "AKW",
    "BU-AKW":"AKW","BU-ALW":"ALW","BU-LCH":"LCH","BU-AHJ":"LIVE",
    "BU-MKR":"MKR","BU-SNB":"SNB"
}

SENDER_LABEL: dict[str, str] = {
    "User": "Agent",
    "Relation": "Patient",
    "Bot": "Bot",
}

_EXCEL_EPOCH = datetime(1899, 12, 30)


def _to_datetime(value) -> Optional[datetime]:
    """
    Unified timestamp coercer.

    SQL Server via pandas  →  pd.Timestamp  (most common)
    Excel file / legacy    →  float serial  (fallback)
    Already a datetime     →  pass-through
    None / NaT / NaN       →  None
    """
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.to_pydatetime()
    if isinstance(value, datetime):
        return value
    # Excel serial float
    try:
        return _EXCEL_EPOCH + timedelta(days=float(value))
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CallTranscript(BaseModel):
    call_id: str
    conversation_link: Optional[str] = None
    agent_name: str
    agent_email: Optional[str] = None
    Patient_Phone: str
    call_date: str
    call_duration_seconds: int = Field(..., ge=0)
    first_response_minutes: int = Field(default=0, ge=0)
    department: str
    topic: Optional[str] = None
    business_unit: Optional[str] = None
    channel: str = "WhatsApp"
    answer_state: Optional[str] = None
    service_level_on_time: bool = True
    is_answered: bool = True
    was_forwarded: bool = False
    transcript: str = Field(..., min_length=1)
    message_count: int = Field(default=0, ge=0)
    bot_handoff_occurred: bool = False

    # -- Validators -----------------------------------------------------------

    @field_validator("call_date")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
            raise ValueError("call_date must be YYYY-MM-DD")
        return v

    @field_validator("department")
    @classmethod
    def validate_department(cls, v: str) -> str:
        if v not in VALID_DEPARTMENTS:
            raise ValueError(f"Unknown department '{v}'. Valid: {VALID_DEPARTMENTS}")
        return v

    @field_validator("transcript")
    @classmethod
    def transcript_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("transcript cannot be blank")
        return v.strip()

    @field_validator("Patient_Phone")
    @classmethod
    def validate_phone_number(cls, v: str) -> str:
        v = str(v).strip()
        if not re.fullmatch(r"\+?\d{9,15}", v):
            raise ValueError("Patient_Phone must be 9–15 digits")
        if v.startswith("+966"):
            v = v[4:]
        elif v.startswith("966") and len(v) > 10:
            v = v[3:]
        return v

    @field_validator("answer_state")
    @classmethod
    def validate_answer_state(cls, v: Optional[str]) -> Optional[str]:
        if v not in {"OnTime", "Late", None}:
            raise ValueError(f"answer_state must be 'OnTime', 'Late', or None — got '{v}'")
        return v

    @model_validator(mode="after")
    def detect_business_unit(self) -> "CallTranscript":
        if self.business_unit is not None:
            return self
        for keyword, code in BUSINESS_UNIT_KEYWORD_MAP.items():
            if keyword in (self.transcript or ""):
                self.business_unit = code
                break
        return self

    # -- Factory --------------------------------------------------------------

    @classmethod
    def from_cm_rows(cls, rows: list[dict]) -> "CallTranscript":
        """
        Build a CallTranscript from a list of row-dicts for one conversation.

        Rows must share the same UniqueId and come from the SQL query above.
        Timestamps are pd.Timestamp objects (SQL Server via pandas).
        Numeric columns (PatientPhoneNumber, ServiceLevelOnTime, etc.) are float64.
        """
        if not rows:
            raise ValueError("rows list is empty")

        # Sort by message timestamp before anything else
        rows_sorted = sorted(
            rows,
            key=lambda r: _to_datetime(r.get("CreationDateTime")) or datetime.min,
        )
        head = rows_sorted[0]

        # -- Timing -----------------------------------------------------------
        start_dt = _to_datetime(head.get("Start_DateTime"))
        archive_dt = _to_datetime(head.get("Archive_DateTime"))
        fwd_dt = _to_datetime(head.get("ForwardedTime"))

        if start_dt is None:
            raise ValueError(f"Start_DateTime missing for UniqueId={head.get('UniqueId')}")

        call_date = start_dt.strftime("%Y-%m-%d")
        duration_s = (
            max(0, int((archive_dt - start_dt).total_seconds()))
            if archive_dt else 0
        )
        # ForwardedTime == Start_DateTime (±5 s) means no real transfer happened
        was_forwarded = (
            fwd_dt is not None
            and abs((fwd_dt - start_dt).total_seconds()) > 5
        )

        # -- Numeric columns (float64 from pandas) ----------------------------
        raw_phone = str(int(head.get("PatientPhoneNumber") or 0))
        first_resp = int(head.get("First_Response_Minutes") or 0)
        sla_on_time = bool(head.get("ServiceLevelOnTime") or 0)
        answered = bool(head.get("IsAnswered") or 0)

        # -- Department & BU --------------------------------------------------
        tag1 = (head.get("Tag1") or "").strip()
        tag2 = (head.get("Tag2") or "").strip()
        department = TAG2_TO_DEPARTMENT.get(tag2, "Helpdesk")
        business_unit = TAG1_TO_BU.get(tag1)   # None → keyword fallback in validator

        # -- Transcript assembly ----------------------------------------------
        lines: list[str] = []
        message_count = 0
        bot_handoff_occurred = False
        first_agent_seen = False

        for row in rows_sorted:
            sender_type = (row.get("SenderType") or "").strip()
            content = (row.get("Content") or "").strip()
            if not content:
                continue

            label = SENDER_LABEL.get(sender_type)

            if label == "Bot":
                if not first_agent_seen:
                    bot_handoff_occurred = True
                continue                         # exclude bot messages from transcript

            if label is None:
                continue                         # skip unknown sender types

            if label == "Agent":
                first_agent_seen = True

            ts_dt = _to_datetime(row.get("CreationDateTime"))
            ts = ts_dt.strftime("%H:%M:%S") if ts_dt else "??:??:??"
            lines.append(f"{label}: {content}")
            message_count += 1

        return cls(
            call_id=str(head.get("UniqueId") or ""),
            conversation_link=head.get("ConversationLink"),
            agent_name=head.get("AgentFullName") or "",
            agent_email=head.get("EmailAddress"),
            Patient_Phone=raw_phone,
            call_date=call_date,
            call_duration_seconds=duration_s,
            first_response_minutes=first_resp,
            department=department,
            topic=tag2 or None,
            business_unit=business_unit,
            channel=head.get("ConversationChannel") or "WhatsApp",
            answer_state=head.get("AnswerState"),
            service_level_on_time=sla_on_time,
            is_answered=answered,
            was_forwarded=was_forwarded,
            transcript="\n".join(lines),
            message_count=message_count,
            bot_handoff_occurred=bot_handoff_occurred,
        )


class BatchCallTranscripts(BaseModel):
    calls: list[CallTranscript] = Field(..., min_length=1, max_length=50)