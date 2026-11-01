from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Optional, Union

import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

from app.services.text_helpers import normalize_arabic_text


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

# Department carries the CM WebStoreName VERBATIM (no translation) — the
# real store names, queried live from
# [CM_DWH].[msc_sub_35033_XYLJV].[dbo].[Webstores] (2026-10-19), are listed
# here only as documentation of the live vocabulary:
#   Andalusia Plus, Approval (stored with a trailing space in CM),
#   Call Center, Confirm-NoShow, Group One, Heroes, Home Care,
#   Medical Concierge, Medical Contact, Patient Experience, VIP
# The department validator passes any non-empty string through unchanged;
# the criteria loader falls back to general.yaml for unmapped departments.
WEBSTORE_TO_DEPARTMENT: dict[str, str] = {}

TAG1_TO_BU: dict[str, str] = {
    "AHJ": "AHJ", "HJH": "AHJ", "MKR": "ADC", "ADC": "ADC", "JDC": "ADC",
    "LCH": "LCH", "SNB": "SNB", "ALW": "ALW", "AKW": "AKW",
}

BUSINESS_UNIT_KEYWORD_MAP: dict[str, str] = {
    "فرع سلطان": "LCH",
    "مكرونة": "ADC",
    "المكرونة": "ADC",
    "سلطان": "LCH",
    "حي الجامعة": "AHJ",
    "سنابل": "SNB",
    "السنابل": "SNB",
    "صحة المرأة": "ALW",
    "المرأة": "ALW",
    "صحة الطفل": "AKW",
    "الطفل": "AKW",
    "مستشفى اندلسية جدة": "AHJ",
    "مستشفى اندلسية": "AHJ",
    "عيادات الأسنان-المكرونة": "ADC",
    "عيادات أندلسية-السنابل": "SNB",
    "عيادات الأسنان-فرع سلطان": "LCH",
    "عيادات اندلسيه-فرع سلطان": "LCH",
    "عيادات أندلسية-صاري": "AKW",

    # Explicit BU labels. Each equivalence group is one branch
    # (crm_leads_validation.LEAD_BU_CODE) and resolves to one code:
    # AHJ/HJH → AHJ, LCH/CHT/ALW → ALW, MKR/ADC/JDC → ADC. A dict keeps one
    # value per key, so list each code once.
    "AKW": "AKW",
    "ALW": "ALW",
    "LCH": "ALW",
    "CHT": "ALW",
    "AHJ": "AHJ",
    "HJH": "AHJ",
    "AFW": "AFW",
    "ADC": "ADC",
    "JDC": "ADC",
    "MKR": "ADC",

    # Attribution/tag forms
    "BU-AKW": "AKW",
    "BU-ALW": "ALW",
    "BU-LCH": "ALW",
    "BU-AHJ": "AHJ",
    "BU-AFW": "AFW",
    "BU-MKR": "ADC",
    "BU-ADC": "ADC",
    "BU-SNB": "SNB",
    "BU-ALW-Derma": "ALW",
    "BU-Clinics-Dental": "AFW"
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
    if value is pd.NaT:
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
    # Every agent who replied in the chat (OwnerName of SenderType "User"
    # rows), in order of first reply; more than one when it was transferred.
    agent_names: list[str] = Field(default_factory=list)
    agent_email: Optional[str] = None
    Patient_Phone: str
    call_date: str
    archive_datetime: Optional[datetime] = None
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
        # Department now carries the CM WebStoreName (NEWFIX_CM_Chat_Search),
        # whose vocabulary is open-ended — a new store name must not crash
        # evaluation. Normalise blanks to the historical default and keep
        # everything else verbatim; the criteria loader falls back to
        # general.yaml for unmapped departments.
        v = (v or "").strip()
        if not v:
            return "Helpdesk"
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

    @field_validator("answer_state", mode="before")
    @classmethod
    def validate_answer_state(cls, v) -> Optional[str]:
        # Keep CM's raw numeric codes without assigning undocumented meanings.
        if v is None:
            return None
        v_str = str(v).strip()
        if not v_str:
            return None
        if v_str not in {"OnTime", "Late"} and not v_str.isdecimal():
            raise ValueError(f"answer_state must be a CM numeric code, 'OnTime', or 'Late' — got '{v_str}'")
        return v_str

    @model_validator(mode="after")
    def detect_business_unit(self) -> "CallTranscript":
        if self.business_unit is not None:
            return self
        # The patient's explicit BU selection takes precedence over branch
        # names in an agent's generic list of locations.
        patient_bu_tags = re.findall(
            r"(?im)^Patient:[^\n]*\bBU-([A-Z]{2,3})(?![A-Z])",
            self.transcript,
        )
        for tag in reversed(patient_bu_tags):
            code = tag.upper()
            selected_bu = (
                BUSINESS_UNIT_KEYWORD_MAP.get(f"BU-{code}")
                or BUSINESS_UNIT_KEYWORD_MAP.get(code)
            )
            if selected_bu:
                self.business_unit = selected_bu
                return self
        text = normalize_arabic_text(self.transcript)
        # A named Sanabel branch is more specific than a passing mention of
        # Andalusia Hospital elsewhere in the conversation.
        if re.search(r"\bفرع\s+(?:ال)?سنابل\b", text):
            self.business_unit = "SNB"
            return self
        # Otherwise use the longest normalised alias, as bank validation does.
        best_len = 0
        for keyword, code in BUSINESS_UNIT_KEYWORD_MAP.items():
            alias = normalize_arabic_text(keyword)
            if alias and alias in text and len(alias) > best_len:
                self.business_unit, best_len = code, len(alias)
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
        archive_dt = (
            _to_datetime(head.get("ArchiveDateTime"))
            or _to_datetime(head.get("Archive_DateTime"))
        )
        fwd_dt = _to_datetime(head.get("ForwardedTime"))

        if start_dt is None:
            raise ValueError(f"Start_DateTime missing for UniqueId={head.get('UniqueId')}")

        call_date = (archive_dt or start_dt).strftime("%Y-%m-%d")
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
        # Department IS the CM WebStoreName, VERBATIM (NEWFIX_CM_Chat_Search
        # joins [Webstores] as ws.WebStoreName) — no translation to the QA
        # department vocabulary. Tag1/Tag2 are commented out of that query,
        # so the legacy tag2 lookup only applies when the store name is
        # blank; a blank department then keeps the "Helpdesk" default.
        tag1 = (head.get("Tag1") or "").strip()
        tag2 = (head.get("Tag2") or "").strip()
        web_store = (head.get("WebStoreName") or "").strip()
        department = (
            web_store
            or TAG2_TO_DEPARTMENT.get(tag2, "Helpdesk")
        )
        business_unit = TAG1_TO_BU.get(tag1)   # None → keyword fallback in validator

        # -- Transcript assembly ----------------------------------------------
        lines: list[str] = []
        message_count = 0
        bot_handoff_occurred = False
        first_agent_seen = False
        agent_names: list[str] = []

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
                owner = (row.get("OwnerName") or "").strip()
                if owner and owner not in agent_names:
                    agent_names.append(owner)

            ts_dt = _to_datetime(row.get("CreationDateTime"))
            ts = ts_dt.strftime("%H:%M:%S") if ts_dt else "??:??:??"
            lines.append(f"{label}: {content}")
            message_count += 1

        return cls(
            call_id=str(head.get("UniqueId") or ""),
            conversation_link=head.get("ConversationLink"),
            agent_name=head.get("AgentFullName") or "",
            agent_names=agent_names,
            agent_email=head.get("EmailAddress"),
            Patient_Phone=raw_phone,
            call_date=call_date,
            archive_datetime=archive_dt,
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
