"""FAQ escalation detection and record-validation helpers."""

from __future__ import annotations

import csv
import re
import unicodedata
from pathlib import Path
from typing import Any


FAQ_CSV_PATH = Path(__file__).with_name("FAQ_latest.csv")
FAQ_REQUIRED_COLUMNS = frozenset(
    {
        "ID",
        "CustomerName",
        "mobile_phone",
        "BU",
        "Date",
        "Inquiry",
        "Response",
        "AgentName",
        "AgentEmail",
        "End Call Result",
    }
)


_ARABIC_CHAR_TRANSLATION = str.maketrans(
    {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ى": "ي",
        "ة": "ه",
        "ؤ": "و",
        "ئ": "ي",
        "ـ": "",
    }
)

_RAISE_OR_ESCALATE_TERMS = (
    "رفع",
    "تصعيد",
    "صعد",
    "raise",
    "raised",
    "escalat",
)
_SEND_OR_TRANSFER_TERMS = (
    "ارسل",
    "حول",
    "تحويل",
    "submit",
    "send",
    "sent",
    "forward",
    "transfer",
)
_REQUEST_TERMS = (
    "طلب",
    "استفسار",
    "شكوي",
    "بلاغ",
    "تذكره",
    "request",
    "inquiry",
    "complaint",
    "ticket",
    "case",
)
_DESTINATION_TERMS = (
    "قسم",
    "جهه",
    "فريق",
    "مختص",
    "معني",
    "مسؤول",
    "اداره",
    "department",
    "team",
    "concerned",
    "responsible",
    "speciali",
)


def normalize_faq_text(value: object) -> str:
    """Return a comparison-friendly Arabic/English representation."""
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.translate(_ARABIC_CHAR_TRANSLATION).casefold()
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def normalize_faq_phone(value: object) -> str:
    """Normalize Saudi local and country-code phone forms to one value."""
    digits = re.sub(r"\D", "", str(value or ""))
    if digits.startswith("00966"):
        digits = digits[5:]
    elif digits.startswith("966"):
        digits = digits[3:]
    if len(digits) == 10 and digits.startswith("0"):
        digits = digits[1:]
    return digits


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def detect_faq_escalation(transcript: str) -> bool:
    """Detect a claim that a request was sent to a responsible department."""
    text = normalize_faq_text(transcript)
    strong_action = _contains_any(text, _RAISE_OR_ESCALATE_TERMS)
    transfer_action = _contains_any(text, _SEND_OR_TRANSFER_TERMS)
    has_request = _contains_any(text, _REQUEST_TERMS)
    has_destination = _contains_any(text, _DESTINATION_TERMS)
    return has_destination and (strong_action or (transfer_action and has_request))


def _normalize_email(value: object) -> str:
    return str(value or "").strip().casefold()


def _numeric_id(value: object) -> tuple[bool, int]:
    raw = str(value or "").strip()
    try:
        return True, int(raw)
    except ValueError:
        return False, -1


def lookup_faq_record(
    *,
    patient_phone: str,
    call_date: str,
    agent_name: str,
    agent_email: str | None,
    csv_path: Path | None = None,
) -> dict[str, Any]:
    """Return the newest same-day FAQ row for the call and agent identity."""
    path = Path(csv_path) if csv_path is not None else FAQ_CSV_PATH
    expected_phone = normalize_faq_phone(patient_phone)
    expected_date = str(call_date or "").strip()
    expected_name = normalize_faq_text(agent_name)
    expected_email = _normalize_email(agent_email)

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            columns = set(reader.fieldnames or [])
            missing_columns = sorted(FAQ_REQUIRED_COLUMNS - columns)
            if missing_columns:
                return {
                    "status": "unavailable",
                    "record": None,
                    "message": (
                        "FAQ CSV is missing required columns: "
                        + ", ".join(missing_columns)
                    ),
                }

            candidates: list[dict[str, Any]] = []
            for row in reader:
                if normalize_faq_phone(row.get("mobile_phone")) != expected_phone:
                    continue
                if str(row.get("Date") or "").strip() != expected_date:
                    continue

                row_email = _normalize_email(row.get("AgentEmail"))
                row_name = normalize_faq_text(row.get("AgentName"))
                email_matches = bool(expected_email and row_email == expected_email)
                name_matches = bool(expected_name and row_name == expected_name)
                if not email_matches and not name_matches:
                    continue

                id_is_numeric, numeric_id = _numeric_id(row.get("ID"))
                candidates.append(
                    {
                        "record": dict(row),
                        "identity_rank": 2 if email_matches else 1,
                        "identity": "email" if email_matches else "name",
                        "id_is_numeric": id_is_numeric,
                        "numeric_id": numeric_id,
                    }
                )
    except (OSError, UnicodeError, csv.Error) as exc:
        return {
            "status": "unavailable",
            "record": None,
            "message": f"FAQ CSV could not be read: {exc}",
        }

    if not candidates:
        return {
            "status": "not_found",
            "record": None,
            "message": "No same-day FAQ record matched the call and agent identity.",
        }

    selected = max(
        candidates,
        key=lambda item: (
            item["identity_rank"],
            item["id_is_numeric"],
            item["numeric_id"],
        ),
    )
    return {
        "status": "found",
        "record": selected["record"],
        "match": {
            "identity": selected["identity"],
            "date": expected_date,
            "faq_id": selected["record"].get("ID"),
        },
        "message": "Same-day FAQ record found.",
    }
