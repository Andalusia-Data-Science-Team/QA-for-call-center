"""FAQ escalation detection and record-validation helpers."""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import pyodbc


FAQ_SQL_PATH = Path(__file__).resolve().parent.parent / "SQL" / "faq_query.sql"
FAQ_PASSCODE_PATH = Path(__file__).resolve().parent.parent / "Passcode.json"
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
        #"AgentEmail",
        "End Call Result",
        "CST Inquiry / Request Details",
    }
)

FAQ_RULES = {
    "C2B_017": ("C2B", "critical"),
    "C2B_021": ("C2B", "critical"),
    "C2C_023": ("C2C", "critical"),
    "C2C_024": ("C2C", "critical"),
}


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
    "تم تقديم طلب",
    "تم رفع طلب",
    "تم تقديم الطلب",
    "raise",
    "raised",
    "escalat",
)
_SEND_OR_TRANSFER_TERMS = (
   "تقديم",
    "تم تقديم طلب",
    "تم رفع طلب",
    "تم تقديم الطلب",
    "forward",
    "transfer",
)
_REQUEST_TERMS = (
    "شكوي",
    "بلاغ",
    "تذكره",
    "request",
    "complaint",
    "ticket",
    "case",
)
_DESTINATION_TERMS = (
    "فريق",
    "مختص",
    "معني",
    "مسؤول",
    "اداره",
    "team",
    "concerned",
    "responsible",
    "specialized",
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


def _agent_message_text(transcript: str) -> str:
    """Return labelled agent messages only."""
    agent_messages: list[str] = []
    for line in str(transcript or "").splitlines():
        match = re.match(r"^\s*([^:：]+)[:：]\s*(.*)$", line)
        if not match:
            continue
        role, content = match.groups()
        normalized_role = normalize_faq_text(role)
        if normalized_role in {"agent", "user", "csr", "advisor", "الموظف"}:
            agent_messages.append(content)
    return "\n".join(agent_messages)


def detect_faq_escalation(transcript: str) -> bool:
    """Detect a claim that a request was sent to a responsible department.
    
    Only checks agent messages to ensure the agent is the one claiming escalation.
    """
    agent_text = _agent_message_text(transcript)
    if not agent_text:
        return False
    text = normalize_faq_text(agent_text)
    strong_action = _contains_any(text, _RAISE_OR_ESCALATE_TERMS)
    transfer_action = _contains_any(text, _SEND_OR_TRANSFER_TERMS)
    has_request = _contains_any(text, _REQUEST_TERMS)
    has_destination = _contains_any(text, _DESTINATION_TERMS)
    return strong_action or (transfer_action and has_request)


def _normalize_email(value: object) -> str:
    return str(value or "").strip().casefold()


def _numeric_id(value: object) -> tuple[bool, int]:
    raw = str(value or "").strip()
    try:
        return True, int(raw)
    except ValueError:
        return False, -1


def _load_db_credentials(db_key: str = "SERVICE_HUB") -> dict[str, str]:
    """Load database credentials from Passcode.json."""
    try:
        with FAQ_PASSCODE_PATH.open("r", encoding="utf-8") as f:
            passcode = json.load(f)
        db_config = passcode.get("DB_NAMES", {}).get(db_key)
        if not db_config:
            raise ValueError(f"Database key '{db_key}' not found in Passcode.json")
        return db_config
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"Failed to load database credentials: {exc}") from exc


def _get_db_connection(db_key: str = "SERVICE_HUB") -> pyodbc.Connection:
    """Create a database connection using credentials from Passcode.json."""
    creds = _load_db_credentials(db_key)
    connection_string = (
        f"DRIVER={creds['driver']};"
        f"SERVER={creds['Server']};"
        f"DATABASE={creds['Database']};"
        f"UID={creds['UID']};"
        f"PWD={creds['PWD']};"
        "charset=utf8;"
    )
    return pyodbc.connect(connection_string)


def lookup_faq_record(
    *,
    patient_phone: str,
    call_date: str,
    agent_name: str,
    agent_email: str | None,
    db_key: str = "SERVICE_HUB",
) -> dict[str, Any]:
    """Return the newest same-day FAQ row for the call and agent identity.
    
    Queries the SQL database instead of reading a CSV file.
    """
    expected_phone = normalize_faq_phone(patient_phone)
    expected_date = str(call_date or "").strip()
    expected_name = normalize_faq_text(agent_name)
    expected_email = _normalize_email(agent_email)

    try:
        # Load SQL query from file
        query = FAQ_SQL_PATH.read_text(encoding="utf-8")
        
        # Get database connection
        conn = _get_db_connection(db_key)
        cursor = conn.cursor()
        
        # Execute query with parameters (using ? placeholders for pyodbc)
        cursor.execute(query, expected_phone, expected_phone, expected_date)
        
        # Fetch all matching rows
        columns = [column[0] for column in cursor.description]
        rows = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        # Check if required columns are present
        missing_columns = sorted(FAQ_REQUIRED_COLUMNS - set(columns))
        if missing_columns:
            return {
                "status": "unavailable",
                "record": None,
                "message": (
                    "FAQ database query missing required columns: "
                    + ", ".join(missing_columns)
                ),
            }
        
        # Convert rows to dictionaries
        candidates: list[dict[str, Any]] = []
        for row in rows:
            row_dict = dict(zip(columns, row))
            
            # Match by agent identity
            row_email = _normalize_email(row_dict.get("AgentEmail"))
            row_name = normalize_faq_text(row_dict.get("AgentName"))
            email_matches = bool(expected_email and row_email == expected_email)
            name_matches = bool(expected_name and row_name == expected_name)
            
            if not email_matches and not name_matches:
                continue
            
            id_is_numeric, numeric_id = _numeric_id(row_dict.get("ID"))
            candidates.append(
                {
                    "record": row_dict,
                    "identity_rank": 2 if email_matches else 1,
                    "identity": "email" if email_matches else "name",
                    "id_is_numeric": id_is_numeric,
                    "numeric_id": numeric_id,
                }
            )
    except (OSError, pyodbc.Error, RuntimeError) as exc:
        return {
            "status": "unavailable",
            "record": None,
            "message": f"FAQ database query failed: {exc}",
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


def missing_faq_evaluation(message: str) -> dict[str, Any]:
    """Build the required missing-record violation without an LLM call."""
    return {
        "faq_status": "violation",
        "summary": message,
        "field_checks": [],
        "faq_flags": [
            {
                "type": "C2B",
                "severity": "critical",
                "description": "C2B_017: FAQ request was not recorded on the call date.",
                "transcript_excerpt": "N/A",
            }
        ],
    }


def normalize_faq_evaluation(data: dict[str, Any]) -> dict[str, Any]:
    """Enforce allowed YAML rule mappings and the public flag schema."""
    field_checks = [
        check for check in (data.get("field_checks") or []) if isinstance(check, dict)
    ]
    normalized_flags: list[dict[str, str]] = []
    seen: set[tuple[str, ...]] = set()

    def add_flag(
        rule_id: str,
        field: object,
        reason: object,
        excerpt: object,
    ) -> None:
        if len(normalized_flags) >= 4:
            return
        resolved_rule = rule_id if rule_id in FAQ_RULES else "C2B_021"
        flag_type, severity = FAQ_RULES[resolved_rule]
        field_name = str(field or "FAQ record").strip()
        evidence = str(excerpt or "N/A").strip() or "N/A"
        explanation = str(reason or "does not match the call facts").strip()
        normalized_evidence = normalize_faq_text(evidence)
        key = (
            (resolved_rule, normalize_faq_text(field_name), normalized_evidence)
            if evidence.upper() == "N/A"
            else (resolved_rule, normalized_evidence)
        )
        if key in seen:
            return
        seen.add(key)
        normalized_flags.append(
            {
                "type": flag_type,
                "severity": severity,
                "description": (
                    f"{resolved_rule}: FAQ mismatch in {field_name}: {explanation}"
                ),
                "transcript_excerpt": evidence,
            }
        )

    for check in field_checks:
        if check.get("matches") is not False:
            continue
        add_flag(
            str(check.get("rule_id") or "C2B_021").strip(),
            check.get("field"),
            check.get("reason"),
            check.get("transcript_excerpt"),
        )

    for raw_flag in data.get("faq_flags") or []:
        if not isinstance(raw_flag, dict):
            continue
        description = str(raw_flag.get("description") or "")
        match = re.search(r"\b(C2[BC]_\d{3})\b", description)
        add_flag(
            str(raw_flag.get("rule_id") or (match.group(1) if match else "C2B_021")),
            raw_flag.get("field") or "FAQ record",
            description or "does not match the call facts",
            raw_flag.get("transcript_excerpt"),
        )

    if data.get("faq_status") == "violation" and not normalized_flags:
        add_flag(
            "C2B_021",
            "FAQ record",
            "The model reported a violation without a valid field-level rule.",
            "N/A",
        )

    return {
        "faq_status": "violation" if normalized_flags else "match",
        "summary": str(data.get("summary") or "").strip(),
        "field_checks": field_checks,
        "faq_flags": normalized_flags,
    }
