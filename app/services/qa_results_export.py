"""Extract the Customer Care sheet schema and export displayed QA calls to XLSX.

The supplied workbook preformats nearly a million rows. Recreate its visible
two-row A:BF schema instead of loading those empty rows into every download.
"""

from __future__ import annotations

from datetime import date, datetime
from io import BytesIO
from math import isfinite
from re import search
from typing import Mapping, Sequence

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from app.services.regulation_catalog import matched_regulation_flags, policy_outcome


# Extracted from "Customer Care" in Quality_Monitoring Form_2026_Inbound.xlsx.
CUSTOMER_CARE_HEADERS = (
    "Month", "Week", "Serial", "Monitoring Type", "HR ID", "BU", "Task",
    "Supervisor ", "Team Leader", "Agent", "1st Day Login", "Call Date",
    "Call Time", "Duration", " Phone Number", "C2C", "C2B", "C2Com", "NC",
    "Pillar", "Attribute", "Comment", "Others",
    "Pillar", "Attribute", "Comment", "Others",
    "Pillar", "Attribute", "Comment", "Others",
    "Pillar", "Attribute", "Comment", "Others",
    "Pillar", "Attribute", "Comment", "Others",
    "Dep", "Pillar", "Attribute", "Comment", "Others", "Monitoring Date",
    "Monitored By", "Last Call Result", "Medical Interest",
    "Know Us From / Call Source", "Clear Comment", "Agent Classification",
    "Profiling Comment", "FCR", "FCR Reason", "Coaching Status",
    "Coaching Date", "Coaching By", "Action Plan",
)
EXTRA_HEADERS = ("All Escalations", "Chat Link", "Call ID")
ERROR_STARTS = (20, 24, 28, 32, 36)  # T, X, AB, AF, AJ
ERROR_COUNT_COLUMNS = {"C2C": "P", "C2B": "Q", "C2Com": "R", "NC": "S"}

# Colors and layout follow the supplied Customer Care sheet's green grid,
# pale green agent fields, light error sections, and yellow-on-dark headers.
GRID_GREEN = "92D050"
PALE_GREEN = "E2F0D9"
PALE_BLUE = "EAF4FB"
COUNT_BLUE = "DDEBF7"
INK = "17365D"
DARK = "101010"
YELLOW = "FFFF00"
COLUMN_WIDTHS = {
    1: 7, 2: 6, 3: 7, 4: 19, 5: 7, 6: 9, 7: 13, 8: 19,
    9: 14, 10: 37, 11: 14, 12: 11, 13: 11, 14: 10, 15: 16,
    16: 6, 17: 6, 18: 9, 19: 6, 40: 7, 41: 18, 42: 29,
    43: 36, 44: 10, 49: 27, 50: 30, 51: 21, 52: 19,
    55: 17, 56: 25, 57: 20, 58: 31, 59: 65, 60: 42, 61: 42,
}


def _clean(value):
    if value is None:
        return None
    if isinstance(value, float) and not isfinite(value):
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value or value.casefold() in {"nan", "nat", "none", "null"}:
            return None
    return value


def _text(value):
    value = _clean(value)
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return "; ".join(filter(None, (_text(item) for item in value))) or None
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return str(value)


def _date_and_time(value):
    value = _clean(value)
    if value is None:
        return None, None
    if isinstance(value, datetime):
        return value.date().isoformat(), value.strftime("%H:%M:%S")
    if isinstance(value, date):
        return value.isoformat(), None
    value = str(value)
    match = search(r"^(\d{4}-\d{2}-\d{2})(?:[T ](\d{2}:\d{2}:\d{2}))?", value)
    return (match.group(1), match.group(2)) if match else (value, None)


def _set(sheet, address, value):
    value = _clean(value)
    if value is None:
        return
    cell = sheet[address]
    cell.value = value
    # User-supplied descriptions and links are data, including leading '='.
    if isinstance(value, str):
        cell.data_type = "s"
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
    cell.alignment = Alignment(
        vertical="center" if is_number else "top",
        horizontal="center" if is_number else None,
        wrap_text=True,
    )


def _flag_text(flag: Mapping, number: int):
    parts = []
    for label, key in (
        ("Pillar", "type"), ("Attribute", "attribute"),
        ("Comment", "comment"), ("Description", "description"),
        ("Evidence", "transcript_excerpt"), ("Severity", "severity"),
    ):
        value = _text(flag.get(key))
        if value:
            parts.append(f"{label}: {value}")
    if flag.get("escalated"):
        parts.append("Escalated: Yes")
    if flag.get("qa_reviewed"):
        parts.append("QA Reviewed: Yes")
    review = _text(flag.get("qa_review_comment"))
    if review:
        parts.append(f"QA Review: {review}")
    return f"{number}. " + " | ".join(parts) if parts else None


def build_customer_care_workbook(calls: Sequence[Mapping]) -> bytes:
    """Make one Customer Care row per displayed call, retaining every flag."""
    projected_calls = []
    for original in calls:
        performance = original.get("agent_performance") or {}
        if not isinstance(performance, Mapping):
            performance = {}
        flags = matched_regulation_flags(original.get("compliance_flags") or [])
        assessment, escalated, classification = policy_outcome(
            flags,
            original.get("overall_assessment") or "pass",
            bool(original.get("escalation_required")),
        )
        projected_calls.append({
            **original,
            "bu": original.get("bu") or original.get("business_unit"),
            "improvements": original.get("improvements") or performance.get("improvements"),
            "compliance_flags": flags,
            "overall_assessment": assessment,
            "escalation_required": escalated,
            "escalation_reason": original.get("escalation_reason") if escalated else None,
            "agent_classification": classification,
            "profiling_comment": (
                original.get("profiling_comment")
                or performance.get("profiling_comment")
                or performance.get("Profiling Comment")
            ) if flags else None,
        })
    calls = projected_calls
    book = Workbook()
    sheet = book.active
    sheet.title = "Customer Care"
    sheet.freeze_panes = "A3"
    sheet.sheet_view.showGridLines = False
    sheet.sheet_view.zoomScale = 85
    thin_green = Side(style="thin", color=GRID_GREEN)
    medium_green = Side(style="medium", color=GRID_GREEN)
    header_border = Border(
        left=thin_green, right=thin_green, top=medium_green, bottom=thin_green,
    )
    data_border = Border(
        left=thin_green, right=thin_green, top=thin_green, bottom=thin_green,
    )

    for column, label in enumerate(CUSTOMER_CARE_HEADERS + EXTRA_HEADERS, 1):
        cell = sheet.cell(1, column, label)
        if column in (1, 2, 9, 11) or 45 <= column <= 48:
            fill, text_color = DARK, YELLOW
        elif column in (3, 4, 13, 14) or 20 <= column <= 23 or 28 <= column <= 31 or 36 <= column <= 39:
            fill, text_color = "FFFFFF", INK
        elif 5 <= column <= 8 or column in (10, 12, 15):
            fill, text_color = PALE_GREEN, INK
        elif 16 <= column <= 19 or 24 <= column <= 27 or 32 <= column <= 35:
            fill, text_color = COUNT_BLUE, INK
        elif 40 <= column <= 44:
            fill, text_color = "D9D9D9", INK
        elif 49 <= column <= 52:
            fill, text_color = YELLOW, INK
        elif 53 <= column <= 58:
            fill, text_color = "548235", "FFFFFF"
        else:
            fill, text_color = PALE_GREEN, INK
        cell.fill = PatternFill("solid", fgColor=fill)
        cell.font = Font(name="Calibri", size=10, color=text_color, bold=True)
        cell.border = header_border
        cell.alignment = Alignment(vertical="center", horizontal="center", wrap_text=True)
        if 20 <= column <= 39:
            offset = (column - 20) % 4
            width = (18, 45, 35, 10)[offset]
        else:
            width = COLUMN_WIDTHS.get(column, 18)
        sheet.column_dimensions[get_column_letter(column)].width = width
    sheet.row_dimensions[1].height = 28
    sheet.row_dimensions[2].height = 22

    # The source workbook uses merged section labels beneath its field names.
    for column in list(range(1, 16)) + list(range(45, 59)) + [59, 60, 61]:
        sheet.merge_cells(start_row=1, start_column=column, end_row=2, end_column=column)
    for start, end, label in (
        (16, 19, "# of Errors"),
        (20, 23, "Error 1"), (24, 27, "Error 2"),
        (28, 31, "Error 3"), (32, 35, "Error 4"),
        (36, 39, "Error 5"), (40, 44, "Company Error"),
    ):
        sheet.merge_cells(start_row=2, start_column=start, end_row=2, end_column=end)
        cell = sheet.cell(2, start, label)
        cell.fill = PatternFill("solid", fgColor=(COUNT_BLUE if start in (16, 24, 32) else "FFFFFF"))
        cell.font = Font(name="Calibri", size=10, color=INK, bold=True)
        cell.border = Border(
            left=thin_green, right=thin_green, top=thin_green, bottom=medium_green,
        )
        cell.alignment = Alignment(horizontal="center", vertical="center")

    for row_index, call in enumerate(calls, 3):
        for column in range(1, len(CUSTOMER_CARE_HEADERS + EXTRA_HEADERS) + 1):
            cell = sheet.cell(row_index, column)
            if 5 <= column <= 12 and column not in (9, 11):
                fill = PALE_GREEN
            elif column in (9, 11, 15) or 53 <= column <= 58:
                fill = PALE_GREEN
            elif column in (13, 14) or 20 <= column <= 23 or 28 <= column <= 31 or 36 <= column <= 39:
                fill = PALE_BLUE
            elif 40 <= column <= 44 or column == 49:
                fill = "FFF2CC"
            else:
                fill = "FFFFFF"
            cell.fill = PatternFill("solid", fgColor=fill)
            cell.font = Font(name="Calibri", size=10, color=INK)
            cell.border = data_border
            cell.alignment = Alignment(vertical="center", wrap_text=True)
        _set(sheet, f"C{row_index}", row_index - 2)
        for column, key in (
            ("E", "agent_id"), ("F", "bu"), ("G", "channel_name"),
            ("H", "supervisor_name"), ("J", "agent_name"),
            ("O", "customer_phone_number"), ("AN", "department"),
            ("AU", "overall_assessment"), ("AW", "channel_name"),
            ("AX", "assessment_reasoning"), ("AY", "agent_classification"),
            ("AZ", "profiling_comment"), ("BC", "coaching_status"),
            ("BF", "improvements"), ("BH", "conversation_link"),
            ("BI", "call_id"),
        ):
            _set(sheet, f"{column}{row_index}", _text(call.get(key)))

        call_date, call_time = _date_and_time(
            call.get("archive_datetime") or call.get("call_date")
        )
        _set(sheet, f"L{row_index}", call_date)
        _set(sheet, f"M{row_index}", call_time)
        duration = _clean(call.get("call_duration_seconds"))
        if duration is not None:
            try:
                _set(sheet, f"N{row_index}", float(duration) / 86400)
                sheet[f"N{row_index}"].number_format = "[h]:mm:ss"
            except (TypeError, ValueError):
                pass
        monitoring_date, _ = _date_and_time(call.get("created_at"))
        _set(sheet, f"AS{row_index}", monitoring_date)

        flags = call.get("compliance_flags") or []
        all_flags = []
        error_counts = {flag_type: 0 for flag_type in ERROR_COUNT_COLUMNS}
        for number, flag in enumerate(flags, 1):
            if not isinstance(flag, Mapping):
                continue
            flag_type = _clean(flag.get("type"))
            if flag_type in error_counts:
                error_counts[flag_type] += 1
            detail = _flag_text(flag, number)
            if detail:
                all_flags.append(detail)
            if number <= len(ERROR_STARTS):
                start = ERROR_STARTS[number - 1]
                for offset, value in enumerate((
                    flag.get("type"), flag.get("attribute"), flag.get("comment"),
                    " | ".join(filter(None, (
                        _text(flag.get("description")), _text(flag.get("severity")),
                        _text(flag.get("transcript_excerpt")),
                    ))),
                )):
                    _set(sheet, f"{get_column_letter(start + offset)}{row_index}", _text(value))
        for flag_type, column in ERROR_COUNT_COLUMNS.items():
            _set(sheet, f"{column}{row_index}", error_counts[flag_type])
        reason = _text(call.get("escalation_reason"))
        if reason:
            all_flags.append(f"Escalation reason: {reason}")
        _set(sheet, f"BG{row_index}", "\n".join(all_flags) or None)
        if _clean(call.get("conversation_link")):
            sheet[f"BH{row_index}"].hyperlink = str(call["conversation_link"])
        sheet.row_dimensions[row_index].height = min(90, 28 + 12 * len(all_flags))

    sheet.auto_filter.ref = f"A2:BI{max(3, len(calls) + 2)}"
    output = BytesIO()
    book.save(output)
    return output.getvalue()
