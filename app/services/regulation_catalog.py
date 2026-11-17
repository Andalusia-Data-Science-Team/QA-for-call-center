"""Resolve flag labels against the existing regulation YAML items only."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import yaml


_POLICIES = Path(__file__).resolve().parent.parent / "criteria" / "policies"
_FILES = (
    _POLICIES / "regulations" / "compliance_regulations.yaml",
    _POLICIES / "regulations" / "reservation_regulations.yaml",
    _POLICIES / "regulations" / "offer_regulations.yaml",
    _POLICIES / "regulations.yaml",
)


def _key(value: object) -> str:
    return " ".join(str(value or "").replace("’", "'").split()).casefold()


@lru_cache(maxsize=1)
def regulation_items() -> tuple[dict[str, str], ...]:
    """Return each item with its exact YAML spelling and owning attribute."""
    items = []
    for path in _FILES:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        pillars = data.get("pillars")
        if pillars is None:  # reservation_regulations.yaml stores attributes at the root
            pillars = {key: value for key, value in data.items()
                       if isinstance(value, dict) and isinstance(value.get("items"), list)}
        for name, pillar in pillars.items():
            for item in pillar.get("items", []):
                if item.get("violation"):
                    items.append({
                        "type": item.get("type") or pillar.get("severity"),
                        "attribute": pillar.get("original_name") or name.replace("_", " "),
                        "comment": item["violation"],
                        "id": item.get("id") or "",
                        "outcome": item.get("outcome") or "",
                    })
    return tuple(items)


def canonicalize_flag(flag: Mapping) -> dict:
    """Add canonical labels when one existing policy item matches unambiguously.

    The original explanation and transcript evidence are left untouched.  A
    generic ``Other`` item is never inferred from prose.
    """
    result = dict(flag)
    result["attribute"] = None
    result["comment"] = None
    candidates = [item for item in regulation_items() if item["type"] == flag.get("type")]
    if not candidates:
        return result

    def unique(items):
        distinct = {}
        for item in items:
            distinct[(item["type"], _key(item["attribute"]), _key(item["comment"]))] = item
        return list(distinct.values())

    attribute = _key(flag.get("attribute"))
    comment = _key(flag.get("comment"))
    if attribute and comment:
        matched = unique(item for item in candidates
                         if _key(item["attribute"]) == attribute and _key(item["comment"]) == comment)
    elif comment and comment != "other":
        matched = unique(item for item in candidates if _key(item["comment"]) == comment)
    else:
        matched = []

    if len(matched) != 1:
        outcome = _key(flag.get("outcome"))
        matched = unique(item for item in candidates if outcome and _key(item["outcome"]) == outcome)
    if len(matched) != 1:
        description = str(flag.get("description") or "")
        matched = unique(item for item in candidates if item["id"]
                         and _key(item["comment"]) != "other" and re.search(
                             rf"(?<![\w]){re.escape(item['id'])}(?![\w])",
                             description, re.IGNORECASE))
        if len(matched) != 1:
            matched = unique(item for item in candidates if _key(item["comment"]) != "other"
                             and _key(item["comment"]) in _key(description))

    if len(matched) == 1:
        result["attribute"] = matched[0]["attribute"]
        result["comment"] = matched[0]["comment"]
    return result


def matched_regulation_flags(flags) -> list[dict]:
    """Keep only findings with an attribute and comment in the current YAML."""
    matched = []
    for flag in flags:
        if not isinstance(flag, Mapping):
            continue
        canonical = canonicalize_flag(flag)
        if canonical["attribute"] and canonical["comment"]:
            matched.append(canonical)
    return matched


def deduplicate_regulation_flags(flags) -> list[dict]:
    """Keep one finding per canonical policy item in a call.

    Different reviewers may cite different excerpts for the same item. For
    stored results, retain the row with a QA review or escalation when one
    exists so its row-level state remains visible on the dashboard.
    """
    by_item: dict[tuple[str, str, str], dict] = {}
    for flag in flags:
        if not isinstance(flag, Mapping):
            continue
        canonical = canonicalize_flag(flag)
        if not (canonical["attribute"] and canonical["comment"]):
            continue
        key = (canonical["type"], canonical["attribute"], canonical["comment"])
        previous = by_item.get(key)
        if previous is None or (
            bool(canonical.get("qa_reviewed")), bool(canonical.get("escalated"))
        ) > (
            bool(previous.get("qa_reviewed")), bool(previous.get("escalated"))
        ):
            by_item[key] = canonical
    return list(by_item.values())


def policy_outcome(flags, assessment: str, escalation_required: bool) -> tuple[str, bool, str]:
    """Derive status and classification solely from matched regulation flags."""
    if assessment == "error":
        return "error", False, "D"
    critical = sum(flag["type"] in {"C2Com", "C2C", "C2B"} for flag in flags)
    noncritical = sum(flag["type"] == "NC" for flag in flags)
    if critical > 1:
        classification = "D"
    elif critical == 1 or noncritical > 2:
        classification = "C"
    elif noncritical:
        classification = "B"
    else:
        classification = "A"

    if not critical:
        return ("needs_review" if noncritical else "pass"), False, classification
    escalated = bool(escalation_required or assessment == "escalate")
    return ("escalate" if escalated else "needs_review"), escalated, classification
