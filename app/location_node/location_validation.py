"""Deterministic, Arabic-first validation of KSA branch/location requests.

Location validation as its own node/feature — see app/bank_node/ for the
independent (but sibling) bank validator. They only share the transcript-
turn split (app.services.text_helpers.split_transcript_by_speaker) and
Arabic text normalisation, not any bank/location-specific logic.

Mirrors the layered-matching philosophy in app/offers_node/crm_offers.py
(exact match → fuzzy match, with ubiquitous/generic words excluded from
scoring) instead of inventing a separate matching strategy.
"""
from __future__ import annotations

import re
from typing import Any

from rapidfuzz import fuzz as _rfuzz

from app.models.input import CallTranscript
from app.services.text_helpers import normalize_arabic_text, split_transcript_by_speaker

LOCATION_FAILURES = {"WRONG_LOCATION", "NO_ADDRESS_PROVIDED"}

# ── Detection ─────────────────────────────────────────────────────────────────
# Strong standalone signals — unambiguous enough to trigger on their own.
_LOCATION_STRONG = re.compile(r"عنوان|موقع|لوكيشن|لوكشن|address|location", re.I)
# Weaker signals that only count as a location request when paired with a
# place-noun. "مكان" deliberately lives here, not in _LOCATION_STRONG: it's
# too overloaded in colloquial Arabic ("عندي مكان فاضي" = "I have a free
# slot", nothing to do with a branch) to trust on its own.
_LOCATION_WEAK = re.compile(r"فين|وين|أين|اين|مكان|ازاي|كيف\s*(اروح|اوصل)", re.I)
_PLACE_NOUN = re.compile(r"فرع|فرعكم|عياد[ةه]|عيادات|مستشفى|مستشفياتكم", re.I)


def detect_location_request(text: str) -> bool:
    """Context-aware detection — avoids matching bare 'فرع'/'مكان' with no
    request context (false positives on ordinary mentions)."""
    norm = normalize_arabic_text(text)
    if not norm:
        return False
    if _LOCATION_STRONG.search(norm):
        return True
    return bool(_LOCATION_WEAK.search(norm) and _PLACE_NOUN.search(norm))


def detect_location_signals(call: CallTranscript) -> tuple[bool, str, str]:
    """Parse the transcript once into (patient_requested, patient_text,
    agent_text) — the location-side equivalent of
    app.bank_node.bank_validation.detect_bank_signals(), independent of it
    (both call the same shared split_transcript_by_speaker(), neither
    depends on the other's module)."""
    patient, agent = split_transcript_by_speaker(call.transcript)
    return detect_location_request(patient), patient, agent


# ── Branch resolution ─────────────────────────────────────────────────────────

_LOC_CONTEXT_FIELDS = ("cr301_branchname", "cr301_area", "cr301_description", "cr18c_region")

_KSA_NAMES = {"ksa", "saudi", "saudi arabia", "السعوديه", "المملكه العربيه السعوديه"}


def _is_ksa(record: dict[str, Any]) -> bool:
    country = normalize_arabic_text(str(record.get("cr301_country") or ""))
    region = normalize_arabic_text(str(record.get("cr18c_region") or ""))
    return country in _KSA_NAMES or region == "ksa"


def _active(record: dict[str, Any]) -> bool:
    names = [normalize_arabic_text(str(record.get(k) or "")) for k in ("statecodename", "statuscodename")]
    return not any(v in {"inactive", "disabled", "draft", "غير نشط", "موقوف"} for v in names if v)


def _name_distinctive_tokens(record: dict[str, Any], pool: list[dict[str, Any]]) -> set[str]:
    """Distinctive tokens from a branch's NAME (not full address) — words that
    do not appear in most other branch names in `pool`. Real branch names
    here share a lot of boilerplate ("عيادات أندلسية فرع ..."), and a patient
    almost never repeats that boilerplate — they say "فرع الأمير سلطان", not
    the DB's full "عيادات أندلسية فرع الأمير سلطان". Matching on distinctive
    tokens (same ubiquity-filtering idea as crm_offers.py's generic-word
    check) instead of requiring the whole DB string as a literal substring is
    what makes that realistic phrasing resolve at all.
    """
    own = set(_tokenize(str(record.get("cr301_branchname") or "")))
    if not own or len(pool) < 3:
        return own
    others = [set(_tokenize(str(r.get("cr301_branchname") or ""))) for r in pool if r is not record]
    n = len(others) or 1
    distinctive = {tok for tok in own if sum(1 for s in others if tok in s) / n < 0.5}
    return distinctive or own


def resolve_branch_candidates(
    query_text: str, locations: list[dict[str, Any]], *, ksa_only: bool = True,
) -> list[dict[str, Any]]:
    """Layered branch resolution, mirroring crm_offers.py's exact→fuzzy strategy.

    1. Distinctive-token overlap against branch names (ubiquity-filtered, see
       _name_distinctive_tokens), ranked by (match COUNT, then ratio) — not
       ratio alone. A branch name that's a superset of another's (e.g. "...
       لطب الأسنان - فرع الأمير سلطان" vs "... فرع الأمير سلطان") would
       otherwise lose to the shorter name just because a shorter distinctive
       set is easier to fully cover: matching 3 of a 4-token name is a
       stronger, more specific signal than matching 2 of a 2-token name, even
       though the latter has a "cleaner" 100% ratio. A tie is returned as
       multiple candidates rather than resolved arbitrarily — the caller
       decides PASS vs AMBIGUOUS.
    2. If no branch has any token overlap at all, fall back to fuzzy
       (rapidfuzz) matching against branch names, so typos ("سلطن" for
       "سلطان") still resolve. Only a single clear fuzzy winner is accepted.
    """
    pool = [r for r in locations if _active(r) and (not ksa_only or _is_ksa(r))]
    query_tokens = set(_tokenize(query_text))
    if not query_tokens or not pool:
        return []

    scored: list[tuple[int, float, dict]] = []
    for record in pool:
        distinctive = _name_distinctive_tokens(record, pool)
        if not distinctive:
            continue
        overlap = distinctive & query_tokens
        if not overlap:
            continue
        ratio = len(overlap) / len(distinctive)
        if ratio >= 0.5:
            scored.append((len(overlap), ratio, record))

    if scored:
        best_count = max(count for count, _, _ in scored)
        top_by_count = [(ratio, record) for count, ratio, record in scored if count == best_count]
        best_ratio = max(ratio for ratio, _ in top_by_count)
        return [record for ratio, record in top_by_count if ratio == best_ratio]

    # Fuzzy fallback — tolerates spelling variants/typos token overlap misses.
    fuzzy_scored: list[tuple[float, dict]] = []
    joined_query = " ".join(sorted(query_tokens))
    for record in pool:
        name = normalize_arabic_text(str(record.get("cr301_branchname") or ""))
        if not name:
            continue
        score = _rfuzz.partial_ratio(joined_query, name)
        if score >= 80:
            fuzzy_scored.append((score, record))
    if not fuzzy_scored:
        return []
    best = max(score for score, _ in fuzzy_scored)
    winners = [record for score, record in fuzzy_scored if score == best]
    return winners if len(winners) == 1 else []


# ── Address-answer validation ─────────────────────────────────────────────────

# Structural/filler words that appear in nearly every Arabic street address
# regardless of branch — they must not count as evidence of a match.
_ADDRESS_FILLER = {
    "شارع", "حي", "الحي", "متفرع", "من", "بعد", "امام", "أمام", "تقاطع", "مع",
    "طريق", "في", "فى", "و", "ال", "بلازا", "مول", "مقابل", "بجوار", "خلف",
}


def _tokenize(text: str) -> list[str]:
    norm = normalize_arabic_text(text)
    return [t for t in norm.split() if t and t not in _ADDRESS_FILLER and len(t) > 1]


def _distinctive_tokens(record: dict[str, Any], all_locations: list[dict[str, Any]]) -> set[str]:
    """Tokens from this branch's name/description/area that do NOT appear in
    most other branches — i.e. actually distinguish it. Same ubiquity-based
    generic-word filtering crm_offers.py uses for offer-name matching."""
    own_text = " ".join(str(record.get(f) or "") for f in _LOC_CONTEXT_FIELDS)
    own_tokens = set(_tokenize(own_text))
    if not own_tokens or len(all_locations) < 3:
        return own_tokens  # too few records to compute ubiquity reliably

    other_token_sets = [
        set(_tokenize(" ".join(str(r.get(f) or "") for f in _LOC_CONTEXT_FIELDS)))
        for r in all_locations
        if r is not record
    ]
    n = len(other_token_sets) or 1
    distinctive = {
        tok for tok in own_tokens
        if sum(1 for s in other_token_sets if tok in s) / n < 0.5
    }
    return distinctive or own_tokens  # never end up with nothing to check


def validate_location_answer(
    branch: dict[str, Any], agent_text: str, all_locations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Score the agent's combined address text against the resolved branch.

    Returns {"overlap_ratio": float, "matched": [...], "distinctive": [...]}.
    Thresholds (documented, not hidden): >=0.6 → sufficient match,
    0.25-0.6 → partial/incomplete, <0.25 → effectively no match.
    """
    distinctive = _distinctive_tokens(branch, all_locations)
    agent_tokens = set(_tokenize(agent_text))
    if not distinctive:
        return {"overlap_ratio": 0.0, "matched": [], "distinctive": []}
    matched = distinctive & agent_tokens
    return {
        "overlap_ratio": len(matched) / len(distinctive),
        "matched": sorted(matched),
        "distinctive": sorted(distinctive),
    }


def _agent_mentions_any_known_address(agent_text: str, all_locations: list[dict[str, Any]]) -> bool:
    """True if the agent's text shares at least one distinctive address token
    with ANY branch in the dataset — used to tell "gave no address at all"
    ("لحظة من فضلك") apart from "gave a real address, just the wrong one"
    ("شارع صاري حي البوادي" when الأمير سلطان was requested). Both cases have
    zero overlap with the REQUESTED branch, but only the second is evidence
    the agent actually attempted an address."""
    agent_tokens = set(_tokenize(agent_text))
    if not agent_tokens:
        return False
    return any(_distinctive_tokens(loc, all_locations) & agent_tokens for loc in all_locations)


def _loc_result(
    outcome: str, reason: str, *, request: bool, applicable: bool = False,
    branch: dict[str, Any] | None = None, overlap: float | None = None,
) -> dict[str, Any]:
    return {
        "outcome": outcome,
        "applicable": applicable,
        "request_detected": request,
        "requested_branch": branch.get("cr301_branchname") if branch else None,
        "match_confidence": round(overlap, 2) if overlap is not None else None,
        "reason": reason,
        "is_violation": outcome in LOCATION_FAILURES,
    }


def validate_location_request(
    call: CallTranscript,
    locations: list[dict[str, Any]],
    *,
    patient_text: str,
    agent_text: str,
) -> dict[str, Any]:
    """Top-level entry: detect → resolve branch → validate the agent's answer.

    `patient_text`/`agent_text` are passed in (already split by the caller)
    rather than re-split here, matching detect_location_signals()'s pattern
    of parsing the transcript once and sharing the result.
    """
    requested = detect_location_request(patient_text)
    if not requested:
        return _loc_result("NOT_APPLICABLE", "No location/address request was detected.", request=False)

    candidates = resolve_branch_candidates(patient_text, locations, ksa_only=True) \
        or resolve_branch_candidates(call.transcript, locations, ksa_only=True)

    if not candidates:
        return _loc_result(
            "BRANCH_UNRESOLVED",
            "No authoritative KSA branch could be resolved from the request.",
            request=True,
        )
    if len(candidates) > 1:
        return _loc_result(
            "AMBIGUOUS_BRANCH",
            f"Request matches {len(candidates)} branches equally — cannot disambiguate without more context.",
            request=True, applicable=True,
        )

    branch = candidates[0]
    if not agent_text.strip():
        return _loc_result(
            "NO_ADDRESS_PROVIDED",
            "The patient asked for a location but the agent gave no address.",
            request=True, applicable=True, branch=branch, overlap=0.0,
        )

    score = validate_location_answer(branch, agent_text, locations)
    overlap = score["overlap_ratio"]
    if overlap >= 0.6:
        outcome = "PASS"
        reason = "The agent's address matches the requested branch's known location."
    elif overlap >= 0.25:
        outcome = "INCOMPLETE_ADDRESS"
        reason = "The agent's address partially matches — some identifying details were omitted."
    elif not _agent_mentions_any_known_address(agent_text, locations):
        # Zero overlap AND nothing address-shaped in the reply at all (e.g.
        # "لحظة من فضلك") — the agent didn't attempt an address, as opposed
        # to giving a real-but-wrong one.
        outcome = "NO_ADDRESS_PROVIDED"
        reason = "The patient asked for a location but the agent's reply contains no address information."
    else:
        outcome = "WRONG_LOCATION"
        reason = "The agent's address does not match the requested branch's known location."

    return _loc_result(outcome, reason, request=True, applicable=True, branch=branch, overlap=overlap)
