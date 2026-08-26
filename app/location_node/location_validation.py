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

import logging
import re
from typing import Any

from rapidfuzz import fuzz as _rfuzz

from app.models.input import CallTranscript
from app.services.text_helpers import normalize_arabic_text, split_transcript_by_speaker, split_transcript_turns

logger = logging.getLogger(__name__)


# ── Terminal debug output ────────────────────────────────────────────────────
# A concise, human-readable "[location] ..." terminal view — a sibling to
# this project's existing "[offers] ..." / "[crm_location] ..." print-style
# logs. Purely additive local-dev/testing observability: it never replaces
# the structured logger.info() calls placed right beside each call below,
# and it never dumps the whole transcript — only the already-extracted
# location-relevant fields validate_location_request() itself computes.
# Two tiny helpers instead of scattering unrelated print() calls throughout
# the validator: _loc_print for a single flat line (blank text = a blank
# separator line, printed with no "[location]" prefix), _loc_print_section
# for a "[location] title:" header followed by its fields indented under it
# (None-valued fields are skipped, so an early return — e.g. branch
# unresolved — naturally only prints the sections it actually computed).

def _loc_print(text: str = "") -> None:
    print(f"[location] {text}" if text else "", flush=True)


def _loc_print_section(title: str, **fields: Any) -> None:
    _loc_print()
    _loc_print(f"{title}:")
    for key, value in fields.items():
        if value is not None:
            _loc_print(f"  {key}: {value}")


# INCOMPLETE_ADDRESS is a violation too, not just PASS/fail's softer middle
# ground: Sections 15/16/24 are explicit that generic/insufficient answers
# (branch name + city only, city-only, etc.) must read as a failure, not a
# tolerated partial credit.
LOCATION_FAILURES = {"WRONG_LOCATION", "NO_ADDRESS_PROVIDED", "INCOMPLETE_ADDRESS"}

# ── Detection ─────────────────────────────────────────────────────────────────
# Unambiguous standalone signal — "عنوان"/"address" is essentially always
# about a facility's address, never used for "send me your location".
_LOCATION_STRONG = re.compile(r"عنوان|address", re.I)
# "موقع"/"لوكيشن"/"location" are ambiguous on their own: they're just as
# commonly used to mean the CUSTOMER's own location (a home-care visit, a
# delivery, a coordinator collecting an address) as an Andalusia branch's.
# See _LOCATION_OF_BRANCH_RE / _CUSTOMER_LOCATION_RE below for how the two
# are told apart before either counts as a location request.
_LOCATION_AMBIGUOUS = re.compile(r"موقع|لوكيشن|لوكشن|location", re.I)
# Weaker signals that only count as a location request when paired with a
# place-noun. "مكان" deliberately lives here, not in _LOCATION_STRONG: it's
# too overloaded in colloquial Arabic ("عندي مكان فاضي" = "I have a free
# slot", nothing to do with a branch) to trust on its own.
_LOCATION_WEAK = re.compile(r"فين|وين|أين|اين|مكان|ازاي|كيف\s*(اروح|اوصل)", re.I)
_PLACE_NOUN = re.compile(r"فرع|فرعكم|عياد[ةه]|عيادات|مستشفى|مستشفياتكم", re.I)

# "موقع/لوكيشن الفرع" (the location OF a named branch/clinic/hospital) is
# unambiguously a facility-location statement, even when it also happens to
# sit inside a sentence that would otherwise look like a customer-location
# phrase (e.g. "هشارك مع حضرتك لوكيشن فرع السنابل" — "حضرتك" there is the
# customer RECEIVING the branch's location, not the subject of it). This
# check is tried before the exclusion below and overrides it.
_LOCATION_OF_BRANCH_RE = re.compile(
    r"(?:موقع|لوكيشن)\s+(?:ال)?(?:فرع|فرعكم|عياد[ةه]|عيادات|مستشفى|مستشفياتكم)", re.I,
)

# Category B (per spec): the CUSTOMER's own location being requested,
# collected, or sent on their behalf — a home-care/delivery/coordinator
# conversation, not an Andalusia facility address. "موقعك"/"موقعكم" (your
# location — possessive suffix), a send/share verb near موقع/لوكيشن
# ("ابعت موقعك", "بترسل لهم الموقع", "شارك اللوكيشن" — deliberately matches
# only conjugated verb stems like يرسل/ترسل/بترسل/هيرسل, not the formal noun
# "إرسال", so "تم إرسال الموقع الخاص بفرع السنابل" — explicitly attributed
# to a branch — still reads as a facility statement), موقع/لوكيشن directly
# followed by a dative pronoun ("موقع حضرتك"), or "نحتاج/يطلب" + موقع/لوكيشن
# (the coordinator asking the customer for THEIR location).
_CUSTOMER_LOCATION_RE = re.compile(
    r"موقعك\b|موقعكم\b|لوكيشنك\b|"
    r"(?:ابعت|رسل|شارك)\S*.{0,20}(?:موقع|لوكيشن)|"
    r"(?:موقع|لوكيشن)\s*(?:ال)?حضرتك|"
    r"(?:موقع|لوكيشن).{0,15}(?:لكم|لهم|لهما|لينا)|"
    r"(?:نحتاج|يطلب).{0,15}(?:موقع|لوكيشن)",
    re.I,
)

# "موقع/لوكيشن/عنوان + المنزل/البيت/المريض" is unambiguously the CUSTOMER's
# own home/service-delivery location (a home-care visit, a delivery, a
# coordinator collecting an address to send someone there) — never an
# Andalusia branch. Checked FIRST, before even the normally-unambiguous
# "عنوان" trigger: "عنوان المنزل" must not count as a facility address
# request just because "عنوان" alone always would. Deliberately narrow —
# only the exact "location word + منزل/بيت/مريض" pairing is excluded, so
# موقع/لوكيشن/عنوان keep triggering normally everywhere else (e.g. "عنوان
# فرع الأمير سلطان؟", "فين موقع المستشفى؟").
_HOME_SERVICE_LOCATION_RE = re.compile(
    r"(?:موقع|لوكيشن|لوكشن|location|address|عنوان)\s*(?:ال)?(?:منزل|بيت|مريض)", re.I,
)


def detect_location_request(text: str) -> bool:
    """Context-aware detection — avoids matching bare 'فرع'/'مكان' with no
    request context (false positives on ordinary mentions), and avoids
    treating a CUSTOMER's own location (home-care/delivery context) as an
    Andalusia branch/facility location request."""
    norm = normalize_arabic_text(text)
    if not norm:
        return False
    if _HOME_SERVICE_LOCATION_RE.search(norm):
        return False
    if _LOCATION_STRONG.search(norm):
        return True
    if _LOCATION_AMBIGUOUS.search(norm):
        if _LOCATION_OF_BRANCH_RE.search(norm):
            return True
        return not _CUSTOMER_LOCATION_RE.search(norm)
    return bool(_LOCATION_WEAK.search(norm) and _PLACE_NOUN.search(norm))


def detect_location_intent(patient_text: str, agent_text: str) -> tuple[bool, bool]:
    """Dedicated location-intent detection layer.

    This is the gate that runs BEFORE any CRM lookup, branch resolution, or
    address scoring — it is deliberately cheap (pure regex over already-
    normalised text) and CRM-independent. Speaker-aware: both sides are
    checked with the SAME context-aware detector (detect_location_request).
    A patient asking "فين فرع السنابل؟" and an agent saying "هشارك مع حضرتك
    لوكيشن فرع السنابل" are symmetric expressions of location intent — an
    explicit location word ("فين", "لوكيشن") is present either way, and
    neither needs CRM data to recognise. Concrete address content with NO
    explicit location word at all ("شارع الأمير سلطان حي المحمدية" with no
    "العنوان" preface) is a separate, CRM-aware signal handled downstream
    once locations have actually been fetched — see
    _agent_gives_address_details()/validate_location_request(). This layer
    intentionally only covers the vocabulary-based half of Trigger B, since
    that's the half answerable without CRM data.

    Returns (patient_has_location_intent, agent_has_location_information).
    """
    return detect_location_request(patient_text), detect_location_request(agent_text)


def is_home_service_location_text(patient_text: str, agent_text: str) -> bool:
    """Diagnostic-only check: does either side's text explicitly pair a
    location word with a home/patient noun (منزل/بيت/مريض) — e.g. "ممكن
    لوكيشن المنزل؟", "نحتاج لوكيشن المريض"? Used purely to give
    skip_location_validation (app.agent.nodes) a more specific skip reason
    to log; the actual routing decision is always
    detect_location_request()/detect_location_intent()/
    location_validation_needed(), never this function."""
    return bool(
        _HOME_SERVICE_LOCATION_RE.search(normalize_arabic_text(patient_text or ""))
        or _HOME_SERVICE_LOCATION_RE.search(normalize_arabic_text(agent_text or ""))
    )


def detect_location_signals(call: CallTranscript) -> tuple[bool, str, str]:
    """Parse the transcript once into (patient_requested, patient_text,
    agent_text) — the location-side equivalent of
    app.bank_node.bank_validation.detect_bank_signals(), independent of it
    (both call the same shared split_transcript_by_speaker(), neither
    depends on the other's module)."""
    patient, agent = split_transcript_by_speaker(call.transcript)
    return detect_location_request(patient), patient, agent


def location_validation_needed(call: CallTranscript, signals: tuple[bool, str, str] | None = None) -> bool:
    """The location-intent gate: patient_has_location_intent OR
    agent_has_location_information (see detect_location_intent()). Either is
    enough to justify a CRM fetch + full branch/address resolution; when
    neither holds, location validation is skipped entirely — no CRM call,
    no branch resolution, no address matching. A false positive here (the
    agent used location vocabulary but validate_location_request() then
    can't resolve a real branch) just costs one extra cached CRM fetch and
    resolves to a non-punitive outcome; a false negative would silently
    skip validation, so this stays intentionally cheap and vocabulary-based
    rather than trying to pre-verify CRM-level address detail.
    """
    _requested, patient_text, agent_text = signals if signals is not None else detect_location_signals(call)
    patient_intent, agent_intent = detect_location_intent(patient_text, agent_text)
    return patient_intent or agent_intent


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


def _ksa_pool(locations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Active, KSA-only records — Egypt (or any other country) rows must
    never become candidates for location validation. Shared by
    resolve_branch_candidates() and validate_location_request()'s own
    filtering-count log line, so the two never drift apart."""
    return [r for r in locations if _active(r) and _is_ksa(r)]


def _name_distinctive_tokens(record: dict[str, Any], pool: list[dict[str, Any]]) -> set[str]:
    """Distinctive tokens from a branch's NAME (not full address) — words that
    do not appear in most other branches' FULL context (name+area+description
    +region — not just their names) in `pool`. Real branch names here share a
    lot of boilerplate ("عيادات أندلسية فرع ..."), and a patient almost never
    repeats that boilerplate — they say "فرع الأمير سلطان", not the DB's full
    "عيادات أندلسية فرع الأمير سلطان". Matching on distinctive tokens (same
    ubiquity-filtering idea as crm_offers.py's generic-word check) instead of
    requiring the whole DB string as a literal substring is what makes that
    realistic phrasing resolve at all.

    Ubiquity is checked against every OTHER record's FULL context, not just
    names: "مستشفى أندلسية جدة" is the only branch with the city baked into
    its own NAME field, but every other current KSA branch is ALSO in جدة
    (via cr301_area) — a name-only ubiquity check would miss that and let a
    bare "جدة" mention resolve uniquely to the hospital.
    """
    own = set(_tokenize(str(record.get("cr301_branchname") or "")))
    if not own or len(pool) < 3:
        return own
    others_full = [
        set(_tokenize(" ".join(str(r.get(f) or "") for f in _LOC_CONTEXT_FIELDS)))
        for r in pool if r is not record
    ]
    n = len(others_full) or 1
    distinctive = {tok for tok in own if sum(1 for s in others_full if tok in s) / n < 0.5}
    return distinctive or own


# Text immediately following a place-noun ("فرع X", "عيادة X", ...), up to
# the next preposition/structural marker that typically starts an address
# description. This is the strongest possible signal of an explicit branch
# reference — used to resolve branch identity BEFORE falling back to
# whole-text token overlap, which a WRONG address can otherwise win: a
# branch is sometimes named after its own street ("فرع الأمير سلطان" on
# "شارع الأمير سلطان"), so if an agent proactively gives one branch's name
# but a DIFFERENT branch's street, matching against the whole blob can let
# that wrong street's extra token count outrank the correct branch-name
# mention. Anchoring to just the words right after the place-noun avoids that.
_BRANCH_ANCHOR_RE = re.compile(
    r"(?:فرع|فرعكم|عياد[ةه]|عيادات|مستشفى|مستشفياتكم)\s+"
    r"(.+?)(?=\s+(?:في|شارع|حي|علي|على|عند|جنب|قريب)|[.,،؟!\n]|$)",
    re.I,
)


def _branch_anchor_text(text: str) -> str:
    return " ".join(m.group(1) for m in _BRANCH_ANCHOR_RE.finditer(text or ""))


def resolve_branch_candidates(
    query_text: str, locations: list[dict[str, Any]], *, ksa_only: bool = True, allow_fuzzy: bool = True,
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
    2. If no branch has any token overlap at all AND allow_fuzzy=True, fall
       back to fuzzy (rapidfuzz) matching against branch names, so typos
       ("سلطن" for "سلطان") still resolve. Only a single clear fuzzy winner
       is accepted. allow_fuzzy=False is used for short, weakly-distinctive
       anchor fragments (e.g. the agent's own proactive branch-name anchor)
       where a low-confidence fuzzy guess would be worse than deferring to
       the caller's broader fallback text.
    """
    pool = _ksa_pool(locations) if ksa_only else [r for r in locations if _active(r)]
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

    if not allow_fuzzy:
        return []

    # Fuzzy fallback — tolerates spelling variants/typos token overlap misses.
    # Matches against each branch's DISTINCTIVE tokens joined, not its raw
    # full name: fuzzy-matching the raw name would let a query containing
    # nothing but a ubiquitous word (e.g. "جدة", which is also literally
    # embedded in "مستشفى أندلسية جدة"'s own name) score a near-perfect
    # partial-ratio hit purely because it's a verbatim substring of a long
    # boilerplate-heavy name string — exactly what the exact-match layer
    # above already deliberately filters out.
    fuzzy_scored: list[tuple[float, dict]] = []
    joined_query = " ".join(sorted(query_tokens))
    for record in pool:
        distinctive = _name_distinctive_tokens(record, pool)
        name = " ".join(sorted(distinctive))
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


def _address_detail_tokens(record: dict[str, Any], all_locations: list[dict[str, Any]]) -> set[str]:
    """Tokens that are genuinely address DETAIL — distinctive words from
    cr301_description that are NOT also part of the branch's own NAME.

    Deliberately narrower than _distinctive_tokens: a branch is sometimes
    named after its own street ("فرع السنابل" sitting on "شارع السنابل"), so
    that shared word alone must not count as "the agent gave an address" —
    only as naming the branch. Used specifically for the Trigger-B gate
    (_agent_gives_address_details), not for scoring an already-applicable
    answer (validate_location_answer keeps using the broader
    _distinctive_tokens, where a legitimate street-name-equals-branch-name
    match should still count in the agent's favour once we already know
    validation applies).
    """
    name_tokens = set(_tokenize(str(record.get("cr301_branchname") or "")))
    desc_only = _distinctive_tokens(record, all_locations) - name_tokens
    return desc_only


def _agent_gives_address_details(agent_text: str, all_locations: list[dict[str, Any]]) -> bool:
    """Trigger-B gate: does the agent's text contain genuine address DETAIL
    (distinctive street/district tokens from SOME branch's description that
    aren't just that branch's own name), not merely a bare branch-name
    mention? Naming a branch alone ("فرع السنابل") must not, on its own,
    count as proactively giving its location."""
    agent_tokens = set(_tokenize(agent_text))
    if not agent_tokens:
        return False
    return any(_address_detail_tokens(loc, all_locations) & agent_tokens for loc in all_locations)


def _agent_mentions_known_location(text: str, all_locations: list[dict[str, Any]]) -> bool:
    """Broader than _agent_gives_address_details: true if the text overlaps
    a real branch's distinctive tokens AT ALL (name OR address detail), not
    just address detail that excludes the branch's own name.

    Used to decide whether an agent TURN is worth keeping as part of the
    location answer once we already know location is being discussed
    (_extract_agent_location_text) — once a request/trigger already exists,
    an answer that's just the branch's own name/street ("شارع صارى" for
    فرع صاري) is still a genuine (if incomplete) attempt and must reach
    scoring, not be silently dropped as if nothing was said.
    _agent_gives_address_details stays deliberately narrower and is used
    only for the Trigger-B "is the agent proactively volunteering a
    location AT ALL, unprompted" gate, where a bare branch-name mention
    must not count on its own.

    Requires either at least TWO distinctive tokens from the SAME branch to
    overlap, or the overlap to cover most (>=60%) of the turn's own
    tokens — not just a bare single coincidental word out of an otherwise
    unrelated, longer turn. A real regression showed an agent's own name
    ("د محمد") coincidentally sharing one word ("محمد") with an unrelated
    branch's street address ("...شارع محمد محمد مطاوع...") inside an
    8-token greeting/department-introduction turn — with a bare
    single-token bar, that greeting was wrongly picked as "the"
    location-bearing turn instead of the real address the agent gave two
    turns later. The ratio branch still accepts a short, genuine (if
    incomplete) answer like "شارع صارى" (a single surviving token —
    "شارع" is filler — that IS the branch's own street name, a 100%
    overlap), which the earlier turn's own coincidental 1-of-8 overlap
    would not meet."""
    tokens = set(_tokenize(text))
    if not tokens:
        return False
    for loc in all_locations:
        overlap = _distinctive_tokens(loc, all_locations) & tokens
        if len(overlap) >= 2 or (overlap and len(overlap) / len(tokens) >= 0.6):
            return True
    return False


def _extract_agent_location_text(
    call: CallTranscript, locations: list[dict[str, Any]], patient_text: str,
) -> str:
    """Extract only the agent turn(s) that actually carry location/address
    information, scoped to LOCAL conversational context around the
    location trigger — never the whole agent-side transcript. This is what
    branch resolution's "provided_address" fallback, the final address
    comparison, and the returned provided_location are built from, so a
    real CRM address mentioned later in an unrelated turn can never
    masquerade as the answer to an earlier, different request.

    1. Find the trigger point: the LAST patient turn (if any) that shows
       location intent (detect_location_request). A pure Trigger-B call
       (agent proactively shares a location, no patient request at all)
       has no trigger point and starts scanning from the beginning.
    2. Group the transcript's agent turns into RUNS of turns that are
       consecutive in the raw transcript — a patient turn in between ends
       a run, so two agent turns "far apart" in the conversation are never
       merged.
    3. Starting from the first run at/after the trigger point, take the
       first run containing at least one turn with a genuine location
       signal (detect_location_request or _agent_gives_address_details).
       Earlier/unrelated runs are never searched once a qualifying one is
       found, and no run beyond it is considered either.
    4. Within that run, keep only the turns that themselves carry a
       location signal OR mention a branch/place-noun (local context, e.g.
       "في فرع المستشفى الرئيسي فقط" right before the address turn) — an
       unrelated turn in the same run ("يلزم وصفة طبية") is dropped.

    Returns "" when no agent turn anywhere at/after the trigger point
    carries a location signal (callers treat this the same as "no address
    provided").
    """
    turns = split_transcript_turns(call.transcript)

    trigger_index = -1
    for i, (speaker, text) in enumerate(turns):
        if speaker == "patient" and detect_location_request(text):
            trigger_index = i  # last matching patient turn wins

    runs: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    for i, (speaker, text) in enumerate(turns):
        if speaker == "agent":
            current.append((i, text))
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)

    candidate_runs = [run for run in runs if run[0][0] > trigger_index] or runs

    for run in candidate_runs:
        if any(detect_location_request(t) or _agent_mentions_known_location(t, locations) for _i, t in run):
            kept = [
                t for _i, t in run
                if detect_location_request(t)
                or _agent_mentions_known_location(t, locations)
                or _PLACE_NOUN.search(normalize_arabic_text(t))
            ]
            return "\n".join(kept).strip()

    return ""


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


def _loc_result(
    outcome: str, reason: str, *, request: bool, applicable: bool = False,
    branch: dict[str, Any] | None = None, overlap: float | None = None,
    provided_location: str | None = None,
) -> dict[str, Any]:
    return {
        "outcome": outcome,
        "applicable": applicable,
        "request_detected": request,
        "requested_branch": branch.get("cr301_branchname") if branch else None,
        # Additive, backward-compatible fields: the authoritative CRM
        # address the agent's answer was checked against, and the location
        # text actually extracted from the chat — so debugging never has to
        # rely solely on console/log output (see README "Debug visibility").
        "crm_location": branch.get("cr301_description") if branch else None,
        "provided_location": provided_location,
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

    Applicable in either of two independent cases (neither depends on the
    other):
      A. the patient asked for a location/address, or
      B. the agent proactively provided one — even with no patient request
         at all — as long as it demonstrably matches a REAL KSA branch's
         distinctive tokens, not just any agent text that happens to
         mention a place-noun.

    `patient_text`/`agent_text` are passed in (already split by the caller)
    rather than re-split here, matching detect_location_signals()'s pattern
    of parsing the transcript once and sharing the result.
    """
    call_id = call.call_id
    requested = detect_location_request(patient_text)
    # Coarse applicability gate — deliberately based on the RAW, whole
    # agent-side text (same signal the location-intent layer already uses
    # at the node level): is location even being discussed in this call at
    # all? Trigger B fires on EITHER genuine address DETAIL (a real
    # branch's distinctive street/district tokens —
    # _agent_gives_address_details) OR explicit location vocabulary
    # alongside a branch/place reference (detect_location_request —
    # "هشارك مع حضرتك لوكيشن فرع السنابل" has no street/district words at
    # all, but "لوكيشن" is an unambiguous statement of intent). A bare
    # branch-name/availability mention with NEITHER signal ("متاح في فرع
    # الأمير سلطان") still correctly does not count. This gate does NOT
    # decide what gets compared — see agent_location_text below for that.
    agent_gave_address_raw = bool(agent_text.strip()) and (
        detect_location_request(agent_text) or _agent_gives_address_details(agent_text, locations)
    )
    _loc_print(f"Location validation started | call_id={call_id}")
    _loc_print(f"intent: patient_request={requested} agent_provided={agent_gave_address_raw}")
    if not requested and not agent_gave_address_raw:
        _loc_print()
        _loc_print("outcome: NOT_APPLICABLE")
        return _loc_result("NOT_APPLICABLE", "No location/address request or agent-provided address was detected.", request=False)

    ksa_pool = _ksa_pool(locations)
    logger.info(
        "location CRM records | call_id=%s fetched=%d ksa_considered=%d",
        call_id, len(locations), len(ksa_pool),
    )
    _loc_print()
    _loc_print(f"CRM records fetched: {len(locations)}")
    _loc_print(f"KSA records considered: {len(ksa_pool)}")

    # The actual location text to resolve/compare against — scoped to the
    # LOCAL agent turn(s) around the request/trigger, never the whole
    # transcript. See _extract_agent_location_text() docstring: this is
    # what fixes both (a) provided_location being the entire concatenated
    # agent transcript, and (b) a real CRM address mentioned in an
    # unrelated later turn being picked up as if it answered an earlier,
    # different request.
    agent_location_text = _extract_agent_location_text(call, locations, patient_text)
    if agent_location_text:
        loc_lines = agent_location_text.split("\n")
        if len(loc_lines) > 1:
            _loc_print_section("extracted chat location", **{
                "branch/context": "\n".join(loc_lines[:-1]), "address": loc_lines[-1],
            })
        else:
            _loc_print_section("extracted chat location", address=agent_location_text)

    # Resolve the branch. Priority: an explicit "فرع/عيادة/مستشفى X" mention
    # in the PATIENT's own request (anchored — the words right after the
    # place-noun, before any address detail starts) beats token overlap,
    # which a WRONG address can otherwise win — a branch is sometimes named
    # after its own street ("فرع الأمير سلطان" on "شارع الأمير سلطان"), so a
    # patient naming the right branch must still resolve to it even if the
    # agent's reply describes a different branch's street. Falls back to
    # the rest of the patient's text, then to the extracted agent location
    # text (never the raw whole transcript — that was the source of the
    # stale/cross-contaminated alias bug: joining anchor matches from
    # unrelated turns anywhere in the conversation). `resolution_source` is
    # tracked for visibility only; it does not change which candidates are
    # returned.
    # The agent's own proactive text can suffer the exact same problem in
    # reverse: an agent naming the RIGHT branch while describing a
    # different, WRONG branch's street ("فرع السنابل موجود في شارع الأمير
    # سلطان") would otherwise let that wrong street's token count outrank
    # the correct explicit self-naming under plain whole-text overlap — so
    # the agent's own anchor is tried before falling back to the whole
    # extracted text, exactly mirroring the patient-side priority above.
    anchor_text = _branch_anchor_text(patient_text)
    agent_anchor_text = _branch_anchor_text(agent_location_text)
    candidates: list[dict[str, Any]] = []
    resolution_source = "none"
    query_used = ""
    for source, text, allow_fuzzy in (
        ("branch_name", anchor_text, True),
        ("patient_text", patient_text, True),
        ("branch_name", agent_anchor_text, False),
        ("provided_address", agent_location_text, True),
    ):
        if not text:
            continue
        found = resolve_branch_candidates(text, locations, ksa_only=True, allow_fuzzy=allow_fuzzy)
        if found:
            candidates, resolution_source, query_used = found, source, text
            break

    if not candidates:
        logger.info(
            "location branch resolution | call_id=%s resolution_source=%s matched_terms=[] "
            "resolved_branch=None candidate_count=0",
            call_id, resolution_source,
        )
        _loc_print_section(
            "branch resolution", source=resolution_source, matched_terms=[],
            resolved_branch="None", candidate_count=0,
        )
        _loc_print()
        _loc_print("outcome: BRANCH_UNRESOLVED")
        _loc_print("reason: no authoritative KSA branch could be resolved from the request")
        return _loc_result(
            "BRANCH_UNRESOLVED",
            "No authoritative KSA branch could be resolved from the request.",
            request=requested, provided_location=agent_location_text or None,
        )
    if len(candidates) > 1:
        # Step 8: identity is ambiguous, but if every tied candidate shares
        # the EXACT same authoritative address, the address itself isn't —
        # validation can safely proceed with it rather than refusing outright.
        descriptions = {normalize_arabic_text(str(c.get("cr301_description") or "")) for c in candidates}
        if len(descriptions) == 1 and next(iter(descriptions)):
            logger.info(
                "location multiple branch records matched | call_id=%s candidate_count=%d "
                "shared_crm_location=%r ambiguity=safe_for_address_validation",
                call_id, len(candidates), candidates[0].get("cr301_description"),
            )
            _loc_print_section(
                "branch resolution", source=resolution_source, candidate_count=len(candidates),
                ambiguity="safe_for_address_validation (shared CRM location)",
                shared_crm_location=candidates[0].get("cr301_description"),
            )
            candidates = candidates[:1]
        else:
            _loc_print_section(
                "branch resolution", source=resolution_source, candidate_count=len(candidates),
            )
            _loc_print()
            _loc_print("outcome: AMBIGUOUS_BRANCH")
            _loc_print(f"reason: request matches {len(candidates)} branches equally — cannot disambiguate without more context")
            return _loc_result(
                "AMBIGUOUS_BRANCH",
                f"Request matches {len(candidates)} branches equally — cannot disambiguate without more context.",
                request=requested, applicable=True, provided_location=agent_location_text or None,
            )

    # The resolution evidence logged here is computed from the SAME query
    # text that actually won (query_used) and the FINAL selected branch —
    # never a leftover from a different, failed attempt — so the log always
    # belongs to the branch it names.
    branch = candidates[0]
    if resolution_source == "provided_address":
        distinctive_terms = _distinctive_tokens(branch, locations)
    else:
        distinctive_terms = _name_distinctive_tokens(branch, ksa_pool or locations)
    matched_terms = sorted(distinctive_terms & set(_tokenize(query_used)))
    logger.info(
        "location branch resolution | call_id=%s resolution_source=%s matched_terms=%s "
        "resolved_branch=%r candidate_count=%d",
        call_id, resolution_source, matched_terms, branch.get("cr301_branchname"), len(candidates),
    )
    logger.info(
        "location CRM reference selected | call_id=%s branch=%r area=%r country=%r description=%r region=%r",
        call_id, branch.get("cr301_branchname"), branch.get("cr301_area"),
        branch.get("cr301_country"), branch.get("cr301_description"), branch.get("cr18c_region"),
    )
    _loc_print_section(
        "branch resolution", source=resolution_source, matched_terms=matched_terms,
        resolved_branch=branch.get("cr301_branchname"), candidate_count=len(candidates),
    )
    _loc_print_section(
        "CRM reference", branch=branch.get("cr301_branchname"), area=branch.get("cr301_area"),
        country=branch.get("cr301_country"), region=branch.get("cr18c_region"),
        address=branch.get("cr301_description"),
    )

    if not agent_location_text:
        # Either the agent said nothing at all, or nothing they said in the
        # local context around the trigger carried a location signal — both
        # read as "no address provided" for this specific request/branch.
        logger.info("location chat location extracted | call_id=%s provided_location=None speaker=Agent", call_id)
        _loc_print()
        _loc_print("outcome: NO_ADDRESS_PROVIDED")
        _loc_print("reason: the patient asked for a location but the agent gave no address")
        return _loc_result(
            "NO_ADDRESS_PROVIDED",
            "The patient asked for a location but the agent gave no address.",
            request=True, applicable=True, branch=branch, overlap=0.0,
        )

    logger.info(
        "location chat location extracted | call_id=%s provided_location=%r speaker=Agent",
        call_id, agent_location_text,
    )
    logger.info(
        "location normalized comparison | call_id=%s crm_normalized=%r chat_normalized=%r",
        call_id, normalize_arabic_text(str(branch.get("cr301_description") or "")), normalize_arabic_text(agent_location_text),
    )

    score = validate_location_answer(branch, agent_location_text, locations)
    overlap = score["overlap_ratio"]
    # Recomputed against the EXTRACTED local text (not the raw whole-call
    # gate above) — this is what actually decides NO_ADDRESS_PROVIDED vs
    # WRONG_LOCATION for the resolved branch specifically.
    agent_gave_address = bool(
        detect_location_request(agent_location_text) or _agent_gives_address_details(agent_location_text, locations)
    )
    if overlap >= 0.6:
        outcome = "PASS"
        reason = "The agent's address matches the requested branch's known location."
    elif overlap >= 0.25:
        outcome = "INCOMPLETE_ADDRESS"
        reason = "The agent's address partially matches — some identifying details were omitted."
    elif not agent_gave_address:
        # Zero overlap with the resolved branch AND nothing address-shaped in
        # the local reply at all (e.g. "لحظة من فضلك") — the agent didn't
        # attempt an address, as opposed to giving a real-but-wrong one.
        outcome = "NO_ADDRESS_PROVIDED"
        reason = "The patient asked for a location but the agent's reply contains no address information."
    else:
        outcome = "WRONG_LOCATION"
        reason = "The agent's address does not match the requested branch's known location — it belongs to a different branch."

    logger.info(
        "location comparison | call_id=%s resolved_branch=%r crm_location=%r chat_location=%r match=%s outcome=%s",
        call_id, branch.get("cr301_branchname"), branch.get("cr301_description"),
        agent_location_text, outcome == "PASS", outcome,
    )
    _loc_print_section(
        "comparison",
        chat_normalized=normalize_arabic_text(agent_location_text),
        crm_normalized=normalize_arabic_text(str(branch.get("cr301_description") or "")),
        match_confidence=round(overlap, 2), match=(outcome == "PASS"),
    )
    _loc_print()
    _loc_print(f"outcome: {outcome}")
    if outcome != "PASS":
        _loc_print(f"reason: {reason}")

    return _loc_result(
        outcome, reason, request=requested, applicable=True, branch=branch, overlap=overlap,
        provided_location=agent_location_text,
    )
