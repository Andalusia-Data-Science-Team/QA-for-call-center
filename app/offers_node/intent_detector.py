"""
intent_detector.py — Offer Intent Detection
============================================
Purpose: Determine whether the patient's message is:

  1. PROACTIVE CHECK   — Mid-booking automatic check (slot already confirmed).
                         Runs silently without any patient keyword trigger.

  2. INQUIRY          — Patient explicitly asked about offers/promotions in
                        their message (e.g. "هل في عروض على الأسنان؟").

  3. NONE             — No offer intent detected; skip the offer flow.

Public API
----------
    detect_offer_intent(user_text, state) -> "proactive" | "inquiry" | "none"

    _patient_asked_about_offers(state) -> bool
        Low-level check: did any recent message mention offers?

    _patient_replied_yes(text, lang) -> bool
    _patient_replied_no(text, lang)  -> bool
        Parse yes/no responses to offer prompts.

    _contains_booking_info(text) -> bool
        Escape hatch: skip offer gate when patient sends phone/payment info.

    _is_price_complaint(text) -> bool
        True when patient declined because the price is too high.
"""
from __future__ import annotations
import re


# ── Offer keyword patterns ────────────────────────────────────────────────────
# Broad match — used to detect ANY offer mention in recent messages.
_OFFER_KW_AR = re.compile(
    r"عرض|عروض|باقة|باقات|خصم|خصومات|تخفيض|تنزيل|اوفر|بروموشن|بروموشون",
    re.IGNORECASE,
)
_OFFER_KW_EN = re.compile(
    r"\boffer|package|promo|promotion|discount|deal\b", re.IGNORECASE
)

# Strict match — guards against verb forms like "اعرض" (I'll show you) that
# share the same root as "عرض" (offer).  Only standalone noun forms trigger.
_OFFER_KW_STRICT = re.compile(
    r"(?<![؀-ۿ])عرض(?![؀-ۿ])"
    r"|(?<![؀-ۿ])عروض"
    r"|باقة|باقات|خصم|تخفيض|تنزيل|اوفر|بروموشن|بروموشون"
    r"|\boffer|package|promo|promotion|discount|deal\b",
    re.IGNORECASE,
)

# ── Patient response patterns ─────────────────────────────────────────────────
_YES_AR = re.compile(
    r"^(نعم|أيوه|ايوه|اوكيه|اوكي|موافق|حسناً|أحجز|احجز|ابغى|ابي|بلى|تمام|صح|ي)\b",
    re.IGNORECASE,
)
_YES_EN = re.compile(
    r"^(yes|yeah|yep|sure|ok|okay|book|i want|i'd like|please)",
    re.IGNORECASE,
)
_NO_AR = re.compile(
    r"^(لا|لأ|مو|مش|ما ابي|ما ابغى|بدون|تجاوز|تجاهل|استمر)\b",
    re.IGNORECASE,
)
_NO_EN = re.compile(
    r"^(no|nope|skip|without|continue|no thanks|not interested)",
    re.IGNORECASE,
)
_IMPLICIT_NO_AR = re.compile(
    r"(اقل|أقل|ارخص|أرخص|اوفر|ارخص|غالي|غالية|مافي.*اقل|محتاج.*اقل|محتاجة.*اقل"
    r"|عرض.*اخر|عرض.*ثاني|عرض.*تاني|في.*عروض.*ثانية|في.*عروض.*تانية"
    r"|cheaper|less expensive|another offer|different offer|lower price)",
    re.IGNORECASE,
)

# ── Exit-intent: patient wants to end the session ────────────────────────────
_EXIT_INTENT = re.compile(
    r"\b(شكرا|شكراً|مشكور|سلام|مع السلامة|ما بدي|مش محتاج"
    r"|لا شكرا|بدون حجز|no thanks|goodbye|thats all|thanks"
    r"|never mind|nevermind|لا مشكورك|مشكوره"
    r"|ما (ابي|ابغى|أبي|أبغى) احجز|ما أبغى احجز|ما اريد احجز)\b",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Public helpers
# ─────────────────────────────────────────────────────────────────────────────

def _patient_asked_about_offers(state: dict) -> bool:
    """True if any of the 6 most recent patient messages mentions offers.

    Guards:
    - Returns False once the patient has declined (``_offer_declined``).
    - Returns False once an offer has been accepted (``_offer_selected``).
    """
    if state.get("_offer_declined"):
        return False
    if state.get("_offer_selected"):
        return False
    messages = state.get("messages") or []
    recent = [m for m in messages[-6:] if m.get("role") == "user"]
    for msg in recent:
        text = msg.get("content") or ""
        if _OFFER_KW_AR.search(text) or _OFFER_KW_EN.search(text):
            return True
    return False


def _patient_replied_yes(text: str, lang: str) -> bool:
    """Patient said yes/agree to an offer prompt."""
    t = text.strip().lower()
    if lang == "ar":
        return bool(_YES_AR.search(t))
    return bool(_YES_EN.search(t))


def _patient_replied_no(text: str, lang: str) -> bool:
    """Patient declined an offer (explicit or implicit price complaint)."""
    t = text.strip().lower()
    if lang == "ar":
        return bool(_NO_AR.search(t)) or bool(_IMPLICIT_NO_AR.search(t))
    return bool(_NO_EN.search(t)) or bool(_IMPLICIT_NO_AR.search(t))


def _is_price_complaint(text: str) -> bool:
    """True when patient declined because the price is too high."""
    return bool(_IMPLICIT_NO_AR.search((text or "").lower()))


def _contains_booking_info(text: str) -> bool:
    """True if message looks like booking info (phone number or payment method).

    Escape hatch: when the patient gives "cash 01245782245" instead of yes/no
    to an offer prompt, skip the offer gate and let patient_info handle it.
    """
    if re.search(r"\d{7,}", text):
        return True
    payment_words = {
        "cash", "كاش", "تامين", "تأمين", "insurance", "visa",
        "card", "نقد", "insured", "مؤمن", "credit", "بطاقة",
    }
    return bool(set(text.lower().split()) & payment_words)


def detect_offer_intent(user_text: str, state: dict) -> str:
    """Return the type of offer intent from the current state + user message.

    Returns
    -------
    "proactive"
        Slot already confirmed — run automatic mid-booking offer check.
        No keyword match needed.
    "inquiry"
        Patient explicitly asked about offers (keyword detected in recent
        messages) — run the inquiry flow.
    "none"
        No offer intent detected — caller should skip the offer node entirely.
    """
    booking_stage = state.get("booking_stage", "routing")
    slot_confirmed = bool(state.get("slot_confirmed"))

    # ── Case A: mid-booking proactive check ──────────────────────────────────
    # Triggered after slot selection automatically, no keyword needed.
    if slot_confirmed and booking_stage not in ("offers_check",):
        return "proactive"

    # ── Case B: patient explicitly asked about offers ─────────────────────────
    if _patient_asked_about_offers(state):
        return "inquiry"

    return "none"
