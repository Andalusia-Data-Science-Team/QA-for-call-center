import re
import unicodedata

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

# Patterns emitted by reasoning/thinking models (Gemma, DeepSeek-R1, etc.)
# before the actual JSON payload.  We strip the entire preamble up to the
# first '{' or '[' so that json.loads receives a clean value.
_THINKING_PREAMBLE_RE = re.compile(
    r"^(?:<think>.*?</think>|<thought>.*?</thought>|\bthought\b.*?(?=\{|\[))",
    re.DOTALL | re.IGNORECASE,
)


def _norm_score(value: object, default: float = 0.5) -> float:
    """
    Normalise a score to the [0.0, 1.0] range expected by QAAnalysisResult.

    LLMs sometimes return scores on a 0–100 scale instead of 0–1.
    Any value > 1 is assumed to be on the 0–100 scale and is divided by 100.
    The result is then clamped to [0.0, 1.0].
    """
    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if v > 1.0:
        v = v / 100.0
    return max(0.0, min(1.0, v))


def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` wrappers and thinking-model preambles.

    Handles:
    * ``\`\`\`json ... \`\`\`` code fences
    * ``<think>...</think>`` / ``<thought>...</thought>`` XML tags
    * Bare ``thought\n{...}`` prefix emitted by some Gemma / DeepSeek builds
    """
    text = text.strip()

    # 1. Strip markdown code fences
    fence_pattern = r"^```(?:json)?\s*\n?(.*?)\n?```$"
    match = re.match(fence_pattern, text, re.DOTALL)
    if match:
        text = match.group(1).strip()

    # 2. Strip <think>…</think> or <thought>…</thought> XML wrappers
    xml_pattern = r"^<(?:think|thought)>.*?</(?:think|thought)>\s*"
    text = re.sub(xml_pattern, "", text, flags=re.DOTALL | re.IGNORECASE).strip()

    # 3. Strip bare "thought" / "thinking" word-preamble that precedes the JSON
    #    (Gemma 3 emits "\nthought\n{" — slice from the first { or [)
    if not text.startswith(("{", "[")):
        first_brace = min(
            (text.find(ch) for ch in ("{", "[") if text.find(ch) != -1),
            default=-1,
        )
        if first_brace != -1:
            text = text[first_brace:]

    return text.strip()


# ---------------------------------------------------------------------------
# Arabic text normalisation helpers
# ---------------------------------------------------------------------------

# Mapping of visually/phonetically equivalent Arabic characters.
# Each key is a variant that should be collapsed to the canonical form (value).
_ARABIC_CHAR_MAP: dict[str, str] = {
    # Alef variants → bare alef
    "أ": "ا",
    "إ": "ا",
    "آ": "ا",
    "ٱ": "ا",
    # Teh marbuta → heh
    "ة": "ه",
    # Yeh variants → yeh
    "ى": "ي",
    "ئ": "ي",
    # Waw with hamza → waw
    "ؤ": "و",
    # Alef maqsura already handled above (ى → ي)
}

# Regex that strips Arabic diacritics (tashkeel / harakat)
_DIACRITICS_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]")


def _normalize_arabic(text: str | None) -> str | None:
    """
    Normalise an Arabic name/word so that visually equivalent spellings
    compare equal.

    Steps applied:
      1. Strip leading/trailing whitespace.
      2. Remove diacritics (tashkeel / harakat).
      3. Collapse alef variants (أ إ آ ٱ) → ا
         teh-marbuta (ة) → ه
         yeh variants (ى ئ) → ي
         waw-with-hamza (ؤ) → و
      4. Collapse multiple internal spaces to a single space.

    Returns ``None`` when the input is ``None`` or an empty string after
    stripping so callers can still detect "no value" cleanly.
    """
    if not text:
        return None
    # Remove diacritics
    text = _DIACRITICS_RE.sub("", text)
    # Collapse equivalent characters
    text = "".join(_ARABIC_CHAR_MAP.get(ch, ch) for ch in text)
    # Normalise Unicode (NFC) and collapse spaces
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def _arabic_like_pattern(text: str | None, first_word_only: bool = False) -> str | None:
    """
    Build a SQL ``LIKE`` pattern for a normalised Arabic token so that the
    database query tolerates alef / teh-marbuta / yeh variants without
    requiring a case-insensitive collation or full-text index.

    Strategy
    ────────
    For each character in the (already normalised) input we emit a character
    class ``[x y]`` when the canonical form has known variants, otherwise we
    emit the character as-is.  A trailing ``%`` wildcard is appended so that
    partial last-name matches still work.

    Example::

        _arabic_like_pattern("احمد")  →  "[اأإآ]حمد%"
        _arabic_like_pattern("اسماء") →  "[اأإآ]سم[اأإآ][ءئ]%"

    When *first_word_only* is True only the first token (space-delimited) of
    the input is used — useful for doctor-name queries where the DB stores the
    full name but we only extracted the first name.
    """
    normalised = _normalize_arabic(text)
    if not normalised:
        return None

    token = normalised.split()[0] if first_word_only else normalised

    # Reverse map: canonical → all its variants (including itself)
    _VARIANTS: dict[str, str] = {
        "ا": "[اأإآٱ]",
        "ه": "[هة]",
        "ي": "[يىئ]",
        "و": "[وؤ]",
        "ء": "[ءئ]",
    }

    pattern_chars: list[str] = [_VARIANTS.get(ch, ch) for ch in token]
    return "".join(pattern_chars) + "%"
