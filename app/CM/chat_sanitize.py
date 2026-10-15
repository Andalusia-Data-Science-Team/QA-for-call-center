"""
Row sanitisation shared by the automated retrieval path.

Mirrors ``Chat_Retriever._sanitize_df`` so both paths coerce the same columns
the same way, without the automation importing ``Chat_Retriever`` — which does
``sys.path`` surgery at import time and pulls in argparse CLI wiring.
"""

from __future__ import annotations

import pandas as pd

# Columns the CallTranscript validators call .strip() on, so a float NaN here
# is a crash rather than a missing value.
_STR_COLUMNS = [
    "Content", "SenderType", "OwnerName", "RelationName",
    "SenderIdName", "ConversationChannel", "Scope",
    "ConversationState", "WebStoreName", "OwnerId", "RelationId",
    "answer_state", "AnswerState", "AgentFullName", "EmailAddress",
]


def sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    """NaN -> '' for text columns, leaving timestamps and numerics alone."""
    df = df.copy()
    for col in _STR_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    # pandas 3 splits 'object' and 'str'; name both explicitly rather than
    # relying on the deprecated backward-compatible overlap.
    mixed = [c for c in df.columns if df[c].dtype == "object" or df[c].dtype == "str"]
    for col in mixed:
        df[col] = df[col].apply(
            lambda v: "" if (isinstance(v, float) and pd.isna(v)) else v
        )
    return df


def sanitize_rows(df: pd.DataFrame) -> list[dict]:
    """Sanitise and return plain dicts, sorted for deterministic transcripts.

    from_cm_rows sorts by CreationDateTime alone, which leaves messages sharing
    a timestamp in arbitrary order. Sorting by (CreationDateTime, MessageId)
    here makes the transcript — and therefore the LLM input and its checksum —
    reproducible.
    """
    df = sanitize_df(df)
    sort_cols = [c for c in ("CreationDateTime", "MessageId") if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, kind="mergesort")
    return df.to_dict(orient="records")
