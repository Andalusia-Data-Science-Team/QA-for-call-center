from CM_DB_Handler import CMDatabaseHandler
from pathlib import Path
import sys
import argparse
import warnings
import logging
from typing import Optional

project_root = Path.cwd().parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.models.input import BatchCallTranscripts, CallTranscript
from pydantic import ValidationError
import pandas as pd

logger = logging.getLogger(__name__)

# Columns that must be strings — NaN → empty string / sensible default
_STR_COLUMNS = [
    "Content", "SenderType", "OwnerName", "RelationName",
    "SenderIdName", "ConversationChannel", "Scope",
    "ConversationState", "WebStoreName", "OwnerId", "RelationId",
]


def get_db_handler(json_file: str = "passcode.json", db_key: str = "CM") -> CMDatabaseHandler:
    return CMDatabaseHandler(json_file=json_file, db_key=db_key)


def retrieve_chats_by_agent_email(
    agent_email: str,
    filter_date: str,
    db_handler: Optional[CMDatabaseHandler] = None,
    sql_file: str = "../SQL/CM_Chat_Search.sql",
):
    handler = db_handler or get_db_handler()
    if not handler.connect():
        raise ConnectionError("Failed to connect to database.")

    try:
        chats = handler.execute_query_from_file(
            sql_file,
            params={
                "ConversationId": None,
                "AgentFullName": None,
                "AgentEmail": agent_email,
                "FilterDate": filter_date,
            },
        )
        return chats
    finally:
        handler.close()


def _sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce NaN floats → empty strings for all text columns so that
    field validators calling .strip() never receive a float.
    """
    df = df.copy()
    for col in _STR_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    # Any remaining object columns that may hold mixed str/float
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(
            lambda v: "" if (isinstance(v, float) and pd.isna(v)) else v
        )

    return df


def _convert_df_to_batch(df: pd.DataFrame) -> BatchCallTranscripts:
    df = _sanitize_df(df)
    calls  = []
    errors = []

    for uid, group in df.groupby("UniqueId", sort=False):
        rows = group.to_dict(orient="records")
        try:
            calls.append(CallTranscript.from_cm_rows(rows))
        except (ValidationError, ValueError) as exc:
            errors.append({"UniqueId": uid, "error": str(exc)})
            logger.warning("Skipped conversation %s: %s", uid, exc)

    if not calls:
        raise ValueError(
            f"No valid conversations could be parsed. "
            f"{len(errors)} conversation(s) failed validation:\n" +
            "\n".join(f"  {e['UniqueId']}: {e['error']}" for e in errors)
        )

    return BatchCallTranscripts(calls=calls)


def df_to_batch(df: pd.DataFrame) -> BatchCallTranscripts:
    uids = list(df["UniqueId"].unique())
    if len(uids) <= 50:
        return _convert_df_to_batch(df)

    first_uids = set(uids[:50])
    df_chunk = df.loc[df["UniqueId"].isin(first_uids)]
    warnings.warn(
        f"Input contains {len(uids)} conversations; truncating to first 50 "
        f"unique conversations to satisfy BatchCallTranscripts limit."
    )
    return _convert_df_to_batch(df_chunk)


def save_batch_json(batch: BatchCallTranscripts, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = batch.model_dump_json()
    out_path.write_text(payload, encoding="utf-8")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Retrieve chats by agent email and save as JSON.")
    parser.add_argument("--agent-email", required=True, help="Agent email to filter chat transcripts.")
    parser.add_argument("--filter-date", required=True, help="Date filter for the chat query.")
    parser.add_argument("--output", default="data/chats.json", help="Output JSON path.")
    parser.add_argument("--json-file", default="passcode.json", help="Credentials JSON file.")
    parser.add_argument("--db-key", default="CM", help="Database key in the credentials file.")
    args = parser.parse_args()

    db_handler = get_db_handler(json_file=args.json_file, db_key=args.db_key)
    chats = retrieve_chats_by_agent_email(args.agent_email, args.filter_date, db_handler=db_handler)

    if chats is None or chats.empty:
        print("No chats retrieved.")
        return

    print(f"Retrieved {len(chats)} rows for {len(chats['UniqueId'].unique())} conversations.")
    batch = df_to_batch(chats)
    saved_path = save_batch_json(batch, Path(args.output))
    print(f"Wrote chats JSON to {saved_path}")


if __name__ == "__main__":
    main()



