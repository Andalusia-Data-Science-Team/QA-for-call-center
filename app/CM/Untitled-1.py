# %%
from CM_DB_Handler import CMDatabaseHandler

# Initialize database handler with credentials from passcode.json
db_handler = CMDatabaseHandler(json_file="passcode.json", db_key="CM")

# Connect to database
if db_handler.connect():
    print("Database connection established successfully.")
else:
    print("Failed to connect to database.")
chats = db_handler.execute_query_from_file("../SQL/CM_Chat_Search.sql", params={
    "ConversationId": None,
    "AgentFullName": None,
    "AgentEmail": "Nada.Abdullah@Andalusiagroup.net",
    "FilterDate": "2026-06-05"
})
print(f"✓ Query executed successfully")
    


# %%
# Display retrieved chats
if chats is not None:
    print(f"\nRetrieved {len(chats)} chat conversations:")
    print(chats['UniqueId'].unique())
else:
    print("No chats retrieved.")
    
# Close connection when done
db_handler.close()

# %%
chats[['Content','AgentFullName','UniqueId','SenderIdName','SenderType','RelationName']].loc[chats['UniqueId'] == 'D8A9B517-DA60-F111-8FCB-00224888EC9A']

# %%


# %%
from pathlib import Path
import sys

# Ensure project root (parent of "app") is on sys.path so "app" can be imported
project_root = Path.cwd().parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.models.input import BatchCallTranscripts, CallTranscript
from pydantic import ValidationError
import pandas as pd

# %%
def df_to_batch(df: pd.DataFrame) -> BatchCallTranscripts:
    """
    Convert a message-level DataFrame (one row per message) into a
    BatchCallTranscripts.  Call this directly in tests or notebooks.
    """
    calls: list[CallTranscript] = []
    errors: list[dict] = []

    for uid, group in df.groupby("UniqueId", sort=False):
        # Convert each row to a plain dict; Timestamps stay as-is for _to_datetime()
        rows = group.to_dict(orient="records")
        try:
            calls.append(CallTranscript.from_cm_rows(rows))
        except (ValidationError, ValueError) as exc:
            # Log and skip bad conversations rather than aborting the whole batch
            errors.append({"UniqueId": uid, "error": str(exc)})

    if errors:
        import logging
        for e in errors:
            logging.warning("Skipped conversation %s: %s", e["UniqueId"], e["error"])

    return BatchCallTranscripts(calls=calls)


# %%


# ── Usage ────────────────────────────────────────────────────────────────────

# Option A: from a live DB connection
#batch = load_batch_from_db("mssql+pyodbc://...")

# Option B: from a DataFrame you already have (notebook / test)
batch = df_to_batch(chats)

# Serialize to JSON for the pipeline ingestion node
payload: str = batch.model_dump_json()

# Or as a dict if you're passing it directly in Python
payload_dict: dict = batch.model_dump()


