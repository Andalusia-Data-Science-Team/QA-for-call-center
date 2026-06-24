"""
CM Portal - Chat History Extractor
====================================
Supports two CM.com APIs:

  1. Conversation History API (Central Services / Conversational Router)
     Endpoint: GET https://api.conversational.cm.com/conversational/conversation-history/v2/
               accounts/{accountId}/session-history
     Auth: X-Cm-Producttoken header

  2. Agent Inbox Conversations API (Mobile Service Cloud)
     Endpoint: POST https://api.robinhq.com/conversations
     Auth: HTTP Basic (username + password, or API key as username)

Usage examples
--------------
# Fetch by router session ID (Conversation History API):
python cm_chat_history_extractor.py \
    --api history \
    --account-id  YOUR_ACCOUNT_ID \
    --product-token YOUR_PRODUCT_TOKEN \
    --session-id   ROUTER_SESSION_ID \
    --include-events \
    --output conversations.json

# Fetch by customer email (Agent Inbox API):
python cm_chat_history_extractor.py \
    --api inbox \
    --inbox-username YOUR_USERNAME \
    --inbox-password YOUR_PASSWORD \
    --email customer@example.com \
    --output conversations.json

# Fetch all (paginated) conversations to Excel:
python cm_chat_history_extractor.py \
    --api inbox \
    --inbox-username YOUR_USERNAME \
    --inbox-password YOUR_PASSWORD \
    --paginate \
    --output conversations.xlsx
"""

import argparse
import json
import sys
import logging
from datetime import datetime
from typing import Optional

import requests

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# =========================================================================== #
#  1.  Conversation History API  (Central Services / Conversational Router)
# =========================================================================== #

HISTORY_BASE = (
    "https://api.conversational.cm.com/conversational/conversation-history/v2"
)


def fetch_session_history(
    account_id: str,
    product_token: str,
    session_id: str,
    include_events: bool = False,
    include_messages: bool = True,
) -> dict:
    """
    GET /accounts/{accountId}/session-history?routerSessionId={sessionId}

    Optional query params:
      includeEvents=true          → include session lifecycle events
      includeMessages=false       → suppress messages (events only)
    """
    url = f"{HISTORY_BASE}/accounts/{account_id}/session-history"
    params = {"routerSessionId": session_id}

    if include_events:
        params["includeEvents"] = "true"
    if not include_messages:
        params["includeMessages"] = "false"

    headers = {
        "X-Cm-Producttoken": product_token,
        "Accept": "application/json",
    }

    log.info("Fetching session history for session %s …", session_id)
    response = requests.get(url, headers=headers, params=params, timeout=30)

    if response.status_code == 401:
        log.error("Authentication failed – check your product token.")
        response.raise_for_status()
    elif response.status_code == 404:
        log.error("Session not found: %s", session_id)
        response.raise_for_status()

    response.raise_for_status()
    data = response.json()
    log.info("Session history retrieved successfully.")
    return data


def delete_session_history(
    account_id: str,
    product_token: str,
    session_id: str,
) -> None:
    """
    DELETE /accounts/{accountId}/session-history/{sessionId}
    Permanently removes session history (GDPR / right-to-erasure).
    """
    # v1 endpoint for delete
    url = (
        f"https://api.conversational.cm.com/conversational/conversation-history/v1"
        f"/accounts/{account_id}/session-history/{session_id}"
    )
    headers = {"X-Cm-Producttoken": product_token}

    log.warning("Deleting session history for session %s …", session_id)
    response = requests.delete(url, headers=headers, timeout=30)
    response.raise_for_status()
    log.info("Session history deleted.")


# =========================================================================== #
#  2.  Agent Inbox Conversations API  (Mobile Service Cloud / robinhq)
# =========================================================================== #

INBOX_URL = "https://api.robinhq.com/conversations"


def fetch_inbox_conversations(
    username: str,
    password: str,
    email_address: Optional[str] = None,
    order_number: Optional[str] = None,
    conversation_id: Optional[str] = None,
    guid: Optional[str] = None,
    skip: int = 0,
) -> dict:
    """
    POST https://api.robinhq.com/conversations

    At least one filter (email_address, order_number, conversation_id, guid)
    should be provided. `skip` is used for pagination.
    """
    payload: dict = {}
    if email_address:
        payload["emailAddress"] = email_address
    if order_number:
        payload["orderNumber"] = order_number
    if conversation_id:
        payload["conversationId"] = conversation_id
    if guid:
        payload["guid"] = guid
    if skip:
        payload["skip"] = skip

    if not payload:
        log.warning(
            "No filters specified – the API may return no results or an error."
        )

    log.info("Fetching inbox conversations (skip=%d) …", skip)
    response = requests.post(
        INBOX_URL,
        json=payload,
        auth=(username, password),
        headers={"Accept": "application/json"},
        timeout=30,
    )

    if response.status_code == 401:
        log.error("Authentication failed – check inbox username/password.")
        response.raise_for_status()
    elif response.status_code == 400:
        log.error("Bad request: %s", response.text)
        response.raise_for_status()

    response.raise_for_status()
    data = response.json()
    count = len(data.get("conversations", []))
    log.info("Retrieved %d conversation(s).", count)
    return data


def fetch_all_inbox_conversations(
    username: str,
    password: str,
    email_address: Optional[str] = None,
    order_number: Optional[str] = None,
    page_size: int = 20,
) -> list:
    """
    Paginate through all conversations using the `skip` parameter.
    """
    all_conversations = []
    skip = 0

    while True:
        data = fetch_inbox_conversations(
            username=username,
            password=password,
            email_address=email_address,
            order_number=order_number,
            skip=skip,
        )
        batch = data.get("conversations", [])
        all_conversations.extend(batch)

        if len(batch) < page_size:
            # Fewer results than a full page → we've reached the end
            break

        skip += page_size

    log.info("Total conversations fetched: %d", len(all_conversations))
    return all_conversations


# =========================================================================== #
#  Output helpers
# =========================================================================== #

def save_json(data: object, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False, default=str)
    log.info("Saved JSON → %s", path)


def save_excel(conversations: list, path: str) -> None:
    """Flatten conversations + messages into an Excel workbook (two sheets)."""
    try:
        import pandas as pd  # type: ignore
    except ImportError:
        log.error("pandas is not installed. Run: pip install pandas openpyxl")
        sys.exit(1)

    # --- Sheet 1: Conversations summary ---
    conv_rows = []
    for c in conversations:
        conv_rows.append(
            {
                "guid": c.get("guid"),
                "subject": c.get("subject"),
                "channel": c.get("channel"),
                "state": c.get("state"),
                "owner_name": (c.get("owner") or {}).get("name"),
                "owner_email": (c.get("owner") or {}).get("emailAddress"),
                "category": c.get("category"),
                "tags": c.get("tags"),
                "linkedOrderNumbers": c.get("linkedOrderNumbers"),
                "webStore": c.get("webStore"),
                "referrer": c.get("referrer"),
                "startDateTime": c.get("startDateTime"),
                "message_count": len(c.get("messages", [])),
            }
        )

    df_conv = pd.DataFrame(conv_rows)

    # --- Sheet 2: Messages (flat) ---
    msg_rows = []
    for c in conversations:
        for m in c.get("messages", []):
            msg_rows.append(
                {
                    "conversation_guid": c.get("guid"),
                    "sentDateTime": m.get("sentDateTime"),
                    "sender_type": (m.get("sender") or {}).get("type"),
                    "sender_name": (m.get("sender") or {}).get("name"),
                    "sender_email": (m.get("sender") or {}).get("emailAddress"),
                    "content": m.get("content"),
                    "scope": m.get("scope"),
                    "rating": m.get("rating"),
                }
            )

    df_msgs = pd.DataFrame(msg_rows)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df_conv.to_excel(writer, sheet_name="Conversations", index=False)
        df_msgs.to_excel(writer, sheet_name="Messages", index=False)

    log.info("Saved Excel → %s", path)


def print_summary(conversations: list) -> None:
    """Pretty-print a summary to stdout."""
    print(f"\n{'='*60}")
    print(f"  Total conversations: {len(conversations)}")
    print(f"{'='*60}")
    for i, c in enumerate(conversations, 1):
        msgs = c.get("messages", [])
        print(
            f"\n[{i}] {c.get('subject', '(no subject)')}"
            f"  |  Channel: {c.get('channel')}"
            f"  |  State: {c.get('state')}"
        )
        print(
            f"     Started: {c.get('startDateTime')}"
            f"  |  Owner: {(c.get('owner') or {}).get('name', 'N/A')}"
        )
        print(f"     Messages ({len(msgs)}):")
        for m in msgs:
            sender = (m.get("sender") or {}).get("name", "Unknown")
            ts = m.get("sentDateTime", "")
            content = (m.get("content") or "")[:120]
            print(f"       [{ts}] {sender}: {content}")
    print()


# =========================================================================== #
#  CLI
# =========================================================================== #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CM Portal – Chat History Extractor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--api",
        required=True,
        choices=["history", "inbox"],
        help=(
            "'history' = Conversation History API (Central Services); "
            "'inbox' = Agent Inbox API (Mobile Service Cloud)"
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path (.json or .xlsx). Default: print to stdout.",
    )

    # --- Conversation History API args ---
    hist = parser.add_argument_group("Conversation History API options")
    hist.add_argument("--account-id", help="CM account GUID (accountId).")
    hist.add_argument(
        "--product-token", help="CM product token (X-Cm-Producttoken)."
    )
    hist.add_argument("--session-id", help="routerSessionId to fetch.")
    hist.add_argument(
        "--include-events",
        action="store_true",
        help="Include session lifecycle events in the response.",
    )
    hist.add_argument(
        "--no-messages",
        action="store_true",
        help="Suppress messages (return events only).",
    )
    hist.add_argument(
        "--delete",
        action="store_true",
        help="DELETE the session history after fetching (use with care).",
    )

    # --- Agent Inbox API args ---
    inbox = parser.add_argument_group("Agent Inbox API options")
    inbox.add_argument("--inbox-username", help="Robin/CM inbox API username.")
    inbox.add_argument("--inbox-password", help="Robin/CM inbox API password.")
    inbox.add_argument("--email", help="Customer email address to filter by.")
    inbox.add_argument("--order-number", help="Order number to filter by.")
    inbox.add_argument("--conversation-id", help="Specific conversation ID.")
    inbox.add_argument("--guid", help="Specific conversation GUID.")
    inbox.add_argument(
        "--paginate",
        action="store_true",
        help="Fetch all pages (automatically increments skip).",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_path: Optional[str] = args.output
    is_excel = output_path and output_path.lower().endswith(".xlsx")

    # ------------------------------------------------------------------ #
    # API: Conversation History
    # ------------------------------------------------------------------ #
    if args.api == "history":
        required = ["account_id", "product_token", "session_id"]
        missing = [r for r in required if not getattr(args, r)]
        if missing:
            parser.error(
                f"--api history requires: {', '.join('--' + r.replace('_', '-') for r in missing)}"
            )

        data = fetch_session_history(
            account_id=args.account_id,
            product_token=args.product_token,
            session_id=args.session_id,
            include_events=args.include_events,
            include_messages=not args.no_messages,
        )

        if args.delete:
            confirm = input(
                "Are you sure you want to DELETE this session history? [yes/no]: "
            )
            if confirm.strip().lower() == "yes":
                delete_session_history(
                    account_id=args.account_id,
                    product_token=args.product_token,
                    session_id=args.session_id,
                )

        if output_path:
            if is_excel:
                # Wrap single-session data as a list if it looks like a conversation
                conversations = data if isinstance(data, list) else [data]
                save_excel(conversations, output_path)
            else:
                save_json(data, output_path)
        else:
            print(json.dumps(data, indent=2, ensure_ascii=False, default=str))

    # ------------------------------------------------------------------ #
    # API: Agent Inbox
    # ------------------------------------------------------------------ #
    elif args.api == "inbox":
        if not args.inbox_username or not args.inbox_password:
            parser.error(
                "--api inbox requires --inbox-username and --inbox-password."
            )

        if args.paginate:
            conversations = fetch_all_inbox_conversations(
                username=args.inbox_username,
                password=args.inbox_password,
                email_address=args.email,
                order_number=args.order_number,
            )
        else:
            result = fetch_inbox_conversations(
                username=args.inbox_username,
                password=args.inbox_password,
                email_address=args.email,
                order_number=args.order_number,
                conversation_id=args.conversation_id,
                guid=args.guid,
            )
            conversations = result.get("conversations", [])

        if output_path:
            if is_excel:
                save_excel(conversations, output_path)
            else:
                save_json({"conversations": conversations}, output_path)
        else:
            print_summary(conversations)


if __name__ == "__main__":
    main()