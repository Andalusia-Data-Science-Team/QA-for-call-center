import requests
import json
import os
from datetime import datetime, timezone

# ==========================================
# CONFIGURATION
# ==========================================
# It's best practice to use environment variables for sensitive tokens
#CM_PRODUCT_TOKEN = os.getenv("CM_PRODUCT_TOKEN", "YOUR_CM_PRODUCT_TOKEN_HERE")
CM_PRODUCT_TOKEN = "00000000-0000-0000-0000-000000000000"
CONVERSATION_ID = os.getenv("CONVERSATION_ID", "e937d210-a464-f111-8fcb-00224888ec9a")

BASE_URL = "https://api.cm.com/conversationhistory/v1"
OUTPUT_FILE = f"cm_chat_history_{CONVERSATION_ID}.json"

# Optional: Filter by date (ISO 8601 format). Set to None if you want all history.
FROM_DATE = "2023-01-01T00:00:00Z" 
TO_DATE = None 

def fetch_chat_history(conversation_id):
    """Fetches all messages for a given conversation, handling pagination."""
    
    if CM_PRODUCT_TOKEN == "YOUR_CM_PRODUCT_TOKEN_HERE":
        raise ValueError("Please set your CM_PRODUCT_TOKEN")

    headers = {
        "X-CM-ProductToken": CM_PRODUCT_TOKEN,
        "Accept": "application/json"
    }

    # Initial endpoint for fetching messages of a specific conversation
    url = f"{BASE_URL}/conversations/{conversation_id}/messages"
    
    params = {}
    if FROM_DATE:
        params["fromDate"] = FROM_DATE
    if TO_DATE:
        params["toDate"] = TO_DATE

    all_messages = []
    page_count = 1

    print(f"Starting extraction for Conversation ID: {conversation_id}...")

    while url:
        try:
            response = requests.get(url, headers=headers, params=params if page_count == 1 else None)
            response.raise_for_status()  # Raises an error for HTTP 4xx/5xx
            
            data = response.json()

            # CM.com usually returns an object with a 'messages' or 'items' array. 
            # Adjust the key below if the exact API response structure differs slightly.
            if isinstance(data, dict) and "messages" in data:
                messages = data.get("messages", [])
                all_messages.extend(messages)
                
                # Handle Pagination: Check if there is a next page URL provided in the response
                url = data.get("nextPageUrl") or data.get("nextPage")
            elif isinstance(data, list):
                # Fallback if the API returns a raw array of messages
                all_messages.extend(data)
                url = None # Assuming no pagination header in list responses
            else:
                print("Unexpected response format.")
                print(json.dumps(data, indent=2))
                break

            print(f"Fetched page {page_count} (Total messages so far: {len(all_messages)})")
            page_count += 1
            params = None # Params are usually included in the nextPageUrl, so we clear them

        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error: {e}")
            print(f"Response body: {e.response.text}")
            break
        except requests.exceptions.RequestException as e:
            print(f"Network/Request Error: {e}")
            break

    return all_messages

def format_timestamp(message):
    """Helper to make timestamps readable in the console output."""
    if "time" in message:
        try:
            # Parse ISO format and convert to readable string
            dt = datetime.fromisoformat(message["time"].replace('Z', '+00:00'))
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            return message["time"]
    return "Unknown time"

def main():
    messages = fetch_chat_history(CONVERSATION_ID)

    if not messages:
        print("No messages found or failed to fetch.")
        return

    # 1. Print a nicely formatted summary to the console
    print("\n" + "="*50)
    print(f"CHAT HISTORY SUMMARY ({len(messages)} messages)")
    print("="*50)
    
    for msg in messages:
        sender = msg.get("sender", "Unknown")
        msg_type = msg.get("type", "text")
        time_str = format_timestamp(msg)
        
        # Handle different message types (text, media, etc.)
        if msg_type == "text":
            content = msg.get("content", {}).get("text", "[No text content]")
        elif msg_type == "media":
            content = "[Media/Attachment]"
        else:
            content = f"[{msg_type.upper()}]"
            
        print(f"[{time_str}] {sender}: {content}")

    # 2. Save raw data to a JSON file for further processing
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(messages, f, indent=4, ensure_ascii=False)
        
    print(f"\nRaw JSON data saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()