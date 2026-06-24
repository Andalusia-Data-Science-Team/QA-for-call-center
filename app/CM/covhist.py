import requests
import json

ACCOUNT_ID = "YOUR_ACCOUNT_ID"
ROUTER_SESSION_ID = "YOUR_ROUTER_SESSION_ID"
PRODUCT_TOKEN = "YOUR_PRODUCT_TOKEN"

url = (
    f"https://api.cm.com/router/conversation-history/v1/"
    f"accounts/{ACCOUNT_ID}/session-history"
)

params = {
    "routerSessionId": ROUTER_SESSION_ID,
    "includeEvents": "true",
    "includeMessages": "true",
    "includeSummary": "true",
    "summaryLanguage": "en",
    "maxSummaryLength": 100,
}

headers = {
    "X-CM-PRODUCTTOKEN": PRODUCT_TOKEN,
    "Accept": "application/json"
}

response = requests.get(
    url,
    headers=headers,
    params=params,
    timeout=30
)

print("Status:", response.status_code)

if not response.ok:
    print(response.text)
    exit()

data = response.json()

print(json.dumps(data, indent=2))

history = data.get("conversationHistory", [])

for item in history:
    sender = item.get("from", {})
    name = sender.get("name", "Unknown")

    print("=" * 80)
    print("Time:", item.get("timeStamp"))
    print("Sender:", name)
    print("Message:", item.get("text"))

history = data.get("conversationHistory", [])

for msg in history:
    sender = msg.get("from", {}).get("name", "Unknown")
    timestamp = msg.get("timeStamp", "")
    text = msg.get("text", "")

    print(f"[{timestamp}] {sender}:")
    print(text)
    print()

    