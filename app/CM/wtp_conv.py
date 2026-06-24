""" import requests
import json
import base64

api_key = "krer4i1R"
api_secret = "spqfVnL9Ze"

credentials = f"{api_key}:{api_secret}"
encoded = base64.b64encode(credentials.encode()).decode()

authorization = f"Basic {encoded}"

print(authorization)

url = "https://api.robinhq.com/conversations"

headers = {
    "Authorization": authorization,
    "Content-Type": "application/json"
}

payload = {
    "emailAddress": "Rafik.Atallah@andalusiagroup.net",
    "orderNumber": "",
    "skip": None,
    "guid": "",
    "conversationId": "ccd3b7b8-b881-f011-b481-000d3a26ed83"
}

response = requests.post(
    url,
    json=payload,
    headers=headers,
    timeout=30
)

print(response.status_code)
print(response.text)

data = response.json()

for conv in data.get("conversations", []):
    print("Subject:", conv.get("subject"))
    print("Channel:", conv.get("channel"))

    for msg in conv.get("messages", []):
        print(
            msg["sentDateTime"],
            msg["sender"]["name"],
            msg["content"]
        ) """
import requests
from requests.auth import HTTPBasicAuth

"""API_KEY = "krer4i1R"
API_SECRET = "spqfVnL9Ze"
"""
API_KEY = "g24x9Rgo"
API_SECRET = "ZAdfx12hKH"

#url = "https://api.robinhq.com/conversations"
#url = "http://online.andalusiaksa.com:7042/search?<yourExpression>=966590760131"
url = "http://online.andalusiaksa.com:7042/orders?Email=&Mobile=966590760131"
url = "https://online.andalusiaksa.com:7042/search?rafik.atallah@andalusiagroup.net=$Email"

payload = {
    # use whichever identifiers you have
    "conversationId": "e937d210-a464-f111-8fcb-00224888ec9a",
    "skip": 0
}

""" headers = {
    "Authorization": "Basic V2hhdHRzQXBwOll4RSZXdyFzVXYyaw==",
    "Content-Type": "application/json"
} """

response = requests.post(
    url,
    #json=payload,
    #headers=headers,
    auth=HTTPBasicAuth(API_KEY, API_SECRET),
    timeout=30
)

print("Status:", response.status_code)
print("Content-Type:", response.headers.get("content-type"))
print("Body:")
print(response.text)

if response.ok:
    data = response.json()

    for conv in data.get("conversations", []):
        print(f"\nConversation: {conv.get('guid')}")
        print(f"Channel: {conv.get('channel')}")

        for msg in conv.get("messages", []):
            sender = msg.get("sender", {}).get("name", "Unknown")
            print(f"{msg.get('sentDateTime')} | {sender}")
            print(msg.get("content"))
            print("-" * 50)