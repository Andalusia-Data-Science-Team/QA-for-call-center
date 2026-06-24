import requests
import base64
import HTTPBasicAuth

API_KEY = "krer4i1R"
API_SECRET = "spqfVnL9Ze"

def get_conversations(skip=0, email_address=None, order_number=None,
                       guid=None, conversation_id=None):
    url = "https://api.robinhq.com/conversations"

    body = {"skip": skip}
    if email_address:
        body["emailAddress"] = email_address
    if order_number:
        body["orderNumber"] = order_number
    if guid:
        body["guid"] = guid
    if conversation_id:
        body["conversationId"] = conversation_id

    response = requests.post(
        url,
        json=body,
        auth=HTTPBasicAuth(API_KEY, API_SECRET),
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def get_all_conversations(email_address="Rafik.atallah@andalusiagroup.net", order_number=None, page_size=50):
    """Paginate through all matching conversations using skip."""
    all_conversations = []
    skip = 0
    while True:
        data = get_conversations(skip=skip, email_address=email_address,
                                  order_number=order_number)
        batch = data.get("conversations", [])
        if not batch:
            break
        all_conversations.extend(batch)
        skip += len(batch)
        if len(batch) < page_size:
            break
    return all_conversations


if __name__ == "__main__":
    conversations = get_all_conversations()
    for conv in conversations:
        print(conv["guid"], conv["subject"], conv["channel"], conv["state"])
        for msg in conv.get("messages", []):
            print("  -", msg["sentDateTime"], msg["sender"]["name"], ":", msg["content"])