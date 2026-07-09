import requests
import json
import os
from datetime import datetime, timezone

# ==========================================
# CONFIGURATION
# ==========================================
# Robin HQ API credentials
API_KEY = os.getenv("ROBIN_API_KEY", "krer4i1R")
API_SECRET = os.getenv("ROBIN_API_SECRET", "oDgXPCWcOa")
CONVERSATION_ID = os.getenv("CONVERSATION_ID", "48F76C0A-7076-F111-AC9A-000D3AA9D522")
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS", "Sara.AbdElShafi@Andalusiagroup.net")
#ORDER_NUMBER = os.getenv("ORDER_NUMBER", "CMO1234")
#GUID = os.getenv("GUID", "2eeb7f72-0239-4104-9d85-d2300d214ff2")

BASE_URL = "https://api.robinhq.com/conversations"


OUTPUT_FILE = f"robin_conversations_{CONVERSATION_ID}.json"

# Optional: Pagination skip parameter
SKIP = 0

# Search API Configuration
SEARCH_URL = os.getenv("SEARCH_URL", "https://api.robinhq.com/showcustomer")
EMAIL_PARAM = os.getenv("EMAIL_PARAM", "Email")  # Parameter name for email

# Expression Search Configuration
EXPRESSION_SEARCH_URL = os.getenv("EXPRESSION_SEARCH_URL", "https://online.andalusiaksa.com/search")
EXPRESSION_PARAM = os.getenv("EXPRESSION_PARAM", "$Expression")  # Parameter name for expression
EXPRESSION_VALUE = os.getenv("EXPRESSION_VALUE", "*")  # Default expression value

import base64
credentials = f"{API_KEY}:{API_SECRET}"
encoded_credentials = base64.b64encode(credentials.encode()).decode()

def fetch_chat_history(conversation_id):
    """Fetches all messages for a given conversation using Robin HQ API."""
    
    if not API_KEY or not API_SECRET:
        raise ValueError("Please set ROBIN_API_KEY and ROBIN_API_SECRET environment variables")

    # Set up headers with API key and secret for authentication
    
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Basic {encoded_credentials}",
    }

    # Robin HQ API endpoint
    url = BASE_URL
    
    # Build the POST request payload
    payload = {
        #"emailAddress": "Esraa.Yousef@Andalusiagroup.net",
        #"orderNumber": ORDER_NUMBER,
        #"skip": 100,
        #"guid": "9110F139-E274-F111-AC9A-000D3AA9D522",
        "conversationId": conversation_id
    }

    all_messages = []
    page_count = 1

    print(f"Starting extraction for Conversation ID: {conversation_id}...")
    print(f"Using endpoint: {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")

    while url:
        try:
            # POST request with API key and secret authentication
            response = requests.post(url, headers=headers, json=payload)
            print(f"\nResponse Status Code: {response.status_code}")
            print(f"Response Headers: {dict(response.headers)}")
            print(f"Response Text (first 500 chars): {response.text[:500]}")
            
            response.raise_for_status()  # Raises an error for HTTP 4xx/5xx
            
            data = response.json()

            # Robin HQ API response handling
            if isinstance(data, dict) and "messages" in data:
                messages = data.get("messages", [])
                all_messages.extend(messages)
                
                # Handle Pagination: Update skip parameter
                if messages and len(messages) > 0:
                    payload["skip"] = payload.get("skip", 0) + len(messages)
                else:
                    url = None
            elif isinstance(data, list):
                # Fallback if the API returns a raw array
                all_messages.extend(data)
                url = None
            elif isinstance(data, dict):
                # Single conversation object returned
                all_messages.append(data)
                url = None
            else:
                print("Unexpected response format.")
                print(json.dumps(data, indent=2))
                break

            print(f"Fetched page {page_count} (Total items so far: {len(all_messages)})")
            page_count += 1

        except requests.exceptions.HTTPError as e:
            print(f"\nHTTP Error: {e}")
            print(f"Response body: {e.response.text}")
            break
        except json.JSONDecodeError as e:
            print(f"\nJSON Decode Error: {e}")
            print(f"Response was not valid JSON. Raw response: {response.text[:1000]}")
            break
        except requests.exceptions.RequestException as e:
            print(f"\nNetwork/Request Error: {e}")
            break

    return all_messages

def search_by_email(email_address):
    """Searches for conversations/data by email address using POST request."""
    email_address = "Sara.AbdElShafi@Andalusiagroup.net"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Basic {encoded_credentials}",
    }

    # Build the URL with email parameter
    params = {
        #"ChannelAccountExternalIdentifier": "customChannel",
        #"RelationExternalIdentifier": email_address,
        #"ConversationGuid": CONVERSATION_ID,
        #"user_email_address": email_address,
        "customer": {
            "email_address": email_address
        }
        
        #EMAIL_PARAM: email_address
    }
    
    print(f"\nStarting search for Email: {email_address}...")
    print(f"Using endpoint: {SEARCH_URL}")
    print(f"Payload: {json.dumps(params, indent=2)}")

    try:
        # POST request with email payload (JSON body)
        response = requests.post(SEARCH_URL, headers=headers, json=params)
        print(f"\nResponse Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print(f"Response Text (first 500 chars): {response.text[:500]}")
        
        response.raise_for_status()  # Raises an error for HTTP 4xx/5xx
        
        data = response.json()
        print(f"\nSearch Results:")
        print(json.dumps(data, indent=2))
        
        return data

    except requests.exceptions.HTTPError as e:
        print(f"\nHTTP Error: {e}")
        print(f"Response body: {e.response.text}")
        return None
    except json.JSONDecodeError as e:
        print(f"\nJSON Decode Error: {e}")
        print(f"Response was not valid JSON. Raw response: {response.text[:1000]}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"\nNetwork/Request Error: {e}")
        return None

def search_by_expression(expression_value):
    """Searches for data by expression using GET request."""
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Basic {encoded_credentials}",
    }

    # Build the URL with expression parameter
    params = {
        EXPRESSION_PARAM: expression_value
    }
    
    print(f"\nStarting search for Expression: {expression_value}...")
    print(f"Using endpoint: {EXPRESSION_SEARCH_URL}")
    print(f"Params: {json.dumps(params, indent=2)}")

    try:
        # GET request with expression parameter
        response = requests.post(EXPRESSION_SEARCH_URL, headers=headers, params=params)
        print(f"\nResponse Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print(f"Response Text (first 500 chars): {response.text[:500]}")
        
        response.raise_for_status()  # Raises an error for HTTP 4xx/5xx
        
        data = response.json()
        print(f"\nExpression Search Results:")
        print(json.dumps(data, indent=2))
        
        return data

    except requests.exceptions.HTTPError as e:
        print(f"\nHTTP Error: {e}")
        print(f"Response body: {e.response.text}")
        return None
    except json.JSONDecodeError as e:
        print(f"\nJSON Decode Error: {e}")
        print(f"Response was not valid JSON. Raw response: {response.text[:1000]}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"\nNetwork/Request Error: {e}")
        return None

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
    
    # Fetch conversation history using POST request
    messages = fetch_chat_history(CONVERSATION_ID)

    if not messages:
        print("No messages found or failed to fetch.")
    else:
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

    # Search by email using GET request
    """ search_results = search_by_email("AbdElRahman.AlQadi@Andalusiagroup.net")
    
    if search_results:
        search_output_file = f"search_results_{EMAIL_ADDRESS.replace('@', '_').replace('.', '_')}.json"
        with open(search_output_file, 'w', encoding='utf-8') as f:
            json.dump(search_results, f, indent=4, ensure_ascii=False)
        print(f"Search results saved to: {search_output_file}") """
    
    # Search by expression using GET request
    """ expression_results = search_by_expression("966539599471")
    
    if expression_results:
        expression_output_file = f"expression_search_results_{EXPRESSION_VALUE.replace('*', 'all').replace('@', '_').replace('.', '_')}.json"
        with open(expression_output_file, 'w', encoding='utf-8') as f:
            json.dump(expression_results, f, indent=4, ensure_ascii=False)
        print(f"Expression search results saved to: {expression_output_file}") """

if __name__ == "__main__":
    main()