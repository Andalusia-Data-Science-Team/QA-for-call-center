import json

with open("/home/ai/Workspace/Rafik/QA/chat_jsons/chat_Esraa Yousef.json", "r", encoding="utf-8") as f:
    messages = json.load(f)

transcript_lines = []
for msg in messages:
    if "agent" in msg:
        transcript_lines.append(f"Agent: {msg['agent']}")
    elif "patient" in msg:
        transcript_lines.append(f"Patient: {msg['patient']}")

payload = {
    "call_id": "CHAT-ESRAA-YOUSEF-001",
    "agent_name": "Esraa Yousef",
    "call_date": None,
    "call_duration_seconds": None,
    "department": None,
    "transcript": "\n".join(transcript_lines)
}

import os

output_dir = os.path.join(os.path.dirname(__file__), "chats")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.json")

with open(output_path, "w", encoding="utf-8") as out_f:
    json.dump(payload, out_f, ensure_ascii=False, indent=2)