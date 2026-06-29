import os
import re
import json
import pymupdf
from dotenv import load_dotenv
from openai import OpenAI
from arabic_reshaper import reshape

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

MODEL = "openai/gpt-oss-120b"

def extract_pdf_text(pdf_path: str) -> str:
    doc = pymupdf.open(pdf_path)

    pages = []

    for page in doc:
        text = page.get_text()
        text = reshape(text)
        pages.append(text)

    return "\n".join(pages)

SYSTEM_PROMPT = """
You are an expert conversation extraction engine.

Convert Andalusia chat transcripts into EXACTLY this JSON schema but only when the human agent start talking:

{
  "call_id": "",
  "agent_name": "",
  "call_date": "YYYY-MM-DD",
  "call_duration_seconds": 1,
  "department": "",
  "transcript": ""
}

Rules:

1. Extract agent name.
2. Extract conversation date.
3. Infer department if possible.
4. Build transcript in this format:

Agent: ...
Patient: ...
Agent: ...

5. Ignore system events:
   - archived conversation
   - reopened conversation
   - bot routing menus

6. Return VALID JSON ONLY.
7. No markdown.
8. No explanations.
9. Ignore Bot messages and start parsing the conversation when the human agent appears.
10. Delete any special characters that are not part of the conversation.
11. Refine the arabic TEXT parsing direction.
"""

def convert_text_to_json(raw_text: str):

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": raw_text
            }
        ]
    )
    content = response.choices[0].message.content
    reshaped_content = reshape(content)

    try:
        result = json.loads(reshaped_content)

    except json.JSONDecodeError:
        from json_repair import repair_json

        fixed = repair_json(reshaped_content)

        result = json.loads(fixed)
    return result

def process_pdf(pdf_path):

    text = extract_pdf_text(pdf_path)

    result = convert_text_to_json(text)

    filename = (
        os.path.basename(pdf_path)
        .replace(".pdf", ".json")
    )

    output_path = os.path.join(
        "output_json",
        filename
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2
        )

    return result

def process_folder(folder):

    pdfs = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".pdf")
    ]

    for pdf in pdfs:

        try:
            process_pdf(pdf)

            print(f"✓ {pdf}")

        except Exception as e:

            print(f"✗ {pdf}")
            print(e)

if __name__ == "__main__":

    process_folder("/home/ai/Workspace/Rafik/QA_System-main/app/chats/")