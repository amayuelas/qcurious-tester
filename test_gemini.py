"""Quick test of Gemini API connectivity."""

import os
from dotenv import load_dotenv
load_dotenv(override=True)

from openai import OpenAI

api_key = os.environ.get("GEMINI_API_KEY", "")
api_base = "https://generativelanguage.googleapis.com/v1beta/openai"
model = "gemini-3-flash-preview"

print(f"API base: {api_base}")
print(f"Model: {model}")
print(f"Key: {api_key[:15]}..." if api_key else "Key: EMPTY")
print()

client = OpenAI(base_url=api_base, api_key=api_key)

try:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Say hello"}],
        max_tokens=10,
        temperature=0.3,
    )
    print(f"Response: {resp.choices[0].message.content}")
    print("SUCCESS")
except Exception as e:
    print(f"ERROR: {e}")
