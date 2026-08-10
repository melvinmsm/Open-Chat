import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import InferenceClient


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = os.getenv("MODEL_ID", "openai/gpt-oss-120b:groq")

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is not set")

print("HF token loaded:", HF_TOKEN[:7] + "...")
print("Model:", MODEL_ID)

client = InferenceClient(
    api_key=HF_TOKEN
)


def ask_llm(prompt: str) -> str:
    completion = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        max_tokens=300,
    )

    return completion.choices[0].message.content 
