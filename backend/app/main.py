from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .huggingface_client import ask_llm


app = FastAPI(
    title="opsLLM API",
    version="1.0.0",
)

 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str


@app.get("/")
def home():
    return {
        "status": "Backend Running"
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    answer = ask_llm(request.message)

    return ChatResponse(
        answer=answer
    )

  

class OpenAIMessage(BaseModel):
    role: str
    content: str


class OpenAIChatRequest(BaseModel):
    model: str
    messages: list[OpenAIMessage]
    temperature: float | None = 0.7
    max_tokens: int | None = 300


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "openai/gpt-oss-120b:groq",
                "object": "model",
                "owned_by": "opsLLM",
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(request: OpenAIChatRequest):
 
    prompt_parts = []

    for message in request.messages:
        prompt_parts.append(
            f"{message.role}: {message.content}"
        )

    prompt = "\n".join(prompt_parts)

    answer = ask_llm(prompt)

    return {
        "id": "opsllm-chat",
        "object": "chat.completion",
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer,
                },
                "finish_reason": "stop",
            }
        ],
    }
