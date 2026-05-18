"""
LLM service - interact with Groq API for text generation.
Replaces Ollama with Groq (free tier, llama3.1:8b, ~500 tokens/sec).
"""

import json
from typing import AsyncGenerator

import httpx
from loguru import logger

from app.core.config import settings

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {settings.groq_api_key}",
        "Content-Type": "application/json",
    }


async def generate(prompt: str, system_prompt: str | None = None) -> str:
    """Generate a response from the LLM (non-streaming)."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": settings.groq_model,
        "messages": messages,
        "stream": False,
        "temperature": 0.7,
        "max_tokens": 1024,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(GROQ_API_URL, headers=_headers(), json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


async def generate_stream(
    prompt: str, system_prompt: str | None = None
) -> AsyncGenerator[str, None]:
    """Stream a response from the LLM token by token."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": settings.groq_model,
        "messages": messages,
        "stream": True,
        "temperature": 0.7,
        "max_tokens": 1024,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream(
            "POST", GROQ_API_URL, headers=_headers(), json=payload
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                # SSE lines look like: "data: {...}" or "data: [DONE]"
                if not line.startswith("data: "):
                    continue
                raw = line[len("data: "):]
                if raw.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(raw)
                    delta = chunk["choices"][0]["delta"]
                    content = delta.get("content")
                    if content:
                        yield content
                except (json.JSONDecodeError, KeyError):
                    continue


async def check_connection() -> bool:
    """Check if Groq API is reachable and the API key is valid."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                "https://api.groq.com/openai/v1/models",
                headers=_headers(),
            )
            return response.status_code == 200
    except Exception as e:
        logger.warning(f"Groq connection check failed: {e}")
        return False