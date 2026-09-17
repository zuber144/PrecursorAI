"""
ai/ollama_client.py — Async HTTP client for local Ollama server.

Provides generate_ollama_json() used by classifier.py and pattern_analyzer.py
for text generation/reasoning with format="json" and think=False.
Gemini is NOT replaced here — embeddings.py continues to use gemini.py
and Gemini's embedding API unchanged.
"""
import logging
from typing import List, Dict, Any
import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


async def generate_ollama_json(
    messages: List[Dict[str, Any]],
    temperature: float = 0.1,
    num_predict: int = 1024,
    timeout: float = 180.0,
) -> str:
    """
    Call Ollama /api/chat with format="json" and think=False to produce
    fast, valid structured JSON output without reasoning overhead.
    """
    base_url = settings.OLLAMA_BASE_URL.rstrip("/").removesuffix("/v1")
    url = f"{base_url}/api/chat"

    payload = {
        "model": settings.OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "format": "json",
        "think": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "").strip()
