from __future__ import annotations

import os

import anthropic
from dotenv import load_dotenv

load_dotenv()

_client: anthropic.Anthropic | None = None

DEFAULT_MODEL = "claude-sonnet-4-6"


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY nao encontrada. "
                "Defina a variavel no arquivo .env ou no ambiente."
            )
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def chat(prompt: str, system: str = "", max_tokens: int = 1024) -> str:
    try:
        client = _get_client()
        kwargs: dict = {
            "model": DEFAULT_MODEL,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        response = client.messages.create(**kwargs)
        return response.content[0].text
    except anthropic.APIError as e:
        raise RuntimeError(f"Erro na API Anthropic: {e}") from e
