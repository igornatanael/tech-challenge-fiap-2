from __future__ import annotations

import time

from src.llm.client import _get_client, DEFAULT_MODEL
from src.observability import log_event


class BaseAgent:
    """Agente com histórico de conversa multi-turn."""

    def __init__(self, system_prompt: str, max_tokens: int = 1500):
        self.system_prompt = system_prompt
        self.history: list[dict] = []
        self.max_tokens = max_tokens
        self.session_id: str | None = None
        self._agent_type = type(self).__name__

    def chat(self, user_message: str) -> str:
        self.history.append({"role": "user", "content": user_message})
        client = _get_client()

        log_event(
            "llm.call.started",
            session_id=self.session_id,
            agent_type=self._agent_type,
            model=DEFAULT_MODEL,
            turn=len(self.history),
        )

        t0 = time.perf_counter()
        response = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=self.max_tokens,
            system=self.system_prompt,
            messages=self.history,
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000)

        assistant_message = response.content[0].text
        self.history.append({"role": "assistant", "content": assistant_message})

        usage = response.usage
        log_event(
            "llm.call.completed",
            session_id=self.session_id,
            agent_type=self._agent_type,
            model=DEFAULT_MODEL,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            elapsed_ms=elapsed_ms,
            turn=len(self.history) // 2,
        )

        return assistant_message

    def reset(self):
        self.history = []
