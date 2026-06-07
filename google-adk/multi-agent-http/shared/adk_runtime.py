"""Small helper around ADK Runner for API request handlers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


@dataclass
class AgentRunResult:
    text: str
    messages: list[str]


class AgentRuntime:
    """Owns one ADK runner and keeps in-memory sessions for chat continuity."""

    def __init__(self, agent: LlmAgent, app_name: str):
        self.app_name = app_name
        self.session_service = InMemorySessionService()
        self.runner = Runner(
            agent=agent,
            app_name=app_name,
            session_service=self.session_service,
        )
        self._session_ids: set[tuple[str, str]] = set()
        self._lock = asyncio.Lock()

    async def _ensure_session(self, user_id: str, session_id: str) -> None:
        session_key = (user_id, session_id)
        if session_key in self._session_ids:
            return

        async with self._lock:
            if session_key in self._session_ids:
                return
            await self.session_service.create_session(
                app_name=self.app_name,
                user_id=user_id,
                session_id=session_id,
            )
            self._session_ids.add(session_key)

    async def run(self, message: str, user_id: str, session_id: str) -> AgentRunResult:
        await self._ensure_session(user_id=user_id, session_id=session_id)

        messages: list[str] = []
        async for event in self.runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=types.Content(
                role="user",
                parts=[types.Part(text=message)],
            ),
        ):
            content = getattr(event, "content", None)
            parts = getattr(content, "parts", None)
            if not parts:
                continue
            for part in parts:
                text = getattr(part, "text", None)
                if text:
                    messages.append(text)

        return AgentRunResult(
            text=messages[-1] if messages else "",
            messages=messages,
        )
