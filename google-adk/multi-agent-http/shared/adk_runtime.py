"""Small helper around ADK Runner for API request handlers."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


@dataclass
class AgentRunResult:
    text: str
    messages: list[str]
    session_id: str


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
        self._session_user: dict[str, str] = {}  # session_id -> user_id
        self._lock = asyncio.Lock()

    async def _create_session(self, user_id: str, session_id: str) -> None:
        await self.session_service.create_session(
            app_name=self.app_name,
            user_id=user_id,
            session_id=session_id,
        )
        self._session_user[session_id] = user_id

    async def get_or_create_session(self, user_id: str, session_id: str | None) -> str:
        if session_id and session_id in self._session_user:
            return session_id
        async with self._lock:
            new_id = session_id or str(uuid.uuid4())
            if new_id not in self._session_user:
                await self._create_session(user_id=user_id, session_id=new_id)
        return new_id

    async def run(self, message: str, user_id: str, session_id: str | None = None) -> AgentRunResult:
        session_id = await self.get_or_create_session(user_id=user_id, session_id=session_id)

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
            session_id=session_id,
        )
