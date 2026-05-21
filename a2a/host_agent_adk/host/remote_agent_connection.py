import sys
from pathlib import Path
from typing import Callable

import httpx
from a2a.client import Client, ClientCallContext, ClientConfig, ClientFactory
from a2a.types import (
    AgentCard,
    SendMessageRequest,
    Task,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
)
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.instrumentation import make_trace_service_parameters

TaskCallbackArg = Task | TaskStatusUpdateEvent | TaskArtifactUpdateEvent
TaskUpdateCallback = Callable[[TaskCallbackArg, AgentCard], Task]


class RemoteAgentConnections:
    """A class to hold the connections to the remote agents."""

    def __init__(self, agent_card: AgentCard, agent_url: str):
        print(f"agent_card: {agent_card}")
        print(f"agent_url: {agent_url}")
        self._httpx_client = httpx.AsyncClient(timeout=120)
        factory = ClientFactory(ClientConfig(httpx_client=self._httpx_client, streaming=False))
        self.agent_client: Client = factory.create(agent_card)
        self.card = agent_card
        self.conversation_name = None
        self.conversation = None
        self.pending_tasks = set()

    def get_agent(self) -> AgentCard:
        return self.card

    async def send_message(self, message_request: SendMessageRequest) -> Task | None:
        context = ClientCallContext(service_parameters=make_trace_service_parameters())
        task: Task | None = None
        async for response in self.agent_client.send_message(message_request, context=context):
            if response.HasField("task"):
                task = response.task
        return task
