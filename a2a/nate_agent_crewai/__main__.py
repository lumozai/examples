"""This file serves as the main entry point for the application.

It initializes the A2A server, defines the agent's capabilities,
and starts the server to handle incoming requests.
"""

import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables FIRST — before any framework imports that might
# auto-configure OTel (e.g. CrewAI's EventListener sets its own TracerProvider
# at module-level when crewai is imported).
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

# Configure OTel instrumentation BEFORE importing crewai.  CrewAI's EventListener
# singleton runs trace.set_tracer_provider(crewai_provider) at import time; if it
# runs first, our subsequent set_tracer_provider() is silently ignored ("Overriding
# not allowed") and remote parent spans won't propagate correctly.
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.instrumentation import configure_openinference, OTelPropagationMiddleware
tracer_provider = configure_openinference(service_name="nate-agent-crewai")

import uvicorn
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes.agent_card_routes import create_agent_card_routes
from a2a.server.routes.jsonrpc_routes import create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
)
from starlette.applications import Starlette
from agent import SchedulingAgent
from agent_executor import SchedulingAgentExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MissingAPIKeyError(Exception):
    """Exception for missing API key."""


def main():
    """Entry point for Nate's Scheduling Agent."""
    host = "localhost"
    port = 10003
    try:
        if not os.getenv("OPENAI_API_KEY"):
            raise MissingAPIKeyError("OPENAI_API_KEY environment variable not set.")

        capabilities = AgentCapabilities(streaming=False)
        skill = AgentSkill(
            id="availability_checker",
            name="Availability Checker",
            description="Check my calendar to see when I'm available for a pickleball game.",
            tags=["schedule", "availability", "calendar"],
            examples=[
                "Are you free tomorrow?",
                "Can you play pickleball next Tuesday at 5pm?",
            ],
        )

        agent_host_url = os.getenv("HOST_OVERRIDE") or f"http://{host}:{port}/"
        agent_card = AgentCard(
            name="Nate Agent",
            description="A friendly agent to help you schedule a pickleball game with Nate.",
            supported_interfaces=[AgentInterface(url=agent_host_url, protocol_binding="JSONRPC")],
            version="1.0.0",
            default_input_modes=SchedulingAgent.SUPPORTED_CONTENT_TYPES,
            default_output_modes=SchedulingAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )

        request_handler = DefaultRequestHandler(
            agent_executor=SchedulingAgentExecutor(),
            task_store=InMemoryTaskStore(),
            agent_card=agent_card,
        )
        routes = create_agent_card_routes(agent_card) + create_jsonrpc_routes(request_handler, "/")
        app = Starlette(routes=routes)
        uvicorn.run(OTelPropagationMiddleware(app), host=host, port=port)

    except MissingAPIKeyError as e:
        logger.error(f"Error: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"An error occurred during server startup: {e}")
        exit(1)


if __name__ == "__main__":
    main()
