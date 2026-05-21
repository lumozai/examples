import logging
import os
import sys
from pathlib import Path

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
from agent import create_agent
from agent_executor import KarleyAgentExecutor
from dotenv import load_dotenv
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.instrumentation import configure_openinference, OTelPropagationMiddleware
tracer_provider = configure_openinference(service_name="karley-agent-adk")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MissingAPIKeyError(Exception):
    """Exception for missing API key."""

    pass


def main():
    """Starts the agent server."""
    host = "localhost"
    port = 10002
    try:
        # Check for API key only if Vertex AI is not configured
        ## if not os.getenv("GOOGLE_GENAI_USE_VERTEXAI") == "TRUE":
        ##     if not os.getenv("GOOGLE_API_KEY"):
        ##         raise MissingAPIKeyError(
        ##             "GOOGLE_API_KEY environment variable not set and GOOGLE_GENAI_USE_VERTEXAI is not TRUE."
        ##         )

        capabilities = AgentCapabilities(streaming=True)
        skill = AgentSkill(
            id="check_schedule",
            name="Check Karley's Schedule",
            description="Checks Karley's availability for a pickleball game on a given date.",
            tags=["scheduling", "calendar"],
            examples=["Is Karley free to play pickleball tomorrow?"],
        )
        agent_card = AgentCard(
            name="Karley Agent",
            description="An agent that manages Karley's schedule for pickleball games.",
            supported_interfaces=[AgentInterface(url=f"http://{host}:{port}/", protocol_binding="JSONRPC")],
            version="1.0.0",
            default_input_modes=["text/plain"],
            default_output_modes=["text/plain"],
            capabilities=capabilities,
            skills=[skill],
        )

        adk_agent = create_agent()
        runner = Runner(
            app_name=agent_card.name.replace(" ", "_"),
            agent=adk_agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )
        agent_executor = KarleyAgentExecutor(runner)

        request_handler = DefaultRequestHandler(
            agent_executor=agent_executor,
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
