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
from app.agent import KaitlynAgent
from app.agent_executor import KaitlynAgentExecutor
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.instrumentation import configure_openinference, OTelPropagationMiddleware
tracer_provider = configure_openinference(service_name="kaitlynn-agent-langgraph")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MissingAPIKeyError(Exception):
    """Exception for missing API key."""


def main():
    """Starts Kaitlyn's Agent server."""
    host = "localhost"
    port = 10004
    try:
        if not os.getenv("OPENAI_API_KEY"):
            raise MissingAPIKeyError("OPENAI_API_KEY environment variable not set.")

        capabilities = AgentCapabilities(streaming=True, push_notifications=True)
        skill = AgentSkill(
            id="schedule_pickleball",
            name="Pickleball Scheduling Tool",
            description="Helps with finding Kaitlyn's availability for pickleball",
            tags=["scheduling", "pickleball"],
            examples=["Are you free to play pickleball on Saturday?"],
        )
        agent_card = AgentCard(
            name="Kaitlynn Agent",
            description="Helps with scheduling pickleball games",
            supported_interfaces=[AgentInterface(url=f"http://{host}:{port}/", protocol_binding="JSONRPC")],
            version="1.0.0",
            default_input_modes=KaitlynAgent.SUPPORTED_CONTENT_TYPES,
            default_output_modes=KaitlynAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )

        request_handler = DefaultRequestHandler(
            agent_executor=KaitlynAgentExecutor(),
            task_store=InMemoryTaskStore(),
            agent_card=agent_card,
        )
        routes = create_agent_card_routes(agent_card) + create_jsonrpc_routes(request_handler, "/")
        app = Starlette(routes=routes)
        uvicorn.run(OTelPropagationMiddleware(app), host=host, port=port)

    except MissingAPIKeyError as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred during server startup: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
