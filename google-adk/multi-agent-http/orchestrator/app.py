"""User-facing FastAPI service that hosts the ADK travel orchestrator agent."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import requests
from fastapi import FastAPI, HTTPException
from opentelemetry import propagate, trace
from pydantic import BaseModel, Field

from google.adk.agents import LlmAgent

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from shared.adk_runtime import AgentRuntime
from shared.config import get_model, get_remote_city_expert_url
from shared.tracing import (
    instrument_fastapi_app,
    instrument_requests,
    set_span_attributes,
)


logger = logging.getLogger(__name__)


def _parse_interests(interests_json: str | list[str] | None) -> list[str]:
    if isinstance(interests_json, list):
        return [str(item).lower() for item in interests_json]
    if not interests_json:
        return []
    try:
        parsed = json.loads(interests_json)
        if isinstance(parsed, list):
            return [str(item).lower() for item in parsed]
    except json.JSONDecodeError:
        pass
    return [item.strip().lower() for item in str(interests_json).split(",") if item.strip()]


def estimate_trip_budget(
    destination: str,
    days: int = 3,
    budget: str = "medium",
    travelers: int = 2,
) -> dict[str, Any]:
    """Estimate a simple trip budget range for lodging, food, transit, and activities."""
    level = budget.lower()
    daily_by_level = {
        "low": {"lodging": 150, "food": 45, "transit": 20, "activities": 35},
        "medium": {"lodging": 240, "food": 80, "transit": 35, "activities": 70},
        "high": {"lodging": 420, "food": 150, "transit": 70, "activities": 140},
    }
    daily = daily_by_level.get(level, daily_by_level["medium"])
    lodging_total = daily["lodging"] * days
    per_person_total = (daily["food"] + daily["transit"] + daily["activities"]) * days
    total = lodging_total + per_person_total * travelers

    return {
        "destination": destination,
        "days": days,
        "travelers": travelers,
        "budget_level": level,
        "estimated_total_usd": total,
        "daily_assumptions_usd": daily,
        "note": "Airfare is excluded because the origin city is unknown.",
    }


def create_itinerary_skeleton(
    destination: str,
    days: int = 3,
    interests_json: str = "[]",
) -> dict[str, Any]:
    """Create a high-level itinerary structure before local recommendations are added."""
    interests = _parse_interests(interests_json)
    day_count = max(1, min(days, 10))
    skeleton = []
    for day in range(1, day_count + 1):
        if day == 1:
            theme = "arrival, easy neighborhood walk, and first meal"
        elif day == day_count:
            theme = "signature activity, flexible meal, and departure buffer"
        else:
            theme = "full day built around the highest-priority interests"
        skeleton.append({"day": day, "theme": theme})

    return {
        "destination": destination,
        "interests": interests,
        "days": skeleton,
    }


def ask_remote_city_expert(
    destination: str,
    interests_json: str = "[]",
    days: int = 3,
    budget: str = "medium",
    season: str = "default",
) -> dict[str, Any]:
    """Ask the remote city expert agent for local recommendations over plain HTTP."""
    remote_url = get_remote_city_expert_url()
    interests = _parse_interests(interests_json)
    payload = {
        "destination": destination,
        "interests": interests,
        "days": days,
        "budget": budget,
        "season": season,
        "user_id": "orchestrator",
        "session_id": f"{destination.lower().replace(' ', '-')}-{days}",
        "context": {
            "called_by": "travel_orchestrator",
            "transport": "plain-http",
            "protocol_note": "This example intentionally does not use A2A.",
        },
    }

    headers: dict[str, str] = {}
    propagate.inject(headers)

    tracer = trace.get_tracer("orchestrator")
    with tracer.start_as_current_span("ask_remote_city_expert") as span:
        span.set_attribute("http.url", remote_url)
        span.set_attribute("travel.destination", destination)
        span.set_attribute("travel.days", days)
        try:
            response = requests.post(remote_url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as exc:
            span.record_exception(exc)
            span.set_attribute("error.type", type(exc).__name__)
            error_body = getattr(exc.response, "text", "") if getattr(exc, "response", None) else ""
            return {
                "error": "remote_city_expert_unavailable",
                "remote_url": remote_url,
                "details": str(exc),
                "response_body": error_body[:1000],
                "recovery_hint": "Start the remote city expert service with: cd remote_city_expert && uv run python app.py",
            }


orchestrator_agent = LlmAgent(
    model=get_model(),
    name="travel_orchestrator",
    description="User-facing travel planning concierge that coordinates local expert recommendations.",
    instruction="""You are a friendly, concise travel planning concierge.

For travel planning requests:
1. Identify the destination, trip length, budget level, season if mentioned, and interests.
2. Use create_itinerary_skeleton to create the basic plan.
3. Use estimate_trip_budget to estimate costs.
4. Always use ask_remote_city_expert to get local recommendations over HTTP.
5. Compose a practical itinerary that names specific places, meals, and pacing.

If important details are missing, make reasonable assumptions and say what you assumed.
Do not mention implementation details unless the user asks.
Keep the final answer easy to scan.""",
    tools=[
        create_itinerary_skeleton,
        estimate_trip_budget,
        ask_remote_city_expert,
    ],
)


instrument_requests()
runtime = AgentRuntime(agent=orchestrator_agent, app_name="travel_orchestrator")
app = FastAPI(title="Travel Orchestrator Agent", version="0.1.0")
instrument_fastapi_app(app, service_name="travel-orchestrator")


class ChatRequest(BaseModel):
    message: str
    user_id: str = "demo_user"
    session_id: str = "demo_session"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    answer: str
    messages: list[str]
    agent: str = "travel_orchestrator"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "travel-orchestrator"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    set_span_attributes(
        {
            "chat.user_id": request.user_id,
            "chat.session_id": request.session_id,
            "agent.name": "travel_orchestrator",
        }
    )

    if request.metadata:
        message = f"{request.message}\n\nClient metadata: {json.dumps(request.metadata)}"
    else:
        message = request.message

    try:
        result = await runtime.run(
            message=message,
            user_id=request.user_id,
            session_id=request.session_id,
        )
    except Exception as exc:
        logger.exception("travel_orchestrator failed to handle chat request")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "travel_orchestrator_failed",
                "type": type(exc).__name__,
                "message": str(exc),
            },
        ) from exc
    return ChatResponse(answer=result.text, messages=result.messages)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        reload=False,
    )
