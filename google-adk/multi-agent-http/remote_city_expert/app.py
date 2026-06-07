"""Remote FastAPI service that hosts the ADK city expert agent."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from google.adk.agents import LlmAgent

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from shared.adk_runtime import AgentRuntime
from shared.config import get_model
from shared.tracing import instrument_fastapi_app, set_span_attributes


logger = logging.getLogger(__name__)


CITY_GUIDE: dict[str, dict[str, Any]] = {
    "san diego": {
        "neighborhoods": {
            "La Jolla": ["beaches", "outdoors", "family-friendly", "views"],
            "Little Italy": ["food", "walkable", "nightlife"],
            "Balboa Park": ["museums", "family-friendly", "culture", "outdoors"],
            "Coronado": ["beaches", "family-friendly", "relaxed"],
            "North Park": ["food", "breweries", "nightlife"],
        },
        "activities": [
            {
                "name": "La Jolla Cove",
                "tags": ["beaches", "outdoors", "views", "family-friendly"],
                "duration_hours": 3,
                "price": "free",
            },
            {
                "name": "San Diego Zoo",
                "tags": ["family-friendly", "wildlife", "outdoors"],
                "duration_hours": 5,
                "price": "premium",
            },
            {
                "name": "Balboa Park museums and gardens",
                "tags": ["culture", "museums", "outdoors", "family-friendly"],
                "duration_hours": 4,
                "price": "medium",
            },
            {
                "name": "Coronado Beach",
                "tags": ["beaches", "relaxed", "family-friendly"],
                "duration_hours": 4,
                "price": "free",
            },
            {
                "name": "Old Town San Diego",
                "tags": ["history", "food", "family-friendly"],
                "duration_hours": 3,
                "price": "low",
            },
            {
                "name": "Torrey Pines State Natural Reserve",
                "tags": ["hiking", "outdoors", "views", "beaches"],
                "duration_hours": 3,
                "price": "low",
            },
        ],
        "restaurants": [
            {
                "name": "Oscar's Mexican Seafood",
                "neighborhood": "Pacific Beach",
                "tags": ["tacos", "seafood", "casual", "low"],
            },
            {
                "name": "The Taco Stand",
                "neighborhood": "La Jolla",
                "tags": ["tacos", "casual", "low"],
            },
            {
                "name": "Civico 1845",
                "neighborhood": "Little Italy",
                "tags": ["italian", "dinner", "medium"],
            },
            {
                "name": "Liberty Public Market",
                "neighborhood": "Point Loma",
                "tags": ["food-hall", "family-friendly", "medium"],
            },
            {
                "name": "Hodad's",
                "neighborhood": "Ocean Beach",
                "tags": ["burgers", "casual", "low"],
            },
        ],
        "seasonality": {
            "default": "Mornings are usually best for beaches and coastal viewpoints. Keep museum or food-hall time as a flexible backup.",
            "summer": "Expect busier beaches and book popular attractions ahead.",
            "winter": "Beach walks are still pleasant, but plan fewer swim-focused blocks.",
        },
    },
    "new york": {
        "neighborhoods": {
            "Central Park": ["outdoors", "family-friendly", "classic"],
            "Lower East Side": ["food", "nightlife", "culture"],
            "Chelsea": ["art", "walkable", "food"],
            "Brooklyn Heights": ["views", "walkable", "family-friendly"],
        },
        "activities": [
            {
                "name": "Central Park loop",
                "tags": ["outdoors", "family-friendly", "classic"],
                "duration_hours": 3,
                "price": "free",
            },
            {
                "name": "American Museum of Natural History",
                "tags": ["museums", "family-friendly", "culture"],
                "duration_hours": 4,
                "price": "medium",
            },
            {
                "name": "High Line and Chelsea Market",
                "tags": ["food", "walkable", "views"],
                "duration_hours": 3,
                "price": "low",
            },
        ],
        "restaurants": [
            {
                "name": "Los Tacos No. 1",
                "neighborhood": "Chelsea",
                "tags": ["tacos", "casual", "low"],
            },
            {
                "name": "Xi'an Famous Foods",
                "neighborhood": "Multiple",
                "tags": ["noodles", "casual", "low"],
            },
        ],
        "seasonality": {
            "default": "Group outdoor walks together and keep one indoor museum or market option each day.",
        },
    },
}


def _normalize_destination(destination: str) -> str:
    destination_key = destination.strip().lower()
    if destination_key in CITY_GUIDE:
        return destination_key
    for known_city in CITY_GUIDE:
        if known_city in destination_key:
            return known_city
    return "san diego"


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


def find_neighborhoods(destination: str, interests_json: str) -> dict[str, Any]:
    """Find neighborhoods that match the traveler's interests."""
    city_key = _normalize_destination(destination)
    interests = _parse_interests(interests_json)
    neighborhoods = CITY_GUIDE[city_key]["neighborhoods"]

    ranked = []
    for name, tags in neighborhoods.items():
        score = len(set(interests).intersection(tags))
        ranked.append({"name": name, "matched_tags": tags, "score": score})

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return {
        "destination": city_key.title(),
        "interests": interests,
        "neighborhoods": ranked[:4],
    }


def find_activities(destination: str, interests_json: str, days: int = 3) -> dict[str, Any]:
    """Find local activities that fit the trip interests and length."""
    city_key = _normalize_destination(destination)
    interests = _parse_interests(interests_json)
    activity_limit = max(3, min(days * 2, 6))

    ranked = []
    for activity in CITY_GUIDE[city_key]["activities"]:
        score = len(set(interests).intersection(activity["tags"]))
        ranked.append({**activity, "score": score})

    ranked.sort(key=lambda item: (item["score"], item["duration_hours"]), reverse=True)
    return {
        "destination": city_key.title(),
        "activities": ranked[:activity_limit],
    }


def find_restaurants(destination: str, interests_json: str, budget: str = "medium") -> dict[str, Any]:
    """Find restaurants that fit the destination, food interests, and budget."""
    city_key = _normalize_destination(destination)
    interests = _parse_interests(interests_json)
    budget = budget.lower()

    ranked = []
    for restaurant in CITY_GUIDE[city_key]["restaurants"]:
        score = len(set(interests).intersection(restaurant["tags"]))
        if budget in restaurant["tags"]:
            score += 1
        ranked.append({**restaurant, "score": score})

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return {
        "destination": city_key.title(),
        "restaurants": ranked[:4],
    }


def get_local_timing_advice(destination: str, season: str = "default") -> dict[str, str]:
    """Get local timing and seasonality advice for the destination."""
    city_key = _normalize_destination(destination)
    seasonality = CITY_GUIDE[city_key]["seasonality"]
    advice = seasonality.get(season.lower(), seasonality["default"])
    return {
        "destination": city_key.title(),
        "season": season,
        "advice": advice,
    }


city_expert_agent = LlmAgent(
    model=get_model(),
    name="remote_city_expert",
    description="Local city expert that recommends neighborhoods, activities, restaurants, and timing tips.",
    instruction="""You are a practical local city expert.

Use your tools before answering:
1. find_neighborhoods for areas that match the traveler.
2. find_activities for things to do.
3. find_restaurants for food recommendations.
4. get_local_timing_advice for pacing and seasonality.

Return concise JSON with these keys:
- destination
- neighborhoods
- activities
- restaurants
- local_timing_advice
- notes

Do not mention that the data is synthetic. If the destination is not in your guide,
offer the closest available sample city and say what assumption you used.""",
    tools=[
        find_neighborhoods,
        find_activities,
        find_restaurants,
        get_local_timing_advice,
    ],
)


runtime = AgentRuntime(agent=city_expert_agent, app_name="remote_city_expert")
app = FastAPI(title="Remote City Expert Agent", version="0.1.0")
instrument_fastapi_app(app, service_name="remote-city-expert")


class RecommendationRequest(BaseModel):
    destination: str
    interests: list[str] = Field(default_factory=list)
    days: int = 3
    budget: str = "medium"
    season: str = "default"
    user_id: str = "demo_user"
    session_id: str = "demo_session"
    context: dict[str, Any] = Field(default_factory=dict)


class RecommendationResponse(BaseModel):
    answer: str
    messages: list[str]
    agent: str = "remote_city_expert"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "remote-city-expert"}


@app.post("/recommend", response_model=RecommendationResponse)
async def recommend(request: RecommendationRequest) -> RecommendationResponse:
    set_span_attributes(
        {
            "travel.destination": request.destination,
            "travel.days": request.days,
            "travel.budget": request.budget,
            "agent.name": "remote_city_expert",
        }
    )

    prompt = f"""
Recommend local options for this trip.

Destination: {request.destination}
Trip length: {request.days} days
Budget: {request.budget}
Season: {request.season}
Interests: {json.dumps(request.interests)}
Context from orchestrator: {json.dumps(request.context)}
"""
    try:
        result = await runtime.run(
            message=prompt,
            user_id=request.user_id,
            session_id=f"remote-{request.session_id}",
        )
    except Exception as exc:
        logger.exception("remote_city_expert failed to handle recommendation request")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "remote_city_expert_failed",
                "type": type(exc).__name__,
                "message": str(exc),
            },
        ) from exc
    return RecommendationResponse(answer=result.text, messages=result.messages)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8001")),
        reload=False,
    )
