# Multi-Agent HTTP Travel Planner

A simple Google ADK example with two FastAPI services and a terminal chat client.

The scenario is a travel planning concierge:

- The user chats with the `travel_orchestrator` agent.
- The orchestrator uses its own LLM and local tools for itinerary structure and budget estimates.
- The orchestrator calls a remote `remote_city_expert` agent over plain HTTP/HTTPS.
- The remote agent uses its own LLM and tools for neighborhoods, activities, restaurants, and timing advice.
- OpenTelemetry trace context is propagated across the HTTP call so Lumoz can show one distributed trace.

This intentionally does not use A2A.

## Architecture

```text
Terminal client
    |
    | POST /chat
    v
Travel Orchestrator FastAPI app :8000
    |
    v
ADK travel_orchestrator agent
    |-- create_itinerary_skeleton tool
    |-- estimate_trip_budget tool
    `-- ask_remote_city_expert tool
            |
            | POST /recommend with W3C trace headers
            v
        Remote City Expert FastAPI app :8001
            |
            v
        ADK remote_city_expert agent
            |-- find_neighborhoods tool
            |-- find_activities tool
            |-- find_restaurants tool
            `-- get_local_timing_advice tool
```

## Shared Environment

```bash
cp .env.example .env
```

Edit `.env`:

```bash
OPENAI_API_KEY=your_openai_api_key
LITELLM_MODEL=openai/gpt-5.4-mini

LUMOZ_API_KEY=your_client_id:your_client_secret
OTEL_ENDPOINT=https://api.lumoz.ai/v1/traces
```

`OTEL_ENDPOINT` and `LUMOZ_API_KEY` are optional for local development, but required to export traces to Lumoz.

Each service has its own `uv` project and virtual environment. Both services load the same root `.env` file.

Remote city expert environment:

```bash
cd remote_city_expert
uv sync
cd ..
```

Orchestrator environment:

```bash
cd orchestrator
uv sync
cd ..
```

Client environment:

```bash
uv sync
```

## Run

Terminal 1:

```bash
cd remote_city_expert
uv run python app.py
```

Terminal 2:

```bash
cd orchestrator
uv run python app.py
```

Terminal 3:

```bash
uv run python client.py
```

Try:

```text
Plan a 3-day San Diego trip with beaches, tacos, and one family-friendly activity.
```

Then follow up:

```text
Make it cheaper and add more outdoor time.
```

## HTTP Endpoints

Orchestrator:

```bash
curl -X POST http://localhost:8000/chat \
  -H "content-type: application/json" \
  -d '{
    "user_id": "demo_user",
    "session_id": "demo_session",
    "message": "Plan a 3-day San Diego trip with beaches, tacos, and family activities"
  }'
```

Remote city expert:

```bash
curl -X POST http://localhost:8001/recommend \
  -H "content-type: application/json" \
  -d '{
    "destination": "San Diego",
    "interests": ["beaches", "tacos", "family-friendly"],
    "days": 3,
    "budget": "medium"
  }'
```

## Lumoz Tracing

Both services call `instrument_fastapi_app`, which installs FastAPI server tracing and OpenInference ADK/LLM tracing. The orchestrator also calls `instrument_requests`, so the `requests.post` call in `ask_remote_city_expert` automatically creates an HTTP client span and injects W3C trace headers into the outbound request.

Expected trace shape:

```text
POST /chat
`-- travel_orchestrator ADK run
    |-- create_itinerary_skeleton tool
    |-- estimate_trip_budget tool
    `-- ask_remote_city_expert tool
        `-- HTTP POST /recommend
            `-- remote_city_expert ADK run
                |-- find_neighborhoods tool
                |-- find_activities tool
                |-- find_restaurants tool
                `-- get_local_timing_advice tool
```

The local recommendation data is intentionally small so the example is easy to understand. Add more cities by extending `CITY_GUIDE` in `remote_city_expert/app.py`.
