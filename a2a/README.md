# A2A Friend Scheduling Demo

A multi-agent application demonstrating how to orchestrate conversations between agents using the A2A protocol to schedule a pickleball game.

This application contains four agents:
- **Host Agent**: Orchestrates the scheduling task — talks to all friend agents and books a court.
- **Karley Agent** (port 10002): Manages Karley's calendar. Built with Google ADK.
- **Nate Agent** (port 10003): Manages Nate's calendar. Built with CrewAI.
- **Kaitlynn Agent** (port 10004): Manages Kaitlynn's calendar. Built with LangGraph.

## Prerequisites

1. **Python 3.12+**
2. **uv** — [installation guide](https://docs.astral.sh/uv/getting-started/installation/)
3. **A `.env` file** in the `a2a` directory (see below)

## Configuration

All agents use **OpenAI by default**. Copy the example file and edit it with your keys:

```bash
cp .env.example .env
```

**Get your Lumoz API key:**
1. Log in to [Lumoz Console](https://console.lumoz.ai)
2. Go to **Settings > API Keys**
3. Click **Create API Key**
4. Copy the key (format: `client_id:client_secret`)

Tracing is optional. The app works with only `OPENAI_API_KEY` set.

**To switch to Gemini** (all agents, no code changes needed), replace the OpenAI key with:

```
GOOGLE_API_KEY=your_google_key_here
GOOGLE_GENAI_USE_VERTEXAI=FALSE
LLM_MODEL=gemini/gemini-2.0-flash
```

## Run the Agents

Start each agent in a separate terminal. **Friend agents must be running before the Host Agent.**

### Terminal 1: Karley Agent
```bash
cd karley_agent_adk
uv run .
```

### Terminal 2: Nate Agent
```bash
cd nate_agent_crewai
uv run .
```

### Terminal 3: Kaitlynn Agent
```bash
cd kaitlynn_agent_langgraph
uv run app/__main__.py
```

### Terminal 4: Host Agent
```bash
cd host_agent_adk
uv run adk web
```

`uv run` will automatically create the virtual environment and install dependencies on first run.

## Interact with the Host Agent

Once all agents are running, open the ADK web UI at **http://localhost:8000**.

Type a message like _"Schedule a pickleball game with all friends this week"_ to start the scheduling flow. The host agent will contact each friend agent, find a common time, and book a court.

## View Traces in Lumoz

1. Run all agents and send a scheduling request via the host
2. Open [Lumoz Console](https://console.lumoz.ai)
3. Go to **Home** and find the agent app cards (`host-agent-adk`, `karley-agent-adk`, `nate-agent-crewai`, `kaitlynn-agent-langgraph`)
4. Navigate to **Telemetry → Traces**
5. Click any trace to explore agent-to-agent calls, tool use, and LLM interactions

## References
- [A2A Python SDK](https://github.com/google/a2a-python)
- [A2A Codelab — Purchasing Concierge](https://codelabs.developers.google.com/intro-a2a-purchasing-concierge#1)
