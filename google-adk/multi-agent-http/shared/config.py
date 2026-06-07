"""Shared configuration for the FastAPI services."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from google.adk.models.lite_llm import LiteLlm


ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")


DEFAULT_MODEL = "openai/gpt-5.4-mini"


def get_model() -> LiteLlm:
    """Build the ADK LiteLLM model from environment configuration."""
    model_name = os.environ.get("LITELLM_MODEL", DEFAULT_MODEL)
    return LiteLlm(model=model_name)


def get_remote_city_expert_url() -> str:
    return os.environ.get("REMOTE_CITY_EXPERT_URL", "http://localhost:8001/recommend")
