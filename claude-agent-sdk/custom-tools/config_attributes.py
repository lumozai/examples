"""Turn customer-controlled ClaudeAgentOptions into OpenInference attributes.

The instrumentor never reads ClaudeAgentOptions, so the system prompt, the
sub-agent definitions and the custom tool schemas never reach a span. They are
all in this process though, so we can emit them ourselves using the standard
OpenInference keys.

Covers only what the customer controls. Built-in tool schemas and Claude Code's
own framing live inside the CLI and are not available here.
"""

from __future__ import annotations

import json
from typing import Any

from openinference.semconv.trace import (
    MessageAttributes,
    SpanAttributes,
    ToolAttributes,
)

# The SDK's own converter, so the schema we report is the schema it sends.
from claude_agent_sdk import _build_input_schema


def _system_prompt_text(system_prompt: Any) -> str | None:
    """system_prompt is a str, or a preset dict whose `append` is the custom part."""
    if system_prompt is None:
        return None
    if isinstance(system_prompt, str):
        return system_prompt
    if isinstance(system_prompt, dict):
        # Only `append` is customer-authored; the preset body lives in the CLI.
        return system_prompt.get("append")
    return None


def config_attributes(options: Any, prompt: str, tools: list | None = None) -> dict[str, Any]:
    """OpenInference attributes for the customer-controlled half of the request.

    `tools` is the list of @tool functions passed to create_sdk_mcp_server. The
    server keeps them in a closure, so they cannot be recovered from `options`.
    """
    attrs: dict[str, Any] = {}

    index = 0
    system_text = _system_prompt_text(getattr(options, "system_prompt", None))
    if system_text:
        attrs[f"{SpanAttributes.LLM_INPUT_MESSAGES}.{index}.{MessageAttributes.MESSAGE_ROLE}"] = "system"
        attrs[f"{SpanAttributes.LLM_INPUT_MESSAGES}.{index}.{MessageAttributes.MESSAGE_CONTENT}"] = system_text
        index += 1

    attrs[f"{SpanAttributes.LLM_INPUT_MESSAGES}.{index}.{MessageAttributes.MESSAGE_ROLE}"] = "user"
    attrs[f"{SpanAttributes.LLM_INPUT_MESSAGES}.{index}.{MessageAttributes.MESSAGE_CONTENT}"] = prompt

    # Custom tools defined in the customer's own code, via SDK MCP servers.
    for i, t in enumerate(tools or []):
        schema = {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": _build_input_schema(t),
            },
        }
        attrs[f"{SpanAttributes.LLM_TOOLS}.{i}.{ToolAttributes.TOOL_JSON_SCHEMA}"] = json.dumps(schema)

    # Sub-agent definitions: each carries a customer-authored system prompt.
    agents = getattr(options, "agents", None) or {}
    if agents:
        attrs["lumoz.agent_definitions"] = json.dumps(
            {
                name: {
                    "description": getattr(a, "description", None),
                    "prompt": getattr(a, "prompt", None),
                    "tools": getattr(a, "tools", None),
                    "model": getattr(a, "model", None),
                }
                for name, a in agents.items()
            }
        )

    # Built-in tools the customer granted. Names only - their schemas are in the CLI.
    allowed = getattr(options, "allowed_tools", None)
    if allowed:
        attrs["lumoz.allowed_tools"] = json.dumps(list(allowed))

    return attrs
