"""Terminal chat client for the travel orchestrator FastAPI app."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path

import requests
from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with the travel orchestrator agent.")
    parser.add_argument(
        "--url",
        default=os.environ.get("ORCHESTRATOR_URL", "http://localhost:8000/chat"),
        help="Orchestrator /chat endpoint.",
    )
    parser.add_argument("--user-id", default="demo_user")
    parser.add_argument(
        "--session-id",
        default=f"demo_session_{uuid.uuid4().hex[:8]}",
        help="Session id for multi-turn chat.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Chatting with {args.url}")
    print(f"Session: {args.session_id}")
    print("Type 'quit' or press Ctrl-D to exit.\n")

    while True:
        try:
            message = input("You: ").strip()
        except EOFError:
            print()
            return

        if not message:
            continue
        if message.lower() in {"quit", "exit"}:
            return

        response = requests.post(
            args.url,
            json={
                "user_id": args.user_id,
                "session_id": args.session_id,
                "message": message,
            },
            timeout=120,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            print(f"\nRequest failed: {exc}")
            try:
                print(json.dumps(response.json(), indent=2))
            except ValueError:
                print(response.text)
            print()
            continue
        body = response.json()
        print(f"\nAssistant: {body.get('answer', '')}\n")


if __name__ == "__main__":
    main()
