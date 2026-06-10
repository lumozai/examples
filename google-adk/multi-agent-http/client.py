"""Terminal chat client for the travel orchestrator FastAPI app."""

from __future__ import annotations

import argparse
import json
import os
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session_id: str | None = None

    print(f"Chatting with {args.url}")
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

        payload: dict = {"user_id": args.user_id, "message": message}
        if session_id:
            payload["session_id"] = session_id

        response = requests.post(args.url, json=payload, timeout=120)
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
        session_id = body.get("session_id", session_id)
        if not payload.get("session_id"):
            print(f"Session: {session_id}")
        print(f"\nAssistant: {body.get('answer', '')}\n")


if __name__ == "__main__":
    main()
