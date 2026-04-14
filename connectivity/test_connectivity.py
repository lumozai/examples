#!/usr/bin/env python3
"""
Lumoz Connectivity Test

Verifies your environment can connect to Lumoz and send traces.
No external dependencies required.

Usage:
    python test_connectivity.py "your-client-id:your-client-secret"
"""

import base64
import json
import socket
import sys
import time
import uuid
from datetime import datetime, UTC
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

ENDPOINT = "https://api.lumoz.ai/proxy/v1/traces"


def main():
    print("=" * 50)
    print("Lumoz Connectivity Test")
    print("=" * 50)

    if len(sys.argv) != 2 or ":" not in sys.argv[1]:
        print("\nUsage: python test_connectivity.py 'client_id:client_secret'")
        sys.exit(1)

    api_key = sys.argv[1]
    print(f"\nAPI Key: {api_key.split(':')[0][:8]}...")
    print(f"Endpoint: {ENDPOINT}")
    print("\nSending test trace...")

    trace_id = uuid.uuid4().hex
    span_id = uuid.uuid4().hex[:16]
    now_ns = int(time.time() * 1_000_000_000)
    service_name = "lumoz-connectivity-test"

    payload = {
        "resourceSpans": [{
            "resource": {
                "attributes": [
                    {"key": "service.name", "value": {"stringValue": service_name}},
                    {"key": "deployment.environment", "value": {"stringValue": "development"}}
                ]
            },
            "scopeSpans": [{
                "scope": {"name": "lumoz.test"},
                "spans": [{
                    "traceId": trace_id,
                    "spanId": span_id,
                    "name": "connectivity_test",
                    "kind": 1,
                    "startTimeUnixNano": str(now_ns),
                    "endTimeUnixNano": str(now_ns + 100_000_000),
                    "status": {"code": 1},
                    "attributes": [
                        {"key": "test.timestamp", "value": {"stringValue": datetime.now(UTC).isoformat()}}
                    ]
                }]
            }]
        }]
    }

    try:
        data = json.dumps(payload).encode()
        req = Request(ENDPOINT, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Basic {base64.b64encode(api_key.encode()).decode()}")

        response = urlopen(req, timeout=15)
        print(f"Status: {response.status} - OK")
        print(f"Trace ID: {trace_id}")
        print(f"\nSuccess! Check Lumoz console for service '{service_name}'")
        print("=" * 50)

    except HTTPError as e:
        if e.code == 401:
            print("FAILED: Authentication error - check your API key")
        else:
            print(f"FAILED: HTTP {e.code}")
        sys.exit(1)
    except socket.timeout:
        print("FAILED: Connection timed out (check firewall)")
        sys.exit(1)
    except URLError as e:
        print(f"FAILED: Connection error (check internet/DNS)")
        sys.exit(1)


if __name__ == "__main__":
    main()
