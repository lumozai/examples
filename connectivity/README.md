# Lumoz Connectivity Test

Verify your environment can connect to Lumoz and send traces.

## Requirements

- Python 3.8+
- Your Lumoz API key

## Usage

Run the test_connectivity with your Lumoz API Key that you got from the console. 

```bash
python test_connectivity.py "your-client-id:your-client-secret"
```

## Expected Output

```
==================================================
Lumoz Connectivity Test
==================================================

API Key: m2m-clie...
Endpoint: https://api.lumoz.ai/proxy/v1/traces

Sending test trace...
Status: 200 - OK
Trace ID: abc123...

Success! Check Lumoz console for service 'lumoz-connectivity-test'
==================================================
```

## View Your Trace

After a successful run, find your trace in the Lumoz Console:

1. Go to **Home**
2. Click the **lumoz-connectivity-test** app card
3. Navigate to the **Telemetry → Traces** tab
4. Click the trace to explore span details

## Troubleshooting

**Connection timed out**: Check firewall allows outbound HTTPS (port 443)

**Authentication error (401)**: Verify API key format is `client_id:client_secret` and credentials are valid

**Connection error**: Check internet connectivity and DNS resolution
