# Gemini-FastAPI Rust branch

This branch is the Rust rewrite track for Gemini-FastAPI. The goal is to remove the always-on Python/FastAPI runtime and keep the OpenAI-compatible Gemini Web gateway small enough for sidecar-style deployment.

## Implemented

- GET /health
- GET /v1/models
- POST /v1/chat/completions
- POST /v1/responses
- Bearer token authentication using server.api_key
- YAML config loading from CONFIG_PATH or config/config.yaml
- Gemini Web cookies: secure_1psid, secure_1psidts, optional secure_1psidcc
- Configured custom Gemini model headers
- Runtime Gemini model discovery via Gemini Web otAQ7b RPC
- Built-in aliases for gemini-3.5-flash, gemini-3.1-pro, and gemini-3.1-flash-lite
- Non-streaming and SSE streaming OpenAI-style responses
- Basic 	ools, 	ool_choice, and esponse_format prompt injection
- Lightweight JSONL request history at storage.path/rust-history.jsonl
- Session/token refresh based on gemini.refresh_interval

## Still being ported

These are intentionally not faked as complete yet:

- Gemini file/image upload
- Generated image download/proxy endpoints
- Full OpenAI tool-call response parsing
- Python-compatible conversation reuse/history semantics
- Google cookie rotation endpoint support
- Deep research/Gems-specific paths

## Run

`ash
CONFIG_PATH=config/config.yaml cargo run --release
`

## Build container

`ash
podman build -f Dockerfile.rust -t gemini-fastapi-rs:local .
`

## Verification performed

The Rust binary was built on the default server, copied to the Seoul VPS, and run on a local-only side port with the existing runtime config. Verified:

- no-auth /v1/models returns 401
- /v1/models returns configured + runtime models
- gemini-3.5-flash chat completion returns ok
- gemini-3.1-pro chat completion returns ok
- chat streaming returns SSE chunks and [DONE]
- /v1/responses non-streaming returns output text
- /v1/responses streaming returns response events and [DONE]
- JSONL history is written under the configured storage path
