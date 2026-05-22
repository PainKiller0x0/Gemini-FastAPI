# Gemini-FastAPI Rust branch

This branch contains a Rust implementation of the core OpenAI-compatible Gemini Web API surface.

## Current scope

Implemented:

- GET /health
- GET /v1/models
- POST /v1/chat/completions
- Bearer token authentication using server.api_key
- YAML config loading from CONFIG_PATH or config/config.yaml
- Gemini Web cookies: secure_1psid, secure_1psidts, optional secure_1psidcc
- Custom Gemini model headers, including gemini-3.5-flash and gemini-3.1-pro
- Non-streaming and SSE streaming OpenAI-style responses

Not yet ported from the Python server:

- /v1/responses
- image upload/image response handling
- LMDB conversation reuse/history
- tool-call protocol conversion
- cookie rotation
- dynamic runtime model discovery from Gemini user status

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
- /v1/models returns configured models
- gemini-3.5-flash chat completion returns ok
- gemini-3.1-pro chat completion returns ok
- streaming chat completion returns SSE chunks and [DONE]
