# Gemini-FastAPI Rust branch

This branch is the Rust rewrite track for Gemini-FastAPI. The goal is to remove the always-on Python/FastAPI runtime and keep the OpenAI-compatible Gemini Web gateway small enough for sidecar-style deployment.

## Implemented

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/responses`
- `GET /images/{filename}` for generated-image proxy files
- Bearer token authentication using `server.api_key`
- YAML config loading from `CONFIG_PATH` or `config/config.yaml`
- Gemini Web cookies: `secure_1psid`, `secure_1psidts`, optional `secure_1psidcc`
- Multiple Gemini clients with round-robin routing and failover
- Configured custom Gemini model headers
- Runtime Gemini model discovery via Gemini Web `otAQ7b` RPC
- Built-in aliases for `gemini-3.5-flash`, `gemini-3.1-pro`, and `gemini-3.1-flash-lite`
- OpenAI tool prompt injection and `tool_calls` response parsing
- `response_format` support for `json_object` and `json_schema` instructions
- OpenAI/Responses text, streaming, and tool-compatible responses
- OpenAI image/file input collection and Gemini content-push upload path
- Generated/web image URL parsing, local download, and tokenized image serving
- Lightweight JSONL request history at `storage.path/rust-history.jsonl`
- Session/token refresh based on `gemini.refresh_interval`

## Notes

- File/image upload requires an authenticated Gemini Web session. With unauthenticated or expired cookies, Gemini currently returns upstream error code `1100`; the Rust service surfaces this clearly as `Gemini API error code: 1100`.
- Streaming is OpenAI-compatible SSE, but it still buffers the Gemini Web response before emitting deltas. True token-by-token Gemini streaming is the next deep porting item.

## Still being ported

- Python-compatible conversation reuse/history metadata semantics
- Full-size generated-image RPC (`c8o8Fe`) before falling back to preview URL download
- Google `RotateCookies` endpoint support
- Deep Research/Gems-specific paths

## Run

```bash
CONFIG_PATH=config/config.yaml cargo run --release
```

## Build container

```bash
podman build -f Dockerfile.rust -t gemini-fastapi-rs:local .
```

## Verification performed

The Rust binary was built on the default server and deployed on the Seoul VPS through Podman/systemd using the existing runtime config. Verified:

- no-auth `/v1/models` returns `401`
- `/health` reports `implementation=rust`
- `/v1/models` returns configured + runtime models
- `gemini-3.5-flash` chat completion returns `rust-ok`
- chat streaming returns SSE chunks and `[DONE]`
- `/v1/responses` non-streaming returns output text
- `/v1/responses` streaming returns response events and `[DONE]`
- OpenAI `tools` request returns `finish_reason=tool_calls`
- `/images/{filename}` serves stored image bytes with token validation
- upload path reaches Gemini and reports upstream `1100` clearly when cookies are unauthenticated
- Podman memory on Seoul VPS is around 4-6 MB for the running Rust gateway
