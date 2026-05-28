# Gemini-FastAPI Rust branch

## Nanobot production notes

这是 `PainKiller0x0/Gemini-FastAPI` 的 Rust 化分支说明。当前目标不是替代上游项目的全部 Python 能力，而是给 nanobot/OBP 提供一个低内存、可守护、OpenAI 兼容的 Gemini Web sidecar。

生产原则：

- 仓库只放源码和脱敏示例配置，真实 Cookie、API Key、worker token 只放运行时配置。
- 普通聊天不应该被 Gemini Web 的图片工具劫持；只有最新一轮用户消息明确要求“生成/绘制图片”才走图片链路。
- 图片生成和视觉附件可以走单独 worker，避免把不稳定的 Web 工具路径污染普通聊天。
- 运行建议使用 Podman/systemd，并让 OBP 从 `/v1/chat/completions` 接入。

This branch is the Rust rewrite track for Gemini-FastAPI. The goal is to remove the always-on Python/FastAPI runtime and keep the OpenAI-compatible Gemini Web gateway small enough for sidecar-style deployment.

## Implemented

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/responses`
- `POST /v1/images/generations` with OpenAI-compatible request/response shape
- `GET /images/{filename}` for generated-image proxy files
- Bearer token authentication using `server.api_key`
- YAML config loading from `CONFIG_PATH` or `config/config.yaml`
- Gemini Web cookies: `secure_1psid`, `secure_1psidts`, optional `secure_1psidcc`
- Optional `cookie_header` for the full Cookie header copied from a known-working Gemini Web session. When set, it takes precedence over the individual cookie fields and is the preferred path for Web tools such as image generation.
- Multiple Gemini clients with round-robin routing and failover
- Configured custom Gemini model headers
- Runtime Gemini model discovery via Gemini Web `otAQ7b` RPC
- Built-in aliases for `gemini-3.5-flash`, `gemini-3.1-pro`, and `gemini-3.1-flash-lite`
- OpenAI tool prompt injection and `tool_calls` response parsing
- `response_format` support for `json_object` and `json_schema` instructions
- OpenAI/Responses text, streaming, and tool-compatible responses
- OpenAI image/file input collection and Gemini content-push upload path
- Generated/web image URL parsing, local download, and tokenized image serving
- Optional image generation backends:
  - `image_generation.backend = "gemini_web"` asks Gemini Web to generate images through the cookie session first
  - `image_generation.backend = "auto"` tries Gemini Web first, then falls back to the configured backend
  - `image_generation.backend = "gemini_worker"` delegates generation to a separate Gemini Web worker service
  - `image_generation.backend = "gemini_api"` uses Gemini native image models, for example `gemini-3.1-flash-image-preview`
  - `image_generation.backend = "imagen_api"` uses Imagen models, for example `imagen-4.0-generate-001`
  - API key is read from `image_generation.api_key` or the configured env var, default `GEMINI_API_KEY`
- Strict image-tool intent detection: normal chat mentioning UI, pictures, ADHD, or image bugs will stay text-only unless the latest user message explicitly asks to generate/draw/create an image
- Optional worker bridge for image generation and vision attachments via `worker_url` + `worker_token_file`
- Lightweight JSONL request history at `storage.path/rust-history.jsonl`
- Session/token refresh based on `gemini.refresh_interval`
- Optional real-generation warmup via `gemini.warm_generate` to reduce cold Gemini Web tail latency; set `active_periods` such as `["07:00-01:30"]` to avoid warming while asleep

## Notes

- File/image upload requires an authenticated Gemini Web session. With unauthenticated or expired cookies, Gemini currently returns upstream error code `1100`; the Rust service surfaces this clearly as `Gemini API error code: 1100`.
- Image generation can use Gemini Web cookies via `backend = "gemini_web"`, a separate worker via `backend = "gemini_worker"`, or official API backends. If the cookie session is unauthenticated or the account/location lacks Web image generation, the endpoint surfaces that clearly instead of silently polluting normal chat.
- If Gemini Web can generate images in your browser but this gateway cannot, copy the full browser `Cookie` header into `gemini.clients[].cookie_header`; some Web tool capability checks depend on more than the three minimal cookies.
- Streaming is OpenAI-compatible SSE, but it still buffers the Gemini Web response before emitting deltas. True token-by-token Gemini streaming is the next deep porting item.

## Image generation

Enable a backend in `config/config.yaml` or your runtime config:

```yaml
image_generation:
  backend: "gemini_worker" # or "disabled", "gemini_web", "auto", "gemini_api", "imagen_api"
  model: "gemini-3.1-flash-image-preview"
  web_model: "gemini-3.5-flash"
  api_key: null
  api_key_env: "GEMINI_API_KEY"
  public_base_url: null
  worker_url: null
  worker_token: null
  worker_token_file: null
  worker_timeout_ms: 180000
```

For production, prefer `worker_token_file` over `worker_token` so the token can live in a secret file or runtime-only mount.

```yaml
image_generation:
  backend: "gemini_worker"
  public_base_url: "https://example.com/gemini-images"
  worker_url: "http://127.0.0.1:8010"
  worker_token_file: "/run/secrets/gemini_worker_token"
  worker_timeout_ms: 300000
```

Call it with an OpenAI-compatible images request:

```bash
curl http://127.0.0.1:8000/v1/images/generations \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-3.1-flash-image-preview",
    "prompt": "A tiny dragon sleeping in a teacup, watercolor",
    "n": 1,
    "size": "1024x1024"
  }'
```

When `public_base_url` is empty, responses return `b64_json` so desktop clients can render without needing a public image URL. If `public_base_url` is set, generated files are stored under `storage.images_path` and returned as tokenized `/images/{filename}` URLs.

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
- `/v1/images/generations` compiles and can route to Gemini Web; with an unauthenticated Web cookie it returns Gemini's explicit signed-in/location refusal
- upload path reaches Gemini and reports upstream `1100` clearly when cookies are unauthenticated
- Podman memory on Seoul VPS is around 4-6 MB for the running Rust gateway
