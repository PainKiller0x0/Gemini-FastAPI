# Gemini-FastAPI Rust 分支 / Rust Branch

> 中文优先说明。English version follows each major section.
>
> This README is Chinese-first. English notes are included after each major section.

## 项目定位

这个分支是 `PainKiller0x0/Gemini-FastAPI` 的 Rust 化部署线，主要给 nanobot / OBP 使用。它保留上游 Python/FastAPI 项目用于对齐能力，但线上优先使用 Rust sidecar：常驻内存更低，适合 Podman/systemd 守护，并提供 OpenAI 兼容接口。

核心目标：

- 作为 Gemini Web 到 OpenAI API 的轻量网关。
- 给 OBP / nanobot 提供 `/v1/chat/completions`、`/v1/responses`、`/v1/images/generations`。
- 尽量使用 Gemini Web 免费能力，同时把普通聊天、生图、识图/OCR 分开，避免互相污染。
- 仓库只提交源码和脱敏示例配置，真实 Cookie、API Key、worker token 只放运行时配置。

## Project Positioning

This branch is the Rust deployment track of `PainKiller0x0/Gemini-FastAPI`, mainly for nanobot / OBP. The upstream Python/FastAPI implementation is kept for compatibility, while production prefers the Rust sidecar for lower memory usage, Podman/systemd supervision, and OpenAI-compatible endpoints.

Goals:

- Act as a lightweight Gemini Web to OpenAI API gateway.
- Provide `/v1/chat/completions`, `/v1/responses`, and `/v1/images/generations` for OBP / nanobot.
- Use Gemini Web free capacity where possible while keeping normal chat, image generation, and vision/OCR paths isolated.
- Keep secrets out of Git. Real cookies, API keys, and worker tokens belong in runtime config only.

## 文档入口

- [架构设计](docs/ARCHITECTURE.md)：模块、seam、adapter、请求流程和后续拆分原则。
- [Nanobot 迭代记录](docs/NANOBOT_ITERATION.md)：我们相对上游做过的改动和回测清单。
- [示例配置](config/config.yaml)：脱敏配置模板。

## Documentation

- [Architecture](docs/ARCHITECTURE.md): modules, seams, adapters, request flow, and refactoring rules.
- [Nanobot iteration log](docs/NANOBOT_ITERATION.md): fork-specific changes and verification checklist.
- [Example config](config/config.yaml): sanitized config template.

## 已实现能力

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/responses`
- `POST /v1/images/generations`
- `GET /images/{filename}`，用于返回生成图片
- `server.api_key` Bearer Token 认证
- `CONFIG_PATH` 或 `config/config.yaml` 加载 YAML 配置
- Gemini Web Cookie：`secure_1psid`、`secure_1psidts`、可选 `secure_1psidcc`
- `cookie_header`：可直接粘贴完整 Gemini Web Cookie header，优先级高于拆分字段
- 多 Gemini client 轮询和失败切换
- 自定义 Gemini Web 模型 header
- Gemini Web 运行时模型发现
- 内置 `gemini-3.5-flash`、`gemini-3.1-pro`、`gemini-3.1-flash-lite` 别名
- OpenAI tool prompt 注入和 `tool_calls` 解析
- `response_format` 的 `json_object` / `json_schema` 指令支持
- OpenAI Chat / Responses 的文本、流式、工具调用兼容
- OpenAI image/file input 收集和 Gemini content-push 上传路径
- Chat/Responses 路由决策已抽到 `src/routing.rs`：集中判断走普通 Gemini、vision worker，还是 image tool
- 严格生图意图判断：只有最新用户消息明确要求生成/绘制图片，才进入图片工具
- 图片生成后端：`disabled`、`gemini_web`、`auto`、`gemini_worker`、`gemini_api`、`imagen_api`
- 更完整的 generated-image RPC (`c8o8Fe`)：会尝试多种 payload 形态，并从嵌套响应中解析全尺寸图片 URL；失败时仍回落到预览图下载
- 视觉识图/OCR worker：配置 `worker_url` + `worker_token_file` 后，图片附件可走 `/vision`
- JSONL 请求历史：`storage.path/rust-history.jsonl`
- Gemini session refresh 和可选真实 warmup，降低冷启动长尾延迟

## Implemented Features

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/responses`
- `POST /v1/images/generations`
- `GET /images/{filename}` for generated-image serving
- Bearer token authentication via `server.api_key`
- YAML config from `CONFIG_PATH` or `config/config.yaml`
- Gemini Web cookies: `secure_1psid`, `secure_1psidts`, optional `secure_1psidcc`
- `cookie_header` support for a full Gemini Web Cookie header, taking precedence over split fields
- Multiple Gemini clients with round-robin and failover
- Custom Gemini Web model headers
- Runtime Gemini Web model discovery
- Built-in aliases for `gemini-3.5-flash`, `gemini-3.1-pro`, and `gemini-3.1-flash-lite`
- OpenAI tool prompt injection and `tool_calls` parsing
- `response_format` support for `json_object` and `json_schema` instructions
- OpenAI Chat / Responses text, streaming, and tool-compatible responses
- OpenAI image/file input collection and Gemini content-push upload path
- Chat/Responses routing decisions live in `src/routing.rs`, centralizing the choice between normal Gemini, vision worker, and image tool
- Strict image intent detection: image tools are used only when the latest user message explicitly asks to generate/draw/create an image
- Image backends: `disabled`, `gemini_web`, `auto`, `gemini_worker`, `gemini_api`, `imagen_api`
- Fuller generated-image RPC (`c8o8Fe`) support: tries multiple payload shapes and recursively extracts full-size image URLs from nested responses; preview download remains the fallback
- Vision/OCR worker path via `worker_url` + `worker_token_file`, using `/vision` for image attachments
- JSONL request history at `storage.path/rust-history.jsonl`
- Gemini session refresh and optional real warmup to reduce cold-start tail latency

## 运行方式

本地运行：

```bash
CONFIG_PATH=config/config.yaml cargo run --release
```

构建容器：

```bash
podman build -f Dockerfile.rust -t gemini-fastapi-rs:local .
```

生产建议：

- 使用 Podman + systemd 守护。
- 服务优先绑定内网、WireGuard 或反向代理后的地址。
- 不建议把 Gemini-FastAPI 直接裸露到公网。
- 对外统一由 OBP 接入和做模型路由/账本统计。

## Run

Run locally:

```bash
CONFIG_PATH=config/config.yaml cargo run --release
```

Build container:

```bash
podman build -f Dockerfile.rust -t gemini-fastapi-rs:local .
```

Production recommendations:

- Use Podman + systemd supervision.
- Bind to private network, WireGuard, or a reverse-proxy-only address.
- Avoid exposing Gemini-FastAPI directly to the public internet.
- Let OBP handle external routing, model policy, and cost/accounting.

## 配置示例

基础配置在 `config/config.yaml`。真实部署时建议使用运行时配置文件，并通过 `CONFIG_PATH` 指向它。

```yaml
gemini:
  clients:
    - id: "client-a"
      secure_1psid: "YOUR_SECURE_1PSID_HERE"
      secure_1psidts: "YOUR_SECURE_1PSIDTS_HERE"
      secure_1psidcc: null
      cookie_header: null
      proxy: null
  chat_mode: "temporary"
  model_strategy: "append"
  warm_generate:
    enabled: false
    interval: 300
    initial_delay: 20
    model: "gemini-3.5-flash"
    prompt: "只回复一个字：好"
    active_periods: []

image_generation:
  backend: "disabled"
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

生产环境如果使用 worker，优先使用 `worker_token_file`，不要把 token 写进仓库：

```yaml
image_generation:
  backend: "gemini_worker"
  public_base_url: "https://example.com/gemini-images"
  worker_url: "http://127.0.0.1:8010"
  worker_token_file: "/run/secrets/gemini_worker_token"
  worker_timeout_ms: 300000
```

## Configuration Example

Base config lives in `config/config.yaml`. For production, use a runtime-only config file and point `CONFIG_PATH` to it.

Prefer `worker_token_file` over inline `worker_token` when using a worker, so the secret can live in a mounted file instead of Git.

## 图片生成

OpenAI 兼容调用示例：

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

返回逻辑：

- `public_base_url` 为空：返回 `b64_json`，桌面客户端可以直接渲染。
- `public_base_url` 不为空：图片保存到 `storage.images_path`，返回带 token 的 `/images/{filename}` URL。
- 普通聊天不会因为出现“画面”“图片”“生图报错”等词自动进入生图工具。

## Image Generation

OpenAI-compatible request example:

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

Response behavior:

- Empty `public_base_url`: returns `b64_json` for desktop clients.
- Non-empty `public_base_url`: stores files under `storage.images_path` and returns tokenized `/images/{filename}` URLs.
- Normal chat does not enter image generation just because it mentions pictures, UI, screenshots, or image-generation bugs.

## 识图 / OCR

当请求里包含 OpenAI image/file input，且配置了 `worker_url` 和 `worker_token_file`，Chat Completions / Responses 会优先走 vision worker：

- Chat Completions 记录为 `chat.completions.vision_worker`
- Responses 记录为 `responses.vision_worker`
- worker contract 当前是 `/vision`，请求包含 prompt、图片 base64、mime、filename

这个设计是为了隔离慢路径：图片理解失败不应该破坏普通文本聊天。

## Vision / OCR

When an OpenAI image/file input is present and `worker_url` + `worker_token_file` are configured, Chat Completions / Responses can use the vision worker first:

- Chat Completions are recorded as `chat.completions.vision_worker`
- Responses are recorded as `responses.vision_worker`
- The current worker contract is `/vision`, with prompt, image base64, mime, and filename

This isolates slow/high-risk vision paths from normal text chat.

## 注意事项

- 文件/图片上传依赖已认证的 Gemini Web session。Cookie 失效时，Gemini 可能返回 upstream error code `1100`。
- 如果浏览器里可以生图，但网关不行，通常需要复制完整 `cookie_header`；部分 Web 工具能力检查不只依赖三段基础 Cookie。
- 当前 SSE 是 OpenAI 兼容输出，但 Gemini Web 本身仍可能有后端抖动；这里做的是网关层兼容和尽量减少额外延迟。
- `gemini.rs` 是深 Module，不建议为了文件行数机械拆分；后续拆分应按 session、upload、frame parser 等真实 seam 做。

## Notes

- File/image upload requires an authenticated Gemini Web session. Expired cookies can surface upstream error code `1100`.
- If image generation works in browser but not through the gateway, copy the full `cookie_header`; some Web tool checks need more than the three minimal cookies.
- SSE is OpenAI-compatible, but Gemini Web itself may still have backend latency variance. The gateway focuses on compatibility and avoiding extra latency.
- `gemini.rs` is intentionally a deep module. Future splits should follow real seams such as session, upload, or frame parser, not line count.

## 仍在迁移 / 后续可做

- Python 兼容的会话复用和历史 metadata 语义。
- Google `RotateCookies` endpoint。
- Deep Research / Gems 等专用路径。
- 如果 `src/main.rs` 继续膨胀，下一步优先拆 SSE chunk 构造，而不是再拆浅工具函数。

## Still Being Ported / Future Work

- Python-compatible conversation reuse and history metadata semantics.
- Google `RotateCookies` endpoint support.
- Deep Research / Gems-specific paths.
- If `src/main.rs` keeps growing, the next useful extraction is SSE chunk construction, not more shallow helper functions.

## 回测清单

最近一次回测项：

- `cargo fmt -- --check`
- `cargo test --locked`，当前 21 个测试
- `cargo build --release --locked`
- 无认证 `/v1/models` 返回 `401`
- `/health` 返回 `implementation=rust`
- `/v1/models` 返回配置模型和运行时模型
- `gemini-3.5-flash` Chat Completion 可正常返回
- Chat streaming 返回 SSE chunks 和 `[DONE]`
- `/v1/responses` 非流式和流式可用
- OpenAI `tools` 请求可返回 `finish_reason=tool_calls`
- `/images/{filename}` 支持 token 校验并返回图片 bytes
- 图片/上传路径在 Cookie 异常时返回清晰错误，而不是污染普通聊天

## Verification Checklist

Latest verification scope:

- `cargo fmt -- --check`
- `cargo test --locked`, currently 21 tests
- `cargo build --release --locked`
- unauthenticated `/v1/models` returns `401`
- `/health` reports `implementation=rust`
- `/v1/models` returns configured and runtime models
- `gemini-3.5-flash` Chat Completion works
- Chat streaming returns SSE chunks and `[DONE]`
- `/v1/responses` works in both non-streaming and streaming modes
- OpenAI `tools` requests can return `finish_reason=tool_calls`
- `/images/{filename}` validates token and serves image bytes
- Image/upload paths surface clear errors when cookies are invalid, without polluting normal chat
