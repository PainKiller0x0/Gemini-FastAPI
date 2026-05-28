# Nanobot Gemini-FastAPI 迭代记录

这份文档记录 `PainKiller0x0/Gemini-FastAPI` fork 相对上游的主要改动。它的定位是 nanobot/OBP 的 Gemini Web sidecar，不是把上游 Python 项目改成另一个大而全平台。架构边界和后续拆分原则见 [docs/ARCHITECTURE.md](ARCHITECTURE.md)。

## 目标

- 给 nanobot 提供 OpenAI 兼容的 `/v1/chat/completions` 和 `/v1/responses`。
- 用 Rust 常驻服务降低内存占用，适合小 VPS 上用 Podman/systemd 守护。
- 尽量使用 Gemini Web 的免费额度，同时保持 OBP 可以观测、统计和路由。
- 普通聊天必须稳定，不能因为 Gemini Web 的图片工具误判而打断对话。

## 已落地

| 模块 | 改动 | 价值 |
| --- | --- | --- |
| Rust sidecar | 新增 `gemini-fastapi-rs`，实现 OpenAI 兼容聊天、Responses、模型列表、健康检查 | 去掉 Python 常驻运行时，内存更低 |
| Gemini Web 模型 | 支持自定义 `x-goog-ext-525001261-jspb` 模型头，内置 `gemini-3.5-flash`、`gemini-3.1-pro`、`gemini-3.1-flash-lite` 别名 | 让 OBP 可以按稳定模型名调用 |
| 会话形态 | 支持 normal / temporary chat 模式 | 减少 Gemini Web 账号聊天列表污染 |
| 会话维护 | 支持定时 refresh 和真实轻量 warmup | 降低冷启动和会话过期带来的长尾延迟 |
| 流式输出 | OpenAI SSE 兼容输出，Gemini 响应后按 chunk 发出 | 保持客户端兼容，QQ/OBP 可直接接 |
| 多模态输入 | 支持 OpenAI image/file input 收集，走 Gemini content-push 或 worker 视觉路径 | 让图片理解链路可用 |
| 图片生成 | 支持 `disabled`、`gemini_web`、`auto`、`gemini_worker`、`gemini_api`、`imagen_api` | 允许按成本/稳定性选择不同后端 |
| 路由决策 | `src/routing.rs` 统一选择普通 Gemini、vision worker 或 image tool | Chat/Responses 不再各自散落判断逻辑 |
| 图片工具防误触 | 只根据“最新用户消息”的明确生图意图触发图片链路；“画面很怪”“画 UI 很烦”“生图报错”不会触发 | 解决闲聊被回复“您登录了吗/地区未开放”的问题 |
| 生成图全尺寸 | `c8o8Fe` 支持多 payload 形态和嵌套 URL 解析，失败再回落预览图 | 提高 Gemini Web 生图落盘质量 |
| Worker 隔离 | `worker_url` + `worker_token_file` 可把生图/视觉任务交给单独 worker | 把高风险慢路径和普通聊天隔离 |
| 安全配置 | 示例配置只放占位符；真实 Cookie、API Key、worker token 放运行时配置 | 避免 secret 进 Git |

## 运行建议

- 代码仓库：`PainKiller0x0/Gemini-FastAPI`，主要分支：`dev-rs`。
- 运行配置：使用 `CONFIG_PATH` 指向服务器本地脱敏外的配置文件。
- 守护方式：推荐 Podman + systemd，端口默认只绑定内网或 WireGuard 地址。
- 对外调用：优先由 OBP 接入 `/v1/chat/completions`，不要把该服务直接裸露到公网。

## 配置要点

```yaml
gemini:
  chat_mode: "temporary"
  model_strategy: "append"
  warm_generate:
    enabled: true
    model: "gemini-3.5-flash"
    prompt: "只回复一个字：好"
    active_periods: ["07:00-01:30"]

image_generation:
  backend: "gemini_worker"
  public_base_url: "https://example.com/gemini-images"
  worker_url: "http://127.0.0.1:8010"
  worker_token_file: "/run/secrets/gemini_worker_token"
  worker_timeout_ms: 300000
```

## 回测清单

每次改 Gemini-FastAPI 后至少跑这些：

- `cargo fmt -- --check`
- `cargo test --locked`
- `cargo build --release --locked`
- `/health` 返回 Rust 实现和可用 client。
- `/v1/models` 能列出 `gemini-3.5-flash`、`gemini-3.1-pro` 等别名。
- 普通聊天不触发 image tool，附件识图可走 vision worker。
- 明确“帮我画一张...”才触发图片链路。
- “画面/图片/生图报错/画 UI 很烦”这类文本讨论仍然走普通聊天。
- `c8o8Fe` 全尺寸图片 URL 解析单测通过。

## 暂不做

- 不把 cookie/token 写进仓库。
- 不做多账号复杂路由，避免引入新状态和新 bug。
- 不把这个 sidecar 变成完整 OBP；模型账本、来源统计、路由策略仍由 OBP 管。
