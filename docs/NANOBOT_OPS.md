# Nanobot Gemini-FastAPI 运维说明

这个分支用于记录 nanobot 侧 Gemini-FastAPI 的线上运维资产。真实 cookie、nonce、API key、图片缓存和 LMDB 数据不进入 Git。

## 主动探针

`/health` 只能证明进程活着，不能证明 Gemini Web cookie 仍然可用。`ops/gemini-active-probe.py` 会通过 OpenAI 兼容接口发送一个极短请求，并把结果写入：

```text
/opt/gemini-fastapi/runtime/active_probe.json
```

安装：

```bash
bash ops/install-active-probe.sh
```

查看：

```bash
systemctl status gemini-fastapi-active-probe.timer
systemctl status gemini-fastapi-active-probe.service
cat /opt/gemini-fastapi/runtime/active_probe.json
```

默认每 10 分钟探测一次。如果 cookie 失效、认证失败、地区限制或上游错误，探针会非 0 退出，并在 JSON 里留下错误信息。

## 运行目录清理

先 dry-run：

```bash
bash ops/cleanup-runtime.sh
```

确认没问题后执行：

```bash
bash ops/cleanup-runtime.sh --apply
```

默认会清理：

- `__pycache__`
- `.ruff_cache`
- 旧的 `bin/gemini-fastapi-rs.bak*`
- 超过 7 天的 `data/images/*.png`

`.venv` 默认保留。如果确认线上只跑 Rust 版本，可以额外传：

```bash
bash ops/cleanup-runtime.sh --apply --remove-venv
```

## 不要提交

- `runtime/`
- `data/`
- `.venv/`
- `.ruff_cache/`
- `bin/`
- `*.bak*`
- 任何 cookie / token / key