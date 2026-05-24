#!/usr/bin/env python3
"""Active Gemini-FastAPI probe.

The normal /health endpoint only proves the process is alive. This probe sends a
small OpenAI-compatible request so cookie/session/auth failures are caught before
nanobot traffic hits them.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


BASE_URL = env("GEMINI_FASTAPI_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
CONFIG_PATH = Path(env("GEMINI_FASTAPI_CONFIG", "/opt/gemini-fastapi/runtime/config.yaml"))
STATE_PATH = Path(env("GEMINI_FASTAPI_PROBE_STATE", "/opt/gemini-fastapi/runtime/active_probe.json"))
MODEL = env("GEMINI_FASTAPI_PROBE_MODEL", "gemini-3.5-flash")
PROMPT = env("GEMINI_FASTAPI_PROBE_PROMPT", "Reply exactly: pong")
TIMEOUT = float(env("GEMINI_FASTAPI_PROBE_TIMEOUT", "25"))


def read_api_key(path: Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    # The server api_key appears before image_generation.api_key in config.yaml.
    match = re.search(r"(?m)^\s*api_key:\s*([\"']?)([^\"'\s#]+)\1\s*(?:#.*)?$", text)
    if not match:
        return None
    value = match.group(2).strip()
    if value.lower() in {"null", "none", "~"}:
        return None
    return value


def extract_content(payload: dict[str, Any]) -> str:
    try:
        message = payload["choices"][0]["message"]
        content = message.get("content")
        if isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=False)
    except Exception:
        return ""


def write_state(state: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_name(f"{STATE_PATH.name}.tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(STATE_PATH)


def main() -> int:
    started = time.perf_counter()
    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    request_id = f"gemini-probe-{int(time.time())}"
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": PROMPT}],
            "stream": False,
            "max_tokens": 32,
            "temperature": 0,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "X-Trace-Id": request_id,
    }
    api_key = read_api_key(CONFIG_PATH)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    state: dict[str, Any] = {
        "checked_at": now,
        "ok": False,
        "base_url": BASE_URL,
        "model": MODEL,
        "request_id": request_id,
        "latency_ms": 0,
    }

    try:
        req = urllib.request.Request(
            f"{BASE_URL}/v1/chat/completions",
            data=body,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            latency_ms = int((time.perf_counter() - started) * 1000)
            payload = json.loads(raw)
            content = extract_content(payload)
            ok = resp.status == 200 and bool(payload.get("choices"))
            state.update(
                {
                    "ok": ok,
                    "http_status": resp.status,
                    "latency_ms": latency_ms,
                    "content_preview": content[:160],
                    "error": "" if ok else "missing choices in response",
                }
            )
    except urllib.error.HTTPError as exc:
        latency_ms = int((time.perf_counter() - started) * 1000)
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        state.update({"http_status": exc.code, "latency_ms": latency_ms, "error": detail or str(exc)})
    except Exception as exc:
        latency_ms = int((time.perf_counter() - started) * 1000)
        state.update({"http_status": 0, "latency_ms": latency_ms, "error": str(exc)})

    write_state(state)
    print(json.dumps(state, ensure_ascii=False, separators=(",", ":")))
    return 0 if state.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())