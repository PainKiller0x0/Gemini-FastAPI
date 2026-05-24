#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_DIR="${GEMINI_FASTAPI_HOME:-/opt/gemini-fastapi}"

install -d -m 0755 "$INSTALL_DIR/ops"
probe_src="$ROOT_DIR/ops/gemini-active-probe.py"
probe_dst="$INSTALL_DIR/ops/gemini-active-probe.py"
if [[ "$(readlink -f "$probe_src")" != "$(readlink -f "$probe_dst" 2>/dev/null || printf '%s' "$probe_dst")" ]]; then
  install -m 0755 "$probe_src" "$probe_dst"
else
  chmod 0755 "$probe_dst"
fi
install -m 0644 "$ROOT_DIR/ops/systemd/gemini-fastapi-active-probe.service" /etc/systemd/system/gemini-fastapi-active-probe.service
install -m 0644 "$ROOT_DIR/ops/systemd/gemini-fastapi-active-probe.timer" /etc/systemd/system/gemini-fastapi-active-probe.timer

systemctl daemon-reload
systemctl enable --now gemini-fastapi-active-probe.timer
systemctl start gemini-fastapi-active-probe.service || true
systemctl --no-pager status gemini-fastapi-active-probe.timer
systemctl --no-pager status gemini-fastapi-active-probe.service || true
