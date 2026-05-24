#!/usr/bin/env bash
set -euo pipefail

ROOT="${GEMINI_FASTAPI_HOME:-/opt/gemini-fastapi}"
APPLY=0
REMOVE_VENV=0
RETENTION_DAYS="${GEMINI_IMAGE_RETENTION_DAYS:-7}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply) APPLY=1 ;;
    --remove-venv) REMOVE_VENV=1 ;;
    --retention-days) RETENTION_DAYS="$2"; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done

run() {
  if [[ "$APPLY" == "1" ]]; then
    echo "+ $*"
    "$@"
  else
    echo "DRY-RUN: $*"
  fi
}

run find "$ROOT" -type d \( -name __pycache__ -o -name .ruff_cache \) -prune -exec rm -rf {} +
run find "$ROOT/bin" -maxdepth 1 -type f -name 'gemini-fastapi-rs.bak*' -delete
run find "$ROOT/data/images" -maxdepth 1 -type f -name '*.png' -mtime "+$RETENTION_DAYS" -delete

if [[ "$REMOVE_VENV" == "1" ]]; then
  run rm -rf "$ROOT/.venv"
else
  echo "keep $ROOT/.venv; pass --remove-venv after confirming Python fallback is unnecessary"
fi