#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -f "$ROOT_DIR/.venv/bin/activate" ]; then
  source "$ROOT_DIR/.venv/bin/activate"
fi

exec python3 -m uvicorn api.app:app --reload --port 8000
