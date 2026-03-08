#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$ROOT_DIR/project practice"

if [ -f "$APP_DIR/.venv/bin/activate" ]; then
  source "$APP_DIR/.venv/bin/activate"
fi

exec python3 -m uvicorn api.app:app --reload --port 8000 --app-dir "$APP_DIR"
