#!/bin/sh
# 使い方: ./run.sh   （環境変数は README 参照）
cd "$(dirname "$0")"
[ -f .venv/bin/activate ] && . .venv/bin/activate
exec python3 -m uvicorn boatbot.app:app --host 0.0.0.0 --port "${PORT:-8000}"
