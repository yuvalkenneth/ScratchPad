#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${LLM_BASE_URL:-http://localhost:8080/v1}"
ROOT_URL="${BASE_URL%/v1}"
PORT="$(printf '%s' "$ROOT_URL" | sed -E 's#^[a-zA-Z]+://[^:/]+:([0-9]+)$#\1#')"

if [[ ! "$PORT" =~ ^[0-9]+$ ]]; then
  echo "Could not determine port from LLM_BASE_URL: $BASE_URL" >&2
  exit 1
fi

PIDS="$(lsof -tiTCP:"$PORT" -sTCP:LISTEN || true)"

if [[ -z "$PIDS" ]]; then
  echo "No local LLM server is listening on port $PORT"
  exit 0
fi

echo "Stopping local LLM server on port $PORT: $PIDS"
kill $PIDS

