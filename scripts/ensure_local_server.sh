#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${LLM_BASE_URL:-http://localhost:8080/v1}"
ROOT_URL="${BASE_URL%/v1}"
HEALTH_URL="${ROOT_URL}/health"
MODELS_URL="${ROOT_URL}/v1/models"
DEFAULT_SCRIPT="/Users/yuvalkenneth/Desktop/local-llms/scripts/run-qwen-4b-server.sh"
START_SCRIPT="${1:-${LLM_START_SCRIPT:-$DEFAULT_SCRIPT}}"
LOG_FILE="${LLM_SERVER_LOG:-/tmp/llama-server.log}"
WAIT_SECONDS="${LLM_SERVER_WAIT_SECONDS:-60}"

is_up() {
  curl -fsS "$HEALTH_URL" >/dev/null 2>&1 || curl -fsS "$MODELS_URL" >/dev/null 2>&1
}

if is_up; then
  echo "Local LLM server already running at $ROOT_URL"
  exit 0
fi

if [[ ! -x "$START_SCRIPT" ]]; then
  echo "Start script is missing or not executable: $START_SCRIPT" >&2
  exit 1
fi

echo "Starting local LLM server with $START_SCRIPT"
nohup "$START_SCRIPT" >"$LOG_FILE" 2>&1 &

for ((i = 1; i <= WAIT_SECONDS; i++)); do
  if is_up; then
    echo "Local LLM server is up at $ROOT_URL"
    exit 0
  fi
  sleep 1
done

echo "Timed out waiting for local LLM server at $ROOT_URL" >&2
echo "Log file: $LOG_FILE" >&2
exit 1
