from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from app.llm.config import LLMConfig
from app.llm.client import LLMClient
from app.llm.prompting import build_system_prompt
from app.llm.runtime import ensure_provider_ready


DEFAULT_SERVER_LOG = Path(os.getenv("LLM_SERVER_LOG", "/tmp/llama-server.log"))


def build_messages() -> list[dict[str, Any]]:
    return [{"role": "system", "content": build_system_prompt()}]


def rebuild_client(config: LLMConfig) -> tuple[LLMConfig, LLMClient]:
    config = ensure_provider_ready(config)
    return config, LLMClient.from_config(config)


def stop_local_server(config: LLMConfig) -> None:
    if config.provider.strip().lower() != "llama_cpp":
        return

    script_path = os.path.join(
        os.path.dirname(__file__),
        "scripts",
        "stop_local_server.sh",
    )
    env = os.environ.copy()
    env["LLM_BASE_URL"] = config.base_url
    subprocess.run([script_path], check=True, env=env)


def parse_command(user_input: str) -> tuple[str, list[str]]:
    parts = shlex.split(user_input)
    return parts[0].lower(), parts[1:]


def fetch_json(url: str) -> Any:
    with urlopen(url, timeout=2) as response:
        return json.loads(response.read().decode("utf-8"))


def read_latest_timing(log_path: Path = DEFAULT_SERVER_LOG) -> dict[str, Any] | None:
    if not log_path.exists():
        return None

    text = log_path.read_text(errors="replace")
    pattern = re.compile(
        r"slot update_slots: id\s+\d+ \| task (?P<task>\d+) \| new prompt, n_ctx_slot = "
        r"(?P<n_ctx>\d+), n_keep = \d+, task\.n_tokens = (?P<task_tokens>\d+).*?"
        r"slot print_timing: id\s+\d+ \| task (?P=task) \|\s*\n"
        r"prompt eval time =\s+(?P<prompt_ms>[0-9.]+) ms /"
        r"\s+(?P<prompt_tokens>\d+) tokens .*?(?P<prompt_tps>[0-9.]+) tokens per second\)\s*\n"
        r"\s+eval time =\s+(?P<eval_ms>[0-9.]+) ms /"
        r"\s+(?P<eval_tokens>\d+) tokens .*?(?P<eval_tps>[0-9.]+) tokens per second\)\s*\n"
        r"\s+total time =\s+(?P<total_ms>[0-9.]+) ms /"
        r"\s+(?P<total_tokens>\d+) tokens",
        re.DOTALL,
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return None

    match = matches[-1]
    groups = match.groupdict()
    return {
        "task": int(groups["task"]),
        "context_limit": int(groups["n_ctx"]),
        "input_tokens": int(groups["task_tokens"]),
        "prompt_eval_ms": float(groups["prompt_ms"]),
        "prompt_eval_tokens": int(groups["prompt_tokens"]),
        "prompt_tokens_per_second": float(groups["prompt_tps"]),
        "output_tokens": int(groups["eval_tokens"]),
        "eval_ms": float(groups["eval_ms"]),
        "output_tokens_per_second": float(groups["eval_tps"]),
        "total_ms": float(groups["total_ms"]),
        "total_tokens": int(groups["total_tokens"]),
    }


def get_server_status(config: LLMConfig) -> str:
    root_url = config.base_url.rstrip("/")
    if root_url.endswith("/v1"):
        root_url = root_url[:-3]

    lines = [f"Configured model: {config.model_name}", f"Base URL: {config.base_url}"]

    health_text = "down"
    model_text = "unknown"
    try:
        health = fetch_json(f"{root_url}/health")
        health_text = "up"
        lines.append(f"Health: {health_text} ({health})")
    except (URLError, TimeoutError, json.JSONDecodeError):
        lines.append("Health: down")
    else:
        try:
            models_payload = fetch_json(f"{root_url}/v1/models")
            models = models_payload.get("data") or []
            if models:
                model_text = ", ".join(model.get("id", "unknown") for model in models)
        except (URLError, TimeoutError, json.JSONDecodeError, AttributeError):
            pass
        lines.append(f"Server models: {model_text}")

    timing = read_latest_timing()
    if timing is None:
        lines.append("Latest timing: unavailable")
        return "\n".join(lines)

    lines.append(
        "Latest timing: "
        f"input {timing['input_tokens']} / ctx {timing['context_limit']}, "
        f"output {timing['output_tokens']}, "
        f"prompt {timing['prompt_tokens_per_second']:.1f} tok/s, "
        f"gen {timing['output_tokens_per_second']:.1f} tok/s, "
        f"total {timing['total_ms'] / 1000:.1f}s"
    )
    return "\n".join(lines)


async def run() -> None:
    config, client = rebuild_client(LLMConfig.from_env())
    messages = build_messages()

    print("Chat started. Type 'exit' or 'quit' to stop.")
    print("Commands: /reset, /reload, /model <model_name> <start_script>, /server-status")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        if user_input.startswith("/"):
            command, arguments = parse_command(user_input)

            if command == "/reset":
                messages = build_messages()
                print("Assistant: Conversation reset.")
                continue

            if command == "/reload":
                config, client = rebuild_client(LLMConfig.from_env())
                messages = build_messages()
                print(
                    "Assistant: Reloaded configuration and reset the conversation. "
                    f"Current model: {config.model_name}"
                )
                continue

            if command == "/model":
                if not arguments:
                    print("Assistant: Usage: /model <model_name> <start_script>")
                    continue

                next_model = arguments[0]
                next_script = arguments[1] if len(arguments) > 1 else None

                if config.provider.strip().lower() == "llama_cpp" and not next_script:
                    print(
                        "Assistant: For llama_cpp, provide a start script so the server "
                        "actually switches models. Usage: /model <model_name> <start_script>"
                    )
                    continue

                next_config = LLMConfig(
                    provider=config.provider,
                    model_name=next_model,
                    base_url=config.base_url,
                    api_key=config.api_key,
                    start_script=next_script or config.start_script,
                )
                try:
                    stop_local_server(config)
                    config, client = rebuild_client(next_config)
                except subprocess.CalledProcessError as exc:
                    print(f"Assistant: Failed to switch models: {exc}")
                    continue

                messages = build_messages()
                print(
                    "Assistant: Switched model and reset the conversation. "
                    f"Current model: {config.model_name}"
                )
                continue

            if command == "/server-status":
                print(f"Assistant: {get_server_status(config)}")
                continue

            print("Assistant: Unknown command.")
            continue

        messages.append({"role": "user", "content": user_input})
        response = await client.get_response(messages)
        messages.append({"role": "assistant", "content": response})
        print(f"Assistant: {response}")


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
