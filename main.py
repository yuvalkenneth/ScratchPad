import asyncio
from typing import Any

from app.llm.config import LLMConfig
from app.llm.client import LLMClient
from app.llm.prompting import build_system_prompt
from app.llm.runtime import ensure_provider_ready


async def run() -> None:
    config = ensure_provider_ready(LLMConfig.from_env())
    client = LLMClient.from_config(config)
    messages: list[dict[str, Any]] = []

    messages.append({"role": "system", "content": build_system_prompt()})

    print("Chat started. Type 'exit' or 'quit' to stop.")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        messages.append({"role": "user", "content": user_input})
        response = await client.get_response(messages)
        messages.append({"role": "assistant", "content": response})
        print(f"Assistant: {response}")


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
