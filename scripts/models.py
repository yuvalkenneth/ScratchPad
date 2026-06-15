from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.llm.openai_compatible import make_sync_client
from app.llm.profiles import config_from_profile, load_model_profiles, resolved_config_metadata
from app.llm.runtime import ensure_provider_ready


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect and check Scratchpad model profiles.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List configured model profiles.")
    list_parser.add_argument("--json", action="store_true")

    show_parser = subparsers.add_parser("show", help="Show one resolved model profile.")
    show_parser.add_argument("profile")
    show_parser.add_argument("--json", action="store_true")

    check_parser = subparsers.add_parser("check", help="Resolve a profile and ping its OpenAI-compatible endpoint.")
    check_parser.add_argument("profile")
    check_parser.add_argument("--start", action="store_true", help="Run the configured start script first when supported.")
    check_parser.add_argument("--provider")
    check_parser.add_argument("--model")
    check_parser.add_argument("--base-url")
    check_parser.add_argument("--api-key")
    check_parser.add_argument("--start-script")
    check_parser.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "list":
        profiles = load_model_profiles()
        rows = [
            {
                "name": profile.name,
                "provider": profile.provider,
                "model": profile.model_name,
                "server_mode": profile.server_mode,
                "description": profile.description,
            }
            for profile in profiles.values()
        ]
        if args.json:
            print(json.dumps(rows, ensure_ascii=True, indent=2))
        else:
            for row in rows:
                print(f"{row['name']}: {row['provider']}/{row['model']} ({row['server_mode']})")
        return 0

    if args.command == "show":
        config = config_from_profile(args.profile)
        metadata = resolved_config_metadata(config, profile=args.profile)
        if args.json:
            print(json.dumps(metadata, ensure_ascii=True, indent=2))
        else:
            print(f"profile: {metadata['profile']}")
            print(f"provider: {metadata['provider']}")
            print(f"model: {metadata['model']}")
            print(f"base_url: {metadata['base_url']}")
            print(f"has_api_key: {metadata['has_api_key']}")
            print(f"has_start_script: {metadata['has_start_script']}")
        return 0

    config = config_from_profile(
        args.profile,
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        start_script=args.start_script,
    )
    if args.start:
        config = ensure_provider_ready(config)
    models = make_sync_client(config).models.list()
    model_ids = [model.id for model in getattr(models, "data", [])]
    result = {
        **resolved_config_metadata(config, profile=args.profile),
        "reachable": True,
        "available_models": model_ids,
    }
    if args.json:
        print(json.dumps(result, ensure_ascii=True, indent=2))
    else:
        print(f"reachable: true")
        print(f"profile: {args.profile}")
        print(f"provider: {config.provider}")
        print(f"model: {config.model_name}")
        if model_ids:
            print(f"available_models: {', '.join(model_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
