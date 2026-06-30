"""Inspect and verify configured Scratchpad model catalog entries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.llm.openai_compatible import make_sync_client
from app.llm.catalog import config_from_model_ref, load_model_catalog, resolved_config_metadata
from app.llm.runtime import ensure_provider_ready


def run(argv: list[str] | None = None) -> int:
    """Run the model-catalog CLI."""
    parser = argparse.ArgumentParser(description="Inspect and check Scratchpad model catalog entries.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List configured model refs.")
    list_parser.add_argument("--json", action="store_true")

    show_parser = subparsers.add_parser("show", help="Show one resolved model ref.")
    show_parser.add_argument("model_ref")
    show_parser.add_argument("--json", action="store_true")

    check_parser = subparsers.add_parser("check", help="Resolve a model ref and ping its OpenAI-compatible endpoint.")
    check_parser.add_argument("model_ref")
    check_parser.add_argument("--start", action="store_true", help="Run the configured start script first when supported.")
    check_parser.add_argument("--provider")
    check_parser.add_argument("--model")
    check_parser.add_argument("--base-url")
    check_parser.add_argument("--api-key")
    check_parser.add_argument("--start-script")
    check_parser.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "list":
        catalog = load_model_catalog()
        rows = []
        for provider in catalog.providers.values():
            for model in provider.models.values():
                rows.append(
                    {
                        "ref": f"custom:{provider.name}:{model.name}",
                        "provider": provider.name,
                        "model": model.name,
                        "server_mode": provider.server_mode,
                        "description": model.description or provider.description,
                    }
                )
        if args.json:
            print(json.dumps(rows, ensure_ascii=True, indent=2))
        else:
            for row in rows:
                print(f"{row['ref']} ({row['server_mode']})")
        return 0

    if args.command == "show":
        config = config_from_model_ref(args.model_ref)
        metadata = resolved_config_metadata(config, model_ref=args.model_ref)
        if args.json:
            print(json.dumps(metadata, ensure_ascii=True, indent=2))
        else:
            print(f"model_ref: {metadata['model_ref']}")
            print(f"provider: {metadata['provider']}")
            print(f"model: {metadata['model']}")
            print(f"base_url: {metadata['base_url']}")
            print(f"has_api_key: {metadata['has_api_key']}")
            print(f"has_start_script: {metadata['has_start_script']}")
            print(f"request_settings: {json.dumps(metadata['request_settings'], ensure_ascii=True, sort_keys=True)}")
        return 0

    config = config_from_model_ref(
        args.model_ref,
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
        **resolved_config_metadata(config, model_ref=args.model_ref),
        "reachable": True,
        "available_models": model_ids,
    }
    if args.json:
        print(json.dumps(result, ensure_ascii=True, indent=2))
    else:
        print(f"reachable: true")
        print(f"model_ref: {args.model_ref}")
        print(f"provider: {config.provider}")
        print(f"model: {config.model_name}")
        if model_ids:
            print(f"available_models: {', '.join(model_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
