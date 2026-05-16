from __future__ import annotations

import argparse
import getpass
import sys

from .config import Config
from .loop import ask, run_agent
from .trace import summarize_trace


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="madharness-mini")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("ask", "run"):
        p = sub.add_parser(name)
        p.add_argument("task")
    p = sub.add_parser("init")
    p.add_argument("--provider")
    p.add_argument("--model")
    p.add_argument("--base-url")
    p.add_argument("--api-key")
    p.add_argument("--no-prompt", action="store_true")
    p = sub.add_parser("trace")
    p.add_argument("trace_id")
    args = parser.parse_args(argv)
    cfg = Config()
    try:
        if args.cmd == "init":
            api_key = args.api_key
            if api_key is None and not cfg.data.get("api_key") and not args.no_prompt:
                if sys.stdin.isatty():
                    value = getpass.getpass("Ключ API (можно оставить пустым): ")
                    api_key = value or None
            path, changes = cfg.initialize(
                provider=args.provider,
                model=args.model,
                base_url=args.base_url,
                api_key=api_key,
            )
            print(f"Настройка записана: {path}")
            if changes:
                names = {
                    "api_key": "api_key",
                    "base_url": "base_url",
                    "created": "config.json",
                    "model": "model",
                    "provider": "provider",
                }
                changed = [names.get(item, item) for item in sorted(set(changes))]
                print("Обновлено: " + ", ".join(changed))
            if not cfg.data.get("api_key"):
                print(
                    "Ключ API не задан. Передайте --api-key или задайте "
                    "MADHARNESS_MINI_API_KEY перед запуском ask/run."
                )
        elif args.cmd in {"ask", "run"}:
            action = ask if args.cmd == "ask" else run_agent
            result, trace = action(args.task, cfg)
            print(result)
            print(f"\nTrace: {trace}", file=sys.stderr)
        else:
            print(summarize_trace(cfg, args.trace_id))
    except RuntimeError as exc:
        raise SystemExit(f"error: {exc}") from exc
