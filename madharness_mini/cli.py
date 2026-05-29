"""Точка входа: команды `ask`, `run`, `init` и `trace`."""

import argparse
import getpass
import sys

from .config import Config
from .loop import ask, run_agent
from .skills import discover_skills
from .skills.activation import list_skill_resources
from .trace import summarize_trace
from .utils import DEFAULT_CONFIG, STATE_DIR


def api_key_prompt(cfg: Config) -> str:
    """Текст подсказки при интерактивном вводе API-ключа.

    Показываем, какой маршрутизатор и модель сейчас в конфиге, и куда
    потом можно дописать настройки вручную.
    """

    base_url = cfg.data.get("base_url") or DEFAULT_CONFIG["base_url"]
    model = cfg.data.get("model") or DEFAULT_CONFIG["model"]
    router = (
        "OpenRouter"
        if base_url == DEFAULT_CONFIG["base_url"]
        else "выбранного маршрутизатора/API"
    )
    return (
        f"Ключ API для {router}.\n"
        f"Маршрутизатор/API: {base_url}\n"
        f"Модель по умолчанию: {model}\n"
        f"Enter - оставить пустым; позже можно переопределить model, base_url "
        f"и api_key в {STATE_DIR}/config.json.\n"
        "Ключ API: "
    )


def main(argv: list[str] | None = None) -> None:
    """Разбираем argv и запускаем выбранную подкоманду.

    init — записать конфиг; ask/run — диалог с моделью;
    trace — краткая сводка по файлу трассы.
    """

    parser = argparse.ArgumentParser(prog="madharness-mini")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("ask", "run"):
        p = sub.add_parser(name)
        p.add_argument("task")
    p = sub.add_parser("init")
    p.add_argument("--model")
    p.add_argument("--base-url")
    p.add_argument("--api-key")
    p.add_argument("--no-prompt", action="store_true")
    p = sub.add_parser("trace")
    p.add_argument("trace_id")
    skills = sub.add_parser("skills")
    skills_sub = skills.add_subparsers(dest="skills_cmd", required=True)
    skills_sub.add_parser("list")
    p = skills_sub.add_parser("show")
    p.add_argument("name")
    skills_sub.add_parser("validate")
    args = parser.parse_args(argv)
    cfg = Config()
    try:
        if args.cmd == "init":
            api_key = args.api_key
            if api_key is None and not cfg.data.get("api_key") and not args.no_prompt:
                if sys.stdin.isatty():
                    value = getpass.getpass(api_key_prompt(cfg))
                    api_key = value or None
            path, changes = cfg.initialize(
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
        elif args.cmd == "skills":
            print(skills_command(cfg, args.skills_cmd, getattr(args, "name", "")))
        else:
            print(summarize_trace(cfg, args.trace_id))
    except RuntimeError as exc:
        raise SystemExit(f"error: {exc}") from exc


def skills_command(cfg: Config, command: str, name: str = "") -> str:
    """Печатаем найденные Agent Skills и диагностику их `SKILL.md`."""

    index = discover_skills(cfg)
    if command == "list":
        if not index.skills:
            return "Навыки не найдены."
        lines = ["Доступные навыки:"]
        for skill_name in index.names():
            skill = index.skills[skill_name]
            lines.append(
                f"- {skill.name}: {skill.description} "
                f"({skill.location(cfg.root)})"
            )
        return "\n".join(lines)
    if command == "show":
        skill = index.skills.get(name)
        if not skill:
            raise RuntimeError(f"skill not found: {name}")
        lines = [
            f"name: {skill.name}",
            f"description: {skill.description}",
            f"location: {skill.location(cfg.root)}",
            f"root: {skill.root_location(cfg.root)}",
            f"source: {skill.source}",
        ]
        if skill.license:
            lines.append(f"license: {skill.license}")
        if skill.compatibility:
            lines.append(f"compatibility: {skill.compatibility}")
        if skill.allowed_tools:
            lines.append("allowed-tools: " + " ".join(skill.allowed_tools))
        if skill.metadata:
            lines.append("metadata:")
            for key, value in sorted(skill.metadata.items()):
                lines.append(f"  {key}: {value}")
        resources = list_skill_resources(skill, cfg.root)
        lines.append("resources:")
        if resources:
            lines.extend(
                f"  - {item.workspace_path} ({item.kind}, {item.bytes} bytes)"
                for item in resources
            )
        else:
            lines.append("  none")
        lines.extend(["", "instructions:", skill.body])
        return "\n".join(lines)
    if command == "validate":
        lines = [f"skills: {len(index.skills)}"]
        for diagnostic in index.diagnostics:
            item = diagnostic.as_dict(cfg.root)
            lines.append(
                f"{item['severity']}: {item['path']}: {item['message']}"
            )
        errors = [
            item for item in index.diagnostics if item.severity == "error"
        ]
        if errors:
            lines.append(f"errors: {len(errors)}")
        else:
            lines.append("OK")
        return "\n".join(lines)
    raise RuntimeError(f"unknown skills command: {command}")
