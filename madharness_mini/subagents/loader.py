"""Discovery встроенных и project-local markdown-субагентов."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

from ..config import Config
from ..policy import Policy
from ..skills.loader import parse_skill_markdown, valid_skill_name
from .types import Subagent, SubagentDiagnostic, SubagentIndex

BUILTIN_SUBAGENTS_DIR = "prompts/subagents"
PROJECT_SUBAGENTS_DIR = ".madharness-mini/subagents"
MAX_SUBAGENT_FILE_BYTES = 200_000
PROFILE_VALUES = {"read-only", "writable"}


def discover_subagents(cfg: Config) -> SubagentIndex:
    """Читаем shipped роли и project-local `.madharness-mini/subagents/*.md`.

    Встроенные роли лежат в `madharness_mini/prompts/subagents`. Проектные
    файлы могут расширить каталог, а перекрыть встроенное имя — только с
    `override: true`, чтобы случайная коллизия не меняла поведение роли.
    """

    found: dict[str, tuple[int, Subagent]] = {}
    diagnostics: list[SubagentDiagnostic] = []

    for subagent, diagnostic_items in _load_builtin_subagents():
        diagnostics.extend(diagnostic_items)
        if subagent:
            found[subagent.name] = (10, subagent)

    project_root, err = Policy(cfg).safe_path(PROJECT_SUBAGENTS_DIR)
    if err:
        diagnostics.append(SubagentDiagnostic("error", PROJECT_SUBAGENTS_DIR, err))
    elif project_root and project_root.exists():
        if not project_root.is_dir():
            diagnostics.append(
                SubagentDiagnostic(
                    "error",
                    PROJECT_SUBAGENTS_DIR,
                    "subagent root is not a directory",
                )
            )
        else:
            for path in sorted(project_root.glob("*.md")):
                subagent, diagnostic_items = load_subagent_file(
                    path,
                    workspace_root=cfg.root,
                    source="project",
                    builtin=False,
                )
                diagnostics.extend(diagnostic_items)
                if subagent:
                    _put_subagent(found, diagnostics, subagent, priority=20)

    return SubagentIndex(
        {name: item[1] for name, item in sorted(found.items())},
        tuple(diagnostics),
    )


def load_subagent_file(
    path: Path,
    *,
    workspace_root: Path | None = None,
    source: str,
    builtin: bool,
) -> tuple[Subagent | None, list[SubagentDiagnostic]]:
    """Читаем один markdown-файл субагента и валидируем frontmatter."""

    diagnostics: list[SubagentDiagnostic] = []
    location = _display_path(path, workspace_root)
    if not path.is_file():
        return None, [
            SubagentDiagnostic("error", location, "subagent file is not a file")
        ]
    try:
        size = path.stat().st_size
    except OSError as exc:
        return None, [
            SubagentDiagnostic("error", location, f"cannot stat subagent file: {exc}")
        ]
    if size > MAX_SUBAGENT_FILE_BYTES:
        return None, [
            SubagentDiagnostic(
                "error",
                location,
                f"subagent file is too large: {size} bytes, limit is {MAX_SUBAGENT_FILE_BYTES}",
            )
        ]
    try:
        raw_text = path.read_text(encoding="utf-8", errors="replace")
        fields, body = parse_skill_markdown(raw_text)
    except ValueError as exc:
        return None, [SubagentDiagnostic("error", location, str(exc))]

    subagent = _subagent_from_fields(
        fields,
        body,
        location=location,
        source=source,
        builtin=builtin,
        diagnostics=diagnostics,
    )
    if any(item.severity == "error" for item in diagnostics):
        return None, diagnostics
    return subagent, diagnostics


def _load_builtin_subagents() -> list[tuple[Subagent | None, list[SubagentDiagnostic]]]:
    """Читаем shipped markdown-файлы ролей из package resources."""

    root = resources.files("madharness_mini").joinpath("prompts", "subagents")
    loaded: list[tuple[Subagent | None, list[SubagentDiagnostic]]] = []
    for item in sorted(root.iterdir(), key=lambda value: value.name):
        if not item.name.endswith(".md"):
            continue
        location = f"madharness_mini/{BUILTIN_SUBAGENTS_DIR}/{item.name}"
        try:
            raw_text = item.read_text(encoding="utf-8", errors="replace")
            fields, body = parse_skill_markdown(raw_text)
        except ValueError as exc:
            loaded.append((None, [SubagentDiagnostic("error", location, str(exc))]))
            continue
        diagnostics: list[SubagentDiagnostic] = []
        subagent = _subagent_from_fields(
            fields,
            body,
            location=location,
            source="builtin",
            builtin=True,
            diagnostics=diagnostics,
        )
        if any(diagnostic.severity == "error" for diagnostic in diagnostics):
            loaded.append((None, diagnostics))
        else:
            loaded.append((subagent, diagnostics))
    return loaded


def _subagent_from_fields(
    fields: dict[str, Any],
    body: str,
    *,
    location: str,
    source: str,
    builtin: bool,
    diagnostics: list[SubagentDiagnostic],
) -> Subagent | None:
    """Превращаем frontmatter + markdown-body в Subagent."""

    name = _scalar(fields.get("name", ""))
    description = _scalar(fields.get("description", ""))
    profile = _scalar(fields.get("profile", ""))
    if not name:
        diagnostics.append(SubagentDiagnostic("error", location, "missing required name"))
    elif not valid_skill_name(name):
        diagnostics.append(
            SubagentDiagnostic("error", location, f"invalid subagent name: {name}")
        )
    if not description:
        diagnostics.append(
            SubagentDiagnostic("error", location, "missing required description")
        )
    if profile not in PROFILE_VALUES:
        diagnostics.append(
            SubagentDiagnostic(
                "error",
                location,
                f"invalid profile: {profile or '<empty>'}; allowed: read-only, writable",
            )
        )
    tools = _parse_tools(fields.get("tools"), location, diagnostics)
    max_turns = _optional_positive_int(
        fields.get("max_turns"), "max_turns", location, diagnostics
    )
    context_max_tokens = _optional_positive_int(
        fields.get("context_max_tokens"),
        "context_max_tokens",
        location,
        diagnostics,
    )
    prompt = body.strip()
    if not prompt:
        diagnostics.append(SubagentDiagnostic("error", location, "missing prompt body"))
    if any(item.severity == "error" for item in diagnostics):
        return None
    return Subagent(
        name=name,
        description=description,
        profile=profile,
        tools=tuple(tools),
        prompt=prompt,
        source=source,
        location=location,
        max_turns=max_turns,
        context_max_tokens=context_max_tokens,
        metadata=_metadata(fields.get("metadata")),
        builtin=_bool_field(fields.get("builtin", builtin)),
        override=_bool_field(fields.get("override", False)),
    )


def _parse_tools(
    raw: Any,
    location: str,
    diagnostics: list[SubagentDiagnostic],
) -> list[str]:
    """Читаем `tools` как JSON-style список строк из YAML frontmatter."""

    if raw is None or raw == "":
        diagnostics.append(SubagentDiagnostic("error", location, "missing required tools"))
        return []
    value = raw
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            diagnostics.append(
                SubagentDiagnostic(
                    "error",
                    location,
                    f"tools must be a JSON-style list of strings: {exc.msg}",
                )
            )
            return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        diagnostics.append(
            SubagentDiagnostic("error", location, "tools must be a list of strings")
        )
        return []
    tools = [item.strip() for item in value if item.strip()]
    if len(set(tools)) != len(tools):
        diagnostics.append(SubagentDiagnostic("error", location, "duplicate tool in tools"))
    if "delegate_task" in tools:
        diagnostics.append(
            SubagentDiagnostic(
                "error",
                location,
                "delegate_task is not allowed inside subagent tools",
            )
        )
    return tools


def _put_subagent(
    found: dict[str, tuple[int, Subagent]],
    diagnostics: list[SubagentDiagnostic],
    subagent: Subagent,
    *,
    priority: int,
) -> None:
    """Добавляем project-local субагента с понятной политикой коллизий."""

    current = found.get(subagent.name)
    if not current:
        found[subagent.name] = (priority, subagent)
        return
    existing = current[1]
    if existing.builtin and not subagent.override:
        diagnostics.append(
            SubagentDiagnostic(
                "warning",
                subagent.location,
                f"subagent shadows builtin name without override: {subagent.name}",
            )
        )
        return
    diagnostics.append(
        SubagentDiagnostic(
            "warning",
            existing.location,
            f"subagent shadowed by higher-priority copy: {subagent.name}",
        )
    )
    found[subagent.name] = (priority, subagent)


def _optional_positive_int(
    raw: Any,
    name: str,
    location: str,
    diagnostics: list[SubagentDiagnostic],
) -> int | None:
    """Читаем положительный integer из frontmatter, если поле задано."""

    if raw in (None, ""):
        return None
    try:
        value = int(str(raw).strip())
    except ValueError:
        diagnostics.append(
            SubagentDiagnostic("error", location, f"{name} must be an integer")
        )
        return None
    if value <= 0:
        diagnostics.append(
            SubagentDiagnostic("error", location, f"{name} must be positive")
        )
        return None
    return value


def _scalar(value: Any) -> str:
    """Приводим frontmatter scalar к строке без лишних пробелов."""

    return str(value or "").strip()


def _bool_field(value: Any) -> bool:
    """Читаем простые boolean-поля frontmatter."""

    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _metadata(value: Any) -> dict[str, str]:
    """Нормализуем metadata в строковый словарь."""

    if not isinstance(value, dict):
        return {}
    return {str(key): str(item) for key, item in value.items()}


def _display_path(path: Path, workspace_root: Path | None) -> str:
    """Показываем путь относительно workspace, если это возможно."""

    if workspace_root is not None:
        try:
            return str(path.resolve().relative_to(workspace_root.resolve()))
        except ValueError:
            pass
    return str(path)
