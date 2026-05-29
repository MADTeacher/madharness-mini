"""Ищем проектные Agent Skills и читаем YAML frontmatter из `SKILL.md`."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..config import Config
from ..policy import Policy
from .types import Skill, SkillDiagnostic, SkillIndex

# Поддерживаем только project-local каталоги: нативный и interoperable `.agents`.
SKILL_ROOTS = (
    (".agents/skills", "agents", 10),
    (".madharness_mini/skills", "native", 20),
)

# Skill-файл должен быть достаточно маленьким, чтобы случайно не забить prompt.
MAX_SKILL_FILE_BYTES = 200_000

NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")


def discover_skills(cfg: Config) -> SkillIndex:
    """Сканируем прямые подпапки skill-каталогов внутри workspace.

    Нативный каталог `.madharness_mini/skills` перекрывает `.agents/skills` при
    совпадении `name`. Ошибки отдельных навыков не валят запуск целиком: они
    попадают в диагностику и доступны через CLI `skills validate`.
    """

    policy = Policy(cfg)
    found: dict[str, tuple[int, Skill]] = {}
    diagnostics: list[SkillDiagnostic] = []

    for raw_root, source, priority in SKILL_ROOTS:
        root, err = policy.skill_root(raw_root)
        if err or root is None:
            diagnostics.append(
                SkillDiagnostic("error", cfg.root / raw_root, err or "invalid skill root")
            )
            continue
        if not root.exists():
            continue
        if not root.is_dir():
            diagnostics.append(
                SkillDiagnostic("error", root, "skill root is not a directory")
            )
            continue
        for child in sorted(path for path in root.iterdir() if path.is_dir()):
            skill, skill_diags = load_skill(child, cfg.root, source)
            diagnostics.extend(skill_diags)
            if skill is None:
                continue
            current = found.get(skill.name)
            if current and current[0] > priority:
                diagnostics.append(
                    SkillDiagnostic(
                        "warning",
                        skill.skill_file,
                        f"skill shadowed by higher-priority copy: {skill.name}",
                    )
                )
                continue
            if current:
                diagnostics.append(
                    SkillDiagnostic(
                        "warning",
                        current[1].skill_file,
                        f"skill shadowed by higher-priority copy: {skill.name}",
                    )
                )
            found[skill.name] = (priority, skill)

    return SkillIndex(
        {name: item[1] for name, item in sorted(found.items())},
        tuple(diagnostics),
    )


def load_skill(root: Path, workspace_root: Path, source: str) -> tuple[Skill | None, list[SkillDiagnostic]]:
    """Читаем один skill-каталог и возвращаем Skill либо понятную диагностику."""

    diagnostics: list[SkillDiagnostic] = []
    resolved_root = root.resolve()
    try:
        resolved_root.relative_to(workspace_root)
    except ValueError:
        return None, [SkillDiagnostic("error", root, "skill directory escapes workspace")]
    skill_file = root / "SKILL.md"
    if not skill_file.exists():
        return None, diagnostics
    if not skill_file.is_file():
        return None, [SkillDiagnostic("error", skill_file, "SKILL.md is not a file")]
    resolved_skill_file = skill_file.resolve()
    try:
        resolved_skill_file.relative_to(resolved_root)
    except ValueError:
        return None, [SkillDiagnostic("error", skill_file, "SKILL.md escapes skill root")]
    try:
        size = resolved_skill_file.stat().st_size
    except OSError as exc:
        return None, [SkillDiagnostic("error", skill_file, f"cannot stat SKILL.md: {exc}")]
    if size > MAX_SKILL_FILE_BYTES:
        return None, [
            SkillDiagnostic(
                "error",
                skill_file,
                f"SKILL.md is too large: {size} bytes, limit is {MAX_SKILL_FILE_BYTES}",
            )
        ]
    try:
        raw_text = resolved_skill_file.read_text(encoding="utf-8", errors="replace")
        fields, body = parse_skill_markdown(raw_text)
    except ValueError as exc:
        return None, [SkillDiagnostic("error", skill_file, str(exc))]

    name = _scalar(fields.get("name", ""))
    description = _scalar(fields.get("description", ""))
    if not name:
        diagnostics.append(SkillDiagnostic("error", skill_file, "missing required name"))
    elif not valid_skill_name(name):
        diagnostics.append(
            SkillDiagnostic("error", skill_file, f"invalid skill name: {name}")
        )
    if not description:
        diagnostics.append(
            SkillDiagnostic("error", skill_file, "missing required description")
        )
    elif len(description) > 1024:
        diagnostics.append(
            SkillDiagnostic("warning", skill_file, "description is longer than 1024 characters")
        )
    if any(item.severity == "error" for item in diagnostics):
        return None, diagnostics

    warnings: list[str] = []
    if name != root.name:
        message = f"name does not match parent directory: {name} != {root.name}"
        warnings.append(message)
        diagnostics.append(SkillDiagnostic("warning", skill_file, message))

    compatibility = _scalar(fields.get("compatibility", ""))
    if compatibility and len(compatibility) > 500:
        diagnostics.append(
            SkillDiagnostic("warning", skill_file, "compatibility is longer than 500 characters")
        )
    skill = Skill(
        name=name,
        description=description,
        root=resolved_root,
        skill_file=resolved_skill_file,
        body=body.strip(),
        raw_text=raw_text,
        source=source,
        license=_scalar(fields.get("license", "")),
        compatibility=compatibility,
        metadata=_metadata(fields.get("metadata")),
        allowed_tools=tuple(_scalar(fields.get("allowed-tools", "")).split()),
        warnings=tuple(warnings),
    )
    return skill, diagnostics


def parse_skill_markdown(text: str) -> tuple[dict[str, Any], str]:
    """Разбираем `---` frontmatter и markdown-body маленьким YAML-subset парсером."""

    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("SKILL.md must start with YAML frontmatter")
    closing = next((index for index in range(1, len(lines)) if lines[index].strip() == "---"), None)
    if closing is None:
        raise ValueError("SKILL.md frontmatter is not closed")
    fields = parse_frontmatter_lines(lines[1:closing])
    body = "\n".join(lines[closing + 1 :])
    return fields, body


def parse_frontmatter_lines(lines: list[str]) -> dict[str, Any]:
    """Парсим простые scalar-поля и одноуровневый `metadata:` без PyYAML."""

    fields: dict[str, Any] = {}
    index = 0
    while index < len(lines):
        raw = lines[index]
        if not raw.strip() or raw.lstrip().startswith("#"):
            index += 1
            continue
        if raw[:1].isspace():
            raise ValueError(f"unexpected indented frontmatter line: {raw.strip()}")
        if ":" not in raw:
            raise ValueError(f"invalid frontmatter line: {raw.strip()}")
        key, value = raw.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError("empty frontmatter key")
        if value in {"|", "|-", "|+", ">", ">-", ">+"}:
            block, index = _read_block_scalar(lines, index + 1, folded=value.startswith(">"))
            fields[key] = block
            continue
        if key == "metadata" and not value:
            mapping, index = _read_indented_mapping(lines, index + 1)
            fields[key] = mapping
            continue
        fields[key] = _unquote(value)
        index += 1
    return fields


def valid_skill_name(name: str) -> bool:
    """Проверяем ограничения имени из спецификации Agent Skills."""

    return bool(NAME_RE.fullmatch(name)) and "--" not in name


def _read_block_scalar(lines: list[str], index: int, *, folded: bool) -> tuple[str, int]:
    """Читаем простейший YAML block scalar, сохраняя достаточную совместимость."""

    collected: list[str] = []
    while index < len(lines):
        raw = lines[index]
        if raw and not raw[:1].isspace():
            break
        collected.append(raw[2:] if raw.startswith("  ") else raw.lstrip())
        index += 1
    if folded:
        return " ".join(line.strip() for line in collected if line.strip()), index
    return "\n".join(collected).strip(), index


def _read_indented_mapping(lines: list[str], index: int) -> tuple[dict[str, str], int]:
    """Читаем `metadata:` как map string->string из вложенных строк."""

    mapping: dict[str, str] = {}
    while index < len(lines):
        raw = lines[index]
        if not raw.strip():
            index += 1
            continue
        if not raw[:1].isspace():
            break
        item = raw.strip()
        if ":" not in item:
            raise ValueError(f"invalid metadata line: {item}")
        key, value = item.split(":", 1)
        mapping[key.strip()] = _unquote(value.strip())
        index += 1
    return mapping, index


def _unquote(value: str) -> str:
    """Снимаем простые кавычки YAML-строки, не пытаясь заменить полноценный YAML."""

    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _scalar(value: Any) -> str:
    """Берём строковое значение frontmatter; сложные типы для scalar-полей игнорируем."""

    return value.strip() if isinstance(value, str) else ""


def _metadata(value: Any) -> dict[str, str]:
    """Нормализуем metadata к string->string, чтобы CLI и трасса были JSON-friendly."""

    if not isinstance(value, dict):
        return {}
    return {str(key): str(item) for key, item in value.items()}
