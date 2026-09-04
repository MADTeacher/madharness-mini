"""Компактный каталог навыков и парсер явного выбора в запросе пользователя."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .types import SkillIndex

MARKER_RE = re.compile(r"@skill[:/]([a-z0-9][a-z0-9-]{0,63})")
DOLLAR_RE = re.compile(r"(?<![\w-])\$([a-z0-9][a-z0-9-]{0,63})(?![\w-])")
PHRASE_RE = re.compile(
    r"(?:используй|использовать|активируй|активировать|use|activate)\s+"
    r"(?:навык|skill)\s+([a-z0-9][a-z0-9-]{0,63})",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ExplicitSkillSelection:
    """Имена навыков, явно запрошенные пользователем и
    неизвестные маркеры навыков."""

    names: tuple[str, ...] = ()
    unknown: tuple[str, ...] = ()

    @property
    def present(self) -> bool:
        """Отличаем явный выбор от обычной задачи без маркеров навыков."""

        return bool(self.names or self.unknown)


def find_explicit_skill_selection(
    task: str,
    available_names: set[str],
) -> ExplicitSkillSelection:
    """Ищем `@skill:name`, `@skill/name`, `$name` и
    фразы вида "используй навык name" в задаче пользователя."""

    requested: list[str] = []
    unknown: list[str] = []
    for regex in (MARKER_RE, PHRASE_RE):
        for match in regex.finditer(task):
            _add_requested(
                match.group(1),
                available_names,
                requested,
                unknown,
            )
    for match in DOLLAR_RE.finditer(task):
        name = match.group(1)
        if name in available_names or "-" in name:
            _add_requested(name, available_names, requested, unknown)
    return ExplicitSkillSelection(
        tuple(requested),
        tuple(unknown),
    )


def render_skill_catalog_for_root(index: SkillIndex, workspace_root) -> str:
    """Формируем каталог навыков относительно workspace 
    без полной загрузки тела навыка."""

    if not index.skills:
        return ""
    # список строк для формирования каталога
    about_skills = """
    # Available Agent Skills

    These project-local skills are available in this `run` session. 
    Use `activate_skill` with an exact `name` to activate a skill when relevant. 
    Do not assume a skill is active until it has been activated.
    
    """
    lines = [about_skills]

    # добавляем навыки в каталог
    for name in index.names():
        skill = index.skills[name]
        lines.append(
            f"- `{skill.name}`: {skill.description} "
            f"(location: `{skill.location(workspace_root)}`)"
        )
    return "\n".join(lines)


def _add_requested(
    name: str,
    available_names: set[str],
    requested: list[str],
    unknown: list[str],
) -> None:
    """Добавляем имя без дублей, разделяя найденные и неизвестные навыки."""

    target = requested if name in available_names else unknown
    if name not in target:
        target.append(name)
