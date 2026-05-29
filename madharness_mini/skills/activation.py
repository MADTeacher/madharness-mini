"""Активация навыков: durable context-фрагмент и список bundled resources."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import Config
from ..context import ContextFragment
from ..utils import ok
from .types import Skill, SkillIndex, SkillResource

# Не даём каталогу ресурсов разрастись в tool observation.
MAX_LISTED_RESOURCES = 200


class SkillRuntime:
    """Состояние навыков одного `run`: что найдено и что уже активировано."""

    def __init__(self, cfg: Config, index: SkillIndex):
        self.cfg = cfg
        self.index = index
        self.active_names: set[str] = set()

    def activate(self, name: str, trigger: str) -> dict[str, Any]:
        """Активируем навык один раз и возвращаем observation с hidden context fragment."""

        skill = self.index.skills.get(name)
        if skill is None:
            return {
                "ok": False,
                "tool": "activate_skill",
                "summary": f"unknown skill: {name}",
                "name": name,
            }
        resources = list_skill_resources(skill, self.cfg.root)
        resource_dicts = [item.as_dict() for item in resources]
        if name in self.active_names:
            return ok(
                "activate_skill",
                f"skill already active: {name}",
                name=name,
                already_active=True,
                skill_root=skill.root_location(self.cfg.root),
                resources=resource_dicts,
            )

        self.active_names.add(name)
        fragment = activation_fragment(skill, self.cfg.root, resources)
        observation = ok(
            "activate_skill",
            f"activated skill: {name}; instructions were added to durable context",
            name=name,
            already_active=False,
            skill_root=skill.root_location(self.cfg.root),
            location=skill.location(self.cfg.root),
            resources=resource_dicts,
        )
        observation["_context_fragments"] = [fragment]
        observation["_skill_event"] = {
            "name": name,
            "trigger": trigger,
            "location": skill.location(self.cfg.root),
            "resources": len(resources),
        }
        return observation

    def resource_event(self, path: Path) -> dict[str, str] | None:
        """Если путь лежит внутри активного skill root, готовим событие трассы."""

        resolved = path.resolve()
        for name in sorted(self.active_names):
            skill = self.index.skills.get(name)
            if not skill:
                continue
            try:
                relative = resolved.relative_to(skill.root)
            except ValueError:
                continue
            if not relative.parts:
                return {
                    "name": name,
                    "path": ".",
                    "skill_root": skill.root_location(self.cfg.root),
                }
            if relative.parts[0] != "SKILL.md":
                return {
                    "name": name,
                    "path": str(relative),
                    "skill_root": skill.root_location(self.cfg.root),
                }
        return None


def activation_fragment(
    skill: Skill,
    workspace_root: Path,
    resources: list[SkillResource] | None = None,
) -> ContextFragment:
    """Упаковываем инструкции навыка в системный фрагмент, защищённый от compaction."""

    resource_lines = _resource_lines(resources or list_skill_resources(skill, workspace_root))
    parts = [
        f"# Active Agent Skill: {skill.name}",
        "",
        f"Location: `{skill.location(workspace_root)}`",
        f"Skill root: `{skill.root_location(workspace_root)}`",
        f"Description: {skill.description}",
    ]
    if skill.compatibility:
        parts.append(f"Compatibility: {skill.compatibility}")
    if skill.license:
        parts.append(f"License: {skill.license}")
    if skill.allowed_tools:
        parts.append(
            "Allowed tools (experimental; global harness policy still applies): "
            + " ".join(skill.allowed_tools)
        )
    if skill.metadata:
        metadata = ", ".join(f"{key}={value}" for key, value in sorted(skill.metadata.items()))
        parts.append(f"Metadata: {metadata}")
    parts.extend(
        [
            "",
            "Follow these skill instructions when they are relevant to the user's task.",
            "",
            "## Instructions",
            "",
            skill.body or "(SKILL.md body is empty.)",
            "",
            "## Bundled resources",
            "",
            resource_lines or "No bundled resources were found.",
        ]
    )
    return ContextFragment(
        id=f"skill:{skill.name}",
        source=skill.location(workspace_root),
        text="\n".join(parts),
        priority=20,
        placement="system",
        transient=False,
    )


def list_skill_resources(skill: Skill, workspace_root: Path) -> list[SkillResource]:
    """Перечисляем файлы внутри skill root, не раскрывая их содержимое."""

    resources: list[SkillResource] = []
    root = skill.root.resolve()
    for path in sorted(root.rglob("*")):
        if len(resources) >= MAX_LISTED_RESOURCES:
            break
        if path.name == "SKILL.md" or not path.is_file():
            continue
        try:
            resolved = path.resolve()
            relative = resolved.relative_to(root)
        except ValueError:
            continue
        kind = relative.parts[0] if relative.parts else "file"
        try:
            workspace_path = str(resolved.relative_to(workspace_root))
        except ValueError:
            continue
        resources.append(
            SkillResource(
                relative_path=str(relative),
                workspace_path=workspace_path,
                kind=kind,
                bytes=resolved.stat().st_size,
            )
        )
    return resources


def _resource_lines(resources: list[SkillResource]) -> str:
    """Форматируем bundled resources как короткий список для системного фрагмента."""

    lines = []
    for item in resources:
        lines.append(
            f"- `{item.relative_path}` ({item.kind}, {item.bytes} bytes; "
            f"workspace path: `{item.workspace_path}`)"
        )
    return "\n".join(lines)
