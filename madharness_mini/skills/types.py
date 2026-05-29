"""Типы данных для найденных и активированных Agent Skills."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SkillDiagnostic:
    """Диагностика discovery: ошибка пропускает skill, предупреждение оставляет его доступным."""

    severity: str
    path: Path
    message: str

    def as_dict(self, workspace_root: Path) -> dict[str, str]:
        """Отдаём диагностику для CLI и трассы без абсолютного шума, если путь внутри workspace."""

        try:
            location = str(self.path.relative_to(workspace_root))
        except ValueError:
            location = str(self.path)
        return {
            "severity": self.severity,
            "path": location,
            "message": self.message,
        }


@dataclass(frozen=True)
class SkillResource:
    """Один bundled-файл навыка, который модель может прочитать по требованию."""

    relative_path: str
    workspace_path: str
    kind: str
    bytes: int

    def as_dict(self) -> dict[str, Any]:
        """Готовим JSON-совместимое описание ресурса для observation и CLI."""

        return {
            "path": self.relative_path,
            "workspace_path": self.workspace_path,
            "kind": self.kind,
            "bytes": self.bytes,
        }


@dataclass(frozen=True)
class Skill:
    """Описание одного валидного `SKILL.md` и его project-local корня."""

    name: str
    description: str
    root: Path
    skill_file: Path
    body: str
    raw_text: str
    source: str
    license: str = ""
    compatibility: str = ""
    metadata: dict[str, str] = field(default_factory=dict)
    allowed_tools: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def location(self, workspace_root: Path) -> str:
        """Показываем путь к `SKILL.md` так, как его сможет использовать модель."""

        try:
            return str(self.skill_file.relative_to(workspace_root))
        except ValueError:
            return str(self.skill_file)

    def root_location(self, workspace_root: Path) -> str:
        """Показываем корень навыка относительно workspace для чтения ресурсов и cwd shell."""

        try:
            return str(self.root.relative_to(workspace_root))
        except ValueError:
            return str(self.root)


@dataclass(frozen=True)
class SkillIndex:
    """Результат discovery: доступные навыки и диагностика сканирования."""

    skills: dict[str, Skill]
    diagnostics: tuple[SkillDiagnostic, ...] = ()

    def names(self) -> list[str]:
        """Возвращаем имена навыков в стабильном порядке для enum и CLI."""

        return sorted(self.skills)
