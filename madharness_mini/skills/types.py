"""Типы данных для найденных и активированных Agent Skills."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SkillDiagnostic:
    """Состояние диагностики навыка: ошибка пропускает skill, 
    предупреждение оставляет его доступным для использования."""

    severity: str
    path: Path
    message: str

    def as_dict(self, workspace_root: Path) -> dict[str, str]:
        """Отдаем диагностику для CLI и трассы """

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
    """Один bundled-файл навыка, который модель может прочитать 
    по требованию из workspace."""

    relative_path: str
    workspace_path: str
    kind: str
    bytes: int

    def as_dict(self) -> dict[str, Any]:
        """Готовим JSON-совместимое описание ресурса для observation 
        и CLI. Название ресурса (kind) указывает на тип файла."""

        return {
            "path": self.relative_path,
            "workspace_path": self.workspace_path,
            "kind": self.kind,
            "bytes": self.bytes,
        }


@dataclass(frozen=True)
class Skill:
    """Описание одного валидного навыка"""

    name: str # имя навыка (например, `docs-writer`)

    # короткое описание, по которому модель потом решает, 
    # нужен навык или нет
    description: str 
    root: Path # корневая директория навыка
    skill_file: Path # путь к файлу навыка (например, `SKILL.md`)
    body: str # тело навыка (без форматтера)
    raw_text: str # исходный текст навыка (с форматтером и телом)

    # откуда навык: native (из .madharness_mini/skills) или 
    # agents (из .agents/skills)
    source: str 
    license: str = "" # лицензия навыка (например, `MIT`)
    compatibility: str = "" # требования окружения (например, `python 3.10`)
    # метаданные навыка (например, `{"author": "Stasko", "version": "1.0"}`)
    metadata: dict[str, str] = field(default_factory=dict)
    # кортеж инструментов, которые может использовать навык
    allowed_tools: tuple[str, ...] = ()
    # кортеж предупреждений (например, о несовпадении имени с папкой)
    warnings: tuple[str, ...] = ()

    def location(self, workspace_root: Path) -> str:
        """Показываем путь к SKILL.md навыка относительно workspace."""

        try:
            return str(self.skill_file.relative_to(workspace_root))
        except ValueError:
            return str(self.skill_file)

    def root_location(self, workspace_root: Path) -> str:
        """Показываем корневую директорию навыка относительно workspace 
        для чтения ресурсов и вызова скриптов с помощью shell"""

        try:
            return str(self.root.relative_to(workspace_root))
        except ValueError:
            return str(self.root)


@dataclass(frozen=True)
class SkillIndex:
    """Доступные навыки и диагностика их обнаружения."""

    skills: dict[str, Skill] # словарь навыков по именам
    diagnostics: tuple[SkillDiagnostic, ...] = () # кортеж диагностик

    def names(self) -> list[str]:
        """Возвращаем список имен навыков в отсортированном порядке"""

        return sorted(self.skills)
