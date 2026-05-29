"""Типы данных для markdown-субагентов и диагностики discovery."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SubagentDiagnostic:
    """Диагностика discovery: ошибка пропускает файл, warning оставляет каталог живым."""

    severity: str
    path: str
    message: str

    def as_dict(self) -> dict[str, str]:
        """Отдаём диагностику в JSON-friendly виде для trace и CLI."""

        return {
            "severity": self.severity,
            "path": self.path,
            "message": self.message,
        }


@dataclass(frozen=True)
class Subagent:
    """Один markdown-субагент: конфигурация из frontmatter и prompt из body."""

    name: str
    description: str
    profile: str
    tools: tuple[str, ...]
    prompt: str
    source: str
    location: str
    max_turns: int | None = None
    context_max_tokens: int | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    builtin: bool = False
    override: bool = False


@dataclass(frozen=True)
class SubagentIndex:
    """Результат discovery: доступные субагенты и диагностика загрузки."""

    subagents: dict[str, Subagent]
    diagnostics: tuple[SubagentDiagnostic, ...] = ()

    def names(self) -> list[str]:
        """Возвращаем имена субагентов в стабильном порядке для enum и CLI."""

        return sorted(self.subagents)
