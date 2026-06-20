"""Провайдер карты проекта на старте сессии (паттерн «Репозиторий»).

Модель лучше ориентируется, если в начале run видит структуру проекта. Карту
собираем через существующий ContextProvider, но сам провайдер живёт верхним
модулем harness, а не в пакете context: тот не должен знать про Config и обход
файловой системы. Обход строго ограничен (глубина, число записей) и уважает
границы политики: пропускает ignored() и пути, чьё имя совпадает с элементом
protected_paths (.git, .env, secrets, ~/.ssh).
"""

from __future__ import annotations

from pathlib import Path

from .config import Config
from .context.fragments import ContextFragment, ContextState
from .utils import ignored

# Идентификатор единственного фрагмента карты проекта.
WORKSPACE_MAP_ID = "workspace-map"


class WorkspaceMapProvider:
    """Один фрагмент с ограниченным деревом каталогов проекта.

    Обход FS выполняется один раз при инициализации (старт сессии): дерево не
    должно меняться каждый ход и нагружать файловую систему. Результат хранится
    в self._text; collect отдаёт его как единственный фрагмент или пустой список.
    """

    def __init__(self, cfg: Config, depth: int, max_entries: int):
        self.cfg = cfg
        self._text = self._build_text(cfg, max(int(depth), 0), max(int(max_entries), 0))

    def _build_text(self, cfg: Config, depth: int, max_entries: int) -> str:
        """Строим текст карты одним ограниченным обходом (FL4).

        Любую ошибку обхода перехватываем и оставляем частичный/пустой текст:
        провайдер карты не должен ронять сессию.
        """

        if max_entries <= 0:
            return ""
        protected_names = _protected_names(cfg)
        lines: list[str] = []
        count = 0
        try:
            for path in sorted(cfg.root.rglob("*")):
                if ignored(path):
                    continue
                rel = path.relative_to(cfg.root)
                if len(rel.parts) > depth:
                    continue
                # Имя любой части пути в protected_paths — пропускаем целиком,
                # как это делает policy.safe_path для файловых инструментов.
                if any(part in protected_names for part in rel.parts):
                    continue
                indent = "  " * (len(rel.parts) - 1)
                suffix = "/" if path.is_dir() else ""
                lines.append(f"{indent}{rel.parts[-1]}{suffix}")
                count += 1
                if count >= max_entries:
                    lines.append("...truncated")
                    break
        except OSError:
            # Частичная карта лучше падения: отдаём то, что успели собрать.
            pass
        if not lines:
            return ""
        return "# Карта проекта\n" + "\n".join(lines)

    def collect(self, state: ContextState) -> list[ContextFragment]:
        """Возвращаем один фрагмент карты или пустой список, если карта пуста."""

        if not self._text:
            return []
        return [
            ContextFragment(
                id=WORKSPACE_MAP_ID,
                source="madharness-mini workspace map",
                text=self._text,
                priority=5,
                placement="system",
                transient=False,
                authority_level="harness",
                context_layer="evidence",
                evictability="normal",
                stability="session",
                applicability="current_project",
            )
        ]


def _protected_names(cfg: Config) -> set[str]:
    """Имена protected_paths для сравнения с частями относительного пути.

    Берём последний сегмент каждого элемента, как policy.safe_path: достаточно
    совпадения по имени каталога/файла, чтобы не подмешивать его в карту.
    """

    names: set[str] = set()
    for item in cfg.data.get("protected_paths", []) or []:
        name = Path(str(item)).expanduser().name or str(item).strip("/").split("/")[-1]
        if name:
            names.add(name)
    return names
