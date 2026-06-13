"""Системный промпт и проектные инструкции AGENTS.md для модели."""

from importlib import resources
from pathlib import Path

# Имя файла с правилами проекта в корне workspace.
PROJECT_DOC_FILENAME = "AGENTS.md"
# Лимит байт для корневого файла, чтобы не раздуть system prompt.
PROJECT_DOC_MAX_BYTES = 32 * 1024


def load_prompt(name: str) -> str:
    """Берём встроенный markdown из madharness_mini/prompts/{name}.md.

    Сейчас используется как минимум `system` — базовые правила агента.
    """

    path = resources.files("madharness_mini").joinpath("prompts", f"{name}.md")
    return path.read_text(encoding="utf-8").rstrip()


def load_project_instructions(cfg: object) -> str:
    """Читаем корневой AGENTS.md workspace с лимитом PROJECT_DOC_MAX_BYTES.

    Учебная базовая версия не ищет вложенные AGENTS.md: модель получает только
    устойчивые правила всего проекта. Пустой файл пропускаем, слишком длинный
    текст обрезаем до лимита.
    """

    root = Path(getattr(cfg, "root"))
    path = root / PROJECT_DOC_FILENAME
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace").rstrip()
    if not text.strip():
        return ""
    data = text.encode("utf-8")
    if len(data) > PROJECT_DOC_MAX_BYTES:
        data = data[:PROJECT_DOC_MAX_BYTES]
    return data.decode("utf-8", errors="ignore").rstrip()
