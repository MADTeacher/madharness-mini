"""Системный промпт и проектные инструкции AGENTS.md для модели."""

from importlib import resources
from pathlib import Path

# Имя файла с правилами проекта в каждой папке workspace.
PROJECT_DOC_FILENAME = "AGENTS.md"
# Суммарный лимит байт при склейке нескольких AGENTS.md (как у Codex).
PROJECT_DOC_MAX_BYTES = 32 * 1024


def load_prompt(name: str) -> str:
    """Берём встроенный markdown из madharness_mini/prompts/{name}.md.

    Сейчас используется как минимум `system` — базовые правила агента.
    """

    path = resources.files("madharness_mini").joinpath("prompts", f"{name}.md")
    return path.read_text(encoding="utf-8").rstrip()


def project_instruction_dirs(root: Path, cwd: Path) -> list[Path]:
    """Список папок от корня workspace до текущей cwd для поиска AGENTS.md.

    Сначала корень, потом каждый уровень вниз — чтобы общие правила шли
    раньше локальных. Если cwd вне root, возвращаем только root.
    """

    root = root.resolve()
    cwd = cwd.resolve()
    try:
        rel = cwd.relative_to(root)
    except ValueError:
        return [root]
    dirs = [root]
    current = root
    for part in rel.parts:
        current = current / part
        dirs.append(current)
    return dirs


def load_project_instructions(cfg: object) -> str:
    """Склеиваем AGENTS.md по цепочке папок с лимитом PROJECT_DOC_MAX_BYTES.

    Пустые файлы пропускаем. При переполнении обрезаем последний кусок —
    модель всё равно получит верхнеуровневые правила проекта.
    """

    combined = bytearray()
    root = Path(getattr(cfg, "root"))
    cwd = Path(getattr(cfg, "cwd"))
    for directory in project_instruction_dirs(root, cwd):
        path = directory / PROJECT_DOC_FILENAME
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace").rstrip()
        if not text.strip():
            continue
        data = text.encode("utf-8")
        prefix = b"\n\n" if combined else b""
        chunk = prefix + data
        remaining = PROJECT_DOC_MAX_BYTES - len(combined)
        if remaining <= 0:
            break
        if len(chunk) > remaining:
            combined.extend(chunk[:remaining])
            break
        combined.extend(chunk)
    return bytes(combined).decode("utf-8", errors="ignore").rstrip()
