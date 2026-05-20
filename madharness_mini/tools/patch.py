"""Инструмент apply_patch и маленький парсер Codex-style patch."""

from pathlib import Path
from typing import Any

from ..utils import fail, obj, ok, strp
from .context import ToolContext
from .specs import ToolSpec


def apply_patch(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    """Применяем текстовый patch в формате Codex (add/update/delete)."""

    parser = PatchParser(ctx)
    try:
        changes = parser.prepare(args["patch"])
    except ValueError as exc:
        return fail("apply_patch", str(exc))
    for path, content in changes.items():
        if content is None:
            path.unlink()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    return ok("apply_patch", f"applied patch to {len(changes)} file(s)")


class PatchParser:
    """Разбираем patch до набора файловых изменений без записи на диск.

    Инструмент сначала валидирует весь patch и только потом применяет изменения,
    чтобы ошибка в одном hunk не оставила workspace в частично изменённом состоянии.
    """

    def __init__(self, ctx: ToolContext):
        self.ctx = ctx

    def prepare(self, patch: str) -> dict[Path, str | None]:
        """Проверяем patch и готовим карту path -> новое содержимое или удаление."""

        lines = patch.splitlines()
        if not lines or lines[0] != "*** Begin Patch":
            raise ValueError("patch must start with *** Begin Patch")
        if lines[-1] != "*** End Patch":
            raise ValueError("patch must end with *** End Patch")

        changes: dict[Path, str | None] = {}
        i = 1
        while i < len(lines) - 1:
            line = lines[i]
            if line.startswith("*** Add File: "):
                i = self._parse_add_file(lines, i, changes)
            elif line.startswith("*** Update File: "):
                i = self._parse_update_file(lines, i, changes)
            elif line.startswith("*** Delete File: "):
                i = self._parse_delete_file(lines, i, changes)
            elif line.startswith("*** Move to: "):
                raise ValueError("Move to is only supported after Update File")
            elif line == "*** End of File":
                i += 1
            else:
                raise ValueError(f"unexpected patch line: {line}")
        return changes

    def _patch_path(self, raw: str) -> Path:
        path, err = self.ctx.policy.safe_path(raw)
        if err or not path:
            raise ValueError(err or f"invalid path: {raw}")
        return path

    def _parse_add_file(
        self, lines: list[str], i: int, changes: dict[Path, str | None]
    ) -> int:
        raw = lines[i].removeprefix("*** Add File: ")
        path = self._patch_path(raw)
        if path.exists() or path in changes:
            raise ValueError(f"file already exists: {raw}")
        i += 1
        new_lines: list[str] = []
        while i < len(lines) - 1 and not lines[i].startswith("*** "):
            if not lines[i].startswith("+"):
                raise ValueError("add file lines must start with +")
            new_lines.append(lines[i][1:])
            i += 1
        changes[path] = "\n".join(new_lines) + ("\n" if new_lines else "")
        return i

    def _parse_delete_file(
        self, lines: list[str], i: int, changes: dict[Path, str | None]
    ) -> int:
        raw = lines[i].removeprefix("*** Delete File: ")
        path = self._patch_path(raw)
        if path in changes:
            raise ValueError(f"file changed more than once: {raw}")
        if not path.is_file():
            raise ValueError(f"not a file: {raw}")
        changes[path] = None
        return i + 1

    def _parse_update_file(
        self, lines: list[str], i: int, changes: dict[Path, str | None]
    ) -> int:
        raw = lines[i].removeprefix("*** Update File: ")
        path = self._patch_path(raw)
        if path in changes:
            raise ValueError(f"file changed more than once: {raw}")
        if not path.is_file():
            raise ValueError(f"not a file: {raw}")
        i += 1
        target_path = None
        if i < len(lines) - 1 and lines[i].startswith("*** Move to: "):
            target_raw = lines[i].removeprefix("*** Move to: ")
            target_path = self._patch_path(target_raw)
            if target_path.exists() or target_path in changes:
                raise ValueError(f"target file already exists: {target_raw}")
            i += 1

        original = path.read_text(encoding="utf-8")
        has_trailing_newline = original.endswith("\n")
        current = original.splitlines()
        saw_hunk = False
        while i < len(lines) - 1 and not lines[i].startswith("*** "):
            if lines[i].startswith("@@"):
                i += 1
            old_lines: list[str] = []
            new_lines: list[str] = []
            while i < len(lines) - 1 and not (
                lines[i].startswith("@@") or lines[i].startswith("*** ")
            ):
                marker = lines[i][:1]
                content = lines[i][1:]
                if marker == " ":
                    old_lines.append(content)
                    new_lines.append(content)
                elif marker == "-":
                    old_lines.append(content)
                elif marker == "+":
                    new_lines.append(content)
                else:
                    raise ValueError(f"invalid hunk line: {lines[i]}")
                i += 1
            if not old_lines and not new_lines:
                raise ValueError("empty update hunk")
            current = self._apply_hunk(current, old_lines, new_lines)
            saw_hunk = True
        if not saw_hunk and target_path is None:
            raise ValueError(f"update has no hunks: {raw}")
        if saw_hunk:
            updated = "\n".join(current)
            if has_trailing_newline:
                updated += "\n"
        else:
            updated = original
        if target_path is None:
            changes[path] = updated
        else:
            changes[path] = None
            changes[target_path] = updated
        return i

    def _apply_hunk(
        self, current: list[str], old_lines: list[str], new_lines: list[str]
    ) -> list[str]:
        matches = []
        if not old_lines:
            raise ValueError("update hunk must include context or removed lines")
        for start in range(len(current) - len(old_lines) + 1):
            if current[start : start + len(old_lines)] == old_lines:
                matches.append(start)
        if len(matches) != 1:
            raise ValueError(f"expected 1 hunk match, found {len(matches)}")
        start = matches[0]
        end = start + len(old_lines)
        return current[:start] + new_lines + current[end:]


APPLY_PATCH_SPEC = ToolSpec(
    "apply_patch",
    "Apply a small Codex-style patch inside the workspace.",
    obj({"patch": strp(req=True)}, ["patch"]),
    apply_patch,
)
