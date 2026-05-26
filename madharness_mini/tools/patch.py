"""Инструмент apply_patch и маленький парсер Codex-style patch."""

from pathlib import Path
from typing import Any

from ..utils import fail, obj, ok, strp
from .context import ToolContext
from .specs import ToolSpec


def patch_failure_data(summary: str) -> dict[str, Any]:
    """Подсказываем модели безопасный следующий шаг после неудачного patch.

    Summary остаётся коротким и совместимым с существующими тестами, а hint
    объясняет, как восстановиться без перехода к shell-скриптам или полной
    перезаписи файла.
    """

    if summary == "expected 1 hunk match, found 0":
        return {
            "hint": (
                "The update hunk did not match the current file. Use read_file or "
                "search_code to reread the exact region, then retry apply_patch with "
                "verbatim current context lines, including spaces."
            ),
            "retryable": True,
        }
    if summary.startswith("expected 1 hunk match, found "):
        return {
            "hint": (
                "The update hunk matched more than one place. Add more surrounding "
                "context lines copied exactly from the current file, then retry "
                "apply_patch."
            ),
            "retryable": True,
        }
    if summary == "invalid hunk line: ":
        return {
            "hint": (
                "The update hunk contains a blank line without a marker. Blank "
                "context lines must still start with one leading space. Reread the "
                "file region and retry apply_patch with exact context markers."
            ),
            "retryable": True,
        }
    if (
        summary.startswith("patch must ")
        or summary.startswith("unexpected patch line")
        or summary.startswith("invalid hunk line")
        or summary.startswith("add file lines must start")
        or summary == "Move to is only supported after Update File"
    ):
        return {
            "hint": (
                "Send only the patch text, starting with *** Begin Patch and ending "
                "with *** End Patch. Do not wrap it in a shell command, Markdown "
                "fence, or extra prose."
            ),
            "retryable": True,
        }
    if summary == "update hunk must include context or removed lines":
        return {
            "hint": (
                "An update hunk needs at least one current context or removed line. "
                "Use read_file to copy exact nearby lines, then retry apply_patch."
            ),
            "retryable": True,
        }
    return {}


def apply_patch(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    """Применяем текстовый patch в формате Codex (add/update/delete)."""

    parser = PatchParser(ctx)
    try:
        changes = parser.prepare(args["patch"])
    except ValueError as exc:
        summary = str(exc)
        return fail("apply_patch", summary, **patch_failure_data(summary))
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


APPLY_PATCH_DESCRIPTION = """Apply a strict Codex-style patch inside the workspace.

Use this for precise edits to existing files, file creation, deletion, and moves.
The patch argument is one multiline string, not a shell command or JSON object.
The parser is strict: keep markers exactly as shown and include enough context
for each update hunk to match exactly one place.
If apply_patch fails, use read_file or search_code to get exact current file text
and retry with verbatim context. Do not switch to write_file or run_shell scripts
for precise edits.
"""


PATCH_ARGUMENT_DESCRIPTION = """Strict Codex-style patch text.

Required shape:
*** Begin Patch
*** Update File: path
@@
 context line begins with one space
-removed line begins with minus
+added line begins with plus
*** End Patch

Supported file operations:
*** Add File: path       then every content line must start with +
*** Update File: path    then one or more @@ hunks, or optional Move to
*** Delete File: path
*** Move to: path        only immediately after Update File

Compact valid example:
*** Begin Patch
*** Update File: hello.txt
@@
 old context
-old text
+new text
 next context
*** End Patch

On failure: reread the current file region with read_file/search_code, copy exact
current lines into the hunk, and retry apply_patch once.
"""

APPLY_PATCH_SPEC = ToolSpec(
    "apply_patch",
    APPLY_PATCH_DESCRIPTION,
    obj({"patch": strp(req=True, desc=PATCH_ARGUMENT_DESCRIPTION)}, ["patch"]),
    apply_patch,
)
