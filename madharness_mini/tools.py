"""Локальные инструменты агентского режима `run`."""

import fnmatch
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .config import Config
from .policy import Policy
from .utils import clipped, fail, ignored, intp, obj, ok, strp


@dataclass
class ToolSpec:
    """Схема инструмента и обработчик его вызова."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any]], dict[str, Any]]

    def schema(self) -> dict[str, Any]:
        """Вернуть описание инструмента в формате Chat Completions."""

        function = {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }
        return {"type": "function", "function": function}


class ToolRegistry:
    """Реестр инструментов, доступных агенту.

    Методы класса выполняют файловые операции, поиск и разрешённые команды
    оболочки внутри workspace. Каждый вызов возвращает JSON-наблюдение единого
    вида, пригодное для отправки обратно модели.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.policy = Policy(cfg)
        self.tools = {
            "list_files": ToolSpec(
                "list_files",
                "List workspace files.",
                obj({"path": strp(".", "directory"), "glob": strp("*", "glob")}),
                self.list_files,
            ),
            "read_file": ToolSpec(
                "read_file",
                "Read a file excerpt.",
                obj(
                    {"path": strp(req=True), "start": intp(1), "end": intp(160)},
                    ["path"],
                ),
                self.read_file,
            ),
            "write_file": ToolSpec(
                "write_file",
                "Write a UTF-8 text file inside the workspace.",
                obj(
                    {"path": strp(req=True), "content": strp(req=True)},
                    ["path", "content"],
                ),
                self.write_file,
            ),
            "replace_text": ToolSpec(
                "replace_text",
                "Replace exact text in a UTF-8 file inside the workspace.",
                obj(
                    {
                        "path": strp(req=True),
                        "old": strp(req=True),
                        "new": strp(req=True),
                        "expected_replacements": intp(1),
                    },
                    ["path", "old", "new"],
                ),
                self.replace_text,
            ),
            "apply_patch": ToolSpec(
                "apply_patch",
                "Apply a small Codex-style patch inside the workspace.",
                obj({"patch": strp(req=True)}, ["patch"]),
                self.apply_patch,
            ),
            "search_code": ToolSpec(
                "search_code",
                "Search text in workspace files.",
                obj({"query": strp(req=True), "glob": strp("*")}, ["query"]),
                self.search_code,
            ),
            "run_shell": ToolSpec(
                "run_shell",
                "Run a safe command in the workspace.",
                obj({"command": strp(req=True)}, ["command"]),
                self.run_shell,
            ),
        }

    def schemas(self) -> list[dict[str, Any]]:
        """Вернуть схемы всех инструментов для передачи модели."""

        return [tool.schema() for tool in self.tools.values()]

    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Выполнить инструмент и оформить ошибки как наблюдение `fail`."""

        tool = self.tools.get(name)
        if not tool:
            return fail(name, "unknown tool")
        try:
            return tool.handler(args)
        except Exception as exc:
            return fail(name, f"{type(exc).__name__}: {exc}")

    def list_files(self, args: dict[str, Any]) -> dict[str, Any]:
        """Найти файлы внутри workspace по простому glob-фильтру имени."""

        base, err = self.policy.safe_path(args.get("path", "."))
        if err:
            return fail("list_files", err)
        pattern = args.get("glob", "*")
        results: list[str] = []
        source = base if base and base.exists() else self.cfg.root
        paths = source.rglob("*") if source.is_dir() else [source]
        for path in paths:
            if (
                path.is_file()
                and not ignored(path)
                and fnmatch.fnmatch(path.name, pattern)
            ):
                results.append(str(path.relative_to(self.cfg.root)))
            if len(results) >= 200:
                return ok(
                    "list_files", "listed 200 files", files=results, truncated=True
                )
        return ok(
            "list_files", f"listed {len(results)} files", files=results, truncated=False
        )

    def read_file(self, args: dict[str, Any]) -> dict[str, Any]:
        """Вернуть диапазон строк UTF-8 файла с номерами строк."""

        path, err = self.policy.safe_path(args["path"])
        if err:
            return fail("read_file", err)
        if not path or not path.is_file():
            return fail("read_file", f"not a file: {args['path']}")
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(int(args.get("start", 1)), 1)
        end = min(int(args.get("end", start + 160)), len(lines))
        excerpt = "\n".join(f"{i}: {lines[i - 1]}" for i in range(start, end + 1))
        return ok(
            "read_file",
            f"read {args['path']}:{start}-{end}",
            content=clipped(excerpt),
            start=start,
            end=end,
        )

    def write_file(self, args: dict[str, Any]) -> dict[str, Any]:
        """Записать UTF-8 текст в workspace, создав родительские каталоги."""

        path, err = self.policy.safe_path(args["path"])
        if err or not path:
            return fail("write_file", err or f"invalid path: {args['path']}")
        path.parent.mkdir(parents=True, exist_ok=True)
        content = args["content"]
        path.write_text(content, encoding="utf-8")
        return ok(
            "write_file", f"wrote {args['path']}", bytes=len(content.encode("utf-8"))
        )

    def replace_text(self, args: dict[str, Any]) -> dict[str, Any]:
        """Точно заменить текст в UTF-8 файле без полной перезаписи моделью."""

        path, err = self.policy.safe_path(args["path"])
        if err or not path:
            return fail("replace_text", err or f"invalid path: {args['path']}")
        if not path.is_file():
            return fail("replace_text", f"not a file: {args['path']}")
        old = args["old"]
        if old == "":
            return fail("replace_text", "old text must not be empty")
        expected = int(args.get("expected_replacements", 1))
        if expected < 1:
            return fail("replace_text", "expected_replacements must be at least 1")
        text = path.read_text(encoding="utf-8")
        count = text.count(old)
        if count != expected:
            return fail(
                "replace_text",
                f"expected {expected} replacements, found {count}",
                replacements=count,
            )
        updated = text.replace(old, args["new"])
        path.write_text(updated, encoding="utf-8")
        return ok(
            "replace_text",
            f"replaced {count} occurrence(s) in {args['path']}",
            replacements=count,
        )

    def apply_patch(self, args: dict[str, Any]) -> dict[str, Any]:
        """Применить небольшой Codex-style patch без внешних утилит."""

        try:
            changes = self._prepare_patch(args["patch"])
        except ValueError as exc:
            return fail("apply_patch", str(exc))
        for path, content in changes.items():
            if content is None:
                path.unlink()
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
        return ok("apply_patch", f"applied patch to {len(changes)} file(s)")

    def _prepare_patch(self, patch: str) -> dict[Path, str | None]:
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
        path, err = self.policy.safe_path(raw)
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

    def search_code(self, args: dict[str, Any]) -> dict[str, Any]:
        """Найти строки файлов workspace, содержащие заданный текст."""

        query = args["query"]
        pattern = args.get("glob", "*")
        matches = []
        for path in self.cfg.root.rglob("*"):
            if (
                ignored(path)
                or not path.is_file()
                or not fnmatch.fnmatch(path.name, pattern)
            ):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for no, line in enumerate(text.splitlines(), 1):
                if query in line:
                    item = {
                        "path": str(path.relative_to(self.cfg.root)),
                        "line": no,
                        "preview": line.strip()[:240],
                    }
                    matches.append(item)
                    if len(matches) >= 100:
                        return ok(
                            "search_code",
                            "found 100 matches",
                            results=matches,
                            truncated=True,
                        )
        return ok(
            "search_code",
            f"found {len(matches)} matches",
            results=matches,
            truncated=False,
        )

    def run_shell(self, args: dict[str, Any]) -> dict[str, Any]:
        """Запустить разрешённую команду оболочки в корне workspace."""

        command = args["command"]
        allowed, reason = self.policy.shell_allowed(command)
        if not allowed:
            return fail("run_shell", reason, command=command)
        proc = subprocess.run(
            shlex.split(command),
            cwd=self.cfg.root,
            text=True,
            capture_output=True,
            timeout=60,
        )
        return ok(
            "run_shell",
            f"exit code {proc.returncode}",
            command=command,
            returncode=proc.returncode,
            stdout=clipped(proc.stdout),
            stderr=clipped(proc.stderr),
        )
