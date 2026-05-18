"""Локальные инструменты агентского режима `run`."""

import fnmatch
import shlex
from dataclasses import dataclass
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
