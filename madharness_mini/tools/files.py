"""Файловые инструменты: список, чтение и запись файлов workspace."""

import fnmatch
from typing import Any

from ..utils import clipped, fail, ignored, obj, ok, strp
from .context import ToolContext
from .specs import ToolSpec


def list_files(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    """Список файлов под path с фильтром glob по имени (до 200 штук)."""

    base, err = ctx.policy.safe_path(args.get("path", "."))
    if err:
        return fail("list_files", err)
    pattern = args.get("glob", "*")
    results: list[str] = []
    source = base if base and base.exists() else ctx.cfg.root
    paths = source.rglob("*") if source.is_dir() else [source]
    for path in paths:
        if (
            path.is_file()
            and not ignored(path)
            and fnmatch.fnmatch(path.name, pattern)
        ):
            results.append(str(path.relative_to(ctx.cfg.root)))
        if len(results) >= 200:
            return ok("list_files", "listed 200 files", files=results, truncated=True)
    return ok(
        "list_files", f"listed {len(results)} files", files=results, truncated=False
    )


def read_file(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    """Читаем фрагмент UTF-8 файла: строки start..end с номерами."""

    path, err = ctx.policy.safe_path(args["path"])
    if err:
        return fail("read_file", err)
    if not path or not path.is_file():
        return fail("read_file", f"not a file: {args['path']}")
    if ctx.trace and ctx.skill_runtime:
        event = ctx.skill_runtime.resource_event(path)
        if event:
            ctx.trace.write("skill_resource_used", tool="read_file", **event)
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


def write_file(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    """Полностью перезаписываем файл в workspace; каталоги создаём сами."""

    scope_error = ctx.write_path_error(args["path"])
    if scope_error:
        return fail("write_file", scope_error)
    path, err = ctx.policy.safe_path(args["path"])
    if err or not path:
        return fail("write_file", err or f"invalid path: {args['path']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    content = args["content"]
    path.write_text(content, encoding="utf-8")
    return ok("write_file", f"wrote {args['path']}", bytes=len(content.encode("utf-8")))

LIST_FILES_DESCRIPTION = """Recursively list files inside the workspace.

Use this to discover repository structure before reading files. Results include
files only, skip ignored folders such as .git and caches, and stop after 200
matches. The glob filter matches each file name, not the full relative path.
"""

READ_FILE_DESCRIPTION = """Read a UTF-8 file excerpt from the workspace.

Use this before editing a file. start and end are 1-based line numbers, and the
returned content includes numbered lines like "12: text" so later edits can
refer to the exact surrounding text.
"""

WRITE_FILE_DESCRIPTION = """Write a complete UTF-8 text file inside the workspace.

This creates parent directories as needed and fully overwrites the target file.
Prefer apply_patch for precise edits to existing files. Do not use write_file as
the fallback for a failed precise edit unless you intentionally need a full-file
rewrite.
"""

LIST_FILES_SPEC = ToolSpec(
    "list_files",
    LIST_FILES_DESCRIPTION,
    obj(
        {
            "path": strp(
                ".",
                "Workspace-relative directory or file to inspect; defaults to .",
            ),
            "glob": strp(
                "*",
                "fnmatch-style pattern matched against file names only; defaults to *",
            ),
        }
    ),
    list_files,
)

READ_FILE_SPEC = ToolSpec(
    "read_file",
    READ_FILE_DESCRIPTION,
    obj(
        {
            "path": strp(req=True, desc="Workspace-relative file path to read."),
            "start": {
                "type": "integer",
                "default": 1,
                "description": "1-based first line number to include; defaults to 1.",
            },
            "end": {
                "type": "integer",
                "default": 160,
                "description": "1-based last line number to include; defaults to start + 160.",
            },
        },
        ["path"],
    ),
    read_file,
)

WRITE_FILE_SPEC = ToolSpec(
    "write_file",
    WRITE_FILE_DESCRIPTION,
    obj(
        {
            "path": strp(
                req=True,
                desc="Workspace-relative file path to create or intentionally fully overwrite.",
            ),
            "content": strp(
                req=True,
                desc="Complete UTF-8 file content to write, including final newline if wanted.",
            ),
        },
        ["path", "content"],
    ),
    write_file,
)
