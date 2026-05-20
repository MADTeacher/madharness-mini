"""Файловые инструменты: список, чтение, запись и точная замена текста."""

import fnmatch
from typing import Any

from ..utils import clipped, fail, ignored, intp, obj, ok, strp
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

    path, err = ctx.policy.safe_path(args["path"])
    if err or not path:
        return fail("write_file", err or f"invalid path: {args['path']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    content = args["content"]
    path.write_text(content, encoding="utf-8")
    return ok("write_file", f"wrote {args['path']}", bytes=len(content.encode("utf-8")))


def replace_text(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    """Заменяем old на new ровно expected_replacements раз в одном файле."""

    path, err = ctx.policy.safe_path(args["path"])
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


LIST_FILES_SPEC = ToolSpec(
    "list_files",
    "List workspace files.",
    obj({"path": strp(".", "directory"), "glob": strp("*", "glob")}),
    list_files,
)

READ_FILE_SPEC = ToolSpec(
    "read_file",
    "Read a file excerpt.",
    obj({"path": strp(req=True), "start": intp(1), "end": intp(160)}, ["path"]),
    read_file,
)

WRITE_FILE_SPEC = ToolSpec(
    "write_file",
    "Write a UTF-8 text file inside the workspace.",
    obj({"path": strp(req=True), "content": strp(req=True)}, ["path", "content"]),
    write_file,
)

REPLACE_TEXT_SPEC = ToolSpec(
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
    replace_text,
)
