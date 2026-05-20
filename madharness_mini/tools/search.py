"""Инструмент поиска точного текста в файлах workspace."""

import fnmatch
from typing import Any

from ..utils import ignored, obj, ok, strp
from .context import ToolContext
from .specs import ToolSpec


def search_code(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    """Ищем подстроку query в файлах workspace (до 100 совпадений)."""

    query = args["query"]
    pattern = args.get("glob", "*")
    matches = []
    for path in ctx.cfg.root.rglob("*"):
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
                    "path": str(path.relative_to(ctx.cfg.root)),
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


SEARCH_CODE_SPEC = ToolSpec(
    "search_code",
    "Search text in workspace files.",
    obj({"query": strp(req=True), "glob": strp("*")}, ["query"]),
    search_code,
)
