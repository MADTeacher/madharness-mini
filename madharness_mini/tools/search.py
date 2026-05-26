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


SEARCH_CODE_DESCRIPTION = """Search for a literal substring in workspace files.

Use this to find names, snippets, and exact text before reading or editing. This
is not regex or semantic search. It scans non-ignored files, returns line
numbers with previews, filters file names with glob, and stops after 100 matches.
"""

SEARCH_CODE_SPEC = ToolSpec(
    "search_code",
    SEARCH_CODE_DESCRIPTION,
    obj(
        {
            "query": strp(
                req=True,
                desc="Literal substring to find; not regex and not semantic search.",
            ),
            "glob": strp(
                "*",
                "fnmatch-style pattern matched against file names only; defaults to *",
            ),
        },
        ["query"],
    ),
    search_code,
)
