"""Инструмент apply_patch и маленький парсер Codex-style patch."""

from pathlib import Path
from typing import Any

from ..utils import clipped, fail, obj, ok, strp
from .context import ToolContext
from .specs import ToolSpec

# Сколько символов актуального файла отдавать модели при несовпадении hunks'а.
# Окно достаточно, чтобы покрыть типичную правку с контекстом, но не раздувает
# observation на больших файлах: дальше контекстный слой всё равно обрежет
# tool-сообщение через clip_tool_content.
MISMATCH_EXCERPT_LIMIT = 4000

# Полуширина окна строк вокруг точки несовпадения: показываем ~10 строк до и
# после, чтобы модель увидела достаточный контекст для rebuilding hunks'а.
MISMATCH_EXCERPT_WINDOW = 10

# Максимум символов проблемной строки hunk'а в observation. Обычно это одна
# короткая строка (часто пустая), но ограничиваем на случай длинного мусора,
# чтобы observation не раздувалось.
BAD_LINE_LIMIT = 200


def patch_failure_data(summary: str) -> dict[str, Any]:
    """Подсказываем модели безопасный следующий шаг после неудачного patch.

    Summary остаётся коротким и совместимым с существующими тестами, а hint
    объясняет, как восстановиться без перехода к shell-скриптам или полной
    перезаписи файла.
    """

    if summary == "expected 1 hunk match, found 0":
        return {
            "hint": (
                "The update hunk did not match the current file. The current file "
                "region is attached as current_excerpt — copy the exact lines from "
                "it (with their surrounding context) and rebuild the hunk, then "
                "retry apply_patch. Do not fall back to write_file."
            ),
            "retryable": True,
        }
    if summary.startswith("expected 1 hunk match, found "):
        return {
            "hint": (
                "The update hunk matched more than one place. The current file "
                "region is attached as current_excerpt — add more surrounding "
                "context lines copied from it to make the match unique, then "
                "retry apply_patch. Do not fall back to write_file."
            ),
            "retryable": True,
        }
    # «invalid hunk line» теперь поднимается как PatchHunkLine и ловится отдельным
    # except в apply_patch, поэтому сюда не доходит — hint живёт в _hunk_line_hint.
    if (
        summary.startswith("patch must ")
        or summary.startswith("unexpected patch line")
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

    # Защищаемся от невалидного вызова без тела patch: иначе args["patch"]
    # ронял handler сырым KeyError, и модель получала техническое сообщение
    # вместо подсказки повторить с корректным аргументом.
    patch = args.get("patch")
    if not isinstance(patch, str) or not patch.strip():
        return fail(
            "apply_patch",
            "missing required argument: patch",
            hint=(
                "Send the patch text in the `patch` argument as one string, "
                "starting with *** Begin Patch and ending with *** End Patch."
            ),
            retryable=True,
        )
    parser = PatchParser(ctx)
    try:
        changes = parser.prepare(patch)
    except PatchMalformed as exc:
        # Diagnostic: показываем модели её собственный край патча, чтобы она
        # увидела, где потеряла структуру (оборванный hunk, dangling '+}' и т.п.),
        # а не общий «must end with *** End Patch».
        return fail(
            "apply_patch",
            str(exc),
            hint=_malformed_hint(exc),
            retryable=True,
            patch_snippet=exc.patch_snippet,
            where=exc.where,
        )
    except PatchHunkMismatch as exc:
        # Self-healing: возвращаем модели актуальный фрагмент файла, чтобы она
        # перестроила hunks из реальных строк, а не падала в write_file.
        return fail(
            "apply_patch",
            str(exc),
            **patch_failure_data(str(exc)),
            current_excerpt=exc.current_excerpt,
            expected_lines=exc.expected_lines,
            match_count=exc.match_count,
        )
    except PatchHunkLine as exc:
        # Self-healing для ошибок формата строки hunk: показываем модели саму
        # проблемную строку, её номер в патче и excerpt реальных строк файла.
        # Типичный случай — пустая строка контекста без ведущего пробела. Раньше
        # здесь был общий hint без excerpt, и модель сбегала в write_file.
        anchor = _best_partial_match(exc.current, exc.old_lines)
        return fail(
            "apply_patch",
            str(exc),
            hint=_hunk_line_hint(exc),
            retryable=True,
            bad_line=clipped(exc.bad_line, limit=BAD_LINE_LIMIT),
            bad_line_number=exc.bad_line_number,
            current_excerpt=_build_mismatch_excerpt(
                exc.current, anchor, exc.old_lines
            ),
        )
    except ValueError as exc:
        summary = str(exc)
        return fail("apply_patch", summary, **patch_failure_data(summary))
    for path, content in changes.items():
        if content is None:
            path.unlink()
        else:
            # already_applied-файлы получили original в changes, пишем его же —
            # это no-op на диске, но сохраняет атомарность цикла записи.
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    already = sorted(str(p) for p in parser.already_applied)
    summary = f"applied patch to {len(changes)} file(s)"
    if already:
        # Явно сообщаем модели, какие файлы уже содержали правку, чтобы она не
        # повторяла тот же патч ещё раз.
        summary += f"; already applied to {len(already)}: " + ", ".join(already)
    result = ok("apply_patch", summary)
    if already:
        result["already_applied_files"] = already
    return result


class PatchHunkMismatch(ValueError):
    """Несовпадение hunks'а: несем модели актуальный фрагмент файла.

    PatchParser поднимает его вместо обычного ValueError, когда update-hunk не
    лёг на файл (0 или >1 совпадений). observation.apply_patch отдаёт модели
    current_excerpt — реальные строки файла с номерами, чтобы следующий patch
    опирался на актуальный текст, а не на устаревшую память о файле.
    """

    def __init__(
        self,
        summary: str,
        *,
        path: Path,
        expected_lines: list[str],
        current_excerpt: str,
        match_count: int,
    ):
        super().__init__(summary)
        self.path = path
        self.expected_lines = expected_lines
        self.current_excerpt = current_excerpt
        self.match_count = match_count


class PatchMalformed(ValueError):
    """Оборванный или несогласованный патч: показываем модели её собственный край.

    Когда патч не обрамлён маркерами *** Begin/End Patch как положено, парсер
    раньше выдавал общий hint. Несём модели её же хвост (или голову), чтобы она
    увидела, где именно потеряла структуру — типично это dangling '+}', пустая
    добавленная строка или незакрытый hunk. patch_snippet несёт обрезанный край.
    """

    def __init__(self, summary: str, *, patch_snippet: str, where: str):
        super().__init__(summary)
        self.patch_snippet = patch_snippet
        # «head» или «tail» — какой край патча показан модели.
        self.where = where


class PatchHunkLine(ValueError):
    """Строка hunk'а с неверным маркером: несем модели саму строку и excerpt файла.

    Самая частая причина — пустая строка контекста без ведущего пробела
    (summary «invalid hunk line: »). Старый код поднимал обычный ValueError, и
    observation несло общий hint без указания, *какая* строка и *где* виновата,
    а excerpt файла вообще не прикладывался. Из-за этого модель не понимала, что
    поправить, и сбегала в write_file — ровно этот паттерн виден в трассах как
    кластер холодных правок.

    Здесь файл уже прочитан (current доступен), поэтому вместе со ссылкой на
    проблемную строку (bad_line, bad_line_number) отдаём excerpt реальных строк
    файла — модели есть из чего перестроить hunk, как и при PatchHunkMismatch.
    """

    def __init__(
        self,
        summary: str,
        *,
        bad_line: str,
        bad_line_number: int,
        current: list[str],
        old_lines: list[str],
    ):
        super().__init__(summary)
        # Проблемная строка патча дословно — типично пустая, таб или мусор.
        self.bad_line = bad_line
        # 1-based номер строки патча (от *** Begin Patch).
        self.bad_line_number = bad_line_number
        # Строки файла на момент разбора hunk'а — для excerpt вокруг частичного
        # совпадения накопленных old_lines.
        self.current = current
        self.old_lines = old_lines


def _patch_snippet(lines: list[str], where: str, limit: int = 8) -> str:
    """Обрезанный край патча для diagnostic-observation.

    where = «tail» показывает последние строки (типичный случай — оборванный
    hunk), «head» — первые. Длинный край обрезаем, чтобы observation не раздулось.
    """

    if where == "tail":
        chunk = lines[-limit:]
        label = f"last {len(chunk)} lines"
    else:
        chunk = lines[:limit]
        label = f"first {len(chunk)} lines"
    return f"[{label}]\n" + "\n".join(chunk)


def _malformed_hint(exc: PatchMalformed) -> str:
    """Подсказка под конкретный край патча: чем именно модель потеряла структуру.

    Для хвоста (нет End Patch) типичен оборванный hunk с dangling '+}' или
    пустой добавленной строкой — упоминаем это явно. Для головы — wrap в
    shell/JSON/prose вместо чистого patch-текста.
    """

    if exc.where == "tail":
        return (
            "The patch text ended without *** End Patch, which usually means the "
            "last hunk is malformed — a dangling '+}' (unbalanced brace), a stray "
            "'+' (empty added line), or a lost closing marker. Look at patch_snippet: "
            "if the tail looks unfinished, re-read the file region and rebuild a "
            "smaller, self-contained hunk ending with *** End Patch. Prefer one "
            "logical change per patch."
        )
    return (
        "The patch text does not start with *** Begin Patch, which usually means "
        "it was wrapped in a shell command, a Markdown fence, JSON object, or "
        "extra prose. Look at patch_snippet: send ONLY the patch text, starting "
        "with *** Begin Patch and ending with *** End Patch."
    )


def _hunk_line_hint(exc: PatchHunkLine) -> str:
    """Подсказка под конкретную проблемную строку hunk'а.

    Указываем модели на её собственную строку (bad_line, bad_line_number) и
    требуем leading space для пустых строк контекста — это типичный провал,
    из-за которого модель раньше сбегала в write_file. Ссылаемся на
    current_excerpt: рядом лежат реальные строки файла для перестройки hunk'а.
    """

    return (
        f"The update hunk has a malformed line at patch line {exc.bad_line_number} "
        f"(shown as bad_line). The most common cause is a blank line with no "
        "marker: Blank context lines must still start with one leading space "
        "(a truly empty line is a parse error). Look at bad_line to see the "
        "offending text, copy the exact surrounding lines from current_excerpt, "
        "fix the marker, and retry apply_patch. Do not fall back to write_file."
    )


def _build_mismatch_excerpt(
    current: list[str],
    anchor: int,
    expected_lines: list[str],
) -> str:
    """Собираем окно строк файла вокруг точки несовпадения с номерами.

    anchor — индекс строки (0-based), вокруг которой показываем контекст:
    позицию наилучшего частичного совпадения или первого точного match'а.
    Формат строк повторяет read_file («12: text»), чтобы модель могла сразу
    использовать те же строки в rebuilding hunks'а. Длинный excerpt обрезаем
    через clipped(), иначе большой файл раздул бы observation.
    """

    start = max(0, anchor - MISMATCH_EXCERPT_WINDOW)
    end = min(len(current), anchor + len(expected_lines) + MISMATCH_EXCERPT_WINDOW)
    lines = [f"{i + 1}: {current[i]}" for i in range(start, end)]
    return clipped("\n".join(lines), limit=MISMATCH_EXCERPT_LIMIT)


def _best_partial_match(
    current: list[str], old_lines: list[str]
) -> int:
    """Индекс строки файла с максимальным перекрытием по old_lines.

    Когда точного совпадения нет (found 0), показываем модели район файла,
    где её ожидаемые строки пересекаются с реальностью сильнее всего. Считаем
    для каждой стартовой позиции число совпавших строк из old_lines и берём
    максимум — простой и понятный для учебного harnessа эвристический якорь.
    """

    if not current or not old_lines:
        return 0
    best_index = 0
    best_score = -1
    for start in range(len(current)):
        score = 0
        for offset, expected in enumerate(old_lines):
            pos = start + offset
            if pos < len(current) and current[pos] == expected:
                score += 1
        if score > best_score:
            best_score = score
            best_index = start
    return best_index


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
            # Несём модели голову её патча, чтобы она увидела, что не так с
            # обрамлением: типично это wrap в shell-команду/JSON/prose вместо
            # чистого patch-текста.
            snippet = _patch_snippet(lines or [patch], "head") if lines else patch
            raise PatchMalformed(
                "patch must start with *** Begin Patch",
                patch_snippet=snippet,
                where="head",
            )
        if lines[-1] != "*** End Patch":
            # Несём модели хвост: типично здесь оборванный hunk с dangling '+}',
            # пустой добавленной строкой или потерянной закрывающей скобкой —
            # модель сама так описала сбой в reasoning. Маркер не «забыт», а
            # потерян из-за несогласованного конца.
            raise PatchMalformed(
                "patch must end with *** End Patch",
                patch_snippet=_patch_snippet(lines, "tail"),
                where="tail",
            )

        changes: dict[Path, str | None] = {}
        # Пути, чьи hunks уже лежат в файле дословно — патч повторно применён.
        # apply_patch вернёт их как already_applied, чтобы модель не зациклилась.
        already_applied: set[Path] = set()
        i = 1
        while i < len(lines) - 1:
            line = lines[i]
            if line.startswith("*** Add File: "):
                i = self._parse_add_file(lines, i, changes)
            elif line.startswith("*** Update File: "):
                i = self._parse_update_file(lines, i, changes, already_applied)
            elif line.startswith("*** Delete File: "):
                i = self._parse_delete_file(lines, i, changes)
            elif line.startswith("*** Move to: "):
                raise ValueError("Move to is only supported after Update File")
            elif line == "*** End of File":
                i += 1
            else:
                raise ValueError(f"unexpected patch line: {line}")
        if already_applied:
            # Пробрасываем collected-множество наверх через атрибут, чтобы
            # apply_patch мог отличить «все hunks уже применены» от обычного успеха.
            self.already_applied = already_applied
        else:
            self.already_applied = set()
        return changes

    def _patch_path(self, raw: str) -> Path:
        scope_error = self.ctx.write_path_error(raw)
        if scope_error:
            raise ValueError(scope_error)
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
        self, lines: list[str], i: int, changes: dict[Path, str | None],
        already_applied: set[Path],
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
        # Число hunks, пропущенных как уже-применённые. Если были только такие
        # skip'ы и ни одного реального применения — файл помечаем already-applied.
        applied_skips = 0
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
                    # Self-healing для ошибок формата строки hunk: несем модели
                    # саму проблемную строку, её номер в патче и excerpt файла,
                    # чтобы она видела, что именно поправить (а не абстрактный
                    # hint), и перестраивала hunk, а не падала в write_file.
                    raise PatchHunkLine(
                        f"invalid hunk line: {lines[i]}",
                        bad_line=lines[i],
                        bad_line_number=i + 1,
                        current=current,
                        old_lines=old_lines,
                    )
                i += 1
            if not old_lines and not new_lines:
                raise ValueError("empty update hunk")
            if not old_lines:
                raise ValueError("update hunk must include context or removed lines")
            matches = self._find_hunk_matches(current, old_lines)
            if not matches and new_lines and self._find_hunk_matches(current, new_lines):
                # Idempotency: old_lines не нашлись, но new_lines уже лежат в
                # файле дословно — hunk был применён ранее. Пропускаем его, не
                # падаем в цикл «патчу одно и то же».
                applied_skips += 1
                continue
            if len(matches) != 1:
                # Self-healing: несём модели актуальный фрагмент файла вокруг
                # точки несовпадения, чтобы она перестроила hunks из реальных
                # строк, а не падала в write_file.
                anchor = matches[0] if matches else _best_partial_match(
                    current, old_lines
                )
                excerpt = _build_mismatch_excerpt(current, anchor, old_lines)
                raise PatchHunkMismatch(
                    f"expected 1 hunk match, found {len(matches)}",
                    path=path,
                    expected_lines=old_lines,
                    current_excerpt=excerpt,
                    match_count=len(matches),
                )
            current = self._apply_matched_hunk(current, matches[0], old_lines, new_lines)
            saw_hunk = True
        if applied_skips and not saw_hunk:
            # Все hunks этого файла уже лежат в файле дословно — повторно
            # применённый патч. Файл не трогаем, помечаем путь как already-applied.
            # Move с уже-применённым содержимым не поддерживаем: это неоднозначно.
            if target_path is not None:
                raise ValueError(f"already-applied update cannot move file: {raw}")
            already_applied.add(path)
            changes[path] = original
            return i
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

    def _find_hunk_matches(
        self, current: list[str], old_lines: list[str]
    ) -> list[int]:
        """Стартовые индексы точных совпадений old_lines в текущем файле."""

        matches = []
        for start in range(len(current) - len(old_lines) + 1):
            if current[start : start + len(old_lines)] == old_lines:
                matches.append(start)
        return matches

    def _apply_matched_hunk(
        self,
        current: list[str],
        start: int,
        old_lines: list[str],
        new_lines: list[str],
    ) -> list[str]:
        """Вырезаем старый блок и вставляем новый по известной позиции match'а."""

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

Blank lines inside a hunk are the #1 cause of apply_patch failures. Every line in
an update hunk MUST start with a marker: one space (context), '-' (removed) or '+'
(added). A truly empty line — no leading space at all — is a PARSE ERROR, even if
the line in the file is empty. So a blank context line is written as a single
space character, never as an empty line:
  valid:   " "   (one space — blank context line)
  INVALID: ""    (zero characters — parse error)
The same applies to blank added lines ("+") and blank removed lines ("-").

Prefer apply_patch over write_file for ANY change to an existing file. A targeted
patch is roughly 10x cheaper in tokens than rewriting the whole file, and it keeps
the context window clean. write_file bloats the window with the full new file text,
which accumulates across edits and crowds out evidence — prefer the small patch.

One patch = one logical change. If you need several independent edits to the same
file, split them into separate apply_patch calls. Long multi-hunk patches are where
models lose the structure (dangling '+}', stray '+', forgotten *** End Patch) —
keep each patch small and self-contained.

Idempotency: if you re-send a hunk that is already in the file, apply_patch does
NOT fail — it reports the file as already_applied in the observation. Use this as
a signal to stop retrying the same change, not as an error.

Self-healing on mismatch: when an update hunk does not match the current file, the
observation includes current_excerpt with the actual file region (with line numbers)
and match_count. Rebuild the hunk from those exact lines and retry apply_patch. This
is the primary recovery path — do not fall back to write_file just because the first
patch did not match.

Common parser errors and how to recover (the current file region is attached as
current_excerpt for the first two cases — use it instead of re-reading):
- "expected 1 hunk match, found 0": context or removed lines do not match the
  current file verbatim. Copy surrounding lines from current_excerpt exactly,
  preserving leading spaces, and retry.
- "expected 1 hunk match, found 2": context matched more than one place. Add a few
  more surrounding context lines copied from current_excerpt to make it unique.
- "invalid hunk line:": a line in the hunk has no marker. The observation shows
  the offending line as bad_line and its patch line number as bad_line_number, and
  attaches current_excerpt (the actual file region). The usual cause is a blank
  context line with no leading space — rewrite it as a single space and retry.
- "Move to is only supported after Update File": put "*** Move to:" on the line
  immediately after "*** Update File:", before any @@ hunks.
- "patch must start with *** Begin Patch" or "unexpected patch line": send only the
  patch text. Do not wrap it in a shell command, Markdown fence, JSON object, or
  extra prose.
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
current lines into the hunk, and retry apply_patch once. Remember that blank
context lines must still start with one space — a truly empty line is a parse
error. To preserve a blank line that exists in the file, write it as " " (one
space) in context, or "+" in an added block — never as a zero-character line.
"""

APPLY_PATCH_SPEC = ToolSpec(
    "apply_patch",
    APPLY_PATCH_DESCRIPTION,
    obj({"patch": strp(req=True, desc=PATCH_ARGUMENT_DESCRIPTION)}, ["patch"]),
    apply_patch,
)
