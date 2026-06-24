"""PNG-рендер «анатомии сессии» — собственно тепловой карты контекста.

В отличие от context_window.png (структура окна), здесь — диагноз: какие типы
фрагментов копятся (блок A), где сессия «болеет» (блок B сигналов), где
срабатывали cold gaps, что агент делал и где harness перепаковывал контекст.
Данные получает готовыми из anatomy_data.build_anatomy_data, чтобы PNG и HTML
пользовались одним источником правды.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .png import (
    BACKGROUND,
    COLD,
    DANGER,
    GRID,
    MUTED,
    PANEL,
    SOURCE_COLORS,
    SOURCE_ORDER,
    TEXT,
    WARNING,
    _Canvas,
    _column_geometry,
    _write_png,
)


# Фирменные цвета дорожек блока B намеренно разнесены с палитрой типов блока A,
# чтобы глаз не путал «красный = tool_output» и «красный = red_token_share».
SIGNAL_COLORS = {
    "cold_gap_score": COLD,
    "window_pressure_score": WARNING,
    "red_token_share": (190, 38, 30),
    "fill_share": (90, 100, 115),
}

SIGNAL_LABELS = {
    "cold_gap_score": "COLD",
    "window_pressure_score": "PRESS",
    "red_token_share": "RED",
    "fill_share": "FILL",
}

# Короткие читаемые подписи типов фрагментов: целые слова вместо обрезки по
# 10 символов (раньше легенда показывала «TOOL OUTPU», «USER MESSA»).
SOURCE_LABELS = {
    "system_instruction": "SYSTEM",
    "developer_instruction": "DEVELOPER",
    "user_message": "USER",
    "context_fragment": "CONTEXT",
    "file_snippet": "FILE",
    "test_result": "TEST",
    "tool_schema": "SCHEMA",
    "assistant_message": "ASSISTANT",
    "tool_output": "TOOL OUT",
    "unknown": "UNKNOWN",
}

# Цвета действий в action track. Имена соответствуют группам из anatomy_data.
ACTION_COLORS = {
    "read": (86, 166, 217),
    "write": (190, 38, 30),
    "test": (100, 180, 93),
    "list": (150, 156, 164),
    "run": (245, 166, 35),
    "other": (120, 120, 130),
}

# Фиксированные зоны рисунка по вертикали. Меняем здесь — переедет весь layout.
WIDTH = 1400
HEIGHT = 860
LEFT = 96  # левая граница области графиков (под метки строк)
RIGHT = WIDTH - 40  # правая граница
GRAPH_W = RIGHT - LEFT
HEADER_H = 70
LEGEND_Y = 92
MASS_Y = 130  # верх блока A
MASS_H = 250
SIGNALS_Y = MASS_Y + MASS_H + 38  # верх блока B
SIGNALS_H = 200
ACTION_Y = SIGNALS_Y + SIGNALS_H + 32
ACTION_H = 26
AXIS_Y = ACTION_Y + ACTION_H + 22


def render_heatmap_png(data: dict[str, Any], out_path: Path) -> None:
    """Пишет PNG тепловой карты «анатомии сессии» (heatmap.png)."""

    canvas = _Canvas(WIDTH, HEIGHT, BACKGROUND)
    columns = data.get("columns") or []
    turn_to_x = _turn_to_x(columns)

    _draw_header(canvas, data)
    _draw_verdict(canvas, data)
    _draw_legend(canvas, data)
    _draw_mass_block(canvas, columns)
    # Дорожка COLD блока B несёт и heatmap-интенсивность, и ромбы-маркеры
    # срабатывания правила cold gap — это один сигнал, не два.
    _draw_signals_block(canvas, columns, data, turn_to_x)
    _draw_action_track(canvas, data, turn_to_x)
    # Швы рисуем последними — поверх всех блоков, чтобы они читались на любой высоте.
    _draw_seams(canvas, data, turn_to_x)
    _draw_axis(canvas, columns)

    _write_png(out_path, canvas.width, canvas.height, canvas.pixels)


def _turn_to_x(columns: list[dict[str, Any]]) -> dict[int, int]:
    """Сопоставляем turn_id → x-координату центра колонки на графике."""

    if not columns:
        return {}
    gap, column_width = _column_geometry(len(columns), GRAPH_W)
    mapping: dict[int, int] = {}
    for index, column in enumerate(columns):
        x = LEFT + index * (column_width + gap) + column_width // 2
        mapping[_safe_int(column.get("turn_id"), index)] = min(x, RIGHT - 1)
    return mapping


def _draw_header(canvas: _Canvas, data: dict[str, Any]) -> None:
    """Шапка с названием и сводными числами сессии."""

    canvas.rect(0, 0, canvas.width, HEADER_H, PANEL)
    canvas.rect(0, HEADER_H - 1, canvas.width, 1, GRID)
    session_id = _ascii(str(data.get("session_id") or "session"))[:40]
    canvas.text(LEFT, 22, "CONTEXT HEATMAP - SESSION ANATOMY", TEXT, scale=2)
    metrics = data.get("metrics") or {}
    line = (
        f"SESSION {session_id}  "
        f"CALLS {_safe_int(metrics.get('model_calls'))}  "
        f"TURNS {_safe_int(metrics.get('turns'))}  "
        f"FINDINGS {_safe_int(metrics.get('findings'))}  "
        f"MAX FILL {(_safe_float(metrics.get('max_fill')) * 100):.0f}%  "
        f"FIX {_safe_float(metrics.get('max_fixed_instruction_cost')):.2f}  "
        f"GOAL {_safe_float(metrics.get('max_goal_anchor_cost')):.2f}"
    )
    # Сводка сессии — ключевая строка, поэтому тёмным TEXT, а не блёклым MUTED.
    canvas.text(LEFT, 48, line, TEXT, scale=1)


def _draw_verdict(canvas: _Canvas, data: dict[str, Any]) -> None:
    """Verdict-указатель справа в шапке: «светофор» + пиковый ход + куда смотреть.

    Точки «светофора» рисуем кругами-примитивами (PNG-шрифт не знает символов
    ●/○), чтобы verdict читался однозначно: залитый квадрат = проблема есть,
    полый = чисто. Состояние берём из verdict.dots как ASCII «1»/«0», а не из
    юникод-символов, чтобы не зависеть от глифов шрифта.
    """

    verdict = data.get("verdict") or {}
    box_x = canvas.width - 380
    box_y = 10
    box_w = 360
    box_h = HEADER_H - 20
    canvas.rect(box_x, box_y, box_w, box_h, PANEL)
    canvas.rect_outline(box_x, box_y, box_w, box_h, GRID)
    canvas.text(box_x + 14, box_y + 10, "VERDICT", MUTED, scale=1)

    # Три точки: RED / COLD / PRESSURE. Цвет точки — цвет дорожки из блока B.
    dots = str(verdict.get("dots") or "000")
    states = (
        ("red_token_share", "R", dots[0] if len(dots) > 0 else "0"),
        ("cold_gap_score", "C", dots[1] if len(dots) > 1 else "0"),
        ("window_pressure_score", "P", dots[2] if len(dots) > 2 else "0"),
    )
    dot_x = box_x + 18
    dot_y = box_y + 28
    for key, _short, state in states:
        color = SIGNAL_COLORS[key]
        if state == "1":
            canvas.rect(dot_x, dot_y, 14, 14, color)
        else:
            canvas.rect_outline(dot_x, dot_y, 14, 14, MUTED)
        dot_x += 26

    label = _ascii(str(verdict.get("label") or "-"))
    canvas.text(box_x + 110, box_y + 32, label, TEXT, scale=1)
    peak = verdict.get("peak_turn")
    if peak is not None:
        canvas.text(
            box_x + 14,
            box_y + 50,
            f"PEAK TURN {_safe_int(peak)}",
            WARNING,
            scale=1,
        )
    hint = _ascii(str(verdict.get("hint") or ""))
    # Подсказка «куда смотреть» — обрезаем, чтобы не вылезла за пределы блока.
    hint = hint[:40]
    canvas.text(box_x + 150, box_y + 50, hint, MUTED, scale=1)


def _draw_legend(canvas: _Canvas, data: dict[str, Any]) -> None:
    """Две части легенды: слева — типы фрагментов (блок A), справа — сигналы (блок B)."""

    canvas.text(LEFT, LEGEND_Y - 18, "FRAGMENT TYPE - BLOCK A", TEXT, scale=1)
    cursor_x = LEFT
    # Берём только типы, реально встречающиеся в сессии, чтобы легенда не пестрила.
    present = _present_source_types(data)
    for source_type in present:
        color = SOURCE_COLORS.get(source_type, SOURCE_COLORS["unknown"])
        label = _ascii(SOURCE_LABELS.get(source_type, source_type.upper()))
        canvas.rect(cursor_x, LEGEND_Y, 12, 12, color)
        canvas.rect_outline(cursor_x, LEGEND_Y, 12, 12, (150, 150, 150))
        canvas.text(cursor_x + 16, LEGEND_Y + 3, label, MUTED, scale=1)
        cursor_x += 16 + len(label) * 6 + 16

    # Сигналы блока B — справа от типов, разделены разрывом.
    cursor_x = LEFT + GRAPH_W // 2
    canvas.text(cursor_x, LEGEND_Y - 18, "QUALITY SIGNAL - BLOCK B", TEXT, scale=1)
    for key in ("cold_gap_score", "window_pressure_score", "red_token_share", "fill_share"):
        color = SIGNAL_COLORS[key]
        label = SIGNAL_LABELS[key]
        canvas.rect(cursor_x, LEGEND_Y, 12, 12, color)
        canvas.rect_outline(cursor_x, LEGEND_Y, 12, 12, (150, 150, 150))
        canvas.text(cursor_x + 16, LEGEND_Y + 3, label, MUTED, scale=1)
        cursor_x += 16 + len(label) * 6 + 16


def _present_source_types(data: dict[str, Any]) -> list[str]:
    """Типы, у которых есть хотя бы один токен хотя бы в одной колонке."""

    present: list[str] = []
    for source_type in SOURCE_ORDER:
        if any(
            _safe_int((col.get("tokens_by_type") or {}).get(source_type)) > 0
            for col in (data.get("columns") or [])
        ):
            present.append(source_type)
    if not present:
        present = list(SOURCE_ORDER)
    return present


def _draw_mass_block(
    canvas: _Canvas,
    columns: list[dict[str, Any]],
) -> None:
    """Блок A — масса фрагментов по типам. Каждая колонка = одно обращение к модели.

    По горизонтали — ходы. Колонка поделена по типам пропорционально их доле в
    окне; высота заливки = fill_share. Цвета — палитра типов.
    """

    canvas.text(LEFT, MASS_Y - 18, "FRAGMENT MASS BY TYPE  % OF WINDOW", TEXT, scale=1)
    canvas.rect(LEFT, MASS_Y, GRAPH_W, MASS_H, PANEL)
    canvas.rect_outline(LEFT, MASS_Y, GRAPH_W, MASS_H, (180, 185, 190))
    # Сетка считается от низа: 0% у основания, 100% у верха — так совпадает с
    # направлением роста столбцов (заливка растёт снизу вверх).
    for ratio in (0.0, 0.25, 0.50, 0.75, 1.00):
        yy = MASS_Y + MASS_H - int(MASS_H * ratio)
        if 0.0 < ratio < 1.0:
            canvas.line(LEFT, yy, RIGHT, yy, GRID)
        label_y = min(max(yy - 3, MASS_Y + 1), MASS_Y + MASS_H - 8)
        canvas.text(40, label_y, f"{int(ratio * 100)}%", MUTED, scale=1)

    if not columns:
        canvas.text(LEFT + 24, MASS_Y + MASS_H // 2 - 8, "NO MODEL CALLS", MUTED, scale=2)
        return

    gap, column_width = _column_geometry(len(columns), GRAPH_W)
    for index, column in enumerate(columns):
        x = LEFT + index * (column_width + gap)
        _draw_mass_column(canvas, column, x, column_width)


def _draw_mass_column(
    canvas: _Canvas,
    column: dict[str, Any],
    x: int,
    width: int,
) -> None:
    """Один столбец блока A: вертикальная «укладка» типов снизу вверх."""

    window_tokens = max(_safe_int(column.get("window_tokens")), 1)
    used_tokens = max(min(_safe_int(column.get("used_tokens")), window_tokens), 0)
    fill_share = used_tokens / window_tokens
    tokens_by_type = column.get("tokens_by_type") or {}

    # Движемся снизу вверх, чтобы типы шли в порядке SOURCE_ORDER от основания.
    cursor = 0
    for source_type in SOURCE_ORDER:
        tokens = max(_safe_int(tokens_by_type.get(source_type)), 0)
        if not tokens:
            continue
        seg_start = min(cursor, used_tokens)
        seg_end = min(cursor + tokens, used_tokens)
        cursor = seg_end
        if seg_end <= seg_start:
            continue
        top = MASS_Y + MASS_H - round(seg_end / window_tokens * MASS_H)
        bottom = MASS_Y + MASS_H - round(seg_start / window_tokens * MASS_H)
        canvas.rect(x, top, width, max(bottom - top, 1), SOURCE_COLORS[source_type])

    # Маркер давления под колонкой: оранжевый ≥ 0.75, красный ≥ 0.90.
    if fill_share >= 0.90:
        canvas.rect(x, MASS_Y + MASS_H + 2, width, 3, DANGER)
    elif fill_share >= 0.75:
        canvas.rect(x, MASS_Y + MASS_H + 2, width, 3, WARNING)


def _draw_signals_block(
    canvas: _Canvas,
    columns: list[dict[str, Any]],
    data: dict[str, Any],
    turn_to_x: dict[int, int],
) -> None:
    """Блок B — heatmap сигналов качества. Цвет = интенсивность × фирменный цвет дорожки.

    На дорожке COLD дополнительно рисуем ромбы ◆ в ходах, где правило cold gap
    сработало (findings): heatmap показывает *величину* сигнала, ромбы — *момент
    срабатывания правила*. Это один сигнал, поэтому живёт в одной строке.
    """

    canvas.text(LEFT, SIGNALS_Y - 18, "QUALITY SIGNALS  INTENSITY = SCORE", TEXT, scale=1)
    canvas.rect(LEFT, SIGNALS_Y, GRAPH_W, SIGNALS_H, PANEL)
    canvas.rect_outline(LEFT, SIGNALS_Y, GRAPH_W, SIGNALS_H, (180, 185, 190))

    signals = ("cold_gap_score", "window_pressure_score", "red_token_share", "fill_share")
    row_h = SIGNALS_H // len(signals)
    thresholds = data.get("thresholds") or {}
    cold_turns = set(_safe_int(t) for t in (data.get("cold_turns") or []))

    for row, key in enumerate(signals):
        row_y = SIGNALS_Y + row * row_h
        canvas.text(40, row_y + row_h // 2 - 3, SIGNAL_LABELS[key], MUTED, scale=1)
        # Базовая линия дорожки + явная метка пустого сигнала: пустая дорожка
        # должна читаться как «0», а не как «данные не посчитаны».
        baseline_y = row_y + row_h - 2
        canvas.line(LEFT, baseline_y, RIGHT, baseline_y, (214, 218, 224))
        if columns and not any(_safe_float(c.get(key)) > 0.0 for c in columns):
            canvas.text(LEFT + 6, row_y + row_h // 2 - 3, "0 - NONE", (158, 162, 168), scale=1)
        # Пороговая линия на дорожке FILL — калибровка глаз под danger/warning.
        if key == "fill_share":
            for thr, color in (
                (thresholds.get("fill_warning", 0.75), WARNING),
                (thresholds.get("fill_danger", 0.90), DANGER),
            ):
                yy = row_y + row_h - int(thr * (row_h - 4)) - 2
                _draw_dashed_h(canvas, LEFT, RIGHT, yy, color, dash=6)

        if not columns:
            continue
        gap, column_width = _column_geometry(len(columns), GRAPH_W)
        base_color = SIGNAL_COLORS[key]
        for index, column in enumerate(columns):
            value = min(max(_safe_float(column.get(key)), 0.0), 1.0)
            if value <= 0.0:
                continue
            x = LEFT + index * (column_width + gap)
            bar_h = max(1, int(value * (row_h - 4)))
            canvas.rect(
                x,
                row_y + row_h - 2 - bar_h,
                column_width,
                bar_h,
                _intensity(base_color, value),
            )

        # На дорожке COLD — ромбы срабатывания правила поверх heatmap.
        if key == "cold_gap_score":
            for turn_id in cold_turns:
                x = turn_to_x.get(turn_id)
                if x is None:
                    continue
                _draw_diamond(canvas, x, row_y + row_h // 2, 5, COLD)


def _draw_action_track(
    canvas: _Canvas,
    data: dict[str, Any],
    turn_to_x: dict[int, int],
) -> None:
    """Цветные квадраты действий: read/write/patch/list/run/test по ходам."""

    canvas.text(40, ACTION_Y + ACTION_H // 2 - 3, "ACTION", MUTED, scale=1)
    # Легенда действий в зазоре над дорожкой: без неё цветные квадратики
    # read/write/test/list/run не читаются.
    legend_y = ACTION_Y - 14
    cursor_x = LEFT
    for group in ("read", "write", "test", "list", "run", "other"):
        color = ACTION_COLORS[group]
        label = group.upper()
        canvas.rect(cursor_x, legend_y, 10, 10, color)
        canvas.rect_outline(cursor_x, legend_y, 10, 10, (150, 150, 150))
        canvas.text(cursor_x + 14, legend_y + 2, label, MUTED, scale=1)
        cursor_x += 14 + len(label) * 6 + 14
    canvas.rect(LEFT, ACTION_Y, GRAPH_W, ACTION_H, PANEL)
    canvas.rect_outline(LEFT, ACTION_Y, GRAPH_W, ACTION_H, (200, 204, 210))
    groups = data.get("action_groups") or {}
    # Инвертируем группу: имя инструмента → имя группы для быстрого поиска цвета.
    name_to_group = {name: group for group, names in groups.items() for name in names}
    for turn_id, names in (data.get("actions_by_turn") or {}).items():
        x = turn_to_x.get(_safe_int(turn_id))
        if x is None:
            continue
        # Несколько действий на ходе — рисуем подряд узкими квадратиками.
        for slot, name in enumerate(names[:4]):
            group = name_to_group.get(str(name), "other")
            color = ACTION_COLORS.get(group, ACTION_COLORS["other"])
            offset = (slot - (len(names[:4]) - 1) / 2) * 6
            canvas.rect(int(x + offset - 3), ACTION_Y + 4, 5, ACTION_H - 8, color)


def _draw_seams(
    canvas: _Canvas,
    data: dict[str, Any],
    turn_to_x: dict[int, int],
) -> None:
    """Швы контекста — только аварийное усечение (truncated/dropped).

    Обычные digest-швы намеренно не рисуем: на длинной сессии свёртка истории
    происходит почти каждый ход, и вертикальные линии превращались в рябящий
    «штрих-код» поверх блоков. Оставляем лишь редкий янтарный маркер усечения —
    он критичен и встречается нечасто, поэтому не мешает чтению.
    """

    top = MASS_Y
    bottom = ACTION_Y + ACTION_H
    for seam in data.get("seams") or []:
        if not bool(seam.get("truncated")):
            continue
        x = turn_to_x.get(_safe_int(seam.get("turn_id")))
        if x is None:
            continue
        _draw_dashed_v(canvas, x, top, bottom, WARNING, dash=6)


def _draw_axis(canvas: _Canvas, columns: list[dict[str, Any]]) -> None:
    """Подписи номеров ходов под графиком."""

    canvas.text(40, AXIS_Y, "TURN", MUTED, scale=1)
    canvas.line(LEFT, AXIS_Y - 6, RIGHT, AXIS_Y - 6, GRID)
    if not columns:
        return
    total = len(columns)
    # Подписываем ~10 равномерных ходов, иначе сольются на длинной сессии.
    step = max(1, total // 10)
    gap, column_width = _column_geometry(total, GRAPH_W)
    for index in range(0, total, step):
        turn_id = _safe_int(columns[index].get("turn_id"), index)
        x = LEFT + index * (column_width + gap) + column_width // 2
        canvas.text(x - 6, AXIS_Y + 4, str(turn_id), MUTED, scale=1)


def _intensity(base: tuple[int, int, int], value: float) -> tuple[int, int, int]:
    """Смешиваем базовый цвет дорожки с фоном панели по интенсивности сигнала.

    Так heatmap блока B читается как «чем темнее/насыщеннее — тем сильнее», без
    жёсткой палитры green→red, которая конфликтовала бы с блоком A. Минимальный
    коэффициент 0.25 не даёт слабым сигналам (fill на flappy2 ~0.35) исчезнуть
    на фоне панели.
    """

    value = max(0.25, min(1.0, value))
    panel = PANEL
    return (
        int(panel[0] + (base[0] - panel[0]) * value),
        int(panel[1] + (base[1] - panel[1]) * value),
        int(panel[2] + (base[2] - panel[2]) * value),
    )


def _draw_diamond(
    canvas: _Canvas, cx: int, cy: int, radius: int, color: tuple[int, int, int]
) -> None:
    """Заполненный ромб-маркер для cold gaps."""

    for dy in range(-radius, radius + 1):
        span = radius - abs(dy)
        canvas.rect(cx - span, cy + dy, span * 2 + 1, 1, color)


def _draw_dashed_h(
    canvas: _Canvas, x0: int, x1: int, y: int, color: tuple[int, int, int], dash: int = 6
) -> None:
    """Горизонтальный пунктир — пороговые линии блока B."""

    cursor = x0
    while cursor < x1:
        canvas.rect(cursor, y, min(dash, x1 - cursor), 1, color)
        cursor += dash * 2


def _draw_dashed_v(
    canvas: _Canvas, x: int, y0: int, y1: int, color: tuple[int, int, int], dash: int = 6
) -> None:
    """Вертикальный пунктир — аварийный шов усечения."""

    cursor = y0
    while cursor < y1:
        canvas.rect(x, cursor, 1, min(dash, y1 - cursor), color)
        cursor += dash * 2


def _safe_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number or number in (float("inf"), float("-inf")):
        return default
    return number


def _ascii(value: str) -> str:
    """PNG-шрифт знает только ASCII, поэтому экзотику заменяем на '?'."""

    return "".join(char if 32 <= ord(char) <= 126 else "?" for char in value)
