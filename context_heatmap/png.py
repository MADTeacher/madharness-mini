"""PNG-renderer краткой диагностической карты контекста."""

from __future__ import annotations

import binascii
import math
import struct
import zlib
from collections import Counter
from pathlib import Path
from typing import Any


SOURCE_ORDER = (
    "system_instruction",
    "developer_instruction",
    "user_message",
    "context_fragment",
    "file_snippet",
    "test_result",
    "tool_schema",
    "assistant_message",
    "tool_output",
    "unknown",
)

SOURCE_COLORS = {
    "system_instruction": (244, 182, 181),
    "developer_instruction": (242, 153, 74),
    "user_message": (86, 166, 217),
    "context_fragment": (156, 207, 117),
    "file_snippet": (111, 194, 176),
    "test_result": (100, 180, 93),
    "tool_schema": (246, 228, 167),
    "assistant_message": (183, 159, 230),
    "tool_output": (194, 58, 82),
    "unknown": (201, 206, 214),
}

BACKGROUND = (247, 248, 250)
PANEL = (255, 255, 255)
TEXT = (32, 33, 36)
MUTED = (95, 99, 104)
GRID = (224, 228, 233)
WARNING = (245, 166, 35)
DANGER = (190, 38, 30)
COLD = (23, 78, 166)


FONT_5X7 = {
    " ": ("00000", "00000", "00000", "00000", "00000", "00000", "00000"),
    "!": ("00100", "00100", "00100", "00100", "00100", "00000", "00100"),
    "#": ("01010", "11111", "01010", "01010", "11111", "01010", "00000"),
    "%": ("11001", "11010", "00100", "01000", "10110", "00110", "00000"),
    "+": ("00000", "00100", "00100", "11111", "00100", "00100", "00000"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    ".": ("00000", "00000", "00000", "00000", "00000", "01100", "01100"),
    "/": ("00001", "00010", "00100", "01000", "10000", "00000", "00000"),
    ":": ("00000", "01100", "01100", "00000", "01100", "01100", "00000"),
    "=": ("00000", "00000", "11111", "00000", "11111", "00000", "00000"),
    "?": ("01110", "10001", "00001", "00010", "00100", "00000", "00100"),
    "_": ("00000", "00000", "00000", "00000", "00000", "00000", "11111"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("00110", "01000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00010", "01100"),
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01110", "10001", "10000", "10000", "10000", "10001", "01110"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01110", "10001", "10000", "10111", "10001", "10001", "01111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("01110", "00100", "00100", "00100", "00100", "00100", "01110"),
    "J": ("00111", "00010", "00010", "00010", "10010", "10010", "01100"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "10101", "01010"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
}


def render_context_window_png(
    report: dict[str, Any],
    packets: list[dict[str, Any]],
    turn_heat: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    out_path: Path,
) -> None:
    """Пишет PNG-снимок структуры контекстного окна (context_window.png).

    Здесь — только давление на окно: WINDOW FILL BY SOURCE TYPE и нижние
    TURN SIGNALS. Диагностика «были ли проблемы» живёт в heatmap.py, который
    рисует «анатомию сессии» поверх того же набора данных.
    """

    calls = _summary_columns(packets, turn_heat, findings)
    width = 1200
    height = 760
    canvas = _Canvas(width, height, BACKGROUND)
    _draw_header(canvas, report, calls)
    _draw_legend(canvas, 72, 92)
    _draw_context_chart(canvas, calls, 72, 160, 1056, 390)
    _draw_sparklines(canvas, calls, 72, 590, 1056, 120)
    _write_png(out_path, canvas.width, canvas.height, canvas.pixels)


def _summary_columns(
    packets: list[dict[str, Any]],
    turn_heat: list[dict[str, Any]],
    findings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Сводим данные одного model call к безопасным численным агрегатам."""

    heat_by_call = {str(row.get("model_call_id") or ""): row for row in turn_heat}
    heat_by_turn = {_safe_int(row.get("turn_id")): row for row in turn_heat}
    findings_by_turn = Counter(_safe_int(item.get("turn_id")) for item in findings)
    if packets:
        columns = []
        for index, packet in enumerate(packets):
            model_call_id = str(packet.get("model_call_id") or "")
            turn_id = _safe_int(packet.get("turn_id"), index)
            heat = heat_by_call.get(model_call_id) or heat_by_turn.get(turn_id) or {}
            fragments = [
                fragment
                for fragment in (packet.get("fragments") or [])
                if isinstance(fragment, dict)
            ]
            tokens_by_type = Counter()
            for fragment in fragments:
                source_type = _source_type(str(fragment.get("source_type") or "unknown"))
                tokens_by_type[source_type] += max(_safe_int(fragment.get("tokens")), 0)
            fragment_tokens = sum(tokens_by_type.values())
            input_tokens = max(_safe_int(packet.get("input_tokens")), 0)
            window_tokens = _safe_int(packet.get("context_window_tokens"))
            if window_tokens <= 0:
                window_tokens = max(input_tokens, fragment_tokens, 1)
            used_tokens = min(max(input_tokens or fragment_tokens, fragment_tokens), window_tokens)
            if used_tokens > fragment_tokens:
                tokens_by_type["unknown"] += used_tokens - fragment_tokens
            columns.append(
                _column_payload(
                    turn_id,
                    model_call_id,
                    window_tokens,
                    used_tokens,
                    tokens_by_type,
                    heat,
                    findings_by_turn.get(turn_id, 0),
                )
            )
        return columns

    columns = []
    for index, heat in enumerate(turn_heat):
        turn_id = _safe_int(heat.get("turn_id"), index)
        used_share = min(
            max(
                _safe_float(heat.get("evidence_density")),
                _safe_float(heat.get("raw_tool_share")),
                _safe_float(heat.get("cold_gap_score")),
                _safe_float(heat.get("red_token_share")),
                0.05,
            ),
            1.0,
        )
        used_tokens = int(1000 * used_share)
        columns.append(
            _column_payload(
                turn_id,
                str(heat.get("model_call_id") or f"turn:{turn_id}"),
                1000,
                used_tokens,
                Counter({"unknown": used_tokens}),
                heat,
                findings_by_turn.get(turn_id, 0),
            )
        )
    return columns


def _column_payload(
    turn_id: int,
    model_call_id: str,
    window_tokens: int,
    used_tokens: int,
    tokens_by_type: Counter[str],
    heat: dict[str, Any],
    findings_count: int,
) -> dict[str, Any]:
    """Нормализуем один столбец перед рисованием."""

    return {
        "turn_id": turn_id,
        "model_call_id": model_call_id,
        "window_tokens": max(window_tokens, 1),
        "used_tokens": max(min(used_tokens, max(window_tokens, 1)), 0),
        "tokens_by_type": tokens_by_type,
        "raw_tool_share": _safe_float(heat.get("raw_tool_share")),
        "assistant_share": _safe_float(heat.get("assistant_share")),
        "evidence_density": _safe_float(heat.get("evidence_density")),
        "positioned_evidence_score": _safe_float(
            heat.get("positioned_evidence_score"),
            1.0,
        ),
        "cold_gap_score": _safe_float(heat.get("cold_gap_score")),
        "red_token_share": _safe_float(heat.get("red_token_share")),
        "normative_status": _safe_float(heat.get("normative_status")),
        "goal_status": _safe_float(heat.get("goal_status")),
        "findings": findings_count,
    }


def _draw_header(
    canvas: _Canvas,
    report: dict[str, Any],
    calls: list[dict[str, Any]],
) -> None:
    """Рисуем заголовок и основные численные признаки сессии."""

    canvas.rect(0, 0, canvas.width, 76, PANEL)
    session_id = _ascii(str(report.get("session_id") or "session"))[:42]
    max_fill = max(
        (call["used_tokens"] / max(call["window_tokens"], 1) for call in calls),
        default=0.0,
    )
    canvas.text(32, 22, "CONTEXT HEATMAP SUMMARY", TEXT, scale=2)
    metrics = (
        f"SESSION {session_id}  CALLS {len(calls)}  "
        f"MAX FILL {max_fill * 100:.0f}%  "
        f"FIX {float(report.get('max_fixed_instruction_cost') or 0):.2f}  "
        f"GOAL {float(report.get('max_goal_anchor_cost') or 0):.2f}  "
        f"PROT {float(report.get('max_normative_status') or 0):.2f}/"
        f"{float(report.get('max_goal_status') or 0):.2f}  "
        f"COLD {float(report.get('max_cold_gap_score') or 0):.2f}  "
        f"ASST {float(report.get('max_assistant_share') or 0):.2f}  "
        f"FINDINGS {int(report.get('findings') or 0)}"
    )
    canvas.text(32, 52, metrics, MUTED, scale=1)


def _draw_legend(canvas: _Canvas, x: int, y: int) -> None:
    """Показываем легенду цветов агрегированных source_type."""

    labels = [
        ("SYS", "system_instruction"),
        ("DEV", "developer_instruction"),
        ("USER", "user_message"),
        ("CTX", "context_fragment"),
        ("FILE", "file_snippet"),
        ("TEST", "test_result"),
        ("SCHEMA", "tool_schema"),
        ("ASST", "assistant_message"),
        ("TOOL", "tool_output"),
        ("UNK", "unknown"),
    ]
    cursor_x = x
    canvas.text(x, y - 22, "SOURCE TYPE COLORS", TEXT, scale=1)
    for label, source_type in labels:
        canvas.rect(cursor_x, y, 14, 14, SOURCE_COLORS[source_type])
        canvas.rect_outline(cursor_x, y, 14, 14, (150, 150, 150))
        canvas.text(cursor_x + 20, y + 3, label, MUTED, scale=1)
        cursor_x += 78
    canvas.rect(cursor_x, y, 14, 14, PANEL)
    canvas.rect_outline(cursor_x, y, 14, 14, (150, 150, 150))
    canvas.text(cursor_x + 20, y + 3, "UNUSED", MUTED, scale=1)


def _draw_context_chart(
    canvas: _Canvas,
    calls: list[dict[str, Any]],
    x: int,
    y: int,
    width: int,
    height: int,
) -> None:
    """Рисуем центральные столбцы заполнения окна."""

    canvas.rect(x, y, width, height, PANEL)
    canvas.rect_outline(x, y, width, height, (180, 185, 190))
    canvas.text(x, y - 26, "WINDOW FILL BY SOURCE TYPE", TEXT, scale=1)
    for ratio, label in (
        (0.0, "0%"),
        (0.25, "25%"),
        (0.5, "50%"),
        (0.75, "75%"),
        (1.0, "100%"),
    ):
        yy = y + int(height * ratio)
        canvas.line(x, yy, x + width, yy, GRID)
        canvas.text(28, yy - 4, label, MUTED, scale=1)
    if not calls:
        canvas.text(x + 24, y + height // 2 - 8, "NO MODEL CALLS", MUTED, scale=2)
        return

    gap, column_width = _column_geometry(len(calls), width)
    # Горизонтальная ось — последовательность model call'ов, поэтому turn 0
    # должен начинаться у левого края, а не выглядеть как "поздний старт".
    start_x = x
    for index, call in enumerate(calls):
        column_x = start_x + index * (column_width + gap)
        _draw_context_column(canvas, call, column_x, y, column_width, height)


def _draw_context_column(
    canvas: _Canvas,
    call: dict[str, Any],
    x: int,
    y: int,
    width: int,
    height: int,
) -> None:
    """Рисуем один агрегированный model call."""

    window_tokens = max(int(call["window_tokens"]), 1)
    used_tokens = max(min(int(call["used_tokens"]), window_tokens), 0)
    tokens_by_type = call["tokens_by_type"]
    cursor = 0
    for source_type in SOURCE_ORDER:
        tokens = max(int(tokens_by_type.get(source_type, 0)), 0)
        if not tokens:
            continue
        segment_start = min(cursor, window_tokens)
        segment_end = min(cursor + tokens, window_tokens)
        cursor = segment_end
        if segment_end <= segment_start:
            continue
        top = y + round(segment_start / window_tokens * height)
        bottom = y + round(segment_end / window_tokens * height)
        canvas.rect(x, top, width, max(bottom - top, 1), SOURCE_COLORS[source_type])
    fill_share = used_tokens / window_tokens
    if fill_share >= 0.90:
        canvas.rect(x, y + height + 4, width, 5, DANGER)
    elif fill_share >= 0.75:
        canvas.rect(x, y + height + 4, width, 5, WARNING)
    if call["red_token_share"] >= 0.50:
        canvas.rect(x, y + height + 18, width, 5, DANGER)


def _draw_sparklines(
    canvas: _Canvas,
    calls: list[dict[str, Any]],
    x: int,
    y: int,
    width: int,
    height: int,
) -> None:
    """Рисуем нижние диагностические полосы по агрегатам turn_heat."""

    canvas.text(x, y - 22, "TURN SIGNALS", TEXT, scale=1)
    metrics = [
        ("TOOL", "raw_tool_share", (194, 58, 82)),
        ("ASST", "assistant_share", SOURCE_COLORS["assistant_message"]),
        ("EVID", "evidence_density", (100, 180, 93)),
        ("POS", "positioned_evidence_score", (86, 166, 217)),
        ("NORM", "normative_status", WARNING),
        ("GOAL", "goal_status", (156, 207, 117)),
        ("COLD", "cold_gap_score", COLD),
    ]
    row_height = height // len(metrics)
    for row, (label, key, color) in enumerate(metrics):
        row_y = y + row * row_height
        canvas.text(28, row_y + 8, label, MUTED, scale=1)
        canvas.rect(x, row_y, width, row_height - 5, PANEL)
        canvas.rect_outline(x, row_y, width, row_height - 5, (220, 224, 230))
        if not calls:
            continue
        gap, column_width = _column_geometry(len(calls), width)
        start_x = x
        for index, call in enumerate(calls):
            value = min(max(float(call.get(key) or 0.0), 0.0), 1.0)
            bar_height = max(1, int((row_height - 9) * value))
            column_x = start_x + index * (column_width + gap)
            canvas.rect(
                column_x,
                row_y + row_height - 6 - bar_height,
                column_width,
                bar_height,
                color,
            )


def _column_geometry(count: int, width: int) -> tuple[int, int]:
    """Подбираем ширину колонок так, чтобы длинная сессия осталась одним PNG."""

    if count <= 0:
        return 0, 1
    gap = 2 if count <= 160 else 1 if count <= 520 else 0
    available = width - gap * max(count - 1, 0)
    if available < count:
        gap = 0
        available = width
    return gap, max(1, min(32, available // count))


def _write_png(path: Path, width: int, height: int, pixels: bytearray) -> None:
    """Записываем RGB PNG без внешних библиотек."""

    raw = bytearray()
    stride = width * 3
    for y in range(height):
        raw.append(0)
        offset = y * stride
        raw.extend(pixels[offset : offset + stride])
    compressed = zlib.compress(bytes(raw), level=9)
    payload = bytearray(b"\x89PNG\r\n\x1a\n")
    payload.extend(_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)))
    payload.extend(_chunk(b"IDAT", compressed))
    payload.extend(_chunk(b"IEND", b""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _chunk(kind: bytes, data: bytes) -> bytes:
    crc = binascii.crc32(kind)
    crc = binascii.crc32(data, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)


class _Canvas:
    """Минимальный RGB canvas с примитивами для отчетного PNG."""

    def __init__(self, width: int, height: int, color: tuple[int, int, int]) -> None:
        self.width = width
        self.height = height
        self.pixels = bytearray(color * (width * height))

    def rect(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        color: tuple[int, int, int],
    ) -> None:
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(self.width, x + max(width, 0))
        y1 = min(self.height, y + max(height, 0))
        if x1 <= x0 or y1 <= y0:
            return
        row = bytes(color) * (x1 - x0)
        for yy in range(y0, y1):
            start = (yy * self.width + x0) * 3
            self.pixels[start : start + len(row)] = row

    def rect_outline(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        color: tuple[int, int, int],
    ) -> None:
        self.rect(x, y, width, 1, color)
        self.rect(x, y + height - 1, width, 1, color)
        self.rect(x, y, 1, height, color)
        self.rect(x + width - 1, y, 1, height, color)

    def line(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        color: tuple[int, int, int],
    ) -> None:
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            self.rect(x0, y0, 1, 1, color)
            if x0 == x1 and y0 == y1:
                break
            twice = 2 * err
            if twice >= dy:
                err += dy
                x0 += sx
            if twice <= dx:
                err += dx
                y0 += sy

    def text(
        self,
        x: int,
        y: int,
        text: str,
        color: tuple[int, int, int],
        scale: int = 1,
    ) -> None:
        cursor = x
        for char in _ascii(text.upper()):
            glyph = FONT_5X7.get(char, FONT_5X7["?"])
            for row_index, row in enumerate(glyph):
                for column_index, value in enumerate(row):
                    if value == "1":
                        self.rect(
                            cursor + column_index * scale,
                            y + row_index * scale,
                            scale,
                            scale,
                            color,
                        )
            cursor += 6 * scale


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
    if math.isnan(number) or math.isinf(number):
        return default
    return number


def _source_type(value: str) -> str:
    if value in SOURCE_COLORS:
        return value
    return "unknown"


def _ascii(value: str) -> str:
    return "".join(char if 32 <= ord(char) <= 126 else "?" for char in value)
