"""Статический HTML-renderer тепловой карты."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

from .io import read_jsonl


FRAGMENT_TYPE_ORDER = {
    "system_instruction": 0,
    "developer_instruction": 1,
    "user_message": 2,
    "context_fragment": 3,
    "file_snippet": 4,
    "test_result": 5,
    "tool_schema": 6,
    "assistant_message": 7,
    "tool_output": 8,
    "unknown": 9,
}

FRAGMENT_TYPE_COLORS = {
    "system_instruction": "#f4b6b5",
    "developer_instruction": "#f2994a",
    "user_message": "#56a6d9",
    "context_fragment": "#9ccf75",
    "file_snippet": "#6fc2b0",
    "test_result": "#64b45d",
    "tool_schema": "#f6e4a7",
    "assistant_message": "#b79fe6",
    "tool_output": "#c23a52",
    "unknown": "#c9ced6",
}


def render_html_report(report_path: Path, out_path: Path) -> None:
    """Создает самодостаточный HTML по соседним JSONL-артефактам."""

    root = report_path.parent
    report = json.loads(report_path.read_text(encoding="utf-8"))
    turn_heat = read_jsonl(root / "turn_heat.jsonl") if (root / "turn_heat.jsonl").exists() else []
    fragment_heat = (
        read_jsonl(root / "fragment_heat.jsonl")
        if (root / "fragment_heat.jsonl").exists()
        else []
    )
    fragments = (
        read_jsonl(root / "fragments.jsonl") if (root / "fragments.jsonl").exists() else []
    )
    packets = read_jsonl(root / "packets.jsonl") if (root / "packets.jsonl").exists() else []
    events = read_jsonl(root / "events.jsonl") if (root / "events.jsonl").exists() else []
    findings = read_jsonl(root / "findings.jsonl") if (root / "findings.jsonl").exists() else []
    payload = {
        "report": report,
        "turn_heat": turn_heat,
        "fragment_heat": fragment_heat,
        "fragments": [_safe_fragment_for_html(fragment) for fragment in fragments],
        "packets": packets,
        "tool_call_names": _tool_call_names(events),
        "findings": findings,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render(payload), encoding="utf-8")


def _render(payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False)
    fragments_by_id = {
        str(fragment.get("fragment_id")): fragment
        for fragment in payload.get("fragments", [])
    }
    rows = _heat_rows(
        payload["fragment_heat"],
        fragments_by_id,
        payload.get("tool_call_names", {}),
    )
    columns = _context_window_columns(
        payload.get("packets", []),
        payload["fragment_heat"],
        fragments_by_id,
        payload.get("tool_call_names", {}),
    )
    findings = _finding_list(payload["findings"])
    session_id = html.escape(str(payload["report"].get("session_id") or "session"))
    return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <title>Context Heatmap: {session_id}</title>
  <style>
    body {{ margin: 0; font: 14px/1.45 -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; color: #202124; background: #f7f8fa; }}
    header {{ padding: 24px 32px 12px; background: #fff; border-bottom: 1px solid #dde1e6; }}
    main {{ display: grid; grid-template-columns: minmax(0, 1fr) 380px; gap: 20px; padding: 20px 32px 32px; }}
    h1 {{ margin: 0 0 8px; font-size: 24px; letter-spacing: 0; }}
    h2 {{ margin: 0 0 12px; font-size: 16px; letter-spacing: 0; }}
    h3 {{ margin: 0 0 8px; font-size: 14px; letter-spacing: 0; }}
    .metrics {{ display: flex; gap: 16px; flex-wrap: wrap; color: #5f6368; }}
    .panel {{ min-width: 0; background: #fff; border: 1px solid #dde1e6; border-radius: 8px; padding: 16px; }}
    .heatmap {{ overflow: auto; }}
    .tabs {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 14px; border-bottom: 1px solid #dde1e6; }}
    .tab-button {{ appearance: none; border: 0; border-bottom: 3px solid transparent; background: transparent; color: #3c4043; cursor: pointer; padding: 8px 10px 9px; font: inherit; font-weight: 700; }}
    .tab-button[aria-selected="true"] {{ color: #174ea6; border-bottom-color: #174ea6; }}
    .tab-panel {{ min-width: 0; }}
    .tab-panel[hidden] {{ display: none; }}
    table {{ border-collapse: collapse; min-width: 760px; width: 100%; }}
    th, td {{ border: 1px solid #e4e7eb; padding: 6px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f1f3f4; position: sticky; top: 0; }}
    .cell {{ min-width: 92px; min-height: 34px; color: #111; font-size: 12px; text-align: center; border-radius: 4px; padding: 4px 6px; }}
    .cell strong {{ display: block; font-size: 13px; }}
    .cell small {{ display: block; color: #3c4043; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 120px; }}
    .fragment-main {{ display: block; font-weight: 700; }}
    .fragment-meta {{ display: block; margin-top: 2px; color: #5f6368; font-size: 11px; font-weight: 400; }}
    .context-window {{ min-width: 0; max-width: 100%; padding-bottom: 4px; }}
    .context-window-track {{ width: 100%; max-width: 100%; overflow-x: auto; padding-bottom: 4px; }}
    .context-window-legend {{ position: sticky; top: 0; z-index: 3; width: 100%; max-width: 100%; overflow-x: auto; margin-bottom: 14px; padding: 8px 0 10px; color: #5f6368; font-size: 12px; background: #fff; scrollbar-gutter: stable; }}
    .context-window-legend-track {{ display: flex; flex-wrap: nowrap; gap: 8px 12px; width: max-content; min-width: max-content; }}
    .legend-item {{ display: inline-flex; flex: 0 0 auto; align-items: center; gap: 5px; white-space: nowrap; }}
    .legend-swatch {{ width: 12px; height: 12px; border-radius: 3px; border: 1px solid rgba(0,0,0,.18); }}
    .context-columns {{ display: flex; align-items: flex-start; gap: 18px; width: max-content; min-width: max-content; }}
    .context-column-wrap {{ width: 116px; flex: 0 0 116px; }}
    .context-column-meta {{ min-height: 52px; margin-bottom: 8px; color: #5f6368; font-size: 12px; }}
    .context-column-meta strong {{ display: block; color: #202124; font-size: 13px; }}
    .estimated {{ color: #8a5a00; font-weight: 700; }}
    .context-column {{ position: relative; width: 72px; height: 460px; margin: 0 auto; background: #fff; border: 1px solid #202124; box-shadow: inset 0 0 0 1px rgba(0,0,0,.04); }}
    .context-unused {{ position: absolute; left: 0; right: 0; bottom: 0; background: #fff; border-top: 1px dashed #b8bec8; pointer-events: none; }}
    .context-request-marker {{ position: absolute; left: -7px; right: -7px; height: 0; border-top: 2px solid #202124; opacity: .34; pointer-events: none; }}
    .context-segment {{ position: absolute; left: 0; width: 100%; min-height: 3px; border: 0; border-top: 1px solid rgba(255,255,255,.72); border-bottom: 1px solid rgba(0,0,0,.18); color: #111; cursor: pointer; padding: 0; overflow: hidden; }}
    .context-segment:hover, .context-segment:focus-visible {{ outline: 2px solid #174ea6; outline-offset: 1px; z-index: 2; }}
    .context-segment.hot {{ box-shadow: inset 0 0 0 2px #7f1d1d; }}
    .context-segment > span {{ display: block; padding: 2px 3px; font-size: 10px; line-height: 1.1; text-align: center; word-break: break-word; }}
    .context-detail {{ position: sticky; top: 16px; align-self: start; max-height: calc(100vh - 32px); overflow: auto; }}
    .context-detail-empty {{ color: #5f6368; }}
    .detail-grid {{ display: grid; gap: 8px; }}
    .detail-row {{ border-top: 1px solid #eceff3; padding-top: 8px; }}
    .detail-row:first-child {{ border-top: 0; padding-top: 0; }}
    .detail-key {{ display: block; color: #5f6368; font-size: 11px; text-transform: uppercase; }}
    .detail-value {{ display: block; overflow-wrap: anywhere; }}
    .type-system_instruction {{ background: #f4b6b5; }}
    .type-developer_instruction {{ background: #f2994a; }}
    .type-user_message {{ background: #56a6d9; }}
    .type-context_fragment {{ background: #9ccf75; }}
    .type-file_snippet {{ background: #6fc2b0; }}
    .type-test_result {{ background: #64b45d; }}
    .type-tool_schema {{ background: #f6e4a7; }}
    .type-assistant_message {{ background: #b79fe6; }}
    .type-tool_output {{ background: #c23a52; color: #fff; }}
    .type-unknown {{ background: #c9ced6; }}
    .cold {{ display: inline-block; margin-left: 4px; color: #1558d6; font-weight: 700; }}
    .finding {{ border-top: 1px solid #eceff3; padding: 10px 0; }}
    .finding:first-child {{ border-top: 0; padding-top: 0; }}
    .sev-critical, .sev-high {{ color: #b3261e; font-weight: 700; }}
    .sev-medium {{ color: #b06000; font-weight: 700; }}
    .sev-low {{ color: #5f6368; font-weight: 700; }}
    code {{ background: #eef1f4; padding: 1px 4px; border-radius: 4px; }}
    @media (max-width: 900px) {{ main {{ grid-template-columns: 1fr; padding: 16px; }} header {{ padding: 20px 16px 10px; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Context Heatmap: <code>{session_id}</code></h1>
    <div class="metrics">
      <span>model calls: {payload['report'].get('model_calls', 0)}</span>
      <span>max red token share: {payload['report'].get('max_red_token_share', 0)}</span>
      <span>fixed instruction cost: {payload['report'].get('max_fixed_instruction_cost', 0)}</span>
      <span>goal anchor cost: {payload['report'].get('max_goal_anchor_cost', 0)}</span>
      <span>protected status: {payload['report'].get('max_normative_status', 0)}/{payload['report'].get('max_goal_status', 0)}</span>
      <span>max cold gap: {payload['report'].get('max_cold_gap_score', 0)}</span>
      <span>findings: {payload['report'].get('findings', 0)}</span>
    </div>
  </header>
  <main>
    <section class="panel">
      <div class="tabs" role="tablist" aria-label="Context heatmap views">
        <button class="tab-button" id="tab-fragment-heat" role="tab" aria-selected="true" aria-controls="panel-fragment-heat" data-tab-target="panel-fragment-heat">Fragment Heat</button>
        <button class="tab-button" id="tab-context-window" role="tab" aria-selected="false" aria-controls="panel-context-window" data-tab-target="panel-context-window">Context Window</button>
        <button class="tab-button" id="tab-findings" role="tab" aria-selected="false" aria-controls="panel-findings" data-tab-target="panel-findings">Findings</button>
      </div>
      <section class="tab-panel heatmap" id="panel-fragment-heat" role="tabpanel" aria-labelledby="tab-fragment-heat">
        <h2>Fragment Heat</h2>
        {rows}
      </section>
      <section class="tab-panel" id="panel-context-window" role="tabpanel" aria-labelledby="tab-context-window" hidden>
        <h2>Context Window</h2>
        {columns}
      </section>
      <section class="tab-panel" id="panel-findings" role="tabpanel" aria-labelledby="tab-findings" hidden>
        <h2>Findings</h2>
        {findings}
      </section>
    </section>
    <aside class="panel context-detail" id="context-detail">
      <h2>Context Detail</h2>
      <div id="context-detail-content" class="context-detail-empty">Выберите сегмент в Context Window, чтобы увидеть безопасные метаданные фрагмента.</div>
    </aside>
  </main>
  <script type="application/json" id="heatmap-data">{html.escape(data)}</script>
  <script>
    (() => {{
      const tabs = Array.from(document.querySelectorAll(".tab-button"));
      const panels = Array.from(document.querySelectorAll(".tab-panel"));
      for (const tab of tabs) {{
        tab.addEventListener("click", () => {{
          for (const item of tabs) {{
            item.setAttribute("aria-selected", String(item === tab));
          }}
          for (const panel of panels) {{
            panel.hidden = panel.id !== tab.dataset.tabTarget;
          }}
        }});
      }}

      const detailRoot = document.getElementById("context-detail-content");
      for (const segment of document.querySelectorAll(".context-segment")) {{
        segment.addEventListener("click", () => {{
          const raw = segment.getAttribute("data-detail") || "{{}}";
          let detail = {{}};
          try {{
            detail = JSON.parse(raw);
          }} catch (_error) {{
            detail = {{ error: "detail payload is invalid" }};
          }}
          renderDetail(detailRoot, detail);
        }});
      }}

      function renderDetail(root, detail) {{
        root.classList.remove("context-detail-empty");
        root.textContent = "";
        const grid = document.createElement("div");
        grid.className = "detail-grid";
        for (const [key, value] of Object.entries(detail)) {{
          const row = document.createElement("div");
          row.className = "detail-row";
          const label = document.createElement("span");
          label.className = "detail-key";
          label.textContent = key;
          const output = document.createElement("span");
          output.className = "detail-value";
          output.textContent = formatValue(value);
          row.append(label, output);
          grid.append(row);
        }}
        root.append(grid);
      }}

      function formatValue(value) {{
        if (Array.isArray(value)) {{
          return value.length ? value.join(", ") : "none";
        }}
        if (value && typeof value === "object") {{
          return JSON.stringify(value);
        }}
        if (value === "" || value === null || value === undefined) {{
          return "none";
        }}
        return String(value);
      }}
    }})();
  </script>
</body>
</html>
"""


def _context_window_columns(
    packets: list[dict],
    fragment_heat: list[dict],
    fragments_by_id: dict[str, dict],
    tool_call_names: dict[str, str],
) -> str:
    """Рисуем полное контекстное окно по каждому обращению к модели."""

    if not packets:
        return "<p>Нет packets.jsonl для отображения Context Window.</p>"
    heat_by_key = {
        (str(row.get("fragment_id")), str(row.get("model_call_id"))): row
        for row in fragment_heat
    }
    legend = _context_window_legend()
    columns = [
        _context_window_column(packet, heat_by_key, fragments_by_id, tool_call_names)
        for packet in packets
    ]
    return (
        '<div class="context-window">'
        f"{legend}"
        '<div class="context-window-track">'
        f'<div class="context-columns">{"".join(columns)}</div>'
        "</div>"
        "</div>"
    )


def _safe_fragment_for_html(fragment: dict) -> dict:
    """Оставляем для HTML только индексные поля, без excerpt-ов и raw content."""

    safe: dict = {}
    for key in (
        "fragment_id",
        "session_id",
        "source_type",
        "source_name",
        "tokens",
        "token_count_method",
        "trust",
        "taint",
        "validity",
        "authority_level",
        "context_layer",
        "evictability",
        "stability",
        "applicability",
        "normative_role",
        "goal_role",
        "target_paths",
        "created_by_event_id",
        "content_hash",
        "metadata",
    ):
        if key in fragment:
            safe[key] = fragment[key]
    return safe


def _context_window_legend() -> str:
    """Показываем, что цвет сегмента означает тип фрагмента, а не heat."""

    items = []
    for source_type, _rank in sorted(FRAGMENT_TYPE_ORDER.items(), key=lambda item: item[1]):
        color = FRAGMENT_TYPE_COLORS.get(source_type, FRAGMENT_TYPE_COLORS["unknown"])
        items.append(
            '<span class="legend-item">'
            f'<span class="legend-swatch" style="background:{color}"></span>'
            f"{html.escape(source_type)}</span>"
        )
    items.append(
        '<span class="legend-item">'
        '<span class="legend-swatch" style="background:#fff"></span>unused</span>'
    )
    return (
        '<div class="context-window-legend">'
        f'<div class="context-window-legend-track">{"".join(items)}</div>'
        "</div>"
    )


def _context_window_column(
    packet: dict,
    heat_by_key: dict[tuple[str, str], dict],
    fragments_by_id: dict[str, dict],
    tool_call_names: dict[str, str],
) -> str:
    model_call_id = str(packet.get("model_call_id") or "")
    turn_id = int(packet.get("turn_id") or 0)
    packet_fragments = [
        item for item in packet.get("fragments") or [] if isinstance(item, dict)
    ]
    input_tokens = _safe_int(packet.get("input_tokens"))
    window_tokens = _safe_int(packet.get("context_window_tokens"))
    fragment_tokens = sum(max(_safe_int(item.get("tokens")), 0) for item in packet_fragments)
    estimated = window_tokens <= 0
    if estimated:
        window_tokens = max(input_tokens, fragment_tokens, 1)
    used_tokens = min(max(input_tokens or fragment_tokens, fragment_tokens), window_tokens)
    unused_tokens = max(window_tokens - used_tokens, 0)
    unused_percent = _percent(unused_tokens, window_tokens)
    unused_top = _percent(used_tokens, window_tokens)
    request_marker = _request_marker(input_tokens, window_tokens) if input_tokens else ""
    segments = [
        _context_window_segment(
            item,
            packet,
            heat_by_key.get((str(item.get("fragment_id")), model_call_id), {}),
            fragments_by_id,
            tool_call_names,
            window_tokens,
        )
        for item in packet_fragments
    ]
    estimated_label = ' <span class="estimated">estimated</span>' if estimated else ""
    return (
        '<article class="context-column-wrap">'
        '<div class="context-column-meta">'
        f"<strong>turn {turn_id}</strong>"
        f"<span>{html.escape(model_call_id or 'model call')}</span><br>"
        f"<span>{used_tokens}/{window_tokens} tokens{estimated_label}</span>"
        "</div>"
        f'<div class="context-column" data-model-call-id="{html.escape(model_call_id, quote=True)}">'
        f'{"".join(segments)}'
        f"{request_marker}"
        f'<div class="context-unused" style="top:{unused_top:.4f}%; height:{unused_percent:.4f}%"></div>'
        "</div>"
        "</article>"
    )


def _context_window_segment(
    packet_fragment: dict,
    packet: dict,
    heat: dict,
    fragments_by_id: dict[str, dict],
    tool_call_names: dict[str, str],
    window_tokens: int,
) -> str:
    fragment_id = str(packet_fragment.get("fragment_id") or "")
    model_call_id = str(packet.get("model_call_id") or "")
    source_type = str(packet_fragment.get("source_type") or "unknown")
    css_type = _source_type_class(source_type)
    start = max(_safe_int(packet_fragment.get("position_start")), 0)
    end = max(_safe_int(packet_fragment.get("position_end")), start)
    tokens = max(_safe_int(packet_fragment.get("tokens")), end - start, 1)
    top = _percent(start, window_tokens)
    height = max(_percent(tokens, window_tokens), 0.6)
    heat_value = float(heat.get("heat") or 0.0)
    hot_class = (
        " hot"
        if heat_value >= 0.75 and not heat.get("excluded_from_red_token_share")
        else ""
    )
    label_html = _fragment_label(fragment_id, fragments_by_id, tool_call_names)
    title = html.escape(
        _fragment_title(fragment_id, fragments_by_id, tool_call_names),
        quote=True,
    )
    detail = _context_segment_detail(
        packet_fragment,
        packet,
        heat,
        fragments_by_id.get(fragment_id, {}),
        tool_call_names,
        window_tokens,
    )
    detail_json = html.escape(json.dumps(detail, ensure_ascii=False), quote=True)
    return (
        f'<button type="button" class="context-segment type-{css_type}{hot_class}" '
        f'data-model-call-id="{html.escape(model_call_id, quote=True)}" '
        f'data-fragment-id="{html.escape(fragment_id, quote=True)}" '
        f'data-source-type="{html.escape(source_type, quote=True)}" '
        f'data-detail="{detail_json}" '
        f'title="{title}" '
        f'style="top:{top:.4f}%; height:{height:.4f}%">'
        f"<span>{label_html}</span>"
        "</button>"
    )


def _context_segment_detail(
    packet_fragment: dict,
    packet: dict,
    heat: dict,
    fragment: dict,
    tool_call_names: dict[str, str],
    window_tokens: int,
) -> dict:
    """Собираем только безопасные метаданные для клика по сегменту."""

    fragment_id = str(packet_fragment.get("fragment_id") or "")
    metadata = fragment.get("metadata") if isinstance(fragment.get("metadata"), dict) else {}
    source_ref = str(metadata.get("source_ref") or "")
    start = max(_safe_int(packet_fragment.get("position_start")), 0)
    end = max(_safe_int(packet_fragment.get("position_end")), start)
    tokens = max(_safe_int(packet_fragment.get("tokens")), end - start, 0)
    input_tokens = max(_safe_int(packet.get("input_tokens")), 1)
    return {
        "label": _fragment_plain_label(fragment_id, fragment, tool_call_names),
        "turn": _safe_int(packet.get("turn_id")),
        "model_call_id": str(packet.get("model_call_id") or ""),
        "fragment_id": fragment_id,
        "source_type": str(
            packet_fragment.get("source_type") or fragment.get("source_type") or "unknown"
        ),
        "source_name": str(fragment.get("source_name") or ""),
        "source_ref": source_ref,
        "tokens": tokens,
        "position_start": start,
        "position_end": end,
        "request_share": round(tokens / input_tokens, 4),
        "window_share": round(tokens / max(window_tokens, 1), 4),
        "heat": _safe_float(heat.get("heat")),
        "ordinary_cost": _safe_float(heat.get("ordinary_cost")),
        "protected_status": _safe_float(heat.get("protected_status")),
        "excluded_from_red_token_share": bool(
            heat.get("excluded_from_red_token_share")
        ),
        "color": str(heat.get("color") or ""),
        "context_layer": str(
            heat.get("context_layer")
            or fragment.get("context_layer")
            or packet_fragment.get("context_layer")
            or "unknown"
        ),
        "authority_level": str(
            heat.get("authority_level")
            or fragment.get("authority_level")
            or packet_fragment.get("authority_level")
            or "unknown"
        ),
        "evictability": str(fragment.get("evictability") or packet_fragment.get("evictability") or ""),
        "stability": str(fragment.get("stability") or packet_fragment.get("stability") or ""),
        "applicability": str(fragment.get("applicability") or packet_fragment.get("applicability") or ""),
        "normative_role": str(fragment.get("normative_role") or packet_fragment.get("normative_role") or ""),
        "goal_role": str(fragment.get("goal_role") or packet_fragment.get("goal_role") or ""),
        "axes": heat.get("axes") if isinstance(heat.get("axes"), dict) else {},
        "reasons": heat.get("reasons") if isinstance(heat.get("reasons"), list) else [],
        "protected_reasons": heat.get("protected_reasons")
        if isinstance(heat.get("protected_reasons"), list)
        else [],
        "confidence": _safe_float(heat.get("confidence") or metadata.get("confidence")),
        "included_because": str(metadata.get("included_because") or ""),
        "content_hash": str(fragment.get("content_hash") or ""),
        "target_paths": fragment.get("target_paths") if isinstance(fragment.get("target_paths"), list) else [],
        "packet_warnings": packet.get("warnings") if isinstance(packet.get("warnings"), list) else [],
    }


def _request_marker(input_tokens: int, window_tokens: int) -> str:
    top = min(_percent(input_tokens, window_tokens), 100.0)
    return (
        '<div class="context-request-marker" '
        f'style="top:{top:.4f}%" title="request tokens estimate"></div>'
    )


def _heat_rows(
    fragment_heat: list[dict],
    fragments_by_id: dict[str, dict],
    tool_call_names: dict[str, str],
) -> str:
    if not fragment_heat:
        return "<p>Нет данных heat для отображения.</p>"
    model_calls = sorted(
        {str(row.get("model_call_id")) for row in fragment_heat},
        key=_model_call_sort_key,
    )
    fragments = sorted(
        {str(row.get("fragment_id")) for row in fragment_heat},
        key=lambda value: _fragment_sort_key(value, fragments_by_id),
    )
    heat_by_key = {
        (str(row.get("fragment_id")), str(row.get("model_call_id"))): row
        for row in fragment_heat
    }
    header = "".join(f"<th>{html.escape(call.split(':')[-1])}</th>" for call in model_calls)
    body_rows = []
    for fragment in fragments:
        cells = []
        for call in model_calls:
            row = heat_by_key.get((fragment, call))
            if not row:
                cells.append("<td></td>")
                continue
            heat = float(row.get("heat") or 0.0)
            color = _semantic_color(str(row.get("color") or ""), heat)
            label = f"{heat:.2f}"
            reasons_text = ", ".join(row.get("reasons") or []) or "no reasons"
            reasons = html.escape(reasons_text)
            cells.append(
                f'<td><div class="cell" title="{reasons}" '
                f'data-reasons="{reasons}" '
                f'style="background:{color}"><strong>{label}</strong>'
                f"<small>{reasons}</small></div></td>"
            )
        label = _fragment_label(fragment, fragments_by_id, tool_call_names)
        title = html.escape(_fragment_title(fragment, fragments_by_id, tool_call_names), quote=True)
        body_rows.append(f'<tr><th title="{title}">{label}</th>{"".join(cells)}</tr>')
    return f"<table><thead><tr><th>fragment</th>{header}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def _fragment_label(
    fragment_id: str,
    fragments_by_id: dict[str, dict],
    tool_call_names: dict[str, str],
) -> str:
    """Делаем строку матрицы читаемой, не раскрывая содержимое фрагмента."""

    fragment = fragments_by_id.get(fragment_id)
    if not fragment:
        return html.escape(_short(fragment_id))
    source_type = str(fragment.get("source_type") or "fragment")
    source_name = str(fragment.get("source_name") or "")
    metadata = fragment.get("metadata") if isinstance(fragment.get("metadata"), dict) else {}
    source_ref = str(metadata.get("source_ref") or "")
    if source_type == "tool_schema":
        tool_index = metadata.get("tool_index")
        detail = f"tools[{tool_index}]" if tool_index is not None else source_ref
        main = f"tool_schema: {source_name or detail or fragment_id}"
    elif source_type == "tool_output":
        tool_name = source_name if source_name and source_name != "tool" else ""
        tool_call_id = str(metadata.get("tool_call_id") or "")
        tool_name = tool_name or str(
            metadata.get("tool_name")
            or tool_call_names.get(tool_call_id)
            or tool_call_id
            or "tool"
        )
        detail = source_ref or _short(fragment_id, 48)
        main = f"tool_output: {tool_name}"
    else:
        detail = source_ref or _short(fragment_id, 48)
        main = f"{source_type}: {source_name}" if source_name else source_type
    return (
        f'<span class="fragment-main">{html.escape(_short(main, 48))}</span>'
        f'<span class="fragment-meta">{html.escape(_short(detail or fragment_id, 56))}</span>'
    )


def _fragment_plain_label(
    fragment_id: str,
    fragment: dict,
    tool_call_names: dict[str, str],
) -> str:
    """Возвращаем тот же смысл подписи без HTML для панели деталей."""

    if not fragment:
        return _short(fragment_id)
    source_type = str(fragment.get("source_type") or "fragment")
    source_name = str(fragment.get("source_name") or "")
    metadata = fragment.get("metadata") if isinstance(fragment.get("metadata"), dict) else {}
    source_ref = str(metadata.get("source_ref") or "")
    if source_type == "tool_schema":
        tool_index = metadata.get("tool_index")
        detail = f"tools[{tool_index}]" if tool_index is not None else source_ref
        return f"tool_schema: {source_name or detail or fragment_id}"
    if source_type == "tool_output":
        tool_name = source_name if source_name and source_name != "tool" else ""
        tool_call_id = str(metadata.get("tool_call_id") or "")
        tool_name = tool_name or str(
            metadata.get("tool_name")
            or tool_call_names.get(tool_call_id)
            or tool_call_id
            or "tool"
        )
        return f"tool_output: {tool_name}"
    if source_name:
        return f"{source_type}: {source_name}"
    return source_type


def _fragment_title(
    fragment_id: str,
    fragments_by_id: dict[str, dict],
    tool_call_names: dict[str, str],
) -> str:
    """Показываем технический идентификатор в tooltip, чтобы связь не терялась."""

    fragment = fragments_by_id.get(fragment_id)
    if not fragment:
        return fragment_id
    source_type = str(fragment.get("source_type") or "fragment")
    source_name = str(fragment.get("source_name") or "")
    metadata = fragment.get("metadata") if isinstance(fragment.get("metadata"), dict) else {}
    source_ref = str(metadata.get("source_ref") or "")
    tool_call_id = str(metadata.get("tool_call_id") or "")
    parts = [fragment_id, f"type={source_type}"]
    if source_name:
        parts.append(f"name={source_name}")
    if tool_call_id:
        parts.append(f"tool_call_id={tool_call_id}")
    if tool_call_id in tool_call_names:
        parts.append(f"tool={tool_call_names[tool_call_id]}")
    if source_ref:
        parts.append(f"ref={source_ref}")
    return " | ".join(parts)


def _fragment_sort_key(fragment_id: str, fragments_by_id: dict[str, dict]) -> tuple:
    """Сортируем строки HTML по роли фрагмента в prompt-пакете."""

    fragment = fragments_by_id.get(fragment_id)
    if not fragment:
        return (len(FRAGMENT_TYPE_ORDER), _natural_sort_key(fragment_id))
    source_type = str(fragment.get("source_type") or "")
    metadata = fragment.get("metadata") if isinstance(fragment.get("metadata"), dict) else {}
    type_rank = FRAGMENT_TYPE_ORDER.get(source_type, len(FRAGMENT_TYPE_ORDER))
    order_hint = _order_hint(source_type, metadata)
    source_ref = str(metadata.get("source_ref") or "")
    source_name = str(fragment.get("source_name") or "")
    return (
        type_rank,
        order_hint,
        _natural_sort_key(source_ref),
        _natural_sort_key(source_name),
        _natural_sort_key(fragment_id),
    )


def _order_hint(source_type: str, metadata: dict) -> int:
    """Достаем локальный порядок внутри группы, если он есть в telemetry."""

    if source_type == "tool_schema" and isinstance(metadata.get("tool_index"), int):
        return int(metadata["tool_index"])
    if isinstance(metadata.get("history_index"), int):
        return int(metadata["history_index"])
    return 0


def _natural_sort_key(value: str) -> tuple:
    """Сравниваем `history[10]` после `history[2]`, а не между 1 и 2."""

    parts = re.split(r"(\d+)", value)
    return tuple(int(part) if part.isdigit() else part for part in parts)


def _model_call_sort_key(value: str) -> tuple:
    """Сортируем колонки turn по числовому suffix model_call_id."""

    prefix, _, suffix = value.rpartition(":")
    if suffix.isdigit():
        return (prefix, 0, int(suffix))
    return (prefix, 1, suffix)


def _tool_call_names(events: list[dict]) -> dict[str, str]:
    """Строим маленькую карту `tool_call_id -> имя`, не встраивая trace в HTML."""

    names: dict[str, str] = {}
    for event in events:
        if event.get("event_type") != "model_output":
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        message = payload.get("message")
        if not isinstance(message, dict):
            continue
        calls = message.get("tool_calls")
        if not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, dict):
                continue
            call_id = str(call.get("id") or "")
            function = call.get("function")
            if not call_id or not isinstance(function, dict):
                continue
            name = str(function.get("name") or "")
            if name:
                names[call_id] = name
    return names


def _finding_list(findings: list[dict]) -> str:
    if not findings:
        return "<p>Нет findings. Холодные дыры не обнаружены.</p>"
    items = []
    for finding in findings[:12]:
        severity = html.escape(str(finding.get("severity") or "low"))
        title = html.escape(str(finding.get("title") or "Finding"))
        explanation = html.escape(str(finding.get("explanation") or ""))
        recommendation = html.escape(str(finding.get("recommendation") or ""))
        cold = '<span class="cold" title="cold gap">cold</span>' if finding.get("kind") == "cold_gap" else ""
        items.append(
            f'<div class="finding"><div class="sev-{severity}">{severity} {cold}</div>'
            f"<strong>{title}</strong><p>{explanation}</p><p>{recommendation}</p></div>"
        )
    return "".join(items)


def _semantic_color(color: str, value: float) -> str:
    """Берём новый semantic color или старый градиент heat."""

    colors = {
        "gray": "#e4e7eb",
        "green": "#dff3e3",
        "yellow": "#fff2bf",
        "orange": "#ffd6a6",
        "red": "#f3a6a0",
    }
    return colors.get(color, _heat_color(value))


def _heat_color(value: float) -> str:
    value = max(0.0, min(1.0, value))
    if value < 0.25:
        return "#dff3e3"
    if value < 0.50:
        return "#fff2bf"
    if value < 0.75:
        return "#ffd6a6"
    return "#f3a6a0"


def _short(value: str, limit: int = 54) -> str:
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _source_type_class(source_type: str) -> str:
    """Ограничиваем CSS-класс известной палитрой фрагментов."""

    if source_type not in FRAGMENT_TYPE_COLORS:
        return "unknown"
    return source_type


def _safe_int(value: object, default: int = 0) -> int:
    """Читаем числа из JSONL без падения на внешних значениях."""

    if isinstance(value, bool):
        return default
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _safe_float(value: object, default: float = 0.0) -> float:
    """Читаем float-поля heat/confidence из внешних отчетов."""

    if isinstance(value, bool):
        return default
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _percent(value: int, total: int) -> float:
    """Переводим токены в процент высоты контекстного окна."""

    return 100.0 * max(value, 0) / max(total, 1)
