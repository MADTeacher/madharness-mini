"""Статический HTML-renderer тепловой карты."""

from __future__ import annotations

import html
import json
from pathlib import Path

from .io import read_jsonl


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
    findings = read_jsonl(root / "findings.jsonl") if (root / "findings.jsonl").exists() else []
    payload = {
        "report": report,
        "turn_heat": turn_heat,
        "fragment_heat": fragment_heat,
        "findings": findings,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render(payload), encoding="utf-8")


def _render(payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False)
    rows = _heat_rows(payload["fragment_heat"])
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
    main {{ display: grid; grid-template-columns: minmax(0, 1fr) 360px; gap: 20px; padding: 20px 32px 32px; }}
    h1 {{ margin: 0 0 8px; font-size: 24px; letter-spacing: 0; }}
    h2 {{ margin: 0 0 12px; font-size: 16px; letter-spacing: 0; }}
    .metrics {{ display: flex; gap: 16px; flex-wrap: wrap; color: #5f6368; }}
    .panel {{ background: #fff; border: 1px solid #dde1e6; border-radius: 8px; padding: 16px; }}
    .heatmap {{ overflow: auto; }}
    table {{ border-collapse: collapse; min-width: 760px; width: 100%; }}
    th, td {{ border: 1px solid #e4e7eb; padding: 6px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f1f3f4; position: sticky; top: 0; }}
    .cell {{ min-width: 92px; min-height: 34px; color: #111; font-size: 12px; text-align: center; border-radius: 4px; padding: 4px 6px; }}
    .cell strong {{ display: block; font-size: 13px; }}
    .cell small {{ display: block; color: #3c4043; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 120px; }}
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
      <span>max cold gap: {payload['report'].get('max_cold_gap_score', 0)}</span>
      <span>findings: {payload['report'].get('findings', 0)}</span>
    </div>
  </header>
  <main>
    <section class="panel heatmap">
      <h2>Fragment Heat</h2>
      {rows}
    </section>
    <aside class="panel">
      <h2>Findings</h2>
      {findings}
    </aside>
  </main>
  <script type="application/json" id="heatmap-data">{html.escape(data)}</script>
</body>
</html>
"""


def _heat_rows(fragment_heat: list[dict]) -> str:
    if not fragment_heat:
        return "<p>Нет данных heat для отображения.</p>"
    model_calls = sorted({str(row.get("model_call_id")) for row in fragment_heat})
    fragments = sorted(
        {str(row.get("fragment_id")) for row in fragment_heat},
        key=lambda value: value[:80],
    )[:60]
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
            color = _heat_color(heat)
            label = f"{heat:.2f}"
            reasons_text = ", ".join(row.get("reasons") or []) or "no reasons"
            reasons = html.escape(reasons_text)
            cells.append(
                f'<td><div class="cell" title="{reasons}" '
                f'data-reasons="{reasons}" '
                f'style="background:{color}"><strong>{label}</strong>'
                f"<small>{reasons}</small></div></td>"
            )
        body_rows.append(
            f"<tr><th>{html.escape(_short(fragment))}</th>{''.join(cells)}</tr>"
        )
    return f"<table><thead><tr><th>fragment</th>{header}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


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
