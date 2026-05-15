#!/usr/bin/env python3
"""
PDF Dataset Dashboard Generator
Scans metadata files + PDF sizes and generates a self-contained HTML dashboard.

Usage:
    python3 scripts/dashboard.py
    python3 scripts/dashboard.py --pdf-dir upsc_pdfs/ --meta-dir dataset_output_final/ --output dashboard.html
    python3 scripts/dashboard.py --open   # auto-open in browser after generating
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_stats(pdf_root: Path, meta_root: Path) -> dict:
    """
    Returns a stats dict with overall totals and per-subject breakdown.
    Reads:
      - *_metadata.json  → pages, chunks, chars, quality, strategy
      - actual PDF files → file size on disk
    """
    subjects: dict[str, dict] = {}

    # Build a quick lookup: pdf stem → file size
    pdf_sizes: dict[str, int] = {}
    if pdf_root.exists():
        for pdf in pdf_root.rglob("*.pdf"):
            pdf_sizes[pdf.stem] = pdf.stat().st_size

    # Walk metadata
    for meta_file in sorted(meta_root.rglob("*_metadata.json")):
        try:
            data = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        subject = data.get("subject") or meta_file.parent.name
        src     = data.get("source_file", meta_file.stem.replace("_metadata", ""))
        stem    = Path(src).stem

        # File size: prefer real file, fall back to chars
        size_bytes = pdf_sizes.get(stem, 0)
        if size_bytes == 0:
            # search partial match
            for k, v in pdf_sizes.items():
                if stem.lower() in k.lower() or k.lower() in stem.lower():
                    size_bytes = v
                    break

        entry = {
            "file":            src,
            "pages_total":     data.get("pages_total", 0),
            "pages_text":      data.get("pages_text", 0),
            "pages_ocr":       data.get("pages_ocr", 0),
            "pages_failed":    data.get("pages_failed", 0),
            "chunks":          data.get("chunks_generated", 0),
            "raw_chars":       data.get("raw_chars", 0),
            "cleaned_chars":   data.get("cleaned_chars", 0),
            "retention_pct":   data.get("char_retention_pct", 0.0),
            "quality":         data.get("mean_quality_score", 0.0),
            "proc_time_sec":   data.get("processing_time_sec", 0.0),
            "size_bytes":      size_bytes,
            "errors":          len(data.get("errors", [])),
        }

        if subject not in subjects:
            subjects[subject] = {"pdfs": []}
        subjects[subject]["pdfs"].append(entry)

    # Aggregate per subject
    result_subjects = []
    for subj, info in sorted(subjects.items()):
        pdfs = info["pdfs"]
        agg = {
            "subject":      subj,
            "pdf_count":    len(pdfs),
            "pages_total":  sum(p["pages_total"]  for p in pdfs),
            "pages_text":   sum(p["pages_text"]   for p in pdfs),
            "pages_ocr":    sum(p["pages_ocr"]    for p in pdfs),
            "pages_failed": sum(p["pages_failed"] for p in pdfs),
            "chunks":       sum(p["chunks"]        for p in pdfs),
            "raw_chars":    sum(p["raw_chars"]     for p in pdfs),
            "cleaned_chars":sum(p["cleaned_chars"] for p in pdfs),
            "size_bytes":   sum(p["size_bytes"]    for p in pdfs),
            "errors":       sum(p["errors"]        for p in pdfs),
            "avg_quality":  (sum(p["quality"] for p in pdfs) / len(pdfs)) if pdfs else 0,
            "proc_time_sec":sum(p["proc_time_sec"] for p in pdfs),
            "pdfs":         sorted(pdfs, key=lambda x: x["pages_total"], reverse=True),
        }
        result_subjects.append(agg)

    totals = {
        "pdf_count":    sum(s["pdf_count"]    for s in result_subjects),
        "pages_total":  sum(s["pages_total"]  for s in result_subjects),
        "pages_text":   sum(s["pages_text"]   for s in result_subjects),
        "pages_ocr":    sum(s["pages_ocr"]    for s in result_subjects),
        "pages_failed": sum(s["pages_failed"] for s in result_subjects),
        "chunks":       sum(s["chunks"]        for s in result_subjects),
        "raw_chars":    sum(s["raw_chars"]     for s in result_subjects),
        "cleaned_chars":sum(s["cleaned_chars"] for s in result_subjects),
        "size_bytes":   sum(s["size_bytes"]    for s in result_subjects),
        "errors":       sum(s["errors"]        for s in result_subjects),
        "proc_time_sec":sum(s["proc_time_sec"] for s in result_subjects),
        "subject_count":len(result_subjects),
    }

    return {"totals": totals, "subjects": result_subjects}


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def fmt_size(b: int) -> str:
    if b == 0: return "—"
    for unit in ("B", "KB", "MB", "GB"):
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} TB"

def fmt_num(n: int | float) -> str:
    return f"{int(n):,}"

def fmt_pct(v: float) -> str:
    return f"{v:.1f}%"

def quality_color(q: float) -> str:
    if q >= 0.95: return "#10b981"   # green
    if q >= 0.85: return "#f59e0b"   # amber
    return "#ef4444"                  # red

def retention_color(r: float) -> str:
    if r >= 95:  return "#10b981"
    if r >= 80:  return "#f59e0b"
    return "#ef4444"


def generate_html(stats: dict) -> str:
    totals   = stats["totals"]
    subjects = stats["subjects"]

    # Prepare chart data
    subj_names  = [s["subject"]         for s in subjects]
    subj_pages  = [s["pages_total"]     for s in subjects]
    subj_sizes  = [round(s["size_bytes"] / 1024 / 1024, 1) for s in subjects]
    subj_chunks = [s["chunks"]          for s in subjects]
    subj_pdfs   = [s["pdf_count"]       for s in subjects]

    strategy_text = totals["pages_text"]
    strategy_ocr  = totals["pages_ocr"]
    strategy_fail = totals["pages_failed"]

    # Build subject cards HTML
    subject_cards = ""
    for s in subjects:
        retention = (s["cleaned_chars"] / s["raw_chars"] * 100) if s["raw_chars"] else 0
        ocr_pct   = (s["pages_ocr"] / s["pages_total"] * 100) if s["pages_total"] else 0
        q_color   = quality_color(s["avg_quality"])
        r_color   = retention_color(retention)
        size_str  = fmt_size(s["size_bytes"])

        pdf_rows = ""
        for p in s["pdfs"]:
            p_ret  = (p["cleaned_chars"] / p["raw_chars"] * 100) if p["raw_chars"] else 0
            p_qc   = quality_color(p["quality"])
            p_rc   = retention_color(p_ret)
            p_size = fmt_size(p["size_bytes"])
            ocr_badge = (
                f'<span class="badge badge-ocr">{p["pages_ocr"]} OCR</span>'
                if p["pages_ocr"] > 0 else ""
            )
            err_badge = (
                f'<span class="badge badge-err">{p["errors"]} err</span>'
                if p["errors"] > 0 else ""
            )
            pdf_rows += f"""
            <tr>
              <td class="pdf-name">{p["file"]}</td>
              <td class="num">{fmt_num(p["pages_total"])}</td>
              <td class="num">{p_size}</td>
              <td class="num">{fmt_num(p["chunks"])}</td>
              <td class="num" style="color:{p_qc};font-weight:600">{p["quality"]:.3f}</td>
              <td class="num" style="color:{p_rc}">{p_ret:.1f}%</td>
              <td class="num">{ocr_badge}{err_badge}</td>
            </tr>"""

        subject_cards += f"""
        <div class="card subject-card" data-subject="{s["subject"]}">
          <div class="card-header">
            <div class="subject-title">{s["subject"]}</div>
            <div class="subject-meta">{s["pdf_count"]} PDF{'s' if s["pdf_count"] != 1 else ''} &nbsp;·&nbsp; {size_str}</div>
          </div>
          <div class="subject-stats">
            <div class="stat-pill">
              <div class="pill-label">Pages</div>
              <div class="pill-value">{fmt_num(s["pages_total"])}</div>
            </div>
            <div class="stat-pill">
              <div class="pill-label">Chunks</div>
              <div class="pill-value">{fmt_num(s["chunks"])}</div>
            </div>
            <div class="stat-pill">
              <div class="pill-label">Quality</div>
              <div class="pill-value" style="color:{q_color}">{s["avg_quality"]:.3f}</div>
            </div>
            <div class="stat-pill">
              <div class="pill-label">Retention</div>
              <div class="pill-value" style="color:{r_color}">{retention:.1f}%</div>
            </div>
            <div class="stat-pill">
              <div class="pill-label">OCR pages</div>
              <div class="pill-value">{fmt_num(s["pages_ocr"])} <span class="pct">({ocr_pct:.0f}%)</span></div>
            </div>
            <div class="stat-pill">
              <div class="pill-label">Proc time</div>
              <div class="pill-value">{s["proc_time_sec"]:.0f}s</div>
            </div>
          </div>
          <details class="pdf-details">
            <summary>Show {s["pdf_count"]} PDF{'s' if s["pdf_count"] != 1 else ''}</summary>
            <table class="pdf-table">
              <thead><tr>
                <th>File</th><th>Pages</th><th>Size</th>
                <th>Chunks</th><th>Quality</th><th>Retention</th><th>Flags</th>
              </tr></thead>
              <tbody>{pdf_rows}</tbody>
            </table>
          </details>
        </div>"""

    # Overall retention
    total_retention = (
        (totals["cleaned_chars"] / totals["raw_chars"] * 100)
        if totals["raw_chars"] else 0
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>UPSC PDF Dataset Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  :root {{
    --bg:       #0f172a;
    --surface:  #1e293b;
    --surface2: #273346;
    --border:   #334155;
    --text:     #e2e8f0;
    --muted:    #94a3b8;
    --accent:   #6366f1;
    --accent2:  #22d3ee;
    --green:    #10b981;
    --amber:    #f59e0b;
    --red:      #ef4444;
  }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 14px;
    line-height: 1.5;
  }}

  /* ── Header ── */
  .header {{
    background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
    border-bottom: 1px solid var(--border);
    padding: 28px 32px 24px;
  }}
  .header h1 {{
    font-size: 22px;
    font-weight: 700;
    color: #fff;
    letter-spacing: -0.3px;
  }}
  .header p {{ color: var(--muted); margin-top: 4px; font-size: 13px; }}

  /* ── Layout ── */
  .page {{ max-width: 1400px; margin: 0 auto; padding: 28px 24px; }}
  .section-title {{
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--muted);
    margin-bottom: 14px;
  }}

  /* ── KPI Cards ── */
  .kpi-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(170px, 1fr));
    gap: 14px;
    margin-bottom: 32px;
  }}
  .kpi {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px 20px;
  }}
  .kpi-label {{ font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.8px; }}
  .kpi-value {{
    font-size: 28px;
    font-weight: 700;
    margin-top: 6px;
    line-height: 1;
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }}
  .kpi-sub {{ font-size: 11px; color: var(--muted); margin-top: 6px; }}

  /* ── Charts ── */
  .charts-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-bottom: 32px;
  }}
  .charts-grid-3 {{
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 16px;
    margin-bottom: 32px;
  }}
  .card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
  }}
  .card-title {{
    font-size: 12px;
    font-weight: 600;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 16px;
  }}
  .chart-wrap {{ position: relative; height: 220px; }}
  .chart-wrap-sm {{ position: relative; height: 180px; }}

  /* ── Subject Cards ── */
  .subjects-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(560px, 1fr));
    gap: 16px;
    margin-bottom: 32px;
  }}
  .subject-card {{ padding: 20px; }}
  .card-header {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 14px;
  }}
  .subject-title {{ font-size: 16px; font-weight: 700; color: #fff; }}
  .subject-meta  {{ font-size: 12px; color: var(--muted); }}
  .subject-stats {{
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-bottom: 14px;
  }}
  .stat-pill {{
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 14px;
    flex: 1;
    min-width: 90px;
  }}
  .pill-label {{ font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.6px; }}
  .pill-value {{ font-size: 16px; font-weight: 700; margin-top: 2px; }}
  .pct {{ font-size: 11px; color: var(--muted); font-weight: 400; }}

  /* ── PDF Detail Table ── */
  .pdf-details summary {{
    font-size: 12px;
    color: var(--accent2);
    cursor: pointer;
    user-select: none;
    padding: 6px 0;
  }}
  .pdf-details summary:hover {{ color: #fff; }}
  .pdf-table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 12px;
    font-size: 12px;
  }}
  .pdf-table th {{
    text-align: left;
    color: var(--muted);
    font-weight: 500;
    padding: 6px 8px;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
  }}
  .pdf-table td {{
    padding: 7px 8px;
    border-bottom: 1px solid rgba(51,65,85,0.5);
    vertical-align: middle;
  }}
  .pdf-table tr:last-child td {{ border-bottom: none; }}
  .pdf-name {{ max-width: 260px; word-break: break-all; color: var(--text); }}
  .num {{ text-align: right; white-space: nowrap; }}

  /* ── Badges ── */
  .badge {{
    display: inline-block;
    font-size: 10px;
    font-weight: 600;
    padding: 2px 6px;
    border-radius: 4px;
    margin-left: 4px;
  }}
  .badge-ocr {{ background: rgba(99,102,241,0.2); color: #818cf8; }}
  .badge-err {{ background: rgba(239,68,68,0.2);  color: #f87171; }}

  /* ── Responsive ── */
  @media (max-width: 900px) {{
    .charts-grid, .charts-grid-3 {{ grid-template-columns: 1fr; }}
    .subjects-grid {{ grid-template-columns: 1fr; }}
    .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }}
  }}
</style>
</head>
<body>

<div class="header">
  <h1>📚 UPSC PDF Dataset Dashboard</h1>
  <p>Extraction statistics across {totals["subject_count"]} subjects · {fmt_num(totals["pdf_count"])} PDFs · Generated from metadata checkpoints</p>
</div>

<div class="page">

  <!-- KPI Row -->
  <div class="section-title">Overall Totals</div>
  <div class="kpi-grid">
    <div class="kpi">
      <div class="kpi-label">Total PDFs</div>
      <div class="kpi-value">{fmt_num(totals["pdf_count"])}</div>
      <div class="kpi-sub">across {totals["subject_count"]} subjects</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Total Pages</div>
      <div class="kpi-value">{fmt_num(totals["pages_total"])}</div>
      <div class="kpi-sub">text: {fmt_num(totals["pages_text"])} · ocr: {fmt_num(totals["pages_ocr"])}</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Total Size</div>
      <div class="kpi-value">{fmt_size(totals["size_bytes"])}</div>
      <div class="kpi-sub">on-disk PDF size</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Total Chunks</div>
      <div class="kpi-value">{fmt_num(totals["chunks"])}</div>
      <div class="kpi-sub">~512 tokens each</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Raw Characters</div>
      <div class="kpi-value">{fmt_size(totals["raw_chars"])}</div>
      <div class="kpi-sub">before cleaning</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Clean Characters</div>
      <div class="kpi-value">{fmt_size(totals["cleaned_chars"])}</div>
      <div class="kpi-sub">retention: {total_retention:.1f}%</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Failed Pages</div>
      <div class="kpi-value" style="{'color:var(--red)' if totals['pages_failed'] > 0 else ''}">{fmt_num(totals["pages_failed"])}</div>
      <div class="kpi-sub">{'⚠ check errors' if totals['pages_failed'] > 0 else '✓ all pages ok'}</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Processing Time</div>
      <div class="kpi-value">{totals["proc_time_sec"]/60:.0f}m</div>
      <div class="kpi-sub">{totals["proc_time_sec"]:.0f} seconds total</div>
    </div>
  </div>

  <!-- Charts Row 1 -->
  <div class="section-title">Distribution</div>
  <div class="charts-grid">
    <div class="card">
      <div class="card-title">Pages by Subject</div>
      <div class="chart-wrap"><canvas id="chartPages"></canvas></div>
    </div>
    <div class="card">
      <div class="card-title">Size (MB) by Subject</div>
      <div class="chart-wrap"><canvas id="chartSize"></canvas></div>
    </div>
  </div>

  <!-- Charts Row 2 -->
  <div class="charts-grid-3">
    <div class="card">
      <div class="card-title">Extraction Strategy (Total Pages)</div>
      <div class="chart-wrap-sm"><canvas id="chartStrategy"></canvas></div>
    </div>
    <div class="card">
      <div class="card-title">Chunks by Subject</div>
      <div class="chart-wrap-sm"><canvas id="chartChunks"></canvas></div>
    </div>
    <div class="card">
      <div class="card-title">PDF Count by Subject</div>
      <div class="chart-wrap-sm"><canvas id="chartPdfs"></canvas></div>
    </div>
  </div>

  <!-- Subject Cards -->
  <div class="section-title">Subject Breakdown</div>
  <div class="subjects-grid">
    {subject_cards}
  </div>

</div>

<script>
const LABELS  = {json.dumps(subj_names)};
const PAGES   = {json.dumps(subj_pages)};
const SIZES   = {json.dumps(subj_sizes)};
const CHUNKS  = {json.dumps(subj_chunks)};
const PDFS    = {json.dumps(subj_pdfs)};

const PALETTE = [
  '#6366f1','#22d3ee','#10b981','#f59e0b','#ef4444',
  '#a78bfa','#34d399','#fbbf24','#f87171','#60a5fa',
  '#e879f9','#4ade80'
];

const baseOpts = {{
  responsive: true,
  maintainAspectRatio: false,
  plugins: {{
    legend: {{ labels: {{ color: '#94a3b8', font: {{ size: 11 }} }} }},
    tooltip: {{ backgroundColor: '#1e293b', borderColor: '#334155', borderWidth: 1,
                titleColor: '#e2e8f0', bodyColor: '#94a3b8' }}
  }},
  scales: {{
    x: {{ ticks: {{ color: '#64748b', font: {{ size: 10 }} }}, grid: {{ color: '#1e293b' }} }},
    y: {{ ticks: {{ color: '#64748b', font: {{ size: 10 }} }}, grid: {{ color: '#273346' }} }}
  }}
}};

function barChart(id, labels, data, label, color) {{
  new Chart(document.getElementById(id), {{
    type: 'bar',
    data: {{
      labels,
      datasets: [{{ label, data, backgroundColor: color || PALETTE,
                    borderRadius: 6, borderSkipped: false }}]
    }},
    options: {{ ...baseOpts, plugins: {{ ...baseOpts.plugins, legend: {{ display: false }} }} }}
  }});
}}

barChart('chartPages',  LABELS, PAGES,  'Pages',    '#6366f1');
barChart('chartSize',   LABELS, SIZES,  'Size (MB)','#22d3ee');
barChart('chartChunks', LABELS, CHUNKS, 'Chunks',   '#10b981');
barChart('chartPdfs',   LABELS, PDFS,   'PDFs',     '#f59e0b');

// Strategy donut
new Chart(document.getElementById('chartStrategy'), {{
  type: 'doughnut',
  data: {{
    labels: ['Text Extraction', 'OCR', 'Failed'],
    datasets: [{{
      data: [{strategy_text}, {strategy_ocr}, {strategy_fail}],
      backgroundColor: ['#10b981', '#6366f1', '#ef4444'],
      borderWidth: 2,
      borderColor: '#1e293b'
    }}]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{
      legend: {{ position: 'bottom', labels: {{ color: '#94a3b8', font: {{ size: 11 }}, padding: 12 }} }},
      tooltip: {{ backgroundColor: '#1e293b', borderColor: '#334155', borderWidth: 1,
                  titleColor: '#e2e8f0', bodyColor: '#94a3b8' }}
    }}
  }}
}});
</script>
</body>
</html>"""

    return html


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Generate PDF dataset dashboard")
    ap.add_argument("--pdf-dir",  default="upsc_pdfs",            type=Path)
    ap.add_argument("--meta-dir", default="dataset_output_final", type=Path)
    ap.add_argument("--output",   default="dashboard.html",        type=Path)
    ap.add_argument("--open",     action="store_true", help="Open in browser after generating")
    args = ap.parse_args()

    base = Path(__file__).parent.parent  # DataSets/
    pdf_root  = args.pdf_dir  if args.pdf_dir.is_absolute()  else base / args.pdf_dir
    meta_root = args.meta_dir if args.meta_dir.is_absolute() else base / args.meta_dir
    out       = args.output   if args.output.is_absolute()   else base / args.output

    print(f"Scanning metadata : {meta_root}")
    print(f"Scanning PDFs     : {pdf_root}")

    stats = collect_stats(pdf_root, meta_root)
    t = stats["totals"]
    print(f"\nFound  {t['subject_count']} subjects · {t['pdf_count']} PDFs · "
          f"{t['pages_total']:,} pages · {t['chunks']:,} chunks")

    html = generate_html(stats)
    out.write_text(html, encoding="utf-8")
    print(f"\nDashboard saved → {out}")

    if args.open:
        if sys.platform == "darwin":
            subprocess.run(["open", str(out)])
        elif sys.platform.startswith("linux"):
            subprocess.run(["xdg-open", str(out)])
        else:
            os.startfile(str(out))


if __name__ == "__main__":
    main()
