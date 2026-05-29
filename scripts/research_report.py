#!/usr/bin/env python3
"""
Research Report Generator — UPSC SLM Project
Generates a self-contained HTML thesis/research document capturing:
  - All datasets created (size, words, chunks, PDFs)
  - Training pipeline phases
  - UPSC Mains question extraction stats
  - Subject-wise breakdown

Usage:
    python3 scripts/research_report.py
    python3 scripts/research_report.py --open   # auto-open in browser
"""

import json
import gc
import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

BASE_DIR    = Path(__file__).parent.parent   # DataSets/
OUTPUT_DIR  = BASE_DIR / "dataset_output_final" / "combined"
OUTPUT_OLD  = BASE_DIR / "dataset_output"
PAPERS_DIR  = BASE_DIR / "upsc_papers" / "mains"
REPORT_FILE = BASE_DIR / "research_report.html"


# ---------------------------------------------------------------------------
# Data collectors
# ---------------------------------------------------------------------------

def count_jsonl(path: Path) -> dict:
    """Count lines, words, chars in a JSONL file."""
    if not path.exists():
        return {"lines": 0, "words": 0, "chars": 0, "size_bytes": 0}
    lines = words = chars = 0
    size  = path.stat().st_size
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj  = json.loads(line)
                text = obj.get("text", "")
                lines += 1
                words += len(text.split())
                chars += len(text)
            except Exception:
                pass
    return {"lines": lines, "words": words, "chars": chars, "size_bytes": size}


def collect_pretrain_stats() -> dict:
    """Stats for unified_pretrain.jsonl — the Stage 1 pretraining corpus."""
    path = OUTPUT_DIR / "unified_pretrain.jsonl"
    stats = count_jsonl(path)
    # Per-subject breakdown
    subjects = defaultdict(lambda: {"chunks": 0, "words": 0})
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj  = json.loads(line)
                    subj = obj.get("subject", "Unknown")
                    text = obj.get("text", "")
                    subjects[subj]["chunks"] += 1
                    subjects[subj]["words"]  += len(text.split())
                except Exception:
                    pass
    stats["subjects"] = dict(subjects)
    return stats


def collect_sft_stats() -> dict:
    """Stats for all SFT JSONL versions."""
    versions = {
        "v1 (NCERT Q+A)":  OUTPUT_DIR / "unified_sft.jsonl",
        "v2 (Coaching)":   OUTPUT_DIR / "unified_sft_v2.jsonl",
        "v3 (Full SFT)":   OUTPUT_DIR / "unified_sft_v3.jsonl",
    }
    result = {}
    for name, path in versions.items():
        result[name] = count_jsonl(path)
    return result


def collect_question_stats() -> dict:
    """Stats for extracted UPSC Mains questions."""
    path = OUTPUT_DIR / "extracted_questions.jsonl"
    if not path.exists():
        return {"total": 0, "by_year": {}, "by_paper": {}, "by_type": {}}
    by_year  = defaultdict(int)
    by_paper = defaultdict(int)
    by_type  = defaultdict(int)
    total    = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                q = json.loads(line)
                total += 1
                by_year[q.get("year",  "?")] += 1
                by_paper[q.get("paper", "?")] += 1
                by_type[q.get("q_type","?")] += 1
            except Exception:
                pass
    return {
        "total":    total,
        "by_year":  dict(sorted(by_year.items())),
        "by_paper": dict(sorted(by_paper.items())),
        "by_type":  dict(sorted(by_type.items(), key=lambda x: -x[1])),
        "size_bytes": path.stat().st_size,
    }


def collect_pdf_stats() -> dict:
    """Count PDFs by year and paper in upsc_papers/mains/."""
    if not PAPERS_DIR.exists():
        return {"total": 0, "by_year": {}, "total_size_bytes": 0}
    by_year  = defaultdict(int)
    total    = 0
    total_sz = 0
    for pdf in PAPERS_DIR.rglob("*.pdf"):
        for part in pdf.parts:
            if part.isdigit() and len(part) == 4:
                by_year[int(part)] += 1
                break
        total    += 1
        total_sz += pdf.stat().st_size
    return {
        "total":            total,
        "by_year":          dict(sorted(by_year.items())),
        "total_size_bytes": total_sz,
    }


def collect_subject_metadata() -> list:
    """Read *_metadata.json from subject folders."""
    rows = []
    for meta in sorted(BASE_DIR.rglob("*_metadata.json")):
        try:
            d    = json.loads(meta.read_text(encoding="utf-8"))
            subj = d.get("subject", meta.parent.name)
            rows.append({
                "subject":      subj,
                "source_file":  d.get("source_file", ""),
                "pages_total":  d.get("pages_total", 0),
                "pages_text":   d.get("pages_text",  0),
                "pages_ocr":    d.get("pages_ocr",   0),
                "chunks":       d.get("chunks_generated", 0),
                "raw_chars":    d.get("raw_chars",    0),
                "cleaned_chars":d.get("cleaned_chars",0),
                "quality":      d.get("mean_quality_score", 0.0),
            })
        except Exception:
            pass
    # Aggregate by subject
    by_subj = defaultdict(lambda: {
        "pdfs": 0, "pages": 0, "pages_ocr": 0, "chunks": 0,
        "raw_chars": 0, "clean_chars": 0, "quality_sum": 0.0,
    })
    for r in rows:
        s = by_subj[r["subject"]]
        s["pdfs"]        += 1
        s["pages"]       += r["pages_total"]
        s["pages_ocr"]   += r["pages_ocr"]
        s["chunks"]      += r["chunks"]
        s["raw_chars"]   += r["raw_chars"]
        s["clean_chars"] += r["cleaned_chars"]
        s["quality_sum"] += r["quality"]
    result = []
    for subj, s in sorted(by_subj.items()):
        avg_q = s["quality_sum"] / s["pdfs"] if s["pdfs"] else 0
        result.append({
            "subject":    subj,
            "pdfs":       s["pdfs"],
            "pages":      s["pages"],
            "pages_ocr":  s["pages_ocr"],
            "chunks":     s["chunks"],
            "raw_chars":  s["raw_chars"],
            "clean_chars":s["clean_chars"],
            "avg_quality":avg_q,
        })
    return result


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_num(n) -> str:
    return f"{int(n):,}"

def fmt_words(n) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.0f}K"
    return str(n)

def fmt_size(b) -> str:
    b = int(b)
    if b == 0: return "—"
    for unit in ("B", "KB", "MB", "GB"):
        if b < 1024:
            return f"{b:.1f} {unit}"
        b //= 1024
    return f"{b} GB"

def pct(a, b) -> str:
    return f"{a/b*100:.1f}%" if b else "—"


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def generate_report(data: dict) -> str:
    pre     = data["pretrain"]
    sft     = data["sft"]
    qs      = data["questions"]
    pdfs    = data["pdfs"]
    subjs   = data["subjects"]
    now     = datetime.now().strftime("%B %d, %Y")

    # ── KPI cards ────────────────────────────────────────────────────────────
    total_pretrain_words = pre["words"]
    total_sft_records    = max((sft.get(k, {}).get("lines", 0) for k in sft), default=0)
    total_sft_words      = max((sft.get(k, {}).get("words", 0) for k in sft), default=0)
    total_chunks         = pre["lines"]
    total_pdfs_ncert     = sum(s["pdfs"]   for s in subjs)
    total_pages          = sum(s["pages"]  for s in subjs)

    # ── Subject table rows ───────────────────────────────────────────────────
    subj_rows = ""
    for s in subjs:
        ret = pct(s["clean_chars"], s["raw_chars"])
        ocr_pct = pct(s["pages_ocr"], s["pages"])
        q_color = ("#10b981" if s["avg_quality"] >= 0.90
                   else "#f59e0b" if s["avg_quality"] >= 0.75
                   else "#ef4444")
        subj_rows += f"""
        <tr>
          <td><strong>{s["subject"]}</strong></td>
          <td class="num">{s["pdfs"]}</td>
          <td class="num">{fmt_num(s["pages"])}</td>
          <td class="num">{fmt_num(s["pages_ocr"])} <span class="muted">({ocr_pct})</span></td>
          <td class="num">{fmt_num(s["chunks"])}</td>
          <td class="num">{fmt_size(s["raw_chars"])}</td>
          <td class="num">{fmt_size(s["clean_chars"])}</td>
          <td class="num">{ret}</td>
          <td class="num" style="color:{q_color};font-weight:700">{s["avg_quality"]:.3f}</td>
        </tr>"""

    # ── SFT version table ────────────────────────────────────────────────────
    sft_rows = ""
    for name, stats in sft.items():
        sft_rows += f"""
        <tr>
          <td><strong>{name}</strong></td>
          <td class="num">{fmt_num(stats["lines"])}</td>
          <td class="num">{fmt_words(stats["words"])}</td>
          <td class="num">{fmt_size(stats["size_bytes"])}</td>
        </tr>"""

    # ── Questions by year ────────────────────────────────────────────────────
    q_year_rows = ""
    for yr, cnt in qs["by_year"].items():
        q_year_rows += f"<tr><td>{yr}</td><td class='num'>{cnt}</td></tr>"

    q_paper_rows = ""
    for paper, cnt in qs["by_paper"].items():
        q_paper_rows += f"<tr><td>{paper}</td><td class='num'>{cnt}</td></tr>"

    q_type_rows = ""
    for qt, cnt in list(qs["by_type"].items())[:12]:
        q_type_rows += f"<tr><td><code>{qt}</code></td><td class='num'>{cnt}</td></tr>"

    # ── PDF by year ──────────────────────────────────────────────────────────
    pdf_year_rows = ""
    for yr, cnt in pdfs["by_year"].items():
        pdf_year_rows += f"<tr><td>{yr}</td><td class='num'>{cnt} PDFs</td></tr>"

    # ── Chart data ───────────────────────────────────────────────────────────
    subj_names   = [s["subject"]                   for s in subjs]
    subj_chunks  = [s["chunks"]                    for s in subjs]
    subj_pages   = [s["pages"]                     for s in subjs]
    subj_quality = [round(s["avg_quality"], 3)     for s in subjs]
    q_type_labels = list(qs["by_type"].keys())[:10]
    q_type_vals   = [qs["by_type"][k] for k in q_type_labels]
    q_year_labels = list(qs["by_year"].keys())
    q_year_vals   = list(qs["by_year"].values())

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>UPSC SLM — Research Dataset Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
:root {{
  --bg:      #0f172a; --surface: #1e293b; --surface2: #273346;
  --border:  #334155; --text:    #e2e8f0; --muted:    #94a3b8;
  --accent:  #6366f1; --accent2: #22d3ee; --green:    #10b981;
  --amber:   #f59e0b; --red:     #ef4444;
}}
body {{ background:var(--bg); color:var(--text);
       font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
       font-size:14px; line-height:1.6; }}

/* Header */
.header {{ background:linear-gradient(135deg,#1e1b4b,#0f172a);
           border-bottom:1px solid var(--border); padding:36px 40px 28px; }}
.header h1 {{ font-size:26px;font-weight:800;color:#fff;letter-spacing:-.5px }}
.header .subtitle {{ color:var(--muted);margin-top:6px;font-size:14px }}
.header .meta {{ color:var(--muted);margin-top:12px;font-size:12px;
                 display:flex;gap:24px;flex-wrap:wrap }}
.header .meta span {{ color:var(--accent2) }}

/* Page */
.page {{ max-width:1300px;margin:0 auto;padding:32px 24px }}
.section {{ margin-bottom:44px }}
.section-title {{ font-size:18px;font-weight:700;color:#fff;margin-bottom:6px;
                  padding-bottom:10px;border-bottom:2px solid var(--accent);
                  display:inline-block }}
.section-sub {{ color:var(--muted);font-size:13px;margin-bottom:20px;margin-top:6px }}

/* KPI */
.kpi-grid {{ display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));
             gap:14px;margin-bottom:32px }}
.kpi {{ background:var(--surface);border:1px solid var(--border);
        border-radius:14px;padding:20px }}
.kpi-label {{ font-size:11px;color:var(--muted);text-transform:uppercase;
              letter-spacing:.8px }}
.kpi-value {{ font-size:30px;font-weight:800;margin-top:6px;line-height:1;
              background:linear-gradient(135deg,var(--accent),var(--accent2));
              -webkit-background-clip:text;-webkit-text-fill-color:transparent;
              background-clip:text }}
.kpi-sub {{ font-size:11px;color:var(--muted);margin-top:5px }}

/* Cards */
.card {{ background:var(--surface);border:1px solid var(--border);
         border-radius:12px;padding:22px }}
.card-title {{ font-size:12px;font-weight:600;color:var(--muted);
               text-transform:uppercase;letter-spacing:.8px;margin-bottom:16px }}
.grid2 {{ display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:20px }}
.grid3 {{ display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-bottom:20px }}
.chart-wrap {{ position:relative;height:240px }}
.chart-sm    {{ position:relative;height:190px }}

/* Tables */
table {{ width:100%;border-collapse:collapse;font-size:13px }}
thead th {{ text-align:left;color:var(--muted);font-weight:600;font-size:11px;
            text-transform:uppercase;letter-spacing:.6px;
            padding:8px 10px;border-bottom:1px solid var(--border) }}
tbody td {{ padding:9px 10px;border-bottom:1px solid rgba(51,65,85,.4);
            vertical-align:middle }}
tbody tr:last-child td {{ border-bottom:none }}
tbody tr:hover {{ background:var(--surface2) }}
.num {{ text-align:right;white-space:nowrap }}
.muted {{ color:var(--muted);font-size:11px }}
code {{ background:rgba(99,102,241,.15);color:#a5b4fc;
        padding:2px 6px;border-radius:4px;font-size:12px }}

/* Pipeline */
.pipeline {{ display:flex;gap:0;align-items:stretch;flex-wrap:wrap;
             margin:20px 0;position:relative }}
.pipe-step {{ flex:1;min-width:180px;background:var(--surface);
              border:1px solid var(--border);border-radius:12px;
              padding:18px 16px;position:relative;margin:4px }}
.pipe-step::after {{ content:'→';position:absolute;right:-18px;top:50%;
                     transform:translateY(-50%);color:var(--muted);font-size:18px;
                     z-index:2 }}
.pipe-step:last-child::after {{ display:none }}
.pipe-num {{ width:28px;height:28px;border-radius:50%;
             background:var(--accent);color:#fff;font-weight:800;font-size:13px;
             display:flex;align-items:center;justify-content:center;
             margin-bottom:10px }}
.pipe-title {{ font-weight:700;font-size:14px;color:#fff;margin-bottom:4px }}
.pipe-sub {{ font-size:12px;color:var(--muted) }}
.pipe-badge {{ display:inline-block;margin-top:8px;font-size:11px;
               font-weight:600;padding:2px 8px;border-radius:20px }}
.badge-green {{ background:rgba(16,185,129,.15);color:#34d399 }}
.badge-blue  {{ background:rgba(99,102,241,.15);color:#818cf8 }}
.badge-amber {{ background:rgba(245,158,11,.15);color:#fbbf24 }}
.badge-red   {{ background:rgba(239,68,68,.15);color:#f87171 }}

/* Training phases */
.phase-grid {{ display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));
               gap:14px;margin-bottom:20px }}
.phase {{ background:var(--surface);border:1px solid var(--border);
          border-radius:12px;padding:20px }}
.phase-header {{ display:flex;align-items:center;gap:10px;margin-bottom:12px }}
.phase-icon {{ font-size:20px }}
.phase-title {{ font-weight:700;font-size:15px;color:#fff }}
.phase-tag {{ font-size:11px;font-weight:600;padding:2px 8px;
              border-radius:20px;margin-left:auto }}
.phase-row {{ display:flex;justify-content:space-between;
              font-size:12px;color:var(--muted);padding:4px 0;
              border-bottom:1px solid rgba(51,65,85,.3) }}
.phase-row:last-child {{ border-bottom:none }}
.phase-val {{ color:var(--text);font-weight:600 }}

/* Abstract box */
.abstract {{ background:var(--surface);border:1px solid var(--border);
             border-left:4px solid var(--accent);border-radius:0 12px 12px 0;
             padding:20px 24px;margin-bottom:28px;font-size:13px;
             color:#cbd5e1;line-height:1.8 }}
.abstract strong {{ color:#fff }}

/* Responsive */
@media(max-width:900px) {{
  .grid2,.grid3,.pipeline,.phase-grid {{ grid-template-columns:1fr }}
  .pipe-step::after {{ display:none }}
}}
@media print {{
  body {{ background:#fff;color:#000 }}
  .card,.phase,.pipe-step,.kpi,.abstract {{ background:#f8fafc;border-color:#e2e8f0 }}
  .header {{ background:#1e1b4b;color:#fff }}
}}
</style>
</head>
<body>

<div class="header">
  <h1>🎓 UPSC SLM — Dataset & Training Research Report</h1>
  <div class="subtitle">Small Language Model Fine-tuning for UPSC Civil Services Preparation</div>
  <div class="meta">
    <div>Generated: <span>{now}</span></div>
    <div>Base Model: <span>Gemma 2 (2B / 9B)</span></div>
    <div>Total PDFs processed: <span>{fmt_num(total_pdfs_ncert + pdfs["total"])}</span></div>
    <div>Total training records: <span>{fmt_words(total_chunks + total_sft_records)}</span></div>
  </div>
</div>

<div class="page">

<!-- ═══════════════════════════════════════════════════════════ ABSTRACT -->
<div class="section">
  <div class="section-title">Abstract</div>
  <p class="section-sub">Overview of the project, motivation and methodology</p>
  <div class="abstract">
    This project builds a <strong>domain-adapted Small Language Model (SLM)</strong> for UPSC Civil
    Services exam preparation. The base model (<strong>Gemma 2</strong>) is fine-tuned in multiple
    progressive stages using a custom-curated dataset derived from NCERT textbooks, Vision IAS coaching
    material, and <strong>{fmt_num(pdfs["total"])} official UPSC Mains question papers</strong>
    (2014–2025). The dataset pipeline converts raw PDFs to structured training data through a
    multi-strategy extraction engine (embedded-text + OCR + <strong>LLM-vision extraction</strong>),
    producing <strong>{fmt_words(total_pretrain_words)} words</strong> of domain pre-training text
    and <strong>{fmt_words(total_sft_words)} words</strong> of supervised fine-tuning Q+A pairs.
    The model is trained to answer UPSC questions in proper answer-writing format — structured,
    factual, and within the required word limit.
  </div>
</div>

<!-- ═══════════════════════════════════════════════════════════ KPIs -->
<div class="section">
  <div class="section-title">Dataset at a Glance</div>
  <p class="section-sub">Aggregate statistics across all pipeline stages</p>
  <div class="kpi-grid">
    <div class="kpi">
      <div class="kpi-label">NCERT / Coaching PDFs</div>
      <div class="kpi-value">{fmt_num(total_pdfs_ncert)}</div>
      <div class="kpi-sub">{fmt_num(total_pages)} pages extracted</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">UPSC Mains Papers</div>
      <div class="kpi-value">{fmt_num(pdfs["total"])}</div>
      <div class="kpi-sub">2014 – 2025 · {len(pdfs["by_year"])} years</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Pretrain Chunks</div>
      <div class="kpi-value">{fmt_words(total_chunks)}</div>
      <div class="kpi-sub">{fmt_words(total_pretrain_words)} words · ~512 tok each</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">SFT Q+A Pairs</div>
      <div class="kpi-value">{fmt_words(total_sft_records)}</div>
      <div class="kpi-sub">{fmt_words(total_sft_words)} words total</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Real UPSC Questions</div>
      <div class="kpi-value">{fmt_num(qs["total"])}</div>
      <div class="kpi-sub">{len(qs["by_year"])} years · 5 papers each</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Subjects Covered</div>
      <div class="kpi-value">{len(subjs)}</div>
      <div class="kpi-sub">History, Polity, Economy, Geo…</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">UPSC Papers Size</div>
      <div class="kpi-value">{fmt_size(pdfs["total_size_bytes"])}</div>
      <div class="kpi-sub">Raw PDF corpus</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Pretrain Corpus</div>
      <div class="kpi-value">{fmt_size(pre["size_bytes"])}</div>
      <div class="kpi-sub">JSONL file size</div>
    </div>
  </div>
</div>

<!-- ═══════════════════════════════════════════════════════════ PIPELINE -->
<div class="section">
  <div class="section-title">Data Pipeline</div>
  <p class="section-sub">End-to-end flow from raw PDFs to training-ready datasets</p>
  <div class="pipeline">
    <div class="pipe-step">
      <div class="pipe-num">1</div>
      <div class="pipe-title">PDF Download</div>
      <div class="pipe-sub">NCERT, UPSC official papers , Government Website and Available free data on internet</div>
      <div class="pipe-badge badge-blue">{fmt_num(total_pdfs_ncert + pdfs["total"])} PDFs</div>
    </div>
    <div class="pipe-step">
      <div class="pipe-num">2</div>
      <div class="pipe-title">Text Extraction</div>
      <div class="pipe-sub">PyMuPDF (embedded text) + Tesseract OCR + Quality scoring</div>
      <div class="pipe-badge badge-green">Per-page strategy</div>
    </div>
    <div class="pipe-step">
      <div class="pipe-num">3</div>
      <div class="pipe-title">LLM Extraction</div>
      <div class="pipe-sub">GPT-4o vision reads PDF images → extracts clean UPSC questions</div>
      <div class="pipe-badge badge-amber">{fmt_num(qs["total"])} questions</div>
    </div>
    <div class="pipe-step">
      <div class="pipe-num">4</div>
      <div class="pipe-title">Chunking</div>
      <div class="pipe-sub">Sentence-boundary chunking, 2000 chars, 10% overlap</div>
      <div class="pipe-badge badge-blue">{fmt_words(total_chunks)} chunks</div>
    </div>
    <div class="pipe-step">
      <div class="pipe-num">5</div>
      <div class="pipe-title">SFT Generation</div>
      <div class="pipe-sub">5 UPSC-style Q+A pairs per chunk via LLM, Gemma 2 chat format</div>
      <div class="pipe-badge badge-green">{fmt_words(total_sft_records)} pairs</div>
    </div>
    <div class="pipe-step">
      <div class="pipe-num">6</div>
      <div class="pipe-title">Fine-tuning</div>
      <div class="pipe-sub">3-stage progressive training on Gemma 2 via QLoRA on Colab</div>
      <div class="pipe-badge badge-amber">3 stages</div>
    </div>
  </div>
</div>

<!-- ═══════════════════════════════════════════════════════════ TRAINING PHASES -->
<div class="section">
  <div class="section-title">Training Phases</div>
  <p class="section-sub">Progressive fine-tuning strategy — each stage builds on the previous checkpoint</p>
  <div class="phase-grid">
    <div class="phase">
      <div class="phase-header">
        <div class="phase-icon">📚</div>
        <div class="phase-title">Stage 1 — Continued Pre-training</div>
        <div class="phase-tag badge-blue">Domain Injection</div>
      </div>
      <div class="phase-row"><span>Data</span><span class="phase-val">unified_pretrain.jsonl</span></div>
      <div class="phase-row"><span>Records</span><span class="phase-val">{fmt_words(total_chunks)} chunks</span></div>
      <div class="phase-row"><span>Words</span><span class="phase-val">{fmt_words(total_pretrain_words)}</span></div>
      <div class="phase-row"><span>Format</span><span class="phase-val">Plain text {"{"}"text":"..."{"}"}</span></div>
      <div class="phase-row"><span>Learning Rate</span><span class="phase-val">2e-4</span></div>
      <div class="phase-row"><span>Epochs</span><span class="phase-val">1</span></div>
      <div class="phase-row"><span>Goal</span><span class="phase-val">UPSC domain vocabulary</span></div>
    </div>
    <div class="phase">
      <div class="phase-header">
        <div class="phase-icon">🎯</div>
        <div class="phase-title">Stage 2A — NCERT SFT</div>
        <div class="phase-tag badge-green">Q+A Learning</div>
      </div>
      <div class="phase-row"><span>Data</span><span class="phase-val">unified_sft.jsonl</span></div>
      <div class="phase-row"><span>Records</span><span class="phase-val">{fmt_words(sft.get("v1 (NCERT Q+A)",{}).get("lines",0))} pairs</span></div>
      <div class="phase-row"><span>Format</span><span class="phase-val">Gemma 2 chat template</span></div>
      <div class="phase-row"><span>Learning Rate</span><span class="phase-val">1e-4</span></div>
      <div class="phase-row"><span>Epochs</span><span class="phase-val">2</span></div>
      <div class="phase-row"><span>Goal</span><span class="phase-val">Answer writing behaviour</span></div>
      <div class="phase-row"><span>Source</span><span class="phase-val">NCERT + coaching chunks</span></div>
    </div>
    <div class="phase">
      <div class="phase-header">
        <div class="phase-icon">🏛️</div>
        <div class="phase-title">Stage 2B — Coaching SFT</div>
        <div class="phase-tag badge-amber">Style Tuning</div>
      </div>
      <div class="phase-row"><span>Data</span><span class="phase-val">unified_sft_v2.jsonl</span></div>
      <div class="phase-row"><span>Records</span><span class="phase-val">{fmt_words(sft.get("v2 (Coaching)",{}).get("lines",0))} pairs</span></div>
      <div class="phase-row"><span>Format</span><span class="phase-val">Gemma 2 chat template</span></div>
      <div class="phase-row"><span>Learning Rate</span><span class="phase-val">5e-5</span></div>
      <div class="phase-row"><span>Epochs</span><span class="phase-val">1</span></div>
      <div class="phase-row"><span>Goal</span><span class="phase-val">Coaching-style long answers</span></div>
      <div class="phase-row"><span>Source</span><span class="phase-val">Vision IAS, Drishti answers</span></div>
    </div>
    <div class="phase">
      <div class="phase-header">
        <div class="phase-icon">📝</div>
        <div class="phase-title">Stage 2C — Real UPSC SFT</div>
        <div class="phase-tag badge-red">Exam Calibration</div>
      </div>
      <div class="phase-row"><span>Data</span><span class="phase-val">extracted_questions.jsonl</span></div>
      <div class="phase-row"><span>Records</span><span class="phase-val">{fmt_num(qs["total"])} real questions</span></div>
      <div class="phase-row"><span>Format</span><span class="phase-val">Gemma 2 chat template</span></div>
      <div class="phase-row"><span>Learning Rate</span><span class="phase-val">2e-5</span></div>
      <div class="phase-row"><span>Epochs</span><span class="phase-val">3</span></div>
      <div class="phase-row"><span>Goal</span><span class="phase-val">Real exam question alignment</span></div>
      <div class="phase-row"><span>Source</span><span class="phase-val">UPSC Mains 2014–2025</span></div>
    </div>
    <div class="phase">
      <div class="phase-header">
        <div class="phase-icon">🔢</div>
        <div class="phase-title">Stage 3 — Prelims MCQ</div>
        <div class="phase-tag badge-blue">MCQ Training</div>
      </div>
      <div class="phase-row"><span>Data</span><span class="phase-val">prelims_mcq.jsonl (pending)</span></div>
      <div class="phase-row"><span>Records</span><span class="phase-val">~15 years × 100 Qs</span></div>
      <div class="phase-row"><span>Format</span><span class="phase-val">MCQ + explanation</span></div>
      <div class="phase-row"><span>Learning Rate</span><span class="phase-val">1e-4</span></div>
      <div class="phase-row"><span>Goal</span><span class="phase-val">Prelims answer + reasoning</span></div>
      <div class="phase-row"><span>Status</span><span class="phase-val" style="color:var(--amber)">⏳ Pending</span></div>
    </div>
    <div class="phase">
      <div class="phase-header">
        <div class="phase-icon">🗣️</div>
        <div class="phase-title">Stage 4 — Hinglish Chat</div>
        <div class="phase-tag badge-green">Mentor Mode</div>
      </div>
      <div class="phase-row"><span>Data</span><span class="phase-val">hinglish_chat.jsonl (pending)</span></div>
      <div class="phase-row"><span>Format</span><span class="phase-val">Hindi+English mixed Q+A</span></div>
      <div class="phase-row"><span>Goal</span><span class="phase-val">Conversational UPSC mentor</span></div>
      <div class="phase-row"><span>Status</span><span class="phase-val" style="color:var(--amber)">⏳ Pending</span></div>
    </div>
  </div>
</div>

<!-- ═══════════════════════════════════════════════════════════ SUBJECT TABLE -->
<div class="section">
  <div class="section-title">NCERT + Coaching — Subject-wise Extraction</div>
  <p class="section-sub">Per-subject PDF extraction statistics feeding the pretrain corpus</p>
  <div class="card">
    <table>
      <thead><tr>
        <th>Subject</th><th>PDFs</th><th>Pages</th><th>OCR Pages</th>
        <th>Chunks</th><th>Raw Size</th><th>Clean Size</th><th>Retention</th><th>Quality</th>
      </tr></thead>
      <tbody>{subj_rows if subj_rows else "<tr><td colspan='9' style='text-align:center;color:var(--muted);padding:20px'>Run pdf_extractor.py to generate metadata</td></tr>"}</tbody>
    </table>
  </div>
</div>

<!-- ═══════════════════════════════════════════════════════════ CHARTS -->
<div class="section">
  <div class="section-title">Dataset Visualisation</div>
  <p class="section-sub">Distribution of data across subjects, years and question types</p>
  <div class="grid2">
    <div class="card">
      <div class="card-title">Chunks by Subject (Pretrain Corpus)</div>
      <div class="chart-wrap"><canvas id="chartChunks"></canvas></div>
    </div>
    <div class="card">
      <div class="card-title">Extraction Quality Score by Subject</div>
      <div class="chart-wrap"><canvas id="chartQuality"></canvas></div>
    </div>
  </div>
  <div class="grid2">
    <div class="card">
      <div class="card-title">UPSC Questions Extracted — By Year</div>
      <div class="chart-wrap"><canvas id="chartQYear"></canvas></div>
    </div>
    <div class="card">
      <div class="card-title">UPSC Questions — By Type</div>
      <div class="chart-wrap"><canvas id="chartQType"></canvas></div>
    </div>
  </div>
  <div class="grid3">
    <div class="card">
      <div class="card-title">SFT Dataset Versions</div>
      <div class="chart-sm"><canvas id="chartSFT"></canvas></div>
    </div>
    <div class="card">
      <div class="card-title">Questions by Paper</div>
      <div class="chart-sm"><canvas id="chartQPaper"></canvas></div>
    </div>
    <div class="card">
      <div class="card-title">UPSC PDFs by Year</div>
      <div class="chart-sm"><canvas id="chartPDFYear"></canvas></div>
    </div>
  </div>
</div>

<!-- ═══════════════════════════════════════════════════════════ UPSC QUESTIONS -->
<div class="section">
  <div class="section-title">UPSC Mains Questions — Extracted Corpus</div>
  <p class="section-sub">
    {fmt_num(qs["total"])} questions extracted from {fmt_num(pdfs["total"])} official UPSC Mains PDFs
    (2014–2025) using GPT-4o vision. Questions span GS1–GS4 and Essay papers.
    Ethics (GS4) case studies preserved as single complete entries including all sub-questions.
  </p>
  <div class="grid3">
    <div class="card">
      <div class="card-title">By Year</div>
      <table><thead><tr><th>Year</th><th>Questions</th></tr></thead>
      <tbody>{q_year_rows if q_year_rows else "<tr><td colspan='2' style='color:var(--muted);text-align:center'>No data yet</td></tr>"}</tbody></table>
    </div>
    <div class="card">
      <div class="card-title">By Paper</div>
      <table><thead><tr><th>Paper</th><th>Questions</th></tr></thead>
      <tbody>{q_paper_rows if q_paper_rows else "<tr><td colspan='2' style='color:var(--muted);text-align:center'>No data yet</td></tr>"}</tbody></table>
    </div>
    <div class="card">
      <div class="card-title">Top Question Types</div>
      <table><thead><tr><th>Type</th><th>Count</th></tr></thead>
      <tbody>{q_type_rows if q_type_rows else "<tr><td colspan='2' style='color:var(--muted);text-align:center'>No data yet</td></tr>"}</tbody></table>
    </div>
  </div>
</div>

<!-- ═══════════════════════════════════════════════════════════ SFT DATASETS -->
<div class="section">
  <div class="section-title">SFT Dataset Versions</div>
  <p class="section-sub">Supervised Fine-Tuning datasets in Gemma 2 chat format</p>
  <div class="card">
    <table>
      <thead><tr><th>Version</th><th>Q+A Pairs</th><th>Words</th><th>File Size</th></tr></thead>
      <tbody>{sft_rows if sft_rows else "<tr><td colspan='4' style='text-align:center;color:var(--muted);padding:20px'>Run generate_sft_dataset.py to create SFT data</td></tr>"}</tbody>
    </table>
  </div>
</div>

<!-- ═══════════════════════════════════════════════════════════ UPSC PAPERS TABLE -->
<div class="section">
  <div class="section-title">UPSC Mains Papers Corpus</div>
  <p class="section-sub">Official question papers downloaded and processed</p>
  <div class="grid2">
    <div class="card">
      <div class="card-title">PDFs by Year</div>
      <table><thead><tr><th>Year</th><th>PDFs</th></tr></thead>
      <tbody>{pdf_year_rows if pdf_year_rows else "<tr><td colspan='2' style='color:var(--muted);text-align:center'>No PDFs found</td></tr>"}</tbody></table>
    </div>
    <div class="card" style="display:flex;flex-direction:column;justify-content:center;gap:16px;padding:32px">
      <div style="text-align:center">
        <div style="font-size:48px;font-weight:800;color:var(--accent2)">{fmt_num(pdfs["total"])}</div>
        <div style="color:var(--muted);margin-top:4px">Total official UPSC PDFs</div>
      </div>
      <div style="text-align:center">
        <div style="font-size:32px;font-weight:700;color:var(--green)">{fmt_size(pdfs["total_size_bytes"])}</div>
        <div style="color:var(--muted);margin-top:4px">Total raw PDF size</div>
      </div>
      <div style="text-align:center">
        <div style="font-size:24px;font-weight:700;color:var(--amber)">{len(pdfs["by_year"])} years</div>
        <div style="color:var(--muted);margin-top:4px">Coverage (2014 – 2025)</div>
      </div>
    </div>
  </div>
</div>

</div><!-- /page -->

<script>
const SUBJ_NAMES   = {json.dumps(subj_names)};
const SUBJ_CHUNKS  = {json.dumps(subj_chunks)};
const SUBJ_QUALITY = {json.dumps(subj_quality)};
const SUBJ_PAGES   = {json.dumps(subj_pages)};
const Q_TYPE_LBLS  = {json.dumps(q_type_labels)};
const Q_TYPE_VALS  = {json.dumps(q_type_vals)};
const Q_YEAR_LBLS  = {json.dumps([str(y) for y in q_year_labels])};
const Q_YEAR_VALS  = {json.dumps(q_year_vals)};
const Q_PAPER_LBLS = {json.dumps(list(qs["by_paper"].keys()))};
const Q_PAPER_VALS = {json.dumps(list(qs["by_paper"].values()))};
const SFT_LBLS     = {json.dumps(list(sft.keys()))};
const SFT_VALS     = {json.dumps([sft[k].get("lines",0) for k in sft.keys()])};
const PDF_YEAR_LBLS= {json.dumps([str(y) for y in pdfs["by_year"].keys()])};
const PDF_YEAR_VALS= {json.dumps(list(pdfs["by_year"].values()))};

const PALETTE = ['#6366f1','#22d3ee','#10b981','#f59e0b','#ef4444',
                 '#a78bfa','#34d399','#fbbf24','#f87171','#60a5fa','#e879f9','#4ade80'];

const TO = {{
  responsive:true, maintainAspectRatio:false,
  plugins:{{
    legend:{{labels:{{color:'#94a3b8',font:{{size:11}}}}}},
    tooltip:{{backgroundColor:'#1e293b',borderColor:'#334155',borderWidth:1,
              titleColor:'#e2e8f0',bodyColor:'#94a3b8'}}
  }},
  scales:{{
    x:{{ticks:{{color:'#64748b',font:{{size:10}}}},grid:{{color:'rgba(51,65,85,.3)'}}}},
    y:{{ticks:{{color:'#64748b',font:{{size:10}}}},grid:{{color:'rgba(51,65,85,.3)'}}}}
  }}
}};

function bar(id, labels, data, label, color) {{
  new Chart(document.getElementById(id), {{
    type:'bar',
    data:{{ labels, datasets:[{{ label, data,
            backgroundColor: color || PALETTE,
            borderRadius:6, borderSkipped:false }}] }},
    options:{{ ...TO, plugins:{{...TO.plugins, legend:{{display:false}} }} }}
  }});
}}

function doughnut(id, labels, data) {{
  new Chart(document.getElementById(id), {{
    type:'doughnut',
    data:{{ labels, datasets:[{{ data, backgroundColor:PALETTE,
            borderWidth:2, borderColor:'#1e293b' }}] }},
    options:{{ responsive:true, maintainAspectRatio:false,
      plugins:{{
        legend:{{position:'bottom',labels:{{color:'#94a3b8',font:{{size:11}},padding:10}}}},
        tooltip:{{backgroundColor:'#1e293b',borderColor:'#334155',borderWidth:1,
                  titleColor:'#e2e8f0',bodyColor:'#94a3b8'}}
      }}
    }}
  }});
}}

bar('chartChunks',  SUBJ_NAMES, SUBJ_CHUNKS,  'Chunks',  '#6366f1');
bar('chartQuality', SUBJ_NAMES, SUBJ_QUALITY, 'Quality', '#10b981');
bar('chartQYear',   Q_YEAR_LBLS,Q_YEAR_VALS,  'Questions','#22d3ee');
bar('chartQType',   Q_TYPE_LBLS,Q_TYPE_VALS,  'Count',   '#f59e0b');
bar('chartPDFYear', PDF_YEAR_LBLS,PDF_YEAR_VALS,'PDFs',  '#a78bfa');
doughnut('chartSFT',   SFT_LBLS,   SFT_VALS);
doughnut('chartQPaper',Q_PAPER_LBLS,Q_PAPER_VALS);
</script>
</body>
</html>"""

    return html


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Generate UPSC SLM research report")
    ap.add_argument("--output", default=str(REPORT_FILE), type=Path,
                    help="Output HTML file path")
    ap.add_argument("--open",  action="store_true", help="Open in browser after generating")
    args = ap.parse_args()

    print("Collecting dataset statistics...")

    data = {
        "pretrain":  collect_pretrain_stats(),
        "sft":       collect_sft_stats(),
        "questions": collect_question_stats(),
        "pdfs":      collect_pdf_stats(),
        "subjects":  collect_subject_metadata(),
    }

    t  = data["pretrain"]
    qs = data["questions"]
    pd = data["pdfs"]

    print(f"  Pretrain chunks : {t['lines']:,}  ({t['words']:,} words)")
    print(f"  SFT versions    : {len(data['sft'])}")
    print(f"  UPSC questions  : {qs['total']:,}  across {len(qs['by_year'])} years")
    print(f"  UPSC PDFs       : {pd['total']:,}  ({pd['total_size_bytes']//1024//1024} MB)")
    print(f"  Subject PDFs    : {sum(s['pdfs'] for s in data['subjects']):,}")

    html = generate_report(data)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"\n✅ Report saved → {out}")

    if args.open:
        if sys.platform == "darwin":
            subprocess.run(["open", str(out)])
        elif sys.platform.startswith("linux"):
            subprocess.run(["xdg-open", str(out)])

if __name__ == "__main__":
    main()
