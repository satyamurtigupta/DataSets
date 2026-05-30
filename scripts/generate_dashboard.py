#!/usr/bin/env python3
"""
generate_dashboard.py
=====================
Generates a self-contained HTML dashboard from unified_pretrain_v2.jsonl
and the rejection log. Run after extracting each subject to update.

Usage:
    python3 scripts/generate_dashboard.py
    python3 scripts/generate_dashboard.py --open      # open in browser after
    python3 scripts/generate_dashboard.py --output custom_dashboard.html
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

BASE_DIR      = Path(__file__).resolve().parent.parent
OUTPUT_DIR    = BASE_DIR / "dataset_output_final" / "combined"
CHUNKS_FILE   = OUTPUT_DIR / "unified_pretrain_v2.jsonl"
REJECT_FILE   = OUTPUT_DIR / "pretrain_v2_rejection_log.jsonl"
DASHBOARD_OUT = BASE_DIR / "dataset_output_final" / "dashboard.html"


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

def load_data():
    chunks = []
    if CHUNKS_FILE.exists():
        with open(CHUNKS_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        chunks.append(json.loads(line))
                    except Exception:
                        pass

    rejections = []
    if REJECT_FILE.exists():
        with open(REJECT_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rejections.append(json.loads(line))
                    except Exception:
                        pass

    return chunks, rejections


def build_subject_data(chunks, rejections):
    subjects = {}

    # Group chunks by subject
    by_subject = defaultdict(list)
    for c in chunks:
        subj = c.get("subject", "unknown").title()
        by_subject[subj].append(c)

    # Group rejections by subject
    rej_by_subject = defaultdict(list)
    for r in rejections:
        subj = r.get("subject", "unknown").title()
        rej_by_subject[subj].append(r)

    for subj, subj_chunks in sorted(by_subject.items()):
        wcs       = [c["word_count"] for c in subj_chunks]
        rej_list  = rej_by_subject.get(subj, [])
        total_processed = len(subj_chunks) + len(rej_list)

        # Word count buckets
        buckets = {
            "80-150":  sum(1 for w in wcs if 80  <= w <= 150),
            "151-250": sum(1 for w in wcs if 151 <= w <= 250),
            "251-350": sum(1 for w in wcs if 251 <= w <= 350),
            "351-450": sum(1 for w in wcs if 351 <= w <= 450),
            "450+":    sum(1 for w in wcs if w   >  450),
        }

        # By section heading
        by_section = defaultdict(list)
        for c in subj_chunks:
            by_section[c.get("section_heading", "—")].append(c)

        # By source file
        by_file = Counter(c["source_file"] for c in subj_chunks)

        # Rejection reasons
        rej_reasons = Counter(r["rejection_reason"] for r in rej_list)

        subjects[subj] = {
            "stats": {
                "total_chunks":     len(subj_chunks),
                "total_words":      sum(wcs),
                "avg_words":        round(sum(wcs) / len(wcs)) if wcs else 0,
                "min_words":        min(wcs) if wcs else 0,
                "max_words":        max(wcs) if wcs else 0,
                "total_rejected":   len(rej_list),
                "pass_rate":        round(100 * len(subj_chunks) / total_processed) if total_processed else 0,
                "unique_sections":  len(by_section),
                "source_files":     dict(by_file),
            },
            "wc_buckets":   buckets,
            "sections":     {
                heading: [
                    {
                        "id":          c["id"],
                        "word_count":  c["word_count"],
                        "char_count":  c["char_count"],
                        "source_file": c["source_file"],
                        "text":        c["text"],
                    }
                    for c in cs
                ]
                for heading, cs in sorted(by_section.items())
            },
            "rejections":    [
                {
                    "reason":       r["rejection_reason"],
                    "page_range":   r.get("page_range", "?"),
                    "batch_num":    r.get("batch_num", "?"),
                    "section":      r.get("section_heading", "?"),
                    "word_count":   r.get("word_count", 0),
                    "source_file":  r.get("source_file", "?"),
                    "text_preview": r.get("text_preview", ""),
                }
                for r in rej_list
            ],
            "rej_reasons":  dict(rej_reasons),
        }

    # ── Build overview block ──
    all_chunks     = chunks
    all_wcs        = [c["word_count"] for c in all_chunks]
    all_rejections = rejections

    overview_buckets = {
        "80-150":  sum(1 for w in all_wcs if 80  <= w <= 150),
        "151-250": sum(1 for w in all_wcs if 151 <= w <= 250),
        "251-350": sum(1 for w in all_wcs if 251 <= w <= 350),
        "351-450": sum(1 for w in all_wcs if 351 <= w <= 450),
        "450+":    sum(1 for w in all_wcs if w   >  450),
    }

    subj_comparison = {
        subj: {
            "chunks":    d["stats"]["total_chunks"],
            "words":     d["stats"]["total_words"],
            "avg_words": d["stats"]["avg_words"],
            "rejected":  d["stats"]["total_rejected"],
            "pass_rate": d["stats"]["pass_rate"],
            "sections":  d["stats"]["unique_sections"],
            "pdfs":      len(d["stats"]["source_files"]),
        }
        for subj, d in subjects.items()
    }

    all_rej_reasons = Counter(r["rejection_reason"] for r in all_rejections)

    subjects["__overview__"] = {
        "stats": {
            "total_chunks":    len(all_chunks),
            "total_words":     sum(all_wcs),
            "avg_words":       round(sum(all_wcs) / len(all_wcs)) if all_wcs else 0,
            "min_words":       min(all_wcs) if all_wcs else 0,
            "max_words":       max(all_wcs) if all_wcs else 0,
            "total_rejected":  len(all_rejections),
            "total_subjects":  len(subjects),
            "pass_rate":       round(100 * len(all_chunks) / (len(all_chunks) + len(all_rejections)))
                               if (all_chunks or all_rejections) else 0,
        },
        "wc_buckets":       overview_buckets,
        "subj_comparison":  subj_comparison,
        "rej_reasons":      dict(all_rej_reasons),
    }

    return subjects


# ---------------------------------------------------------------------------
# HTML generator
# ---------------------------------------------------------------------------

def generate_html(subjects_data: dict, output_path: Path):
    # Escape </script> so it cannot break the inline <script> block
    data_json    = json.dumps(subjects_data, ensure_ascii=False).replace("</", "<\\/")
    ov           = subjects_data.get("__overview__", {})
    ov_stats     = ov.get("stats", {})
    subject_keys = sorted(k for k in subjects_data if k != "__overview__")

    # Build sidebar buttons as static HTML (no JS dependency)
    btn_parts = []
    for k in subject_keys:
        n = subjects_data[k]["stats"]["total_chunks"]
        btn_parts.append(
            '<button id="btn-{k}" class="nav-btn w-full text-left px-3 py-2 rounded-lg'
            ' text-sm text-slate-300 flex items-center justify-between hover:bg-slate-800"'
            ' onclick="loadSubject(\'{k}\')">'
            '<span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{k}</span>'
            '<span style="font-size:11px;background:#334155;padding:1px 6px;border-radius:9999px'
            ';color:#94a3b8;flex-shrink:0">{n}</span>'
            '</button>'.format(k=k, n=n)
        )
    sidebar_buttons = "\n      ".join(btn_parts)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>UPSC SLM — Dataset Dashboard</title>
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  body {{ font-family: system-ui, sans-serif; background:#0f172a; color:#e2e8f0; }}
  ::-webkit-scrollbar{{width:6px}} ::-webkit-scrollbar-track{{background:#1e293b}}
  ::-webkit-scrollbar-thumb{{background:#475569;border-radius:3px}}
  .chunk-card{{transition:border-color .15s}} .chunk-card:hover{{border-color:#6366f1}}
  .nav-btn{{transition:all .15s; cursor:pointer}}
  .nav-btn.active{{background:#4f46e5!important; color:#fff!important}}
  .tab-btn{{transition:all .15s; border-bottom:2px solid transparent}}
  .tab-btn.active{{border-bottom-color:#6366f1; color:#818cf8}}
  mark{{background:#fbbf24;color:#1e293b;border-radius:2px;padding:0 2px}}
  .stat-card{{background:linear-gradient(135deg,#1e293b,#0f172a);border:1px solid #334155}}
  .ov-row:hover{{background:#1e293b}}
</style>
</head>
<body class="min-h-screen">

<!-- TOP BAR -->
<div class="bg-slate-900 border-b border-slate-700 px-5 py-3 flex items-center justify-between sticky top-0 z-50">
  <div class="flex items-center gap-3">
    <div class="w-2 h-2 rounded-full bg-green-400 animate-pulse"></div>
    <span class="font-bold text-indigo-400">UPSC SLM</span>
    <span class="text-slate-500 text-sm">Pretrain Dataset Dashboard</span>
  </div>
  <div class="flex gap-5 text-sm text-slate-400">
    <span id="hdr-s">—</span>
    <span id="hdr-c">—</span>
    <span id="hdr-w">—</span>
  </div>
</div>

<div class="flex" style="height:calc(100vh - 49px)">

  <!-- SIDEBAR -->
  <div class="w-44 bg-slate-900 border-r border-slate-700 flex flex-col flex-shrink-0 overflow-y-auto">

    <!-- Overview button -->
    <div class="px-2 pt-3 pb-1">
      <button id="btn-overview"
        class="nav-btn w-full text-left px-3 py-2 rounded-lg text-sm font-semibold flex items-center gap-2 text-amber-300 bg-amber-950 hover:bg-amber-900"
        onclick="showOverview()">
        <span>🌐</span><span>Overview</span>
      </button>
    </div>

    <div class="px-3 pt-3 pb-1 text-xs font-semibold text-slate-600 uppercase tracking-wider">Subjects</div>
    <div class="flex flex-col gap-1 px-2 pb-4">
      {sidebar_buttons}
    </div>
  </div>

  <!-- MAIN -->
  <div class="flex-1 overflow-y-auto bg-slate-950">

    <!-- ═══════════════════ OVERVIEW PANEL ═══════════════════ -->
    <div id="panel-overview" class="p-6">

      <h2 class="text-lg font-bold text-amber-300 mb-4">Dataset Overview — All Subjects</h2>

      <!-- Global stats -->
      <div class="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
        <div class="stat-card rounded-xl p-4">
          <div class="text-xs text-slate-500 mb-1">Subjects loaded</div>
          <div class="text-3xl font-bold text-amber-400">{ov_stats.get("total_subjects", 0)}</div>
        </div>
        <div class="stat-card rounded-xl p-4">
          <div class="text-xs text-slate-500 mb-1">Total chunks</div>
          <div class="text-3xl font-bold text-indigo-400">{ov_stats.get("total_chunks", 0):,}</div>
        </div>
        <div class="stat-card rounded-xl p-4">
          <div class="text-xs text-slate-500 mb-1">Total words</div>
          <div class="text-3xl font-bold text-emerald-400">{ov_stats.get("total_words", 0) // 1000}K</div>
        </div>
        <div class="stat-card rounded-xl p-4">
          <div class="text-xs text-slate-500 mb-1">Overall pass rate</div>
          <div class="text-3xl font-bold text-amber-400">{ov_stats.get("pass_rate", 0)}%</div>
        </div>
        <div class="stat-card rounded-xl p-4">
          <div class="text-xs text-slate-500 mb-1">Avg words/chunk</div>
          <div class="text-3xl font-bold text-sky-400">{ov_stats.get("avg_words", 0)}</div>
        </div>
        <div class="stat-card rounded-xl p-4">
          <div class="text-xs text-slate-500 mb-1">Min / Max words</div>
          <div class="text-2xl font-bold text-slate-300">{ov_stats.get("min_words", 0)} / {ov_stats.get("max_words", 0)}</div>
        </div>
        <div class="stat-card rounded-xl p-4">
          <div class="text-xs text-slate-500 mb-1">Total rejected</div>
          <div class="text-3xl font-bold text-rose-400">{ov_stats.get("total_rejected", 0)}</div>
        </div>
        <div class="stat-card rounded-xl p-4">
          <div class="text-xs text-slate-500 mb-1">Pending subjects</div>
          <div class="text-3xl font-bold text-slate-400">{10 - ov_stats.get("total_subjects", 0)}</div>
          <div class="text-xs text-slate-600">of 10 planned</div>
        </div>
      </div>

      <!-- Subject comparison table -->
      <div class="mb-6">
        <div class="text-sm font-semibold text-slate-400 mb-2">Subject Comparison</div>
        <div class="overflow-x-auto rounded-xl border border-slate-800">
          <table class="w-full text-sm">
            <thead>
              <tr class="bg-slate-800 text-slate-400 text-xs uppercase">
                <th class="text-left px-4 py-2">Subject</th>
                <th class="text-right px-3 py-2">PDFs</th>
                <th class="text-right px-3 py-2">Chunks</th>
                <th class="text-right px-3 py-2">Words</th>
                <th class="text-right px-3 py-2">Avg</th>
                <th class="text-right px-3 py-2">Sections</th>
                <th class="text-right px-3 py-2">Rejected</th>
                <th class="text-right px-3 py-2">Pass%</th>
                <th class="px-3 py-2">Coverage</th>
              </tr>
            </thead>
            <tbody id="ov-table-body" class="divide-y divide-slate-800"></tbody>
          </table>
        </div>
      </div>

      <!-- Two charts side by side -->
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        <div class="stat-card rounded-xl p-4">
          <div class="text-sm font-semibold text-slate-400 mb-3">Chunks per Subject</div>
          <canvas id="ov-bar-chart" height="220"></canvas>
        </div>
        <div class="stat-card rounded-xl p-4">
          <div class="text-sm font-semibold text-slate-400 mb-3">Word Count Distribution (All)</div>
          <canvas id="ov-wc-chart" height="220"></canvas>
        </div>
      </div>

      <!-- Rejection reasons across all subjects -->
      <div id="ov-rej-block" class="stat-card rounded-xl p-4 hidden">
        <div class="text-sm font-semibold text-slate-400 mb-3">Rejection Reasons (All Subjects)</div>
        <div id="ov-rej-chips" class="flex flex-wrap gap-2"></div>
      </div>
    </div>

    <!-- ═══════════════════ SUBJECT PANEL ═══════════════════ -->
    <div id="panel-subject" class="hidden p-6">

      <!-- Stats row -->
      <div class="grid grid-cols-2 lg:grid-cols-4 xl:grid-cols-7 gap-3 mb-5">
        <div class="stat-card rounded-lg p-3"><div class="text-xs text-slate-500 mb-1">Chunks</div><div id="s-chunks" class="text-2xl font-bold text-indigo-400">—</div></div>
        <div class="stat-card rounded-lg p-3"><div class="text-xs text-slate-500 mb-1">Words</div><div id="s-words" class="text-2xl font-bold text-emerald-400">—</div></div>
        <div class="stat-card rounded-lg p-3"><div class="text-xs text-slate-500 mb-1">Avg words</div><div id="s-avg" class="text-2xl font-bold text-sky-400">—</div></div>
        <div class="stat-card rounded-lg p-3"><div class="text-xs text-slate-500 mb-1">Min / Max</div><div id="s-minmax" class="text-lg font-bold text-slate-300">—</div></div>
        <div class="stat-card rounded-lg p-3"><div class="text-xs text-slate-500 mb-1">Sections</div><div id="s-sections" class="text-2xl font-bold text-purple-400">—</div></div>
        <div class="stat-card rounded-lg p-3"><div class="text-xs text-slate-500 mb-1">Rejected</div><div id="s-rejected" class="text-2xl font-bold text-rose-400">—</div></div>
        <div class="stat-card rounded-lg p-3"><div class="text-xs text-slate-500 mb-1">Pass rate</div><div id="s-passrate" class="text-2xl font-bold text-amber-400">—</div></div>
      </div>

      <!-- Source files -->
      <div class="mb-4 flex flex-wrap gap-2" id="s-files"></div>

      <!-- Tabs -->
      <div class="flex gap-6 border-b border-slate-800 mb-4">
        <button onclick="showTab('chunks')"   id="tab-chunks"   class="tab-btn active pb-2 text-sm font-medium text-slate-300">Chunks</button>
        <button onclick="showTab('sections')" id="tab-sections" class="tab-btn pb-2 text-sm font-medium text-slate-500">Sections</button>
        <button onclick="showTab('chart')"    id="tab-chart"    class="tab-btn pb-2 text-sm font-medium text-slate-500">Distribution</button>
        <button onclick="showTab('rejects')"  id="tab-rejects"  class="tab-btn pb-2 text-sm font-medium text-slate-500">Rejections</button>
      </div>

      <!-- Chunks tab -->
      <div id="view-chunks">
        <div class="flex flex-wrap gap-2 mb-4">
          <input id="search-input" type="text" placeholder="Search text, headings, IDs…"
            class="flex-1 min-w-48 bg-slate-800 border border-slate-600 rounded-lg px-4 py-2 text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-indigo-500"
            oninput="filterChunks()">
          <select id="section-filter" class="bg-slate-800 border border-slate-600 rounded-lg px-3 py-2 text-sm text-slate-300 focus:outline-none" onchange="filterChunks()">
            <option value="">All sections</option>
          </select>
          <select id="file-filter" class="bg-slate-800 border border-slate-600 rounded-lg px-3 py-2 text-sm text-slate-300 focus:outline-none" onchange="filterChunks()">
            <option value="">All files</option>
          </select>
          <span id="chunk-count-label" class="text-sm text-slate-500 self-center whitespace-nowrap"></span>
        </div>
        <div id="chunk-list" class="flex flex-col gap-3"></div>
        <div id="load-more-wrap" class="flex justify-center mt-4 hidden">
          <button onclick="loadMore()" class="px-6 py-2 bg-indigo-700 hover:bg-indigo-600 rounded-lg text-sm font-medium">Load more</button>
        </div>
      </div>

      <!-- Sections tab -->
      <div id="view-sections" class="hidden">
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-2" id="section-grid"></div>
      </div>

      <!-- Chart tab -->
      <div id="view-chart" class="hidden max-w-xl">
        <canvas id="wc-chart" height="230"></canvas>
      </div>

      <!-- Rejections tab -->
      <div id="view-rejects" class="hidden">
        <div id="rej-summary" class="flex flex-wrap gap-2 mb-4"></div>
        <div id="rej-list" class="flex flex-col gap-3"></div>
        <div id="rej-empty" class="hidden text-slate-500 text-sm">No rejection log for this subject yet. Re-run extraction to populate.</div>
      </div>
    </div>

  </div><!-- /MAIN -->
</div>

<script>
const DATA = {data_json};

let currentSubject = null;
let currentChunks  = [];
let filteredChunks = [];
let displayedCount = 0;
const PAGE_SIZE    = 30;
let wcChart = null, ovBarChart = null, ovWcChart = null;

// ─────────────────────────────────────────────────────────────────────────────
// INIT — sidebar buttons are static HTML; just update header + overview table
// ─────────────────────────────────────────────────────────────────────────────
try {{
  var ov = DATA['__overview__'];
  var nSubjects = Object.keys(DATA).filter(function(k){{ return k !== '__overview__'; }}).length;
  document.getElementById('hdr-s').textContent = nSubjects + ' subjects';
  document.getElementById('hdr-c').textContent = ov.stats.total_chunks.toLocaleString() + ' chunks';
  document.getElementById('hdr-w').textContent = (ov.stats.total_words/1000).toFixed(1) + 'K words';
  buildOverviewTable();
}} catch(e) {{ console.error('Init error:', e); }}

// Charts load after CDN scripts
window.onload = function() {{
  try {{ buildOverviewCharts(); }} catch(e) {{ console.warn('Charts unavailable:', e); }}
}};

// ─────────────────────────────────────────────────────────────────────────────
// OVERVIEW
// ─────────────────────────────────────────────────────────────────────────────
function showOverview() {{
  document.getElementById('panel-overview').classList.remove('hidden');
  document.getElementById('panel-subject').classList.add('hidden');
  document.getElementById('btn-overview').classList.add('active');
  document.querySelectorAll('[id^="btn-"]:not(#btn-overview)').forEach(b => b.classList.remove('active'));
  currentSubject = null;
}}

// Table only — no Chart.js needed
function buildOverviewTable() {{
  const ov       = DATA['__overview__'];
  const subjects = Object.keys(DATA).filter(k => k !== '__overview__').sort();
  const maxChunks = Math.max.apply(null, subjects.map(function(s) {{ return DATA[s].stats.total_chunks; }}));

  var tbody = document.getElementById('ov-table-body');
  subjects.forEach(function(subj) {{
    var d = DATA[subj];
    var s = d.stats;
    var barW = Math.round(s.total_chunks / maxChunks * 100);
    var passColor = s.pass_rate >= 90 ? 'color:#34d399' : s.pass_rate >= 80 ? 'color:#fbbf24' : 'color:#f87171';
    var nPdfs = Object.keys(s.source_files || {{}}).length;
    var row = document.createElement('tr');
    row.className = 'ov-row cursor-pointer';
    row.setAttribute('onclick', 'loadSubject("' + subj + '")');
    row.innerHTML =
      '<td class="px-4 py-2 font-medium" style="color:#e2e8f0">' + subj + '</td>' +
      '<td class="px-3 py-2 text-right" style="color:#94a3b8">' + nPdfs + '</td>' +
      '<td class="px-3 py-2 text-right font-mono" style="color:#a5b4fc">' + s.total_chunks.toLocaleString() + '</td>' +
      '<td class="px-3 py-2 text-right font-mono" style="color:#6ee7b7">' + (s.total_words/1000).toFixed(1) + 'K</td>' +
      '<td class="px-3 py-2 text-right" style="color:#94a3b8">' + s.avg_words + 'w</td>' +
      '<td class="px-3 py-2 text-right" style="color:#d8b4fe">' + s.unique_sections + '</td>' +
      '<td class="px-3 py-2 text-right" style="color:#fca5a5">' + s.total_rejected + '</td>' +
      '<td class="px-3 py-2 text-right font-bold" style="' + passColor + '">' + s.pass_rate + '%</td>' +
      '<td class="px-3 py-2 w-28"><div class="bg-slate-800 rounded-full h-2 w-full">' +
      '<div class="bg-indigo-500 h-2 rounded-full" style="width:' + barW + '%"></div></div></td>';
    tbody.appendChild(row);
  }});

  // Rejection reasons chips (no Chart.js needed)
  if (ov.rej_reasons && Object.keys(ov.rej_reasons).length > 0) {{
    document.getElementById('ov-rej-block').classList.remove('hidden');
    var chips = document.getElementById('ov-rej-chips');
    Object.entries(ov.rej_reasons).sort(function(a,b){{return b[1]-a[1];}}).forEach(function(entry) {{
      var span = document.createElement('span');
      span.className = 'px-3 py-1 rounded-full text-sm border border-rose-800 text-rose-300 bg-rose-950';
      span.textContent = entry[0] + ': ' + entry[1];
      chips.appendChild(span);
    }});
  }}
}}

// Charts only — requires Chart.js
function buildOverviewCharts() {{
  var ov       = DATA['__overview__'];
  var subjects = Object.keys(DATA).filter(function(k){{ return k !== '__overview__'; }}).sort();
  var colors   = ['#6366f1','#22c55e','#f59e0b','#3b82f6','#ec4899','#14b8a6','#f97316','#8b5cf6','#06b6d4','#84cc16'];

  var barCtx = document.getElementById('ov-bar-chart').getContext('2d');
  ovBarChart = new Chart(barCtx, {{
    type: 'bar',
    data: {{
      labels: subjects,
      datasets: [{{ label:'Chunks', data: subjects.map(function(s){{return DATA[s].stats.total_chunks;}}),
        backgroundColor: colors, borderRadius:5 }}]
    }},
    options: {{ responsive:true, plugins:{{ legend:{{display:false}} }},
      scales:{{ x:{{ticks:{{color:'#94a3b8'}},grid:{{color:'#1e293b'}}}},
               y:{{ticks:{{color:'#94a3b8'}},grid:{{color:'#1e293b'}}}} }} }}
  }});

  var wcCtx = document.getElementById('ov-wc-chart').getContext('2d');
  var bk = ov.wc_buckets;
  ovWcChart = new Chart(wcCtx, {{
    type: 'doughnut',
    data: {{
      labels: Object.keys(bk),
      datasets: [{{ data: Object.values(bk),
        backgroundColor: ['#6366f1','#818cf8','#a5b4fc','#c7d2fe','#e0e7ff'],
        borderWidth:0 }}]
    }},
    options: {{ responsive:true, cutout:'60%',
      plugins:{{ legend:{{ position:'right', labels:{{ color:'#94a3b8', boxWidth:12 }} }} }} }}
  }});
}}

// ─────────────────────────────────────────────────────────────────────────────
// SUBJECT VIEW
// ─────────────────────────────────────────────────────────────────────────────
function loadSubject(subj) {{
  currentSubject = subj;
  const d = DATA[subj];

  // Panel switch
  document.getElementById('panel-overview').classList.add('hidden');
  document.getElementById('panel-subject').classList.remove('hidden');

  // Sidebar highlights
  document.getElementById('btn-overview').classList.remove('active');
  document.querySelectorAll('[id^="btn-"]:not(#btn-overview)').forEach(b => {{
    b.classList.toggle('active', b.id === 'btn-' + subj);
  }});

  // Stats
  document.getElementById('s-chunks').textContent  = d.stats.total_chunks.toLocaleString();
  document.getElementById('s-words').textContent   = (d.stats.total_words/1000).toFixed(1)+'K';
  document.getElementById('s-avg').textContent     = d.stats.avg_words+'w';
  document.getElementById('s-minmax').textContent  = d.stats.min_words+' / '+d.stats.max_words;
  document.getElementById('s-sections').textContent= d.stats.unique_sections;
  document.getElementById('s-rejected').textContent= d.stats.total_rejected;
  document.getElementById('s-passrate').textContent= d.stats.pass_rate+'%';

  // Source files
  const filesDiv = document.getElementById('s-files');
  filesDiv.innerHTML = '';
  Object.entries(d.stats.source_files||{{}}).forEach(([f,n]) => {{
    filesDiv.innerHTML += `<span class="text-xs bg-slate-800 border border-slate-700 px-3 py-1 rounded-full text-slate-400">📄 ${{f}} (${{n}})</span>`;
  }});

  // Flat chunk list
  currentChunks = [];
  Object.entries(d.sections||{{}}).forEach(([heading, chunks]) => {{
    chunks.forEach(c => {{ c._heading = heading; currentChunks.push(c); }});
  }});

  // Section filter
  const secFilter = document.getElementById('section-filter');
  secFilter.innerHTML = '<option value="">All sections</option>';
  Object.keys(d.sections||{{}}).sort().forEach(h => {{
    const o=document.createElement('option'); o.value=h; o.textContent=h.length>50?h.slice(0,50)+'…':h; secFilter.appendChild(o);
  }});

  // File filter
  const fileFilter = document.getElementById('file-filter');
  fileFilter.innerHTML = '<option value="">All files</option>';
  Object.keys(d.stats.source_files||{{}}).forEach(f => {{
    const o=document.createElement('option'); o.value=f; o.textContent=f.length>40?f.slice(0,40)+'…':f; fileFilter.appendChild(o);
  }});

  // Sections grid
  const secGrid = document.getElementById('section-grid');
  secGrid.innerHTML = '';
  Object.entries(d.sections||{{}}).sort((a,b)=>b[1].length-a[1].length).forEach(([h,cs]) => {{
    const tw = cs.reduce((s,c)=>s+c.word_count,0);
    const el = document.createElement('div');
    el.className = 'flex justify-between items-center px-4 py-3 rounded-lg border border-slate-800 hover:border-indigo-700 cursor-pointer';
    el.innerHTML =
      '<div class="min-w-0 mr-3">'
      + '<div class="text-sm font-medium text-slate-200 truncate">' + escHtml(h) + '</div>'
      + '<div class="text-xs text-slate-500 mt-0.5">' + tw.toLocaleString() + ' words</div>'
      + '</div>'
      + '<div class="text-right flex-shrink-0">'
      + '<div class="text-lg font-bold text-indigo-400">' + cs.length + '</div>'
      + '<div class="text-xs text-slate-500">chunks</div>'
      + '</div>';
    el.addEventListener('click', function() {{ filterBySection(h); }});
    secGrid.appendChild(el);
  }});

  document.getElementById('search-input').value = '';
  document.getElementById('section-filter').value = '';
  document.getElementById('file-filter').value = '';
  showTab('chunks');
  filterChunks();
  buildSubjectChart(d);
  buildRejectionsTab(d);
}}

// ─────────────────────────────────────────────────────────────────────────────
// CHUNK FILTER + RENDER
// ─────────────────────────────────────────────────────────────────────────────
function filterChunks() {{
  const q = document.getElementById('search-input').value.toLowerCase().trim();
  const s = document.getElementById('section-filter').value;
  const f = document.getElementById('file-filter').value;
  filteredChunks = currentChunks.filter(c => {{
    if (s && c._heading !== s) return false;
    if (f && c.source_file !== f) return false;
    if (q) return c.text.toLowerCase().includes(q)||c._heading.toLowerCase().includes(q)||c.id.toLowerCase().includes(q);
    return true;
  }});
  displayedCount = 0;
  document.getElementById('chunk-list').innerHTML = '';
  document.getElementById('chunk-count-label').textContent = filteredChunks.length+' of '+currentChunks.length+' chunks';
  loadMore();
}}

function filterBySection(h) {{
  document.getElementById('section-filter').value = h;
  showTab('chunks');
  filterChunks();
}}

function loadMore() {{
  const q    = document.getElementById('search-input').value.toLowerCase().trim();
  const list = document.getElementById('chunk-list');
  filteredChunks.slice(displayedCount, displayedCount+PAGE_SIZE).forEach(c => {{
    const text = q ? highlight(c.text,q) : escHtml(c.text);
    const head = q ? highlight(c._heading,q) : escHtml(c._heading);
    list.innerHTML += `
      <div class="chunk-card border border-slate-800 rounded-xl p-4 hover:bg-slate-900">
        <div class="flex items-start justify-between mb-2 gap-2">
          <div class="flex-1 min-w-0">
            <span class="text-xs font-mono text-indigo-400">${{escHtml(c.id)}}</span>
            <div class="text-sm font-semibold text-slate-300 mt-0.5 truncate" title="${{escHtml(c._heading)}}">${{head}}</div>
          </div>
          <div class="flex gap-2 flex-shrink-0">
            <span class="text-xs bg-slate-800 px-2 py-1 rounded-full text-slate-400">${{c.word_count}}w</span>
            <span class="text-xs bg-slate-800 px-2 py-1 rounded-full text-slate-500 max-w-28 truncate" title="${{escHtml(c.source_file)}}">${{c.source_file.replace('.pdf','')}}</span>
          </div>
        </div>
        <div class="text-sm text-slate-400 leading-relaxed whitespace-pre-wrap">${{text}}</div>
      </div>`;
  }});
  displayedCount += PAGE_SIZE;
  document.getElementById('load-more-wrap').classList.toggle('hidden', displayedCount >= filteredChunks.length);
}}

// ─────────────────────────────────────────────────────────────────────────────
// CHARTS
// ─────────────────────────────────────────────────────────────────────────────
function buildSubjectChart(d) {{
  const ctx = document.getElementById('wc-chart').getContext('2d');
  if (wcChart) wcChart.destroy();
  wcChart = new Chart(ctx, {{
    type:'bar',
    data:{{ labels:Object.keys(d.wc_buckets), datasets:[{{
      label:'Chunks', data:Object.values(d.wc_buckets),
      backgroundColor:['#6366f1','#818cf8','#a5b4fc','#c7d2fe','#e0e7ff'], borderRadius:5
    }}] }},
    options:{{ responsive:true, plugins:{{ legend:{{display:false}},
      title:{{display:true,text:'Word Count Distribution',color:'#94a3b8',font:{{size:13}}}} }},
      scales:{{ x:{{ticks:{{color:'#94a3b8'}},grid:{{color:'#1e293b'}}}},
               y:{{ticks:{{color:'#94a3b8'}},grid:{{color:'#1e293b'}}}} }} }}
  }});
}}

// ─────────────────────────────────────────────────────────────────────────────
// REJECTIONS
// ─────────────────────────────────────────────────────────────────────────────
function buildRejectionsTab(d) {{
  const summary = document.getElementById('rej-summary');
  const list    = document.getElementById('rej-list');
  const empty   = document.getElementById('rej-empty');
  summary.innerHTML = ''; list.innerHTML = '';
  if (!d.rejections || d.rejections.length===0) {{ empty.classList.remove('hidden'); return; }}
  empty.classList.add('hidden');
  Object.entries(d.rej_reasons||{{}}).sort((a,b)=>b[1]-a[1]).forEach(([r,n]) => {{
    summary.innerHTML += `<span class="px-3 py-1 rounded-full text-sm border border-rose-800 text-rose-300 bg-rose-950">${{r}}: ${{n}}</span>`;
  }});
  d.rejections.forEach(r => {{
    list.innerHTML += `
      <div class="border border-slate-800 border-l-4 border-l-rose-700 rounded-xl p-4 bg-slate-900">
        <div class="flex flex-wrap gap-2 mb-2">
          <span class="text-xs bg-rose-950 text-rose-300 px-2 py-0.5 rounded-full">${{escHtml(r.reason)}}</span>
          <span class="text-xs bg-slate-800 text-slate-400 px-2 py-0.5 rounded-full">pages ${{r.page_range}}</span>
          <span class="text-xs bg-slate-800 text-slate-400 px-2 py-0.5 rounded-full">batch ${{r.batch_num}}</span>
          <span class="text-xs bg-slate-800 text-slate-400 px-2 py-0.5 rounded-full">${{r.word_count}}w</span>
          <span class="text-xs bg-slate-800 text-slate-500 px-2 py-0.5 rounded-full truncate max-w-xs">${{escHtml(r.source_file)}}</span>
        </div>
        <div class="text-xs font-medium text-indigo-300 mb-1">${{escHtml(r.section)}}</div>
        <div class="text-sm text-slate-500 italic">${{escHtml(r.text_preview)}}</div>
      </div>`;
  }});
}}

// ─────────────────────────────────────────────────────────────────────────────
// TABS + HELPERS
// ─────────────────────────────────────────────────────────────────────────────
function showTab(name) {{
  ['chunks','sections','chart','rejects'].forEach(t => {{
    document.getElementById('view-'+t).classList.toggle('hidden', t!==name);
    document.getElementById('tab-'+t).classList.toggle('active', t===name);
  }});
}}

function highlight(text, q) {{
  return escHtml(text).replace(new RegExp('('+escRegex(q)+')','gi'),'<mark>$1</mark>');
}}
function escHtml(s)  {{ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }}
function escRegex(s) {{ return s.replace(/[.*+?^${{}}()|[\\]\\\\]/g,'\\\\$&'); }}
</script>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Generate HTML dashboard from pretrain JSONL")
    ap.add_argument("--output", default=str(DASHBOARD_OUT), help="Output HTML path")
    ap.add_argument("--open",   action="store_true",        help="Open in browser after generating")
    args = ap.parse_args()

    output_path = Path(args.output)

    print("Loading data ...")
    chunks, rejections = load_data()

    if not chunks:
        print("ERROR: No chunks found in %s" % CHUNKS_FILE)
        print("  Run the extraction script first.")
        sys.exit(1)

    print("  Chunks loaded    : %d" % len(chunks))
    print("  Rejections loaded: %d" % len(rejections))

    print("Building subject data ...")
    subjects_data = build_subject_data(chunks, rejections)

    for subj, d in subjects_data.items():
        if subj == "__overview__":
            continue
        print("  %-15s : %d chunks | %d sections | %d rejected" % (
            subj, d["stats"]["total_chunks"],
            d["stats"]["unique_sections"],
            d["stats"]["total_rejected"],
        ))

    print("Generating dashboard -> %s" % output_path)
    generate_html(subjects_data, output_path)

    size_kb = output_path.stat().st_size // 1024
    print("Done. Dashboard size: %d KB" % size_kb)

    if args.open:
        print("Opening in browser ...")
        if sys.platform == "darwin":
            subprocess.run(["open", str(output_path)])
        elif sys.platform == "linux":
            subprocess.run(["xdg-open", str(output_path)])
        else:
            os.startfile(str(output_path))


if __name__ == "__main__":
    main()
