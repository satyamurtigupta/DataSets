#!/usr/bin/env python3
"""
extract_ncert_marker.py
=======================
High-quality NCERT PDF extraction using pymupdf4llm.

pymupdf4llm uses PyMuPDF's layout analysis to output clean Markdown,
automatically separating text from figures, equations, and tables.
It produces proper paragraph boundaries and section headings — solving
the mid-sentence chunking problem of the old pdf_extractor.py.

Improvements over old extractor:
  - Semantic chunking (by heading, not character count)
  - Skips equation-heavy paragraphs automatically
  - Strips NCERT page markers, page numbers, figure labels
  - Proper heading hierarchy from document structure
  - Built-in quality checks (9 rules)
  - Subject-wise, resume, dry-run, force modes

Usage:
    python3 scripts/extract_ncert_marker.py
    python3 scripts/extract_ncert_marker.py --subject History
    python3 scripts/extract_ncert_marker.py --subject History Polity
    python3 scripts/extract_ncert_marker.py --resume
    python3 scripts/extract_ncert_marker.py --force --subject Physics
    python3 scripts/extract_ncert_marker.py --dry-run
    python3 scripts/extract_ncert_marker.py --quality-check
    python3 scripts/extract_ncert_marker.py --min-words 100 --max-words 1500
    python3 scripts/extract_ncert_marker.py --output custom_pretrain.jsonl

Requirements:
    pip install pymupdf4llm
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR      = Path(__file__).resolve().parent.parent
PDF_DIR       = BASE_DIR / "upsc_pdfs"
OUTPUT_DIR    = BASE_DIR / "dataset_output_final" / "combined"
OUTPUT_FILE   = OUTPUT_DIR / "unified_pretrain_v2.jsonl"
PROGRESS_FILE = OUTPUT_DIR / "pretrain_v2_progress.json"
AUDIT_FILE    = OUTPUT_DIR / "pretrain_v2_audit.jsonl"

# ---------------------------------------------------------------------------
# Quality thresholds
# ---------------------------------------------------------------------------

MIN_WORDS_DEFAULT = 80
MAX_WORDS_DEFAULT = 1200
ALPHA_RATIO_MIN   = 0.60
BIGRAM_REP_MAX    = 0.35
SINGLE_CHAR_MAX   = 0.15
COMMA_CLUSTER_MAX = 2

ALL_SUBJECTS = [
    "Art", "Biology", "Chemistry", "Economics",
    "Geography", "History", "Physics", "Polity",
    "Science", "Sociology",
]

# ---------------------------------------------------------------------------
# Markdown cleaning
# ---------------------------------------------------------------------------

# NCERT year markers: 2018-19, 2020-21
_NCERT_YEAR_RE    = re.compile(r"\b20\d\d[-–]\d\d\b")
# Lone page numbers on their own line
_PAGE_NUM_RE      = re.compile(r"^\s*\d{1,4}\s*$", re.MULTILINE)
# Markdown image tags
_IMAGE_TAG_RE     = re.compile(r"!\[.*?\]\(.*?\)", re.DOTALL)
# Large blank-line blocks (where figures appear in markdown output)
_BLANK_BLOCK_RE   = re.compile(r"(\n\s*){4,}")
# Markdown bold/italic that leaked into body
_MD_STYLE_RE      = re.compile(r"(\*{1,2}|_{1,2})([^\*_]+)\1")
# Heading pattern
_HEADING_RE       = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)
# Equation line heuristic: line has multiple math operators or LaTeX commands
_EQ_LINE_RE       = re.compile(r"[=+\-\*/\\^]{3,}|\\frac|\\sum|\\int|\$\$|\\\(")
# Lines that are just a number or roman numeral (table of contents stubs)
_NUM_ONLY_RE      = re.compile(r"^\s*[ivxIVX\d]{1,5}\.?\s*$")
# NCERT exercise markers
_EXERCISE_RE      = re.compile(
    r"^\s*(exercise|activity|exercises|intext question|think and discuss|let us recall"
    r"|check your progress|do you know|box \d|summary|objectives|key concepts)",
    re.IGNORECASE,
)


def clean_markdown(text: str) -> str:
    """Strip page artifacts from pymupdf4llm Markdown output."""
    # Remove NCERT year markers
    text = _NCERT_YEAR_RE.sub("", text)
    # Remove bare page numbers on their own line
    text = _PAGE_NUM_RE.sub("", text)
    # Remove image tags
    text = _IMAGE_TAG_RE.sub("", text)
    # Collapse large blank blocks (figure placeholders)
    text = _BLANK_BLOCK_RE.sub("\n\n", text)
    # Remove markdown link markup but keep link text
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove NCERT spaced-out running headers.
    # These appear as "C RAFT T RADITIONS OF I NDIA : P AST, P RESENT AND F UTURE"
    # — words where the first letter is separated from the rest by a space.
    # Pattern: 3+ occurrences of "[capital] [2+ capitals]" on the same line.
    def _remove_spaced_headers(m_text: str) -> str:
        cleaned_lines = []
        for line in m_text.splitlines():
            # Count "[A-Z] [A-Z]{2,}" patterns — spaced first-letter
            spaced = len(re.findall(r"\b[A-Z] [A-Z]{2,}", line))
            if spaced >= 2:
                continue   # drop this line — it's a spaced-out display header
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines)

    text = _remove_spaced_headers(text)

    # Remove lines that are chapter running headers ending with a page number:
    # "C OLONIAL R ULE AND C RAFTS 17"  or  "COLONIAL RULE AND CRAFTS 17"
    text = re.sub(
        r"^[A-Z][A-Z \t]{8,}\s+\d{1,3}\s*$",
        "",
        text,
        flags=re.MULTILINE,
    )
    # Collapse 3+ blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_equation_lines(text: str) -> str:
    """
    Remove lines that are clearly equation/formula lines.
    Keep lines that have equation notation but are mostly prose.
    """
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned.append(line)
            continue
        # Skip lines that are pure equation (> 40% non-alpha/space chars
        # AND match equation pattern)
        non_alpha = sum(1 for c in stripped if not c.isalpha() and not c.isspace())
        ratio = non_alpha / max(len(stripped), 1)
        if ratio > 0.45 and _EQ_LINE_RE.search(stripped):
            continue
        # Skip NUM_ONLY lines
        if _NUM_ONLY_RE.match(stripped):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


# ---------------------------------------------------------------------------
# Heading / section detection
# ---------------------------------------------------------------------------

_SKIP_HEADING_PATTERNS = [
    re.compile(r"^\d+[\.\d]*$"),                         # pure number
    re.compile(r"^\s*(exercise|activity|summary|objectives|let us|intext"
               r"|check your|think|do you know|box \d|figure|table"
               r"|table of contents|index|glossary|bibliography"
               r"|references|appendix)\b", re.I),
    re.compile(r"^(answer|solution)\s+\d", re.I),        # "Answer 3.1"
]


def _is_skip_heading(heading: str) -> bool:
    h = heading.strip()
    if len(h) <= 2:
        return True
    for pat in _SKIP_HEADING_PATTERNS:
        if pat.match(h):
            return True
    return False


# ---------------------------------------------------------------------------
# Fragment trimmer
# ---------------------------------------------------------------------------

def _trim_leading_fragments(text: str) -> str:
    """
    Remove leading sentence fragments caused by multi-column PDF layouts.

    pymupdf4llm reads text boxes in page order. When a PDF has floating
    text boxes or multi-column layouts, the end of a sentence (from one
    box) appears before the beginning of the next sentence (from another
    box). This produces leading fragments like:
        "with artisans and their crafts.\n\nof knowledge.\n\nFlexibility..."

    Strategy per paragraph:
    1. If paragraph starts with uppercase -> keep as-is
    2. If paragraph starts with lowercase AND contains a sentence boundary
       (. or ! or ?) followed by an uppercase letter -> trim to that point
    3. If paragraph is entirely lowercase fragments -> skip it entirely
    """
    paragraphs = re.split(r"\n\n+", text.strip())
    kept = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        first_char = para.lstrip()[0] if para.lstrip() else ""

        if first_char.isupper():
            # Good paragraph — starts clean
            kept.append(para)
            continue

        # Starts lowercase — try to find a sentence boundary inside
        # Pattern: '. ' or '! ' or '? ' followed by an uppercase letter
        m = re.search(r"[.!?]\s+([A-Z])", para)
        if m:
            # Trim to start from the found uppercase
            trimmed = para[m.start() + 2:].strip()
            if len(trimmed.split()) >= 10:
                kept.append(trimmed)
            # else: the trimmed result is too short — discard
        # else: pure fragment line (e.g. "of knowledge.") — discard entirely

    return "\n\n".join(kept)


# ---------------------------------------------------------------------------
# Semantic chunker
# ---------------------------------------------------------------------------

def parse_markdown_to_chunks(
    markdown: str,
    subject: str,
    source_file: str,
    id_prefix: str,
    min_words: int,
    max_words: int,
) -> list:
    """
    Parse Markdown into semantic chunks.
    Each chunk = one named section (heading + body prose).

    Strategy:
    1. Split on headings (#, ##, ###, ####)
    2. Skip exercise/activity/summary headings
    3. Merge sections that are too short into the next section
    4. Split sections that are too long at paragraph boundaries
    """
    cleaned  = clean_markdown(markdown)
    cleaned  = remove_equation_lines(cleaned)
    lines    = cleaned.splitlines()

    # ── Build list of (heading_level, heading_text, body_text) ──
    sections = []
    cur_heading = subject
    cur_level   = 1
    cur_body    = []

    for line in lines:
        m = re.match(r"^(#{1,4})\s+(.+)$", line.strip())
        if m:
            level   = len(m.group(1))
            heading = m.group(2).strip()

            if _is_skip_heading(heading):
                # Treat as body text
                cur_body.append(heading)
                continue

            body = _trim_leading_fragments("\n".join(cur_body))
            if body:
                sections.append((cur_level, cur_heading, body))

            cur_heading = heading
            cur_level   = level
            cur_body    = []
        else:
            cur_body.append(line)

    # Flush last section
    if cur_body:
        body = _trim_leading_fragments("\n".join(cur_body))
        if body:
            sections.append((cur_level, cur_heading, body))

    # ── Merge short sections and split long ones ──
    chunks    = []
    chunk_idx = 0
    buf_heading = ""
    buf_text    = ""

    def _make_chunk(heading: str, text: str) -> Optional[dict]:
        text = text.strip()
        wc   = len(text.split())
        if wc < 20:
            return None
        chunk_id = f"{id_prefix}_{chunk_idx:04d}"
        return {
            "id":              chunk_id,
            "text":            text,
            "subject":         subject,
            "source_file":     source_file,
            "section_heading": heading,
            "word_count":      wc,
            "char_count":      len(text),
        }

    def _flush_buf(heading: str, text: str) -> list:
        nonlocal chunk_idx
        results = []
        text = text.strip()
        if not text:
            return results

        wc = len(text.split())

        if wc <= max_words:
            c = _make_chunk(heading, text)
            if c:
                results.append(c)
                chunk_idx += 1
        else:
            # Split at paragraph boundaries
            paras   = text.split("\n\n")
            half    = []
            running = 0
            part    = 1

            for para in paras:
                pw = len(para.split())
                if running + pw > max_words and half:
                    c = _make_chunk(
                        f"{heading} (part {part})",
                        "\n\n".join(half)
                    )
                    if c:
                        results.append(c)
                        chunk_idx += 1
                    half    = [para]
                    running = pw
                    part   += 1
                else:
                    half.append(para)
                    running += pw

            if half:
                c = _make_chunk(
                    f"{heading} (part {part})" if part > 1 else heading,
                    "\n\n".join(half)
                )
                if c:
                    results.append(c)
                    chunk_idx += 1

        return results

    for level, heading, body in sections:
        body_wc = len(body.split())

        if not buf_text:
            buf_heading = heading
            buf_text    = body
        elif body_wc >= min_words:
            # Current buffer is ready — flush and start new
            chunks.extend(_flush_buf(buf_heading, buf_text))
            buf_heading = heading
            buf_text    = body
        else:
            # Short section — merge into buffer
            buf_text = buf_text + "\n\n" + heading + "\n" + body

    # Flush final buffer
    if buf_text:
        chunks.extend(_flush_buf(buf_heading, buf_text))

    return chunks


# ---------------------------------------------------------------------------
# Quality checks
# ---------------------------------------------------------------------------

def _alpha_ratio(text: str) -> float:
    chars = [c for c in text if c.isalpha() or c.isspace()]
    return len(chars) / max(len(text), 1)


def _bigram_rep(text: str) -> float:
    words   = text.lower().split()
    if len(words) < 4:
        return 0.0
    bigrams = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
    return 1.0 - len(set(bigrams)) / max(len(bigrams), 1)


def _single_char_density(text: str) -> float:
    tokens  = text.split()
    if not tokens:
        return 0.0
    singles = sum(1 for t in tokens if len(re.sub(r"[^a-zA-Z]", "", t)) == 1)
    return singles / len(tokens)


def _comma_clusters(text: str) -> int:
    return len(re.findall(r",{3,}", text))


def _chunk_sig(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text[:300].lower()).strip()
    return hashlib.md5(normalized.encode()).hexdigest()


def quality_check(chunk: dict, seen_sigs: set, min_words: int, max_words: int) -> dict:
    """
    Run 9 quality checks on a chunk.
    Returns: {"passed": bool, "reasons": [str], "metrics": {str: float}}
    """
    text    = chunk.get("text", "")
    wc      = chunk.get("word_count", 0)
    reasons = []
    metrics: dict = {}

    # QC-1  Word count
    if wc < min_words:
        reasons.append(f"QC1_TOO_SHORT({wc}w)")
    if wc > max_words:
        reasons.append(f"QC1_TOO_LONG({wc}w)")

    # QC-2  Alpha ratio
    ar = _alpha_ratio(text)
    metrics["alpha_ratio"] = round(ar, 3)
    if ar < ALPHA_RATIO_MIN:
        reasons.append(f"QC2_LOW_ALPHA({ar:.2f})")

    # QC-3  Bigram repetition
    br = _bigram_rep(text)
    metrics["bigram_rep"] = round(br, 3)
    if br > BIGRAM_REP_MAX:
        reasons.append(f"QC3_REPETITION({br:.2f})")

    # QC-4  Single-char density  (diagram / figure OCR remnants)
    sc = _single_char_density(text)
    metrics["single_char"] = round(sc, 3)
    if sc > SINGLE_CHAR_MAX:
        reasons.append(f"QC4_DIAGRAM_OCR({sc:.2f})")

    # QC-5  Comma clusters  (table / dotted-line OCR)
    cc = _comma_clusters(text)
    metrics["comma_clusters"] = cc
    if cc >= COMMA_CLUSTER_MAX:
        reasons.append(f"QC5_COMMA_CLUSTER({cc})")

    # QC-6  Mid-sentence start
    first = text.lstrip()
    if first and first[0].islower():
        reasons.append("QC6_MID_SENTENCE")

    # QC-7  Equation-heavy paragraph
    lines    = [l.strip() for l in text.splitlines() if l.strip()]
    eq_lines = sum(1 for l in lines if _EQ_LINE_RE.search(l))
    eq_ratio = eq_lines / max(len(lines), 1)
    metrics["eq_ratio"] = round(eq_ratio, 3)
    if eq_ratio > 0.30:
        reasons.append(f"QC7_EQUATION_HEAVY({eq_ratio:.2f})")

    # QC-8  NCERT exercise block leaked in
    if _EXERCISE_RE.match(text.strip()):
        reasons.append("QC8_EXERCISE_BLOCK")

    # QC-9  Near-duplicate
    sig = _chunk_sig(text)
    metrics["sig"] = sig
    if sig in seen_sigs:
        reasons.append("QC9_DUPLICATE")
    else:
        seen_sigs.add(sig)

    return {
        "passed":  len(reasons) == 0,
        "reasons": reasons,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

def load_progress() -> set:
    if PROGRESS_FILE.exists():
        return set(json.loads(PROGRESS_FILE.read_text()))
    return set()


def save_progress(done: set):
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(json.dumps(sorted(done), indent=2))


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_pdf(pdf_path: Path) -> Optional[str]:
    """Run pymupdf4llm on a single PDF. Returns Markdown or None on failure."""
    try:
        import pymupdf4llm
        return pymupdf4llm.to_markdown(str(pdf_path))
    except Exception as e:
        print(f"    ERROR: {e}")
        return None


def get_subject_pdfs(subject_dir: Path) -> list:
    return sorted(subject_dir.glob("**/*.pdf"))


# ---------------------------------------------------------------------------
# Per-PDF processing
# ---------------------------------------------------------------------------

def process_pdf(
    pdf_path: Path,
    subject: str,
    out_f,
    audit_f,
    seen_sigs: set,
    done_ids: set,
    args,
    stats: dict,
) -> int:
    pdf_id = f"{subject}::{pdf_path.name}"

    if args.resume and not args.force and pdf_id in done_ids:
        print(f"  SKIP : {pdf_path.name}")
        stats["pdfs_skipped"] += 1
        return 0

    print(f"  [{pdf_path.name}] extracting ...", end="", flush=True)
    t0 = time.time()

    if args.dry_run:
        wc_est = pdf_path.stat().st_size // 1500   # rough estimate
        print(f"  DRY RUN (~{wc_est} chunks estimated)")
        return 0

    markdown = extract_pdf(pdf_path)
    elapsed  = time.time() - t0

    if not markdown or len(markdown.strip()) < 200:
        print(f" FAILED (empty in {elapsed:.0f}s)")
        stats["pdfs_failed"] += 1
        done_ids.add(pdf_id)
        save_progress(done_ids)
        return 0

    safe  = re.sub(r"[^a-z0-9]", "_", pdf_path.stem.lower())[:40]
    prefix = f"{subject.lower()}_{safe}"

    raw_chunks = parse_markdown_to_chunks(
        markdown,
        subject     = subject,
        source_file = pdf_path.name,
        id_prefix   = prefix,
        min_words   = args.min_words,
        max_words   = args.max_words,
    )

    kept = 0
    rej  = 0

    for chunk in raw_chunks:
        qc = quality_check(chunk, seen_sigs, args.min_words, args.max_words)
        stats["total_raw"] += 1

        audit_rec = {
            "id":       chunk["id"],
            "subject":  subject,
            "source":   pdf_path.name,
            "passed":   qc["passed"],
            "reasons":  qc["reasons"],
            "metrics":  qc["metrics"],
            "wc":       chunk.get("word_count", 0),
        }

        if qc["passed"]:
            out_f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            kept += 1
            stats["total_kept"] += 1
        else:
            rej += 1
            stats["total_rejected"] += 1
            for r in qc["reasons"]:
                key = r.split("(")[0]
                stats["rejection_reasons"][key] = \
                    stats["rejection_reasons"].get(key, 0) + 1

        if audit_f:
            audit_f.write(json.dumps(audit_rec, ensure_ascii=False) + "\n")

    done_ids.add(pdf_id)
    save_progress(done_ids)
    stats["pdfs_done"] += 1

    print(f" {kept} kept / {rej} rejected  ({elapsed:.0f}s)")
    return kept


# ---------------------------------------------------------------------------
# Quality-check-only mode
# ---------------------------------------------------------------------------

def run_quality_check_only(path: Path, min_words: int, max_words: int):
    print(f"Quality checking: {path}")
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Records         : {len(records):,}")
    seen_sigs   = set()
    pass_count  = 0
    fail_count  = 0
    reason_ctr: dict = {}

    for r in records:
        qc = quality_check(r, seen_sigs, min_words, max_words)
        if qc["passed"]:
            pass_count += 1
        else:
            fail_count += 1
            for reason in qc["reasons"]:
                key = reason.split("(")[0]
                reason_ctr[key] = reason_ctr.get(key, 0) + 1

    total = len(records)
    print()
    print("=" * 60)
    print("  QUALITY CHECK REPORT")
    print("=" * 60)
    print(f"  Passed : {pass_count:>6,}  ({100*pass_count/total:.1f}%)")
    print(f"  Failed : {fail_count:>6,}  ({100*fail_count/total:.1f}%)")
    if reason_ctr:
        print()
        print("  By check:")
        for k, v in sorted(reason_ctr.items(), key=lambda x: -x[1]):
            print(f"    {k:<35} {v:>5,}  ({100*v/total:.1f}%)")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------------

def print_report(stats: dict):
    total_raw  = stats["total_raw"]
    total_kept = stats["total_kept"]
    total_rej  = stats["total_rejected"]

    print()
    print("=" * 65)
    print("  EXTRACTION COMPLETE")
    print("=" * 65)
    print(f"  PDFs processed   : {stats['pdfs_done']:>6,}")
    print(f"  PDFs skipped     : {stats['pdfs_skipped']:>6,}")
    print(f"  PDFs failed      : {stats['pdfs_failed']:>6,}")
    print()
    print(f"  Raw chunks       : {total_raw:>6,}")
    print(f"  Kept (passed QC) : {total_kept:>6,}  ({100*total_kept/max(total_raw,1):.1f}%)")
    print(f"  Rejected         : {total_rej:>6,}  ({100*total_rej/max(total_raw,1):.1f}%)")

    if stats["rejection_reasons"]:
        print()
        print("  Rejection reasons:")
        for k, v in sorted(stats["rejection_reasons"].items(), key=lambda x: -x[1]):
            print(f"    {k:<35} {v:>5,}")

    print()
    print(f"  Output           : {OUTPUT_FILE}")
    print("=" * 65)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Extract NCERT PDFs with pymupdf4llm into high-quality training chunks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 scripts/extract_ncert_marker.py
  python3 scripts/extract_ncert_marker.py --subject History Polity
  python3 scripts/extract_ncert_marker.py --resume
  python3 scripts/extract_ncert_marker.py --force --subject Physics
  python3 scripts/extract_ncert_marker.py --dry-run
  python3 scripts/extract_ncert_marker.py --quality-check
        """,
    )

    ap.add_argument(
        "--subject", nargs="+", default=None, metavar="NAME",
        help=f"Subjects to process. Choices: {', '.join(ALL_SUBJECTS)}. Default: all.",
    )
    ap.add_argument(
        "--resume", action="store_true",
        help="Skip PDFs already in progress file. Appends to output.",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="Re-process even if in progress file (use with --subject to re-do one subject).",
    )
    ap.add_argument(
        "--output", default=str(OUTPUT_FILE), metavar="PATH",
        help=f"Output JSONL file. Default: {OUTPUT_FILE}",
    )
    ap.add_argument(
        "--min-words", type=int, default=MIN_WORDS_DEFAULT,
        help=f"Min words per chunk (default: {MIN_WORDS_DEFAULT}).",
    )
    ap.add_argument(
        "--max-words", type=int, default=MAX_WORDS_DEFAULT,
        help=f"Max words per chunk (default: {MAX_WORDS_DEFAULT}).",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="List PDFs without extracting.",
    )
    ap.add_argument(
        "--quality-check", action="store_true",
        help="Run quality checks on existing output file only.",
    )
    ap.add_argument(
        "--quality-check-file", default=None, metavar="PATH",
        help="JSONL file to quality-check. Default: the output file.",
    )
    ap.add_argument(
        "--no-audit", action="store_true",
        help="Skip writing per-chunk audit log.",
    )
    ap.add_argument(
        "--pdf-dir", default=str(PDF_DIR), metavar="DIR",
        help=f"Root PDF directory. Default: {PDF_DIR}",
    )

    args = ap.parse_args()

    # ── Quality check only mode ──
    if args.quality_check:
        qc_path = Path(args.quality_check_file or args.output)
        if not qc_path.exists():
            print(f"ERROR: File not found: {qc_path}")
            sys.exit(1)
        run_quality_check_only(qc_path, args.min_words, args.max_words)
        return

    # ── Validate pymupdf4llm ──
    try:
        import pymupdf4llm  # noqa: F401
    except ImportError:
        print("ERROR: pymupdf4llm not installed.")
        print("Run: pip install pymupdf4llm")
        sys.exit(1)

    pdf_dir     = Path(args.pdf_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Resolve subjects ──
    if args.subject:
        subjects = args.subject
        invalid  = [s for s in subjects if s not in ALL_SUBJECTS]
        if invalid:
            print(f"ERROR: Unknown subject(s): {invalid}")
            print(f"Valid: {', '.join(ALL_SUBJECTS)}")
            sys.exit(1)
    else:
        subjects = ALL_SUBJECTS

    # ── Load progress ──
    done_ids: set = set()
    if args.resume:
        done_ids = load_progress()
        print(f"Resume: {len(done_ids)} PDFs already processed.")

    write_mode = "a" if args.resume else "w"
    seen_sigs:  set = set()

    stats: dict = {
        "pdfs_done":         0,
        "pdfs_skipped":      0,
        "pdfs_failed":       0,
        "total_raw":         0,
        "total_kept":        0,
        "total_rejected":    0,
        "rejection_reasons": {},
    }

    print()
    print("=" * 65)
    print("  NCERT PDF EXTRACTION  (pymupdf4llm)")
    print("=" * 65)
    print(f"  Subjects  : {', '.join(subjects)}")
    print(f"  PDF dir   : {pdf_dir}")
    print(f"  Output    : {output_path}")
    print(f"  Min words : {args.min_words}")
    print(f"  Max words : {args.max_words}")
    print(f"  Resume    : {args.resume}")
    print(f"  Dry run   : {args.dry_run}")
    print("=" * 65)
    print()

    with open(output_path, write_mode, encoding="utf-8") as out_f:
        audit_f = None
        if not args.no_audit and not args.dry_run:
            audit_path = output_path.parent / (output_path.stem + "_audit.jsonl")
            audit_f = open(audit_path, write_mode, encoding="utf-8")

        for subject in subjects:
            subject_dir = pdf_dir / subject
            if not subject_dir.exists():
                print(f"[{subject}] Not found at {subject_dir} — skipping")
                continue

            pdfs = get_subject_pdfs(subject_dir)
            print(f"[{subject}] {len(pdfs)} PDF(s)")

            if args.force:
                to_remove = {d for d in done_ids if d.startswith(f"{subject}::")}
                if to_remove:
                    done_ids -= to_remove
                    print(f"  Force: cleared {len(to_remove)} cached PDF(s) for {subject}")

            subject_total = 0
            for pdf_path in pdfs:
                n = process_pdf(
                    pdf_path  = pdf_path,
                    subject   = subject,
                    out_f     = out_f,
                    audit_f   = audit_f,
                    seen_sigs = seen_sigs,
                    done_ids  = done_ids,
                    args      = args,
                    stats     = stats,
                )
                subject_total += n
                # Flush after each PDF so partial results are usable
                out_f.flush()

            print(f"  [{subject}] total chunks written: {subject_total}\n")

        if audit_f:
            audit_f.close()

    if not args.dry_run:
        print_report(stats)


if __name__ == "__main__":
    main()
