#!/usr/bin/env python3
"""
extract_ncert_gemini.py
=======================
NCERT PDF extraction using Gemini File API (Approach C).

Uploads each PDF natively to Gemini — no page splitting, no OCR heuristics.
Gemini reads layout, columns, and formatting directly and returns clean
prose chunks in structured JSON, skipping exercises, activities, and captions.

Usage:
    python3 scripts/extract_ncert_gemini.py
    python3 scripts/extract_ncert_gemini.py --subject History
    python3 scripts/extract_ncert_gemini.py --subject History Polity
    python3 scripts/extract_ncert_gemini.py --resume
    python3 scripts/extract_ncert_gemini.py --dry-run
    python3 scripts/extract_ncert_gemini.py --model gemini-1.5-flash
    python3 scripts/extract_ncert_gemini.py --quality-check

Requirements:
    pip install google-generativeai
    GEMINI_API_KEY set in env/.env
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR      = Path(__file__).resolve().parent.parent
PDF_DIR       = BASE_DIR / "upsc_pdfs"
ENV_FILE      = BASE_DIR / "env" / ".env"
OUTPUT_DIR    = BASE_DIR / "dataset_output_final" / "combined"
OUTPUT_FILE      = OUTPUT_DIR / "unified_pretrain_v2.jsonl"
PROGRESS_FILE    = OUTPUT_DIR / "pretrain_v2_gemini_progress.json"
REJECTION_LOG    = OUTPUT_DIR / "pretrain_v2_rejection_log.jsonl"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_MODEL    = "gemini-2.5-flash"
MIN_WORDS        = 80
MAX_WORDS        = 500
RATE_LIMIT_WAIT  = 5      # seconds between API calls (free tier: 15 RPM)
MAX_RETRIES      = 3
RETRY_WAIT       = 20     # seconds after rate-limit / transient error
MAX_OUTPUT_TOKENS = 32768  # per batch call
PAGES_PER_BATCH  = 40     # split large PDFs into batches of this many pages

ALL_SUBJECTS = [
    "Art", "Biology", "Chemistry", "Economics",
    "Geography", "History", "Physics", "Polity",
    "Science", "Sociology",
]

# PDFs confirmed as non-educational (YouTube transcripts, answer keys, etc.)
# These are skipped automatically regardless of --subject or --force.
SKIP_FILES = {
    "[Hindi (auto-generated)] How Did They Track Khamenei [DownSub.com].pdf",
}

# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """You are extracting training data from an NCERT textbook PDF for an AI language model.

YOUR TASK:
Read the entire PDF and extract clean educational prose. Output each chunk as one JSON object per line (JSONL format).

WHAT TO EXTRACT (keep):
- Explanatory paragraphs: definitions, historical narrative, scientific explanations, concepts
- Descriptive sections about people, events, places, processes, phenomena
- Any text that teaches something and reads as complete, flowing prose

WHAT TO SKIP COMPLETELY (do not include):
- Exercise sections, questions at end of chapters
- Activity boxes, "Do This", "Think and Discuss", "Let Us Recall"
- "Intext Questions", "Check Your Progress", "Box" sidebars
- Figure captions (e.g. "Fig. 3.1 — Diagram showing...")
- Table of contents, index, bibliography, glossary
- Page numbers, running headers/footers
- Chapter number lines (e.g. "Chapter 4")
- Any incomplete sentence or sentence fragment

CHUNKING RULES:
- Each chunk must be 120 to 450 words — complete, self-contained prose
- Do NOT cut a sentence in the middle — always end at a full stop
- Each chunk must start with an uppercase letter and a complete sentence
- If a section is very long, split it into multiple consecutive chunks
- Attach the nearest chapter/section heading as section_heading
- If no heading is nearby, use the subject area as section_heading

OUTPUT FORMAT — CRITICAL:
- Output ONE JSON object per line. No JSON array. No markdown code fences. No explanation.
- Each line must be a complete, valid JSON object with exactly two fields.
- All double quotes inside text values must be escaped as \"
- No newlines inside text values — use a space instead
- The text field must be on the same line as section_heading

Correct format (one object per line):
{"section_heading": "The Mughal Empire", "text": "The Mughal Empire was one of the largest empires in Indian history. It was founded by Babur in 1526 after the First Battle of Panipat."}
{"section_heading": "The Mughal Empire", "text": "Akbar, who ruled from 1556 to 1605, is often regarded as the greatest of the Mughal emperors. He expanded the empire significantly."}

Now extract all qualifying prose from this PDF and output one JSON object per line."""

# ---------------------------------------------------------------------------
# Env loader
# ---------------------------------------------------------------------------

def load_env() -> dict:
    env = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env

# ---------------------------------------------------------------------------
# Progress tracker
# ---------------------------------------------------------------------------

def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        try:
            return json.loads(PROGRESS_FILE.read_text())
        except Exception:
            return {}
    return {}

def save_progress(progress: dict) -> None:
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))

# ---------------------------------------------------------------------------
# JSON parser — robust against LLM wrapping output in markdown fences
# ---------------------------------------------------------------------------

def parse_json_response(raw: str) -> Optional[List[dict]]:
    """
    Parse JSONL or JSON array from LLM response.

    Strategy (most robust to least):
    1. JSONL  — parse each line independently as a JSON object
    2. JSON array — find first [ ... last ] and parse as array
    3. Partial — collect any valid {..} objects found anywhere in the text
    """
    results = []

    # ── Strategy 1: JSONL (one object per line) ──
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        # Skip markdown fences and prose lines
        if line.startswith("```") or not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict) and "text" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            pass

    if results:
        return results

    # ── Strategy 2: JSON array ──
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    text = text.strip()

    start = text.find("[")
    end   = text.rfind("]")
    if start != -1 and end != -1:
        try:
            data = json.loads(text[start:end + 1])
            if isinstance(data, list):
                return [d for d in data if isinstance(d, dict) and "text" in d]
        except json.JSONDecodeError:
            pass

    # ── Strategy 3: find any {...} blocks with both required fields ──
    for m in re.finditer(r'\{[^{}]{20,}\}', text, re.DOTALL):
        try:
            obj = json.loads(m.group())
            if isinstance(obj, dict) and "text" in obj and "section_heading" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            pass

    return results if results else None

# ---------------------------------------------------------------------------
# Quality filter
# ---------------------------------------------------------------------------

def passes_quality(chunk: dict, seen_sigs: set) -> tuple:
    """
    Basic quality checks on a single chunk dict.
    Returns (passed: bool, reason: str).
    """
    text = chunk.get("text", "").strip()
    words = text.split()
    wc = len(words)

    if wc < MIN_WORDS:
        return False, "TOO_SHORT"
    if wc > MAX_WORDS * 1.5:
        return False, "TOO_LONG"

    # Alpha ratio
    alpha = sum(1 for c in text if c.isalpha())
    if alpha / max(len(text), 1) < 0.55:
        return False, "LOW_ALPHA"

    # Must start with uppercase
    first = text.lstrip()[0] if text.lstrip() else ""
    if first.islower():
        return False, "MID_SENTENCE"

    # Near-duplicate check
    sig = hashlib.md5(text[:300].encode()).hexdigest()
    if sig in seen_sigs:
        return False, "DUPLICATE"
    seen_sigs.add(sig)

    # Exercise block leaked in
    exercise_re = re.compile(
        r"^\s*(exercise|activity|exercises|intext question|think and discuss"
        r"|let us recall|check your progress|do you know|answer the following"
        r"|fill in the blank|match the following)",
        re.IGNORECASE,
    )
    if exercise_re.search(text[:120]):
        return False, "EXERCISE_BLOCK"

    return True, "OK"

# ---------------------------------------------------------------------------
# PDF page splitter — creates temp PDFs for batched processing
# ---------------------------------------------------------------------------

def split_pdf_into_batches(pdf_path: Path, pages_per_batch: int) -> List[Path]:
    """
    Split a PDF into temporary sub-PDFs of pages_per_batch pages each.
    Returns list of temp file paths. Caller must delete them when done.
    Requires PyMuPDF (fitz) — installed as part of pymupdf4llm.
    Returns [pdf_path] unchanged if fitz not available or PDF is small enough.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return [pdf_path]

    doc = fitz.open(str(pdf_path))
    total_pages = doc.page_count

    if total_pages <= pages_per_batch:
        doc.close()
        return [pdf_path]

    import tempfile
    batches = []
    start = 0
    batch_num = 0
    while start < total_pages:
        end = min(start + pages_per_batch, total_pages)
        sub = fitz.open()
        sub.insert_pdf(doc, from_page=start, to_page=end - 1)
        tmp = tempfile.NamedTemporaryFile(
            suffix=f"_batch{batch_num}.pdf",
            delete=False,
            prefix=f"{pdf_path.stem}_",
        )
        sub.save(tmp.name)
        sub.close()
        batches.append(Path(tmp.name))
        start = end
        batch_num += 1

    doc.close()
    return batches


# ---------------------------------------------------------------------------
# Upload one file to Gemini and wait until ACTIVE
# ---------------------------------------------------------------------------

def upload_and_wait(client, pdf_path: Path) -> tuple:
    """
    Upload a file to Gemini File API and wait for it to become ACTIVE.
    Returns (uploaded_file, error_str_or_None).
    """
    from google.genai import types as gtypes

    try:
        with open(str(pdf_path), "rb") as fh:
            uploaded = client.files.upload(
                file=fh,
                config=gtypes.UploadFileConfig(mime_type="application/pdf"),
            )
    except Exception as e:
        return None, "upload failed: %s" % e

    wait_total = 0
    while uploaded.state == gtypes.FileState.PROCESSING:
        time.sleep(3)
        wait_total += 3
        if wait_total > 180:
            return None, "file processing timed out after 180s"
        try:
            uploaded = client.files.get(name=uploaded.name)
        except Exception:
            pass

    if uploaded.state != gtypes.FileState.ACTIVE:
        return None, "file in unexpected state: %s" % uploaded.state

    return uploaded, None


# ---------------------------------------------------------------------------
# Call Gemini on one uploaded file and return raw text
# ---------------------------------------------------------------------------

def call_gemini_on_file(client, model_name: str, uploaded_file) -> tuple:
    """
    Run extraction prompt on an already-uploaded Gemini file.
    Returns (raw_text, error_str_or_None).
    """
    from google.genai import types as gtypes

    raw_response = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[uploaded_file, EXTRACTION_PROMPT],
                config=gtypes.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                ),
            )
            raw_response = response.text
            break
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower():
                print("      rate limit — waiting %ds (attempt %d/%d)" % (RETRY_WAIT, attempt, MAX_RETRIES))
                time.sleep(RETRY_WAIT)
            elif attempt < MAX_RETRIES:
                print("      error: %s — retrying in 10s" % e)
                time.sleep(10)
            else:
                return None, "generation failed after %d attempts: %s" % (MAX_RETRIES, e)

    if not raw_response:
        return None, "empty response from model"

    return raw_response, None


# ---------------------------------------------------------------------------
# Build and quality-filter chunks from parsed JSON items
# ---------------------------------------------------------------------------

def build_chunks(
    items: List[dict],
    subject: str,
    source_file: str,
    id_prefix: str,
    chunk_idx_start: int,
    seen_sigs: set,
    rejection_log_fh,
    batch_num: int,
    page_start: int,
    page_end: int,
) -> tuple:
    """
    Convert raw parsed items into filtered chunk dicts.
    Writes rejected chunks with full detail to rejection_log_fh (JSONL).
    Returns (kept: list, reject_tally: dict, next_chunk_idx: int).
    """
    kept         = []
    reject_tally = {}   # reason -> count
    chunk_idx    = chunk_idx_start
    timestamp    = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    for item in items:
        if not isinstance(item, dict):
            continue
        text    = str(item.get("text", "")).strip()
        heading = str(item.get("section_heading", subject)).strip()

        if not text:
            continue

        wc = len(text.split())
        chunk = {
            "id":              "%s_%04d" % (id_prefix, chunk_idx),
            "text":            text,
            "subject":         subject,
            "source_file":     source_file,
            "section_heading": heading,
            "word_count":      wc,
            "char_count":      len(text),
        }

        passed, reason = passes_quality(chunk, seen_sigs)
        if passed:
            kept.append(chunk)
            chunk_idx += 1
        else:
            reject_tally[reason] = reject_tally.get(reason, 0) + 1

            # Write full rejection record to log
            if rejection_log_fh:
                log_entry = {
                    "timestamp":       timestamp,
                    "subject":         subject,
                    "source_file":     source_file,
                    "batch_num":       batch_num,
                    "page_range":      "%d-%d" % (page_start, page_end),
                    "section_heading": heading,
                    "rejection_reason": reason,
                    "word_count":      wc,
                    "char_count":      len(text),
                    "text_preview":    text[:300],   # first 300 chars for inspection
                    "text_full":       text,          # full text for deep analysis
                }
                rejection_log_fh.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    return kept, reject_tally, chunk_idx


# ---------------------------------------------------------------------------
# PDF processor — batches large PDFs, merges results
# ---------------------------------------------------------------------------

def process_pdf(
    pdf_path: Path,
    subject: str,
    client,
    model_name: str,
    id_prefix: str,
    seen_sigs: set,
    rejection_log_fh,
) -> tuple:
    """
    Process a single PDF through Gemini, splitting into page batches if large.
    Returns (chunks_kept: list, reject_tally: dict, error: str or None)
    """
    # ── Get total page count for page-range labels ──
    try:
        import fitz
        _doc = fitz.open(str(pdf_path))
        total_pages = _doc.page_count
        _doc.close()
    except Exception:
        total_pages = n_batches * PAGES_PER_BATCH  # fallback estimate

    # ── Split into batches if PDF is large ──
    batches   = split_pdf_into_batches(pdf_path, PAGES_PER_BATCH)
    is_split  = len(batches) > 1
    n_batches = len(batches)

    if is_split:
        print("    %d batches of %d pages each" % (n_batches, PAGES_PER_BATCH))

    all_kept         = []
    all_reject_tally = {}   # reason -> count across all batches
    chunk_idx        = 0
    temp_files       = [b for b in batches if b != pdf_path]  # track temps to delete

    for batch_num, batch_path in enumerate(batches, start=1):
        # Calculate which pages this batch covers (1-indexed for human readability)
        page_start = (batch_num - 1) * PAGES_PER_BATCH + 1
        page_end   = min(batch_num * PAGES_PER_BATCH, total_pages)

        label = "batch %d/%d (pages %d-%d)" % (batch_num, n_batches, page_start, page_end) \
                if is_split else pdf_path.name
        print("    uploading %s ..." % label, end=" ", flush=True)

        uploaded, err = upload_and_wait(client, batch_path)
        if err:
            print("ERROR: %s" % err)
            continue

        print("ok", flush=True)

        raw, err = call_gemini_on_file(client, model_name, uploaded)

        # Always delete uploaded file from Gemini to avoid storage buildup
        try:
            client.files.delete(name=uploaded.name)
        except Exception:
            pass

        if err:
            print("    ERROR: %s" % err)
            continue

        parsed = parse_json_response(raw)
        if not parsed:
            snippet = raw[:200].replace("\n", " ")
            print("    WARNING: JSON parse failed for %s. Snippet: %s" % (label, snippet))
            continue

        kept, reject_tally, chunk_idx = build_chunks(
            parsed, subject, pdf_path.name, id_prefix, chunk_idx, seen_sigs,
            rejection_log_fh, batch_num, page_start, page_end,
        )
        all_kept.extend(kept)
        for reason, count in reject_tally.items():
            all_reject_tally[reason] = all_reject_tally.get(reason, 0) + count

        total_rejected_batch = sum(reject_tally.values())
        print("    %s — kept %d / rejected %d" % (label, len(kept), total_rejected_batch))

        # Rate limit pause between batch calls
        if batch_num < n_batches:
            time.sleep(RATE_LIMIT_WAIT)

    # ── Delete temp batch files ──
    for tmp in temp_files:
        try:
            tmp.unlink()
        except Exception:
            pass

    total_rejected = sum(all_reject_tally.values())

    if not all_kept and total_rejected == 0:
        return [], {}, "no chunks produced — all batches may have failed"

    # Print rejection breakdown if any rejections occurred
    if all_reject_tally:
        print("    Rejection breakdown:")
        for reason, count in sorted(all_reject_tally.items(), key=lambda x: -x[1]):
            print("      %-20s : %d" % (reason, count))

    return all_kept, all_reject_tally, None

# ---------------------------------------------------------------------------
# Quality check mode
# ---------------------------------------------------------------------------

def run_quality_check(output_file: Path) -> None:
    if not output_file.exists():
        print(f"ERROR: file not found: {output_file}")
        sys.exit(1)

    chunks = []
    with open(output_file) as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    total = len(chunks)
    if total == 0:
        print("File is empty.")
        return

    word_counts = [c.get("word_count", 0) for c in chunks]
    subjects    = {}
    for c in chunks:
        s = c.get("subject", "unknown")
        subjects[s] = subjects.get(s, 0) + 1

    avg_wc  = sum(word_counts) / total
    min_wc  = min(word_counts)
    max_wc  = max(word_counts)
    in_range = sum(1 for w in word_counts if MIN_WORDS <= w <= MAX_WORDS)

    print(f"\nQuality Report — {output_file.name}")
    print(f"  Total chunks   : {total:,}")
    print(f"  Avg word count : {avg_wc:.0f}")
    print(f"  Min / Max      : {min_wc} / {max_wc}")
    print(f"  In range       : {in_range:,} ({100*in_range/total:.1f}%)")
    print(f"\n  Chunks per subject:")
    for s, n in sorted(subjects.items()):
        print(f"    {s:<15} {n:>5}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="NCERT PDF extraction via Gemini File API")
    ap.add_argument("--subject",       nargs="+", help="Process only these subjects")
    ap.add_argument("--file",          help="Process a single PDF by filename (e.g. 'History6.pdf'). Use with --subject to locate it.")
    ap.add_argument("--preview",       action="store_true", help="Print extracted chunks to terminal instead of writing to file (use with --file for manual review)")
    ap.add_argument("--resume",        action="store_true", help="Skip already-processed PDFs")
    ap.add_argument("--force",         action="store_true", help="Re-process even if in progress file")
    ap.add_argument("--dry-run",       action="store_true", help="List PDFs without extracting")
    ap.add_argument("--model",         default=DEFAULT_MODEL, help="Gemini model (default: %s)" % DEFAULT_MODEL)
    ap.add_argument("--output",        default=str(OUTPUT_FILE), help="Output JSONL path")
    ap.add_argument("--quality-check", action="store_true", help="Run quality report on existing output")
    ap.add_argument("--pdf-dir",       default=str(PDF_DIR), help="Root PDF directory")
    args = ap.parse_args()

    output_path = Path(args.output)

    if args.quality_check:
        run_quality_check(output_path)
        return

    # ── Load env ──
    env = load_env()
    api_key = env.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("ERROR: GEMINI_API_KEY is not set in env/.env")
        print("  Get a key at https://aistudio.google.com and add:")
        print("  GEMINI_API_KEY=your_key_here")
        sys.exit(1)

    # ── Init Gemini (new google-genai SDK) ──
    try:
        from google import genai as google_genai
        client = google_genai.Client(api_key=api_key)
    except ImportError:
        print("ERROR: google-genai not installed.")
        print("  Run: python3 -m pip install google-genai")
        sys.exit(1)

    # ── Subject list ──
    subjects = args.subject if args.subject else ALL_SUBJECTS
    pdf_root = Path(args.pdf_dir)

    # ── Output dir ──
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Progress ──
    progress = load_progress() if args.resume else {}
    if args.force:
        progress = {}

    # ── Collect PDFs ──
    pdf_list = []
    for subj in subjects:
        subj_dir = pdf_root / subj
        if not subj_dir.exists():
            print("WARNING: directory not found — %s" % subj_dir)
            continue
        pdfs = sorted(subj_dir.glob("*.pdf"))
        if not pdfs:
            print("WARNING: no PDFs in %s" % subj_dir)
            continue
        for pdf in pdfs:
            pdf_list.append((subj, pdf))

    # ── Filter to single file if --file was given ──
    if args.file:
        target = args.file.strip()
        matched = [(s, p) for s, p in pdf_list if p.name == target]
        if not matched:
            # Try case-insensitive partial match
            matched = [(s, p) for s, p in pdf_list if target.lower() in p.name.lower()]
        if not matched:
            print("ERROR: No PDF found matching '%s'" % target)
            print("Available PDFs in selected subjects:")
            for s, p in pdf_list:
                print("  %s/%s" % (s, p.name))
            sys.exit(1)
        if len(matched) > 1:
            print("Multiple matches for '%s' — please be more specific:" % target)
            for s, p in matched:
                print("  %s/%s" % (s, p.name))
            sys.exit(1)
        pdf_list = matched
        print("Single-file mode: %s/%s" % (matched[0][0], matched[0][1].name))

    print("=" * 65)
    print("  NCERT PDF EXTRACTION  (Gemini File API)")
    print("=" * 65)
    print("  Subjects  : %s" % ", ".join(subjects))
    print("  Model     : %s" % args.model)
    print("  PDFs      : %d" % len(pdf_list))
    print("  Output    : %s" % ("PREVIEW (terminal)" if args.preview else str(output_path)))
    print("  Resume    : %s" % args.resume)
    print("  Dry run   : %s" % args.dry_run)
    print("=" * 65)

    if args.dry_run:
        print("\nDry run — PDFs that would be processed:")
        for subj, pdf in pdf_list:
            pdf_key = "%s/%s" % (subj, pdf.name)
            if pdf.name in SKIP_FILES:
                status = "SKIP (blocklist)"
            elif pdf_key in progress and args.resume:
                status = "SKIP (done)"
            else:
                status = "PROCESS"
            print(f"  [{status}] {subj}/{pdf.name}")
        return

    # ── Extract ──
    total_kept     = 0
    total_rejected = 0
    total_failed   = 0
    seen_sigs      = set()      # global dedup across all PDFs

    out_fh      = None if args.preview else open(output_path, "a", encoding="utf-8")
    reject_log_fh = None if args.preview else open(REJECTION_LOG, "a", encoding="utf-8")

    for subj, pdf_path in pdf_list:
        pdf_key = "%s/%s" % (subj, pdf_path.name)

        if pdf_path.name in SKIP_FILES:
            print("  SKIP (blocklist) : %s" % pdf_key)
            continue

        if pdf_key in progress and args.resume:
            print("  SKIP (done) : %s" % pdf_key)
            continue

        id_prefix = re.sub(r"[^a-z0-9]", "_", subj.lower()) + \
                    "_" + re.sub(r"[^a-z0-9]", "_", pdf_path.stem.lower()[:30])

        print("\n  [%s] %s" % (subj, pdf_path.name))

        kept, reject_tally, error = process_pdf(
            pdf_path, subj, client, args.model, id_prefix, seen_sigs, reject_log_fh
        )
        n_rejected = sum(reject_tally.values()) if isinstance(reject_tally, dict) else 0

        if error:
            print("    ERROR: %s" % error)
            total_failed += 1
            progress[pdf_key] = {"status": "error", "error": error}
        else:
            if args.preview:
                print("\n" + "=" * 70)
                print("  PREVIEW — %d chunks extracted from %s" % (len(kept), pdf_path.name))
                print("=" * 70)
                for i, chunk in enumerate(kept, start=1):
                    print("\n--- Chunk %d of %d | Section: %s | Words: %d ---" % (
                        i, len(kept), chunk["section_heading"], chunk["word_count"]
                    ))
                    print(chunk["text"])
                print("\n" + "=" * 70)
                print("  END PREVIEW — %d kept / %d rejected" % (len(kept), n_rejected))
                print("  To write to file, re-run WITHOUT --preview")
                print("=" * 70)
            else:
                for chunk in kept:
                    out_fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                out_fh.flush()

                progress[pdf_key] = {
                    "status":   "done",
                    "kept":     len(kept),
                    "rejected": n_rejected,
                }
                save_progress(progress)

            total_kept     += len(kept)
            total_rejected += n_rejected

            if not args.preview:
                print("    total for this PDF — kept %d / rejected %d" % (len(kept), n_rejected))

        # Rate limit pause between PDFs
        time.sleep(RATE_LIMIT_WAIT)

    if out_fh:
        out_fh.close()
    if reject_log_fh:
        reject_log_fh.close()

    if not args.preview:
        # ── Summary ──
        print("\n" + "=" * 65)
        print("  EXTRACTION COMPLETE")
        print("=" * 65)
        done = len([p for p in progress.values() if p.get("status") == "done"])
        print("  PDFs processed  : %d" % done)
        print("  PDFs failed     : %d" % total_failed)
        print("  Chunks kept     : %d" % total_kept)
        print("  Chunks rejected : %d" % total_rejected)
        if total_kept + total_rejected > 0:
            pct = 100 * total_kept / (total_kept + total_rejected)
            print("  Pass rate       : %.1f%%" % pct)
        print("  Output          : %s" % output_path)
        print("  Rejection log   : %s" % REJECTION_LOG)
        print("=" * 65)


if __name__ == "__main__":
    main()
