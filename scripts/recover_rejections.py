#!/usr/bin/env python3
"""
recover_rejections.py
=====================
Recover rejected chunks from the rejection log and append them to the main
pretrain JSONL, filtered by rejection reason.

Usage:
    # Recover all TOO_SHORT records (default)
    python3 scripts/recover_rejections.py

    # Recover records with word count >= 40 (custom floor)
    python3 scripts/recover_rejections.py --min-words 40

    # Recover a different reason type
    python3 scripts/recover_rejections.py --reason MID_SENTENCE

    # Recover multiple reasons
    python3 scripts/recover_rejections.py --reason TOO_SHORT --reason MID_SENTENCE

    # Dry run — show what would be added without writing
    python3 scripts/recover_rejections.py --dry-run

    # Preview the text of records that would be added
    python3 scripts/recover_rejections.py --preview
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR     = Path(__file__).resolve().parent.parent
OUTPUT_DIR   = BASE_DIR / "dataset_output_final" / "combined"
REJECT_FILE  = OUTPUT_DIR / "pretrain_v2_rejection_log.jsonl"
OUTPUT_FILE  = OUTPUT_DIR / "unified_pretrain_v2.jsonl"


# ---------------------------------------------------------------------------
# ID generator  (matches format used by extract_ncert_gemini.py)
# ---------------------------------------------------------------------------

def make_id(subject: str, source_file: str, idx: int) -> str:
    """Generate an ID matching the existing format: subj_filekey_NNNN"""
    subj = re.sub(r"[^a-z0-9]+", "_", subject.lower()).strip("_")
    stem = Path(source_file).stem
    fkey = re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")[:30]
    return "%s_%s_%04d" % (subj, fkey, idx)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_existing_ids(output_file: Path) -> set:
    """Return set of all IDs already in the output file."""
    ids = set()
    if output_file.exists():
        with open(output_file, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        if "id" in rec:
                            ids.add(rec["id"])
                    except Exception:
                        pass
    return ids


def next_idx(output_file: Path) -> int:
    """Return the next available sequential index across all records."""
    count = 0
    if output_file.exists():
        with open(output_file, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    count += 1
    return count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Recover rejected chunks into the main pretrain JSONL.")
    parser.add_argument(
        "--reason", action="append", dest="reasons", metavar="REASON",
        help="Rejection reason to recover (can be specified multiple times). Default: TOO_SHORT",
    )
    parser.add_argument(
        "--min-words", type=int, default=0,
        help="Only recover records with at least this many words (0 = no filter, default: 0)",
    )
    parser.add_argument(
        "--max-words", type=int, default=0,
        help="Only recover records with at most this many words (0 = no filter, default: 0)",
    )
    parser.add_argument(
        "--subject", action="append", dest="subjects", metavar="SUBJECT",
        help="Only recover records for this subject (can be repeated). Default: all subjects",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be written without actually writing",
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Show a preview of each record's text that would be added",
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_FILE,
        help="Output JSONL file to append to (default: unified_pretrain_v2.jsonl)",
    )
    parser.add_argument(
        "--reject-log", type=Path, default=REJECT_FILE,
        help="Rejection log file to read from",
    )
    args = parser.parse_args()

    # Defaults
    reasons = set(args.reasons) if args.reasons else {"TOO_SHORT"}
    subjects_filter = {s.lower() for s in args.subjects} if args.subjects else None

    if not args.reject_log.exists():
        print("ERROR: Rejection log not found: %s" % args.reject_log)
        sys.exit(1)

    # Load all rejections
    all_rejections = []
    with open(args.reject_log, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    all_rejections.append(json.loads(line))
                except Exception:
                    pass

    print("Rejection log loaded: %d total records" % len(all_rejections))
    print("Filtering by reason(s): %s" % ", ".join(sorted(reasons)))

    # Filter
    candidates = []
    for r in all_rejections:
        if r.get("rejection_reason") not in reasons:
            continue
        if subjects_filter and r.get("subject", "").lower() not in subjects_filter:
            continue
        wc = r.get("word_count", 0)
        if args.min_words and wc < args.min_words:
            continue
        if args.max_words and wc > args.max_words:
            continue
        text = r.get("text_full", "").strip()
        if not text:
            continue
        candidates.append(r)

    print("Candidates to recover: %d" % len(candidates))
    if not candidates:
        print("Nothing to recover. Exiting.")
        sys.exit(0)

    # Load existing IDs to avoid duplicates
    existing_ids = load_existing_ids(args.output)
    print("Existing records in output: %d" % len(existing_ids))

    idx = next_idx(args.output)
    to_write = []

    for r in candidates:
        text      = r["text_full"].strip()
        subject   = r.get("subject", "unknown")
        source    = r.get("source_file", "unknown")
        heading   = r.get("section_heading", "")
        wc        = r.get("word_count", len(text.split()))
        cc        = r.get("char_count", len(text))

        rec_id = make_id(subject, source, idx)

        # Skip if already present (shouldn't happen but be safe)
        if rec_id in existing_ids:
            idx += 1
            continue

        record = {
            "id":              rec_id,
            "text":            text,
            "subject":         subject,
            "source_file":     source,
            "section_heading": heading,
            "word_count":      wc,
            "char_count":      cc,
        }
        to_write.append(record)
        existing_ids.add(rec_id)
        idx += 1

    print("")
    if args.preview:
        print("=" * 60)
        for i, rec in enumerate(to_write, 1):
            print("[%d] %s | %s | %dw" % (i, rec["subject"], rec["section_heading"], rec["word_count"]))
            print("    " + rec["text"][:200] + ("..." if len(rec["text"]) > 200 else ""))
            print("")
        print("=" * 60)
        print("")

    if args.dry_run:
        print("DRY RUN — would append %d records to: %s" % (len(to_write), args.output))
        by_subject = {}
        for rec in to_write:
            subj = rec["subject"]
            by_subject.setdefault(subj, 0)
            by_subject[subj] += 1
        for subj, count in sorted(by_subject.items()):
            print("  %-15s : %d records" % (subj, count))
        return

    # Append
    with open(args.output, "a", encoding="utf-8") as fh:
        for rec in to_write:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("Appended %d records to: %s" % (len(to_write), args.output))
    by_subject = {}
    for rec in to_write:
        subj = rec["subject"]
        by_subject.setdefault(subj, 0)
        by_subject[subj] += 1
    for subj, count in sorted(by_subject.items()):
        print("  %-15s : %d records added" % (subj, count))

    print("")
    print("Done. Regenerate dashboard:")
    print("  python3 scripts/generate_dashboard.py --open")


if __name__ == "__main__":
    main()
