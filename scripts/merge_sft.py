#!/usr/bin/env python3
"""
merge_sft.py — Merge multiple SFT JSONL datasets into a single clean,
deduplicated, shuffled training file for Gemma 2 Stage 2 fine-tuning.

Usage:
    python3 scripts/merge_sft.py                     # merge all available sources
    python3 scripts/merge_sft.py --balance            # merge + cap subjects/q_types
    python3 scripts/merge_sft.py --stats-only         # print stats, don't write
    python3 scripts/merge_sft.py --inputs a.jsonl b.jsonl   # specific files
"""

import argparse
import collections
import hashlib
import json
import pathlib
import random
import re
import sys


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = pathlib.Path(__file__).parent.parent / "dataset_output_final" / "combined"

# Source definitions: (tag, default_filename, priority)
# Lower priority number = higher priority (kept during dedup)
SOURCES = [
    ("past_papers",       "past_papers_questions.jsonl",  1),
    ("scraped",           "scraped_coaching.jsonl",        2),
    ("sft_v3",            "unified_sft_v3.jsonl",          3),
    ("sft_v2",            "unified_sft_v2.jsonl",          4),
]

BALANCE_SUBJECT_CAP = 3000
BALANCE_QTYPE_CAP   = 4000
SHUFFLE_SEED        = 42

# Quality filter thresholds
MIN_QUESTION_WORDS  = 8
MIN_ANSWER_WORDS    = 80

# Corruption pattern: question contains something like "3.22 Figure" or "6.11(b)"
CORRUPTION_RE = re.compile(r'\b\d+\.\d+\s*(?:Figure|Table|Box|Exhibit|Graph|Chart)', re.IGNORECASE)


# ---------------------------------------------------------------------------
# Gemma format helpers
# ---------------------------------------------------------------------------

def parse_gemma_text(text: str):
    """
    Extract (question_text, answer_text) from a Gemma-formatted string.
    Returns (None, None) on parse failure.
    """
    user_match  = re.search(r'<start_of_turn>user\n(.*?)<end_of_turn>', text, re.DOTALL)
    model_match = re.search(r'<start_of_turn>model\n(.*?)<end_of_turn>', text, re.DOTALL)
    if not user_match or not model_match:
        return None, None
    return user_match.group(1).strip(), model_match.group(1).strip()


def validate_gemma(record: dict) -> bool:
    """Return True if the record passes Gemma format validation."""
    text = record.get("text", "")
    user_count  = text.count("<start_of_turn>user")
    model_count = text.count("<start_of_turn>model")
    if user_count != 1 or model_count != 1:
        return False
    q, a = parse_gemma_text(text)
    if not q or not a:
        return False
    return True


# ---------------------------------------------------------------------------
# Quality filter
# ---------------------------------------------------------------------------

def passes_quality(record: dict) -> bool:
    """Return True if the record passes all quality checks."""
    text     = record.get("text", "")
    question = record.get("question", "")

    # Must contain Gemma turn markers
    if "<start_of_turn>user" not in text:
        return False

    # Parse question/answer from text (authoritative source)
    q_text, a_text = parse_gemma_text(text)
    if q_text is None:
        return False

    # Use stored question field if available, else fall back to parsed
    effective_question = question.strip() if question.strip() else q_text

    # Question length
    if len(effective_question.split()) < MIN_QUESTION_WORDS:
        return False

    # Answer length
    if len(a_text.split()) < MIN_ANSWER_WORDS:
        return False

    # Corruption check
    if CORRUPTION_RE.search(effective_question):
        return False

    return True


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def question_hash(question: str) -> str:
    normalised = question.lower().strip()
    return hashlib.md5(normalised.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_source(path: pathlib.Path, tag: str, priority: int) -> list:
    """Load records from a JSONL file, tag them, and return a list."""
    records = []
    if not path.exists():
        print(f"  [WARNING] File not found, skipping: {path}", file=sys.stderr)
        return records

    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"  [WARNING] {path.name}:{lineno} JSON decode error: {exc}", file=sys.stderr)
                continue
            # Tag and normalise
            rec["source"]   = tag
            rec["_priority"] = priority
            # Keep only expected fields (plus internal _priority for dedup)
            records.append(rec)

    return records


def keep_fields(record: dict) -> dict:
    """Return a clean record with only the fields we want to keep."""
    return {k: record[k] for k in ("id", "text", "subject", "question", "q_type", "source")
            if k in record}


# ---------------------------------------------------------------------------
# Balancing
# ---------------------------------------------------------------------------

def balance_by(records: list, key: str, cap: int) -> list:
    """Cap records so that no value of `key` exceeds `cap`."""
    counts: dict = collections.defaultdict(int)
    result = []
    for rec in records:
        val = rec.get(key, "unknown")
        if counts[val] < cap:
            result.append(rec)
            counts[val] += 1
    return result


# ---------------------------------------------------------------------------
# Stats printing
# ---------------------------------------------------------------------------

def print_stats(
    loaded_counts: dict,
    filtered_counts: dict,
    deduped_counts: dict,
    final_records: list,
    output_path: pathlib.Path | None,
):
    source_order = [s[0] for s in SOURCES]
    # Include any extra sources not in the default list
    all_sources = list(dict.fromkeys(source_order + list(loaded_counts.keys())))

    col_w = [20, 10, 16, 12]
    sep   = "─" * 58

    print()
    print("=" * 60)
    print("DATASET MERGE SUMMARY")
    print("=" * 60)
    header = (
        f"{'Source':<{col_w[0]}}"
        f"{'Loaded':>{col_w[1]}}"
        f"{'After filter':>{col_w[2]}}"
        f"{'After dedup':>{col_w[3]}}"
    )
    print(header)
    print(sep)

    total_loaded = total_filtered = total_deduped = 0
    for src in all_sources:
        if src not in loaded_counts:
            continue
        l = loaded_counts.get(src, 0)
        f = filtered_counts.get(src, 0)
        d = deduped_counts.get(src, 0)
        total_loaded   += l
        total_filtered += f
        total_deduped  += d
        print(
            f"{src:<{col_w[0]}}"
            f"{l:>{col_w[1]},}"
            f"{f:>{col_w[2]},}"
            f"{d:>{col_w[3]},}"
        )

    print(sep)
    print(
        f"{'TOTAL':<{col_w[0]}}"
        f"{total_loaded:>{col_w[1]},}"
        f"{total_filtered:>{col_w[2]},}"
        f"{total_deduped:>{col_w[3]},}"
    )

    if final_records:
        final_count = len(final_records)
        print(f"\n(After balancing / final write: {final_count:,})")

        subject_counts: dict = collections.Counter(r.get("subject", "unknown") for r in final_records)
        qtype_counts:   dict = collections.Counter(r.get("q_type",   "unknown") for r in final_records)

        print("\nBy subject:")
        for subj, cnt in sorted(subject_counts.items(), key=lambda x: -x[1]):
            print(f"  {subj:<30} {cnt:>6,}")

        print("\nBy question type:")
        for qt, cnt in sorted(qtype_counts.items(), key=lambda x: -x[1]):
            print(f"  {qt:<30} {cnt:>6,}")

    if output_path:
        print(f"\nSaved → {output_path}")
    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# Interleaved shuffle
# ---------------------------------------------------------------------------

def interleave_shuffle(records: list, seed: int = SHUFFLE_SEED) -> list:
    """
    Shuffle within each source first, then interleave sources so records from
    the same source are spread across the file, then do a final light shuffle
    to mix the interleaved order slightly.
    """
    rng = random.Random(seed)

    # Group by source
    by_source: dict = collections.defaultdict(list)
    for rec in records:
        by_source[rec.get("source", "unknown")].append(rec)

    # Shuffle each group independently
    for grp in by_source.values():
        rng.shuffle(grp)

    # Round-robin interleave
    interleaved = []
    iters = [iter(grp) for grp in by_source.values()]
    while iters:
        next_iters = []
        rng.shuffle(iters)          # randomise which source goes next each round
        for it in iters:
            try:
                interleaved.append(next(it))
                next_iters.append(it)
            except StopIteration:
                pass
        iters = next_iters

    # Final shuffle with the same seed (deterministic)
    rng.shuffle(interleaved)
    return interleaved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge SFT JSONL datasets for Gemma 2 fine-tuning."
    )
    parser.add_argument(
        "--inputs", nargs="+", metavar="FILE",
        help="Override default source files (paths relative to combined/ dir or absolute)."
    )
    parser.add_argument(
        "--balance", action="store_true",
        help=f"Cap subjects at {BALANCE_SUBJECT_CAP} and q_types at {BALANCE_QTYPE_CAP}."
    )
    parser.add_argument(
        "--stats-only", action="store_true",
        help="Print statistics but do not write the output file."
    )
    parser.add_argument(
        "--output", metavar="FILE",
        default=str(BASE_DIR / "unified_sft_final.jsonl"),
        help="Output file path (default: dataset_output_final/combined/unified_sft_final.jsonl)."
    )
    return parser.parse_args()


def resolve_inputs(args) -> list:
    """
    Return a list of (tag, path, priority) tuples based on CLI args or defaults.
    """
    if args.inputs:
        resolved = []
        for i, inp in enumerate(args.inputs):
            p = pathlib.Path(inp)
            if not p.is_absolute():
                p = BASE_DIR / p
            # Try to match a known source tag
            tag      = p.stem  # fallback
            priority = 99 - i  # higher index = lower priority
            for s_tag, s_file, s_priority in SOURCES:
                if p.name == s_file or p.stem == s_tag:
                    tag      = s_tag
                    priority = s_priority
                    break
            resolved.append((tag, p, priority))
        return resolved
    else:
        return [(tag, BASE_DIR / fname, prio) for tag, fname, prio in SOURCES]


def main():
    args = parse_args()
    inputs = resolve_inputs(args)
    output_path = pathlib.Path(args.output)

    print(f"\nLoading {len(inputs)} source(s)...")

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    loaded_counts:   dict = {}
    filtered_counts: dict = {}

    all_records: list = []

    for tag, path, priority in inputs:
        recs = load_source(path, tag, priority)
        loaded_counts[tag] = len(recs)
        print(f"  {tag:<20} loaded  {len(recs):>6,}  from {path.name}")

        # Quality filter
        filtered = [r for r in recs if passes_quality(r)]
        filtered_counts[tag] = len(filtered)
        dropped = len(recs) - len(filtered)
        if dropped:
            print(f"  {tag:<20} dropped {dropped:>6,}  (quality filter)")

        all_records.extend(filtered)

    # ------------------------------------------------------------------
    # Deduplication (priority-aware)
    # ------------------------------------------------------------------
    # Sort so highest-priority (lowest number) comes first
    all_records.sort(key=lambda r: r.get("_priority", 99))

    seen:         dict = {}   # hash → source_tag of winner
    deduped_records: list = []
    dup_counts:   dict = collections.defaultdict(int)

    for rec in all_records:
        q_field = rec.get("question", "")
        # Fall back to parsing question from text
        if not q_field.strip():
            q_text, _ = parse_gemma_text(rec.get("text", ""))
            q_field = q_text or ""

        h = question_hash(q_field)
        if h not in seen:
            seen[h] = rec.get("source", "unknown")
            deduped_records.append(rec)
        else:
            dup_counts[rec.get("source", "unknown")] += 1

    # Build dedup counts per source
    deduped_counts: dict = collections.Counter(r.get("source") for r in deduped_records)

    if dup_counts:
        print(f"\n  Duplicates removed by source: " +
              ", ".join(f"{src}={cnt:,}" for src, cnt in dup_counts.items()))

    # ------------------------------------------------------------------
    # Gemma format validation
    # ------------------------------------------------------------------
    valid_records = []
    invalid_count = 0
    for rec in deduped_records:
        if validate_gemma(rec):
            valid_records.append(rec)
        else:
            invalid_count += 1

    if invalid_count:
        print(f"  Dropped {invalid_count:,} records that failed Gemma format validation.")

    # ------------------------------------------------------------------
    # Balancing (optional)
    # ------------------------------------------------------------------
    final_records = valid_records

    if args.balance:
        before = len(final_records)
        final_records = balance_by(final_records, "subject", BALANCE_SUBJECT_CAP)
        after_subj = len(final_records)
        final_records = balance_by(final_records, "q_type",  BALANCE_QTYPE_CAP)
        after_qtype = len(final_records)
        print(f"\n  Balancing: {before:,} → {after_subj:,} (subject cap) "
              f"→ {after_qtype:,} (q_type cap)")

    # ------------------------------------------------------------------
    # Shuffle + interleave
    # ------------------------------------------------------------------
    final_records = interleave_shuffle(final_records, seed=SHUFFLE_SEED)

    # Clean internal fields before output
    final_records = [keep_fields(r) for r in final_records]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    out_path_for_stats = None if args.stats_only else output_path
    print_stats(
        loaded_counts,
        filtered_counts,
        deduped_counts,
        final_records,
        out_path_for_stats,
    )

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    if args.stats_only:
        print("(--stats-only: output file not written)")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for rec in final_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Done. Wrote {len(final_records):,} records to {output_path}\n")


if __name__ == "__main__":
    main()
