# =============================================================================
# clean_sft_dataset.py
# =============================================================================
# Cleans unified_sft_v3.jsonl before Round 2 training.
#
# What this script removes:
#   1. Answers that reference "the passage" / "this passage" — invalid because
#      at inference time there is no passage, only the question.
#   2. Answers that are too short (under 60 words) — not useful for training.
#   3. Answers containing NCERT page markers like "314 / Social and Political Life"
#   4. Answers with common OCR artifacts from PDF extraction.
#
# What this script keeps:
#   Everything else — the 99%+ of records that are valid.
#
# Usage:
#   python3 scripts/clean_sft_dataset.py
#   python3 scripts/clean_sft_dataset.py --input path/to/file.jsonl --dry-run
# =============================================================================

import json
import re
import os
import sys
import argparse
import shutil
from collections import Counter
from datetime import datetime


# ---------------------------------------------------------------------------
# Bad patterns to check in the answer text
# ---------------------------------------------------------------------------

# These phrases mean the answer is referring to a source chunk/passage
# that does not exist at inference time -- invalid for training
PASSAGE_PHRASES = [
    "this passage",
    "the passage",
    "as described in the passage",
    "as mentioned in the passage",
    "as discussed in the passage",
    "according to the passage",
    "the passage states",
    "the passage discusses",
    "the passage explains",
    "the passage highlights",
    "the text states",
    "the text discusses",
    "the excerpt",
    "this text",
    "the above passage",
    "the given passage",
]

# NCERT page number patterns like "314 / Social and Political Life"
# or "Chapter 5 / History" that leaked from PDF extraction
PAGE_NUMBER_PATTERN = re.compile(
    r'\b\d{2,4}\s*/\s*[A-Z][a-zA-Z\s]{3,30}'
)

# OCR artifacts from bad PDF extraction
OCR_ARTIFACT_PATTERN = re.compile(
    r'(\bl\s+[A-Z]|>{1,3}\s+\*\*|^l\s)', re.MULTILINE
)

# Lines that are just standalone numbers (page numbers)
STANDALONE_NUMBER_PATTERN = re.compile(
    r'^\s*\d{1,4}\s*$', re.MULTILINE
)


def extract_answer(text: str) -> str:
    """Pull just the model answer from the full chat template text."""
    if "<start_of_turn>model\n" in text:
        answer = text.split("<start_of_turn>model\n")[-1]
        answer = answer.replace("<end_of_turn>", "").strip()
        return answer
    return text


def check_record(record: dict) -> tuple:
    """
    Check a single record for quality issues.

    Returns:
        (is_clean: bool, reason: str)
        is_clean = True means keep the record
        reason = why it was removed (empty string if kept)
    """
    text = record.get("text", "")
    answer = extract_answer(text)
    answer_lower = answer.lower()
    word_count = len(answer.split())

    # Check 1 -- passage references
    for phrase in PASSAGE_PHRASES:
        if phrase in answer_lower:
            return False, f"passage_reference: '{phrase}'"

    # Check 2 -- too short (under 60 words is not useful for training)
    if word_count < 60:
        return False, f"too_short: {word_count} words"

    # Check 3 -- NCERT page number leaks
    if PAGE_NUMBER_PATTERN.search(answer):
        match = PAGE_NUMBER_PATTERN.search(answer)
        return False, f"page_number_leak: '{match.group()[:40]}'"

    # Check 4 -- OCR artifacts
    if OCR_ARTIFACT_PATTERN.search(answer):
        return False, "ocr_artifact"

    # Check 5 -- answer is just numbers or whitespace
    if STANDALONE_NUMBER_PATTERN.search(answer):
        lines = [l.strip() for l in answer.split("\n") if l.strip()]
        standalone_num_lines = sum(
            1 for l in lines if re.match(r'^\d{1,4}$', l)
        )
        if standalone_num_lines > 2:
            return False, f"standalone_numbers: {standalone_num_lines} numeric lines"

    return True, ""


def clean_dataset(input_path: str, output_path: str, dry_run: bool = False):
    """
    Read input JSONL, filter bad records, write clean JSONL.
    """
    print(f"Input  : {input_path}")
    print(f"Output : {output_path}")
    print(f"Mode   : {'DRY RUN (no files written)' if dry_run else 'LIVE'}")
    print("-" * 60)

    kept = []
    removed = []
    removal_reasons = Counter()

    with open(input_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  Warning: skipping malformed JSON at line {i+1}: {e}")
                removed.append((record if 'record' in dir() else {}, "malformed_json"))
                removal_reasons["malformed_json"] += 1
                continue

            is_clean, reason = check_record(record)

            if is_clean:
                kept.append(record)
            else:
                removed.append((record, reason))
                removal_reasons[reason.split(":")[0]] += 1

    total = len(kept) + len(removed)
    removed_count = len(removed)
    kept_count = len(kept)

    print(f"Total records read : {total:,}")
    print(f"Records kept       : {kept_count:,}  ({kept_count/total*100:.1f}%)")
    print(f"Records removed    : {removed_count:,}  ({removed_count/total*100:.1f}%)")
    print()
    print("Removal breakdown:")
    for reason, count in removal_reasons.most_common():
        print(f"  {reason:<35} {count:>5,}")

    # Show samples of removed records
    print()
    print("Sample removed records:")
    shown = {}
    for record, reason in removed:
        reason_type = reason.split(":")[0]
        if reason_type not in shown:
            shown[reason_type] = True
            text = record.get("text", "")
            answer = extract_answer(text)
            print(f"\n  Reason : {reason}")
            print(f"  ID     : {record.get('id', 'unknown')}")
            print(f"  Answer : {answer[:200]}...")

    if dry_run:
        print("\nDry run complete -- no files written")
        return

    # Backup original
    backup_path = input_path + f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(input_path, backup_path)
    print(f"\nBackup saved : {backup_path}")

    # Write cleaned output
    with open(output_path, "w", encoding="utf-8") as f:
        for record in kept:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Clean file written : {output_path}")
    print(f"Records in clean file : {kept_count:,}")


def main():
    parser = argparse.ArgumentParser(
        description="Clean unified_sft_v3.jsonl before Round 2 training"
    )
    parser.add_argument(
        "--input",
        default="/Users/satyamurti/Downloads/DataSets/dataset_output_final/combined/unified_sft_v3.jsonl",
        help="Path to input JSONL file"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to output clean JSONL (default: same folder, _clean suffix)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyse only, do not write any files"
    )
    args = parser.parse_args()

    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = base + "_clean" + ext

    clean_dataset(args.input, args.output, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
