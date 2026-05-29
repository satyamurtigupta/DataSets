# =============================================================================
# dataset_quality_check.py
# =============================================================================
# Runs 3 independent quality checks on any MCQ JSONL file (synthetic or real).
#
# CHECK 1 — Structural Integrity
#   Every required Alpaca field exists, answer is a/b/c/d, all 4 options
#   are present in the question text, explanation is non-empty and substantive.
#
# CHECK 2 — Instruction Sanity
#   Instruction is a real human-readable sentence (not a truncated chunk ID,
#   exercise question number, OCR artifact, or mid-sentence fragment).
#   Source context starts with a capital letter (not mid-sentence).
#
# CHECK 3 — Content Grounding
#   The explanation explicitly names the correct answer letter.
#   For synthetic MCQs: proper nouns introduced in the explanation that are
#   absent from both the question and the source context are flagged as
#   potential hallucinations.
#
# Usage:
#   python3 scripts/dataset_quality_check.py --input path/to/file.jsonl
#   python3 scripts/dataset_quality_check.py --input file.jsonl --clean-output cleaned.jsonl
#   python3 scripts/dataset_quality_check.py --input file.jsonl --verbose
# =============================================================================

import json
import os
import re
import sys
import argparse
from typing import List, Tuple, Optional
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_ALPACA_FIELDS = [
    "instruction", "input", "question", "answer", "explanation",
    "subject", "question_type", "difficulty", "exam",
]

VALID_ANSWERS = {"a", "b", "c", "d"}

VALID_QUESTION_TYPES = {
    "statement_based", "assertion_reason", "match_pairs",
    "factual", "conceptual", "how_many_correct", "batch",
}

VALID_DIFFICULTIES = {"easy", "medium", "difficult"}

# Proper nouns that appear in MCQ language itself — not hallucinations
EXPLANATION_STOPWORDS = {
    "Statement", "Option", "Hence", "Therefore", "However",
    "Correct", "Incorrect", "Because", "According", "Consider",
    "Which", "These", "This", "Thus", "Both", "Neither", "Only",
    "Based", "Note", "Also", "While", "Since", "Although",
}

# OCR and PDF artifact patterns
ARTIFACT_PATTERN = re.compile(
    r'\|\)|'         # |)
    r'\|\('          # |(
    r'|<[A-Z\']+>'   # <MCRO>, <BI>
    r'|[~]{2,}'      # ~~~
    r'|\\x[0-9a-f]{2}'  # hex escapes
    r'|\x00'         # null bytes
)

# Exercise question pattern: "6. Compare the..." or "3) Explain..."
EXERCISE_PATTERN = re.compile(r'^\s*\d{1,2}[\.\)]\s+[A-Za-z]')

# Equation/formula topic pattern
EQUATION_PATTERN = re.compile(r'^\d+[\.\d]+\s')


# ---------------------------------------------------------------------------
# Check 1: Structural Integrity
# ---------------------------------------------------------------------------

def check_structural(record: dict) -> List[str]:
    """
    Validates that the record has all required fields in the correct format.

    Returns a list of error strings. Empty list = passed.
    """
    errors = []
    record_id = record.get("id", record.get("source_id", "unknown"))

    # 1a. Required fields present and non-empty
    for field in REQUIRED_ALPACA_FIELDS:
        val = record.get(field)
        if val is None:
            errors.append(f"MISSING_FIELD:{field}")
        elif isinstance(val, str) and not val.strip():
            errors.append(f"EMPTY_FIELD:{field}")

    # 1b. Answer is a single valid letter
    answer = str(record.get("answer", "")).lower().strip()
    # For batch records the answer field is "Q1: (a), Q2: (c)..."
    if record.get("question_type") != "batch":
        if answer not in VALID_ANSWERS:
            errors.append(f"INVALID_ANSWER:{answer!r}")
        else:
            # 1c. The correct option appears in the question text
            question = record.get("question", "")
            if f"({answer})" not in question.lower() and f"{answer})" not in question.lower():
                errors.append(f"ANSWER_OPTION_MISSING_IN_QUESTION:({answer})")

    # 1d. All 4 options present in question text
    question = record.get("question", "")
    for opt in ["a", "b", "c", "d"]:
        if f"({opt})" not in question.lower():
            # Batch questions intentionally skip this
            if record.get("question_type") != "batch":
                errors.append(f"MISSING_OPTION_IN_QUESTION:({opt})")

    # 1e. Explanation is substantive
    explanation = record.get("explanation", "")
    word_count = len(explanation.split())
    if word_count == 0:
        errors.append("EMPTY_EXPLANATION")
    elif word_count < 15:
        errors.append(f"SHORT_EXPLANATION:{word_count}_words")

    # 1f. Question type is known
    qt = record.get("question_type", "")
    if qt and qt not in VALID_QUESTION_TYPES:
        errors.append(f"UNKNOWN_QUESTION_TYPE:{qt}")

    # 1g. Difficulty is known
    diff = record.get("difficulty", "")
    if diff and diff not in VALID_DIFFICULTIES:
        errors.append(f"UNKNOWN_DIFFICULTY:{diff}")

    return errors


# ---------------------------------------------------------------------------
# Check 2: Instruction Sanity
# ---------------------------------------------------------------------------

def check_instruction_sanity(record: dict) -> List[str]:
    """
    Validates that the instruction is a real human-readable sentence and
    the input context is clean and complete.

    Returns a list of error strings. Empty list = passed.
    """
    errors = []
    instruction = record.get("instruction", "")
    inp         = record.get("input", "")
    question    = record.get("question", "")

    # 2a. Instruction contains NCERT exercise question number
    if EXERCISE_PATTERN.match(instruction):
        errors.append("EXERCISE_NUMBER_IN_INSTRUCTION")

    # 2b. Instruction is too short to be meaningful
    if len(instruction.split()) < 6:
        errors.append(f"SHORT_INSTRUCTION:{len(instruction.split())}_words")

    # 2c. Instruction ends mid-word or mid-sentence (truncated chunk ID leaked in)
    if instruction and instruction[-1] not in ".?!\"'":
        # Allow instructions that end with a subject name or topic word
        last_word = instruction.split()[-1] if instruction.split() else ""
        if last_word and last_word[-1].islower() and len(last_word) > 3:
            errors.append("TRUNCATED_INSTRUCTION")

    # 2d. OCR/PDF artifacts in instruction or input
    if ARTIFACT_PATTERN.search(instruction):
        errors.append("ARTIFACT_IN_INSTRUCTION")
    if ARTIFACT_PATTERN.search(inp):
        errors.append("ARTIFACT_IN_INPUT")
    if ARTIFACT_PATTERN.search(question):
        errors.append("ARTIFACT_IN_QUESTION")

    # 2e. Source context starts mid-sentence (chunk boundary error)
    # Only applies to context_based and generate_by_topic formats
    fmt = record.get("format", "")
    if fmt in ("context_based", "generate_by_topic") and inp:
        first_char = inp.lstrip().lstrip("\"'([{")[0:1]
        if first_char and first_char.islower():
            errors.append("INPUT_STARTS_MID_SENTENCE")

    # 2f. Equation/formula leaked as topic in instruction
    if EQUATION_PATTERN.search(instruction):
        errors.append("EQUATION_REFERENCE_IN_INSTRUCTION")

    # 2g. Chunk ID leaked into instruction (looks like an ID, not a topic)
    # IDs contain underscores and digits like "history_ncert_class_11_0304"
    if re.search(r'\b[a-z]+_[a-z]+_\d{2,}', instruction):
        errors.append("CHUNK_ID_IN_INSTRUCTION")

    return errors


# ---------------------------------------------------------------------------
# Check 3: Content Grounding
# ---------------------------------------------------------------------------

def check_content_grounding(record: dict) -> List[str]:
    """
    Validates that the explanation is grounded:
      - The correct answer letter is explicitly referenced in the explanation.
      - For synthetic MCQs: proper nouns in the explanation that do not appear
        in the source context or question are flagged as likely hallucinations.

    Returns a list of error strings. Empty list = passed.
    """
    errors = []

    answer      = str(record.get("answer", "")).lower().strip()
    explanation = record.get("explanation", "")
    question    = record.get("question", "")
    inp         = record.get("input", "")
    record_id   = record.get("id", "")
    fmt         = record.get("format", "")

    # Skip grounding check for batch records
    if record.get("question_type") == "batch":
        return errors

    # 3a. Correct answer letter is mentioned in explanation
    expl_lower = explanation.lower()
    answer_mentioned = (
        f"({answer})" in expl_lower
        or f"option {answer}" in expl_lower
        or f"option ({answer})" in expl_lower
        or f"answer is {answer}" in expl_lower
        or f"answer is ({answer})" in expl_lower
        or f"correct answer is ({answer})" in expl_lower
        or f"correct answer is {answer}" in expl_lower
        or (len(answer) == 1 and f" {answer})" in expl_lower)
    )
    if not answer_mentioned:
        errors.append(f"ANSWER_NOT_REFERENCED_IN_EXPLANATION:({answer})")

    # 3b. Hallucination check — only for synthetic records.
    # Real UPSC records (dataset_type == "real_upsc_sft") legitimately reference
    # facts not in the question text; that is what a good explanation does.
    is_synthetic = (
        "synth" in record_id
        or record.get("dataset_type") == "synthetic_sft"
    )

    if is_synthetic and inp:
        # Extract proper nouns (capitalized words >= 4 chars) from explanation
        expl_proper = set(re.findall(r'\b([A-Z][a-z]{3,})\b', explanation))
        # Remove common explanation language that is never hallucination
        expl_proper -= EXPLANATION_STOPWORDS

        # Build reference set: everything in input + question
        reference_text = inp + " " + question
        ref_proper = set(re.findall(r'\b([A-Z][a-z]{3,})\b', reference_text))

        # Also check lowercase version (some words are cased differently)
        ref_lower = set(w.lower() for w in ref_proper)

        ungrounded = [
            w for w in expl_proper
            if w not in ref_proper and w.lower() not in ref_lower
        ]

        if len(ungrounded) > 4:
            errors.append(
                f"LIKELY_HALLUCINATION:{len(ungrounded)}_ungrounded_nouns"
                f":{','.join(sorted(ungrounded)[:6])}"
            )

    # 3c. Explanation repeats the question text verbatim (low quality copy-paste)
    if len(question) > 40 and question[:40].lower() in explanation.lower():
        errors.append("EXPLANATION_COPIES_QUESTION")

    return errors


# ---------------------------------------------------------------------------
# Full record audit
# ---------------------------------------------------------------------------

def audit_record(record: dict) -> dict:
    """Run all 3 checks on a single record. Returns audit result dict."""
    s_errors = check_structural(record)
    i_errors = check_instruction_sanity(record)
    g_errors = check_content_grounding(record)

    all_errors = s_errors + i_errors + g_errors
    return {
        "id"              : record.get("id", record.get("source_id", "unknown")),
        "passed"          : len(all_errors) == 0,
        "structural"      : s_errors,
        "instruction"     : i_errors,
        "grounding"       : g_errors,
        "all_errors"      : all_errors,
        "error_count"     : len(all_errors),
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def print_report(audits: List[dict], total: int, verbose: bool = False):
    passed  = sum(1 for a in audits if a["passed"])
    failed  = total - passed
    pass_pct = 100 * passed / total if total else 0

    print("=" * 65)
    print("  DATASET QUALITY REPORT")
    print("=" * 65)
    print(f"  Total records      : {total:,}")
    print(f"  PASSED all checks  : {passed:,}  ({pass_pct:.1f}%)")
    print(f"  FAILED >= 1 check  : {failed:,}  ({100-pass_pct:.1f}%)")
    print()

    # Per-check summary
    s_failed = sum(1 for a in audits if a["structural"])
    i_failed = sum(1 for a in audits if a["instruction"])
    g_failed = sum(1 for a in audits if a["grounding"])

    print("  CHECK 1 — Structural Integrity")
    print(f"    Failed : {s_failed:,} / {total:,} ({100*s_failed/total:.1f}%)")
    print()
    print("  CHECK 2 — Instruction Sanity")
    print(f"    Failed : {i_failed:,} / {total:,} ({100*i_failed/total:.1f}%)")
    print()
    print("  CHECK 3 — Content Grounding")
    print(f"    Failed : {g_failed:,} / {total:,} ({100*g_failed/total:.1f}%)")
    print()

    # Error frequency table
    error_freq: Counter = Counter()
    for a in audits:
        for e in a["all_errors"]:
            # Strip trailing detail (keep just the error code)
            code = e.split(":")[0]
            error_freq[code] += 1

    if error_freq:
        print("  Most common errors:")
        for code, count in error_freq.most_common(15):
            print(f"    {count:5d}  {code}")
        print()

    # Verbose: show first 10 failed records with details
    if verbose:
        print("  Failed record samples (first 10):")
        shown = 0
        for a in audits:
            if not a["passed"] and shown < 10:
                print(f"    [{a['id']}]")
                for e in a["all_errors"]:
                    print(f"      - {e}")
                shown += 1
        print()

    print("=" * 65)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace):
    input_path = args.input
    if not os.path.exists(input_path):
        print(f"ERROR: file not found — {input_path}")
        sys.exit(1)

    print(f"Loading : {input_path}")
    records = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"  WARNING: skipping invalid JSON line — {e}")

    print(f"Records : {len(records):,}")
    print()

    # Run audits
    audits = [audit_record(r) for r in records]

    # Print report
    print_report(audits, len(records), verbose=args.verbose)

    # Write cleaned output (records that passed all checks)
    if args.clean_output:
        passed_records = [r for r, a in zip(records, audits) if a["passed"]]
        with open(args.clean_output, "w", encoding="utf-8") as f:
            for r in passed_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Cleaned output : {args.clean_output}")
        print(f"Kept           : {len(passed_records):,} / {len(records):,} records")
        print()

    # Write audit log (one line per record with error details)
    if args.audit_log:
        with open(args.audit_log, "w", encoding="utf-8") as f:
            for a in audits:
                f.write(json.dumps(a, ensure_ascii=False) + "\n")
        print(f"Audit log      : {args.audit_log}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run 3 quality checks on an MCQ training JSONL file."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to the JSONL file to check."
    )
    parser.add_argument(
        "--clean-output", default=None,
        help="Write passing records to this JSONL file."
    )
    parser.add_argument(
        "--audit-log", default=None,
        help="Write per-record audit results to this JSONL file."
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show first 10 failed records with error details."
    )
    args = parser.parse_args()
    run(args)
