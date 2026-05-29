# =============================================================================
# pretrain_quality_check.py
# =============================================================================
# NLP data quality checks on unified_pretrain.jsonl (NCERT text chunks).
#
# 11 checks (no heavy ML dependencies — runs on plain Python 3.9):
#
#  NLP-1   Language coherence    — bigram repetition score detects looping/OCR
#  NLP-2   Sentence structure    — avg sentence length flags fragmented text
#  NLP-3   Vocabulary richness   — type-token ratio flags copy-pasted boilerplate
#  NLP-4   OCR noise score       — symbol density: maths, arrows, garbled chars
#  NLP-5   Alphabetic ratio      — low alpha% = tables, answer keys, equations
#  NLP-6   Near-duplicate        — MD5 of first 200 chars catches copy chunks
#  NLP-7   Boundary coherence    — chunk starts mid-sentence (lowercase first word)
#  NLP-8   Heading coherence     — heading is equation ref, single char, or chunk ID
#  NLP-9   Single-char density   — >15% single-letter tokens = diagram/figure OCR
#  NLP-10  Comma cluster         — 3+ commas in a row = dotted-line / table OCR
#  NLP-11  Garbled word ratio    — words with impossible consonant clusters (OCR noise)
#
# Usage:
#   python3 scripts/pretrain_quality_check.py
#   python3 scripts/pretrain_quality_check.py --clean-output cleaned_pretrain.jsonl
#   python3 scripts/pretrain_quality_check.py --verbose --subject History
#   python3 scripts/pretrain_quality_check.py --report-only   # print report, no file
# =============================================================================

import json
import os
import re
import sys
import hashlib
import argparse
from collections import Counter, defaultdict
from typing import Optional, List, Tuple

INPUT_PATH  = "/Users/satyamurti/Downloads/DataSets/dataset_output_final/combined/unified_pretrain.jsonl"
OUTPUT_DIR  = "/Users/satyamurti/Downloads/DataSets/dataset_output_final/combined"

# ---------------------------------------------------------------------------
# NLP helpers
# ---------------------------------------------------------------------------

# OCR / math symbol patterns that do not belong in natural-language NCERT text
_OCR_SYMBOLS = re.compile(
    r'\|\)|'           # |)
    r'\|\(|'           # |(
    r'<[A-Z\']{2,}>|'  # <MCRO>, <BI'>
    r'[~]{3,}|'        # ~~~
    r'[=]{5,}|'        # =====
    r'\x00|'           # null byte
    r'[•‣◦]{3,}|'  # bullet clusters
    r'[^\x00-\x7F]{4,}'           # long runs of non-ASCII (garbled OCR)
)

# Equation / formula heading: "14.7", "10.6 MICRO...", "6.4.2"
_EQUATION_HEADING = re.compile(r'^\d+[\.\d]+\s*')

# Answer-key pattern: chunks that are just MCQ answer lists
_ANSWER_KEY = re.compile(r'^\s*(\d+[\.\)]\s*[\(\[]?[a-dA-D][\)\]]?\s*){5,}')

# Chemistry / physics formula dense text (many digits, brackets, superscripts)
_FORMULA_DENSE = re.compile(r'(\d+[\^]?\d*\s*[×x\*]\s*\d+|10[−\-]\d+|kJ\s*mol|m\s*s[−\-]1)', re.IGNORECASE)


def bigram_repetition_score(text: str) -> float:
    """
    Returns 0.0 (no repetition) to 1.0 (fully repetitive).
    High score = text has the same bigrams cycling over and over.
    Threshold for rejection: > 0.35
    """
    words = text.lower().split()
    if len(words) < 20:
        return 0.0
    bigrams = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
    return 1.0 - len(set(bigrams)) / len(bigrams)


def avg_sentence_length(text: str) -> float:
    """
    Average word count per sentence.
    < 5  = fragmented (tables, bullet points, answer keys)
    > 40 = overly long (rare in NCERT)
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s for s in sentences if len(s.split()) > 2]
    if not sentences:
        return 0.0
    return sum(len(s.split()) for s in sentences) / len(sentences)


def type_token_ratio(text: str) -> float:
    """
    Unique words / total words.
    Very low (<0.30) = repetitive boilerplate or looping OCR.
    Healthy NCERT text: 0.45-0.70
    """
    words = re.findall(r'\b[a-z]{2,}\b', text.lower())
    if len(words) < 20:
        return 1.0
    return len(set(words)) / len(words)


def alpha_ratio(text: str) -> float:
    """Fraction of characters that are alphabetic."""
    if not text:
        return 0.0
    return sum(1 for ch in text if ch.isalpha()) / len(text)


def ocr_noise_score(text: str) -> float:
    """
    Count OCR/math noise patterns per 100 chars.
    > 2.0 per 100 chars = too noisy.
    """
    hits = len(_OCR_SYMBOLS.findall(text))
    return hits / max(len(text), 1) * 100


def chunk_signature(text: str) -> str:
    """MD5 of first 200 normalised chars — for near-duplicate detection."""
    normalised = re.sub(r'\s+', ' ', text[:200].strip().lower())
    return hashlib.md5(normalised.encode()).hexdigest()


def starts_mid_sentence(text: str) -> bool:
    """True if the chunk starts with a lowercase letter (boundary error)."""
    stripped = text.lstrip()
    if not stripped:
        return False
    # Allow digits (e.g. numbered list items that are mid-chunk)
    return stripped[0].islower()


def heading_is_bad(heading: str) -> Optional[str]:
    """
    Returns the failure reason if the heading is not a real section title.
    Returns None if the heading is fine.
    """
    if not heading or len(heading.strip()) < 2:
        return "empty"
    h = heading.strip()
    if re.match(r'^[A-Za-z]\s*$', h):
        return "single_char"
    if _EQUATION_HEADING.match(h):
        return "equation_ref"
    if re.search(r'[_]{2,}', h):
        return "chunk_id_format"
    if re.search(r'[<>~\|]{2,}', h):
        return "ocr_artifact"
    if re.search(r'[^\x00-\x7F]{5,}', h):
        return "non_ascii_garbage"
    return None


def is_answer_key(text: str) -> bool:
    """True if chunk is just an MCQ answer list (1. (a) 2. (b) ...)."""
    return bool(_ANSWER_KEY.match(text.strip()))


def is_formula_dense(text: str) -> bool:
    """True if chunk is dominated by equations/formulas, not prose."""
    hits = len(_FORMULA_DENSE.findall(text))
    words = len(text.split())
    return hits > 0 and (hits / max(words, 1)) > 0.05


def single_char_density(text: str) -> float:
    """
    NLP-9: Fraction of whitespace-split tokens that are a single letter.

    Normal prose: < 5% (articles like 'a', 'I' are normal).
    Diagram / figure OCR: > 15% because the OCR reads lines of dots/dashes
    as repeated single letters: 't, t, t, t, t...' or 'I I I I I'.

    The biology chunk example had 44% single-char tokens.
    Threshold for rejection: > 0.15
    """
    tokens = text.split()
    if not tokens:
        return 0.0
    singles = sum(1 for t in tokens if len(re.sub(r'[^a-zA-Z]', '', t)) == 1)
    return singles / len(tokens)


def comma_cluster_count(text: str) -> int:
    """
    NLP-10: Count of runs of 3 or more commas (e.g. ",,,,,,,").

    This pattern comes from OCR of dotted lines, table borders, or
    figure separators. Any occurrence >= 1 is suspicious; >= 2 is rejection.
    'IL,,,,,,,,,,,,,,,,,, J,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,' had 2 clusters.

    Threshold for rejection: >= 2 clusters
    """
    return len(re.findall(r',{3,}', text))


# Consonant clusters that cannot appear in English words:
# e.g. 'mcrospores', 'Tr.cmsve:rse', 'an.U:er', 'nDllier', 'woll!ayors'
# Pattern: 5+ consonants in sequence with no vowel
_CONSONANT_CLUSTER = re.compile(r'\b[^aeiouAEIOU\W\d_]{5,}\b')

# Valid long consonant starts (legitimate English words like "strength", "through")
_VALID_CONSONANT_WORDS = {
    'strength', 'through', 'growth', 'stress', 'strands', 'structure',
    'straight', 'string', 'strip', 'strict', 'strong', 'stiff', 'swift',
    'lymph', 'rhythm', 'myth', 'glyph', 'crypt', 'graft', 'craft', 'draft',
    'shift', 'shelf', 'flesh', 'fresh', 'press', 'dress', 'class', 'grass',
    'cross', 'bless', 'glass', 'blood', 'flood', 'floor', 'clock', 'block',
}


def garbled_word_ratio(text: str) -> float:
    """
    NLP-11: Fraction of words that look garbled (impossible consonant clusters).

    OCR garbles words like:
      'microspores'  ->  'mcrospores'     (vowel dropped)
      'transverse'   ->  'Tr.cmsve:rse'  (punctuation inserted mid-word)
      'another'      ->  'an.U:er'       (OCR misread)

    These words have 5+ consonants with no vowel — impossible in English.
    Threshold for rejection: > 5% of words are garbled
    """
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text)
    if not words:
        return 0.0
    garbled = [
        w for w in words
        if _CONSONANT_CLUSTER.match(w) and w.lower() not in _VALID_CONSONANT_WORDS
    ]
    return len(garbled) / len(words)


# ---------------------------------------------------------------------------
# Per-chunk audit
# ---------------------------------------------------------------------------

THRESHOLDS = {
    "bigram_rep"   : 0.35,   # fail if > this
    "single_char_density" : 0.15,   # fail if > this (diagram OCR)
    "comma_clusters"      : 2,      # fail if >= this many clusters
    "garbled_word_ratio"  : 0.05,   # fail if > this (5% garbled words)
    "avg_sent_len" : 5.0,    # fail if < this (fragmented)
    "ttr"          : 0.30,   # fail if < this (repetitive)
    "alpha_ratio"  : 0.45,   # fail if < this (too many symbols)
    "ocr_noise"    : 2.0,    # fail if > this per 100 chars
    "word_count"   : 60,     # fail if < this (too short)
}


def audit_chunk(chunk: dict, sig_counter: Counter) -> dict:
    """Run all NLP checks on a single chunk. Return audit dict."""
    text    = chunk.get("text", "").strip()
    heading = chunk.get("section_heading", "") or ""
    cid     = chunk.get("id", "unknown")
    subject = chunk.get("subject", "unknown")
    wc      = len(text.split())

    failures = []

    # NLP-1: Bigram repetition
    br = bigram_repetition_score(text)
    if br > THRESHOLDS["bigram_rep"]:
        failures.append(f"NLP1_HIGH_REPETITION:{br:.2f}")

    # NLP-2: Sentence structure
    asl = avg_sentence_length(text)
    if 0 < asl < THRESHOLDS["avg_sent_len"]:
        failures.append(f"NLP2_FRAGMENTED_SENTENCES:{asl:.1f}_words_avg")

    # NLP-3: Vocabulary richness
    ttr = type_token_ratio(text)
    if ttr < THRESHOLDS["ttr"]:
        failures.append(f"NLP3_LOW_VOCABULARY_RICHNESS:{ttr:.2f}")

    # NLP-4: OCR noise
    ons = ocr_noise_score(text)
    if ons > THRESHOLDS["ocr_noise"]:
        failures.append(f"NLP4_OCR_NOISE:{ons:.2f}_per_100chars")

    # NLP-5: Alphabetic ratio
    ar = alpha_ratio(text)
    if ar < THRESHOLDS["alpha_ratio"]:
        failures.append(f"NLP5_LOW_ALPHA_RATIO:{ar:.2f}")

    # NLP-6: Near-duplicate
    sig = chunk_signature(text)
    if sig_counter[sig] > 1:
        failures.append(f"NLP6_NEAR_DUPLICATE")

    # NLP-7: Boundary coherence
    if starts_mid_sentence(text):
        failures.append("NLP7_STARTS_MID_SENTENCE")

    # NLP-8: Heading coherence
    hbad = heading_is_bad(heading)
    if hbad:
        failures.append(f"NLP8_BAD_HEADING:{hbad}")

    # NLP-9: Single-char token density (diagram/figure OCR)
    scd = single_char_density(text)
    if scd > THRESHOLDS["single_char_density"]:
        failures.append(f"NLP9_DIAGRAM_OCR:{scd:.2f}_single_char_density")

    # NLP-10: Comma clusters (dotted-line / table border OCR)
    cc = comma_cluster_count(text)
    if cc >= THRESHOLDS["comma_clusters"]:
        failures.append(f"NLP10_COMMA_CLUSTERS:{cc}_runs")

    # NLP-11: Garbled word ratio (consonant clusters that can't exist in English)
    gwr = garbled_word_ratio(text)
    if gwr > THRESHOLDS["garbled_word_ratio"]:
        failures.append(f"NLP11_GARBLED_WORDS:{gwr:.2f}_ratio")

    # Bonus: content type flags (informational, not rejection criteria by default)
    flags = []
    if is_answer_key(text):
        flags.append("ANSWER_KEY")
    if is_formula_dense(text):
        flags.append("FORMULA_DENSE")
    if wc < THRESHOLDS["word_count"]:
        flags.append(f"SHORT:{wc}_words")

    return {
        "id"           : cid,
        "subject"      : subject,
        "word_count"   : wc,
        "passed"       : len(failures) == 0,
        "failures"     : failures,
        "flags"        : flags,
        "metrics"      : {
            "bigram_rep"        : round(br, 3),
            "avg_sent_len"      : round(asl, 1),
            "ttr"               : round(ttr, 3),
            "alpha_ratio"       : round(ar, 3),
            "ocr_noise"         : round(ons, 3),
            "single_char_density": round(scd, 3),
            "comma_clusters"    : cc,
            "garbled_word_ratio": round(gwr, 3),
        },
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(audits: List[dict], total: int, verbose: bool = False):
    passed  = sum(1 for a in audits if a["passed"])
    failed  = total - passed

    print()
    print("=" * 70)
    print("  PRETRAIN DATA — NLP QUALITY REPORT")
    print("=" * 70)
    print(f"  Total chunks       : {total:,}")
    print(f"  PASSED all checks  : {passed:,}  ({100*passed/total:.1f}%)")
    print(f"  FAILED >= 1 check  : {failed:,}  ({100*failed/total:.1f}%)")
    print()

    # Per-check breakdown
    check_counts = Counter()
    for a in audits:
        for f in a["failures"]:
            code = f.split(":")[0]
            check_counts[code] += 1

    checks = [
        ("NLP1_HIGH_REPETITION",         "NLP-1   Bigram repetition (looping text)"),
        ("NLP2_FRAGMENTED_SENTENCES",    "NLP-2   Fragmented sentences (<5 words avg)"),
        ("NLP3_LOW_VOCABULARY_RICHNESS", "NLP-3   Low vocabulary richness (TTR < 0.30)"),
        ("NLP4_OCR_NOISE",               "NLP-4   OCR noise (symbols, artifacts)"),
        ("NLP5_LOW_ALPHA_RATIO",         "NLP-5   Low alphabetic ratio (tables/equations)"),
        ("NLP6_NEAR_DUPLICATE",          "NLP-6   Near-duplicate chunk"),
        ("NLP7_STARTS_MID_SENTENCE",     "NLP-7   Starts mid-sentence (bad boundary)"),
        ("NLP8_BAD_HEADING",             "NLP-8   Bad section heading"),
        ("NLP9_DIAGRAM_OCR",             "NLP-9   Diagram/figure OCR (>15% single-char tokens)"),
        ("NLP10_COMMA_CLUSTERS",         "NLP-10  Comma clusters (dotted-line OCR)"),
        ("NLP11_GARBLED_WORDS",          "NLP-11  Garbled words (impossible consonant clusters)"),
    ]

    print("  Per-check failure count:")
    for code, label in checks:
        cnt = check_counts.get(code, 0)
        bar = "#" * min(int(cnt / total * 40), 40)
        print(f"    {label}")
        print(f"      {cnt:5d} / {total}  ({100*cnt/total:.1f}%)  {bar}")
        print()

    # Subject breakdown of failures
    subj_fail = Counter()
    subj_total = Counter()
    for a in audits:
        subj_total[a["subject"]] += 1
        if not a["passed"]:
            subj_fail[a["subject"]] += 1

    print("  Failure rate by subject:")
    for subj in sorted(subj_total, key=lambda s: -subj_fail.get(s, 0)):
        t = subj_total[subj]
        f = subj_fail.get(subj, 0)
        print(f"    {subj:20s}  {f:4d} / {t:4d}  ({100*f/max(t,1):.1f}%)")
    print()

    # Content flag counts (informational)
    flag_counts = Counter()
    for a in audits:
        for fl in a["flags"]:
            flag_counts[fl.split(":")[0]] += 1

    if flag_counts:
        print("  Informational flags (not rejection criteria):")
        for flag, cnt in flag_counts.most_common():
            print(f"    {flag:30s}: {cnt:5d} ({100*cnt/total:.1f}%)")
        print()

    # Metric distribution summary
    metrics_list = [a["metrics"] for a in audits]
    print("  Metric distribution (all chunks):")
    for key, label in [
        ("bigram_rep",   "Bigram rep score  (higher=worse, reject >0.35)"),
        ("ttr",          "Type-token ratio  (lower=worse, reject <0.30)"),
        ("alpha_ratio",  "Alpha ratio       (lower=worse, reject <0.45)"),
        ("avg_sent_len", "Avg sentence len  (lower=worse, reject <5.0)"),
    ]:
        vals = sorted(m[key] for m in metrics_list)
        n = len(vals)
        print(f"    {label}")
        print(f"      min={vals[0]:.2f}  p10={vals[n//10]:.2f}  "
              f"p50={vals[n//2]:.2f}  p90={vals[9*n//10]:.2f}  max={vals[-1]:.2f}")
        print()

    if verbose:
        print("  Worst 15 chunks by failure count:")
        worst = sorted(audits, key=lambda a: len(a["failures"]), reverse=True)[:15]
        for a in worst:
            if not a["failures"]:
                break
            print(f"    [{a['id']}]  subject={a['subject']}")
            for f in a["failures"]:
                print(f"      - {f}")
        print()

    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace):
    path = args.input

    print(f"Loading : {path}")
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  WARNING: bad JSON line — {e}")

    if args.subject:
        chunks = [c for c in chunks if c.get("subject", "").lower() == args.subject.lower()]
        print(f"Filtered to subject '{args.subject}': {len(chunks)} chunks")

    print(f"Chunks  : {len(chunks):,}")

    # Pre-compute signature counts for near-duplicate detection
    print("Computing signatures for duplicate detection...")
    sig_counter: Counter = Counter()
    for c in chunks:
        sig_counter[chunk_signature(c.get("text", ""))] += 1

    print("Running NLP checks...")
    audits = [audit_chunk(c, sig_counter) for c in chunks]

    print_report(audits, len(chunks), verbose=args.verbose)

    # Write clean output
    if args.clean_output and not args.report_only:
        passed_chunks = [c for c, a in zip(chunks, audits) if a["passed"]]
        out_path = args.clean_output
        with open(out_path, "w", encoding="utf-8") as f:
            for c in passed_chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"\nClean output   : {out_path}")
        print(f"Kept           : {len(passed_chunks):,} / {len(chunks):,} chunks")

        # Show subject breakdown of what was kept
        kept_subj = Counter(c.get("subject", "unknown") for c in passed_chunks)
        orig_subj = Counter(c.get("subject", "unknown") for c in chunks)
        print("\n  Kept per subject:")
        for subj in sorted(orig_subj, key=lambda s: -orig_subj[s]):
            k = kept_subj.get(subj, 0)
            t = orig_subj[subj]
            print(f"    {subj:20s}: {k:4d} / {t:4d}  ({100*k/max(t,1):.1f}% kept)")

    # Write audit log
    if args.audit_log and not args.report_only:
        with open(args.audit_log, "w", encoding="utf-8") as f:
            for a in audits:
                f.write(json.dumps(a, ensure_ascii=False) + "\n")
        print(f"Audit log      : {args.audit_log}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NLP quality checks on NCERT pretrain chunks."
    )
    parser.add_argument(
        "--input", default=INPUT_PATH,
        help="Path to the pretrain JSONL file."
    )
    parser.add_argument(
        "--clean-output",
        default=os.path.join(OUTPUT_DIR, "unified_pretrain_clean.jsonl"),
        help="Write passing chunks to this file."
    )
    parser.add_argument(
        "--audit-log",
        default=os.path.join(OUTPUT_DIR, "pretrain_audit_log.jsonl"),
        help="Write per-chunk audit results to this file."
    )
    parser.add_argument(
        "--subject", default=None,
        help="Filter to one subject (e.g. History)."
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show worst 15 failing chunks with details."
    )
    parser.add_argument(
        "--report-only", action="store_true",
        help="Print report only — do not write any output files."
    )
    args = parser.parse_args()
    run(args)
