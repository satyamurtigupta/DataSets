# =============================================================================
# extract_prelims_mcq.py
# =============================================================================
# Extracts UPSC Prelims MCQs from the PYQ PDF and builds an SFT dataset.
#
# PDF has TWO formats:
#   Format A (year-wise, pages 8-118):
#       2025 questions → then 2025 answers+explanations
#       2024 questions → then 2024 answers+explanations
#       2023 questions → then 2023 answers+explanations
#
#   Format B (topic-wise, pages 119+):
#       Section A = Ancient & Medieval History
#       Section B = Modern History  ... etc.
#       Within each section: questions then answers on adjacent pages
#       Year embedded in question text as "(2021)", "(2016)" etc.
#
# Output:
#   mcq_raw.jsonl   -- every extracted Q+A+Explanation record
#   mcq_sft.jsonl   -- SFT training conversation pairs
#
# Usage:
#   python3 scripts/extract_prelims_mcq.py
#   python3 scripts/extract_prelims_mcq.py --dry-run
# =============================================================================

import re
import json
import os
import argparse
import random
from collections import defaultdict
from datetime import datetime

try:
    import fitz
except ImportError:
    raise ImportError("Install PyMuPDF: pip install pymupdf")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR    = "/Users/satyamurti/Downloads/DataSets"
PDF_PATH    = f"{BASE_DIR}/upsc_pdfs/Prelimns/Prelims PYQ (2011-2025).pdf"
OUT_DIR     = f"{BASE_DIR}/dataset_output_final/combined"
RAW_JSONL   = f"{OUT_DIR}/mcq_raw.jsonl"
SFT_JSONL   = f"{OUT_DIR}/mcq_sft.jsonl"

GEMMA_TEMPLATE = "<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n{answer}<end_of_turn>"

# ---------------------------------------------------------------------------
# Section → Subject mapping
# ---------------------------------------------------------------------------
SECTION_SUBJECT_MAP = {
    "A": "Ancient and Medieval History",
    "B": "Modern History",
    "C": "Art and Culture",
    "D": "Polity",
    "E": "Indian Economy",
    "F": "Environment and Ecology",
    "G": "Geography",
    "H": "Science and Technology",
    "I": "International Relations",
    "J": "Social Issues",
}

# Keywords in page headers that identify subject sections
SECTION_HEADER_MAP = {
    "ancient and medieval"   : "Ancient and Medieval History",
    "ancient history"        : "Ancient and Medieval History",
    "medieval history"       : "Ancient and Medieval History",
    "modern history"         : "Modern History",
    "art and culture"        : "Art and Culture",
    "art & culture"          : "Art and Culture",
    "polity"                 : "Polity",
    "indian economy"         : "Indian Economy",
    "economy"                : "Indian Economy",
    "environment and ecology": "Environment and Ecology",
    "environment"            : "Environment and Ecology",
    "geography"              : "Geography",
    "science and technology" : "Science and Technology",
    "science & technology"   : "Science and Technology",
    "international relations": "International Relations",
    "social issues"          : "Social Issues",
}

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------
# Year-paper header: "Prelims 2025 Question Paper"
YEAR_PAPER_RE   = re.compile(r'[Pp]relims\s+(20\d{2})\s+[Qq]uestion', re.IGNORECASE)

# Year tag inside question text: "(2021)" or "(2016)" — only 2011-2025
INLINE_YEAR_RE  = re.compile(r'\((201[1-9]|202[0-5])\)')

# Section letter — two formats found in this PDF:
#   Format 1 (topic-wise cover page): "A\nSection\n" — letter BEFORE the word "Section"
#   Format 2 (some pages): "Section A" or "Section A - Ancient..." — letter AFTER
SECTION_LETTER_RE     = re.compile(r'\bSection\s+([A-J])\b', re.IGNORECASE)
SECTION_LETTER_REV_RE = re.compile(r'^([A-J])\s*\n\s*Section', re.IGNORECASE | re.MULTILINE)

# Marks the start of the topic-wise section (page 119+)
TOPIC_SECTION_TRIGGER_RE = re.compile(
    r'previous\s+years?\s+questions?|topic.wise\s+questions?|SECTION\s*[:\-]?\s*[A-J]',
    re.IGNORECASE
)

# Question start: "  1.  Consider the following..."
QSTART_RE       = re.compile(r'^\s*(\d{1,3})[.)]\s+(.+)', re.DOTALL)

# Option line: "(a) Option text"
OPTION_RE       = re.compile(r'^\s*\(([a-dA-D1-4])\)\s+(.+)')

# Answer line in explanation section: "75.  (a)  The fiscal deficit..."
ANS_RE          = re.compile(r'^\s*(\d{1,3})\.\s+\(([a-dA-D](?:/[a-dA-D])?)\)\s*(.*)')

# Real UPSC question starters — sub-statements never begin with these
QUESTION_STARTERS = (
    "consider", "which", "with reference", "what", "who", "when", "where",
    "how", "in the context", "in respect", "regarding", "about", "among",
    "according", "as per", "the following", "select", "identify",
    "match", "arrange", "given below", "statements", "pairs",
)

def is_real_question(text: str) -> bool:
    """
    Return True if text looks like a real UPSC MCQ question.
    Return False only for obvious numbered sub-statements inside questions.

    Sub-statements look like:
      "Different kinds of surgical instruments were in common use by the 1st century AD"
      "Transplant of internal organs in the human body began in the 3rd century AD"

    Real questions look like:
      "Consider the following statements regarding..."
      "Which of the following is correct?"
      "With reference to ancient India..."
    """
    t = text.strip()

    # Too short to be a real UPSC question
    if len(t) < 50:
        return False

    t_lower = t.lower()

    # Has a question mark — definitely a real question
    if "?" in t:
        return True

    # Starts with a recognized question word/phrase
    for starter in QUESTION_STARTERS:
        if t_lower.startswith(starter):
            return True

    # Long enough to almost certainly be a real question
    if len(t) > 150:
        return True

    # Short text starting with lowercase — likely a sub-statement
    if t[0].islower():
        return False

    return True


# ---------------------------------------------------------------------------
# Step 1: Read all pages
# ---------------------------------------------------------------------------
def read_pdf(path: str) -> list:
    doc = fitz.open(path)
    pages = []
    for i in range(doc.page_count):
        text = doc[i].get_text()
        pages.append((i + 1, text))
    doc.close()
    return pages


# ---------------------------------------------------------------------------
# Step 2: Detect subject from page header lines
# ---------------------------------------------------------------------------
def detect_subject_from_header(text: str) -> str:
    first_lines = " ".join(text.split("\n")[:5]).lower()
    for keyword, subject in SECTION_HEADER_MAP.items():
        if keyword in first_lines:
            return subject
    return None


# ---------------------------------------------------------------------------
# Step 3: Extract questions from a block of text
# ---------------------------------------------------------------------------
def extract_questions_from_text(text: str, year: str,
                                subject: str, page_num: int) -> list:
    """
    Parse questions from a page of text.
    Returns list of question dicts.
    """
    questions = []
    lines = text.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        m = QSTART_RE.match(line)

        if m:
            q_num = int(m.group(1))
            if q_num > 200:   # skip page numbers
                i += 1
                continue

            q_lines = [m.group(2).strip()]

            # Extract inline year if present: (2021)
            inline_yr = INLINE_YEAR_RE.search(" ".join(q_lines))
            q_year = inline_yr.group(1) if inline_yr else year

            i += 1
            options = {}

            while i < len(lines):
                l = lines[i].strip()

                opt_m = OPTION_RE.match(l)
                if opt_m:
                    letter = opt_m.group(1).lower()
                    # normalize 1/2/3/4 to a/b/c/d (some sections use numbers)
                    if letter.isdigit():
                        letter = chr(ord('a') + int(letter) - 1)
                    opt_text = opt_m.group(2).strip()
                    # option may continue on next lines
                    j = i + 1
                    while j < len(lines):
                        nl = lines[j].strip()
                        if OPTION_RE.match(nl) or QSTART_RE.match(nl) or ANS_RE.match(nl):
                            break
                        if nl:
                            opt_text += " " + nl
                        j += 1
                    options[letter] = opt_text.strip()
                    i = j
                elif QSTART_RE.match(l) or ANS_RE.match(l):
                    break
                else:
                    if l:
                        q_lines.append(l)
                    i += 1

            # Clean question text - remove inline year tags
            q_text = " ".join(q_lines)
            q_text = INLINE_YEAR_RE.sub("", q_text).strip()
            q_text = re.sub(r'\s+', ' ', q_text).strip()

            if len(q_text) < 20 or len(options) < 2:
                continue
            # Filter out sub-statements that look like numbered list items
            if not is_real_question(q_text):
                continue

            questions.append({
                "q_num"   : q_num,
                "year"    : q_year,
                "subject" : subject,
                "question": q_text,
                "options" : options,
                "page"    : page_num,
            })
        else:
            i += 1

    return questions


# ---------------------------------------------------------------------------
# Step 4: Extract answers from a block of text
# ---------------------------------------------------------------------------
def extract_answers_from_text(text: str, year: str) -> dict:
    """
    Parse answer+explanation lines from a page.
    Returns dict: {q_num: {"correct": "a", "explanation": "..."}}
    """
    answers = {}
    lines   = text.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i]
        m = ANS_RE.match(line)
        if m:
            q_num   = int(m.group(1))
            correct = m.group(2).lower().split("/")[0]
            expl    = [m.group(3).strip()] if m.group(3).strip() else []
            i += 1

            while i < len(lines):
                l = lines[i]
                if ANS_RE.match(l):
                    break
                if l.strip():
                    expl.append(l.strip())
                i += 1

            explanation = " ".join(expl).strip()
            explanation = re.sub(r'\s+', ' ', explanation)
            explanation = explanation.replace(
                "PW ONLYIAS SUPER HINT", "\n\nExam Tip:"
            )

            answers[q_num] = {
                "correct"    : correct,
                "explanation": explanation,
            }
        else:
            i += 1

    return answers


# ---------------------------------------------------------------------------
# Step 5: Main extraction — walk all pages with context tracking
# ---------------------------------------------------------------------------
def extract_all(pages: list) -> list:
    """
    Walk all pages, track year/subject context, extract Q+A.
    Returns list of complete MCQ records.
    """
    current_year     = "Unknown"
    current_subject  = "General Studies"
    current_section  = None
    in_topic_section = False   # True once we pass into Section A / topic-wise part

    # Temporary storage: questions waiting for answers
    pending_questions = {}   # key: (year_str, q_num) -> question dict
    final_records     = []

    def flush_answers(answers: dict, year: str):
        """
        Match pending questions with newly found answers.
        First tries exact (year, q_num) match.
        Falls back to any pending question with same q_num —
        needed for topic-wise sections where inline year in question
        differs from page-header year on answer pages.
        """
        for q_num, ans in answers.items():
            key = (year, q_num)
            q   = None

            if key in pending_questions:
                q = pending_questions.pop(key)
            else:
                # Fallback: find any pending question with this q_num
                # Take the one with the closest page to current answer page
                candidates = [(k, v) for k, v in pending_questions.items()
                              if k[1] == q_num]
                if candidates:
                    # Pick the one with smallest page difference
                    best_key = min(candidates, key=lambda x: x[0])[0]
                    q = pending_questions.pop(best_key)

            if q is None:
                continue

            q_year = q["year"] if q["year"] != "Unknown" else year
            final_records.append({
                "id"         : f"prelims_{q_year}_q{q_num:03d}_{q['subject'][:4].lower()}",
                "year"       : q_year,
                "subject"    : q["subject"],
                "q_num"      : q_num,
                "question"   : q["question"],
                "options"    : q["options"],
                "correct"    : ans["correct"],
                "explanation": ans["explanation"],
            })

    for page_num, text in pages:
        if not text.strip():
            continue

        # Detect year paper header
        yr_m = YEAR_PAPER_RE.search(text)
        if yr_m:
            current_year = yr_m.group(1)

        # Detect section letter — check both "Section A" and "A\nSection" formats.
        # Also check full page text (not just top 200) for the reversed format,
        # since the letter and word appear on separate lines.
        page_top = text[:300]
        sec_m = SECTION_LETTER_RE.search(page_top)
        if sec_m is None:
            sec_m = SECTION_LETTER_REV_RE.search(page_top)

        if sec_m:
            sec_letter = sec_m.group(1).upper()
            if sec_letter in SECTION_SUBJECT_MAP:
                current_section  = sec_letter
                current_subject  = SECTION_SUBJECT_MAP[sec_letter]
                in_topic_section = True   # crossed into topic-wise section

        # Also trigger topic-wise mode if "Previous Years Questions" appears
        # even on pages where the section letter is not in the top 300 chars
        if not in_topic_section and TOPIC_SECTION_TRIGGER_RE.search(text[:400]):
            in_topic_section = True

        # Detect subject from header keywords
        hdr_subj = detect_subject_from_header(text)
        if hdr_subj:
            current_subject = hdr_subj

        # Count option lines vs answer lines to classify page
        opt_lines = sum(1 for l in text.split("\n") if OPTION_RE.match(l))
        ans_lines = sum(1 for l in text.split("\n") if ANS_RE.match(l))

        if opt_lines > ans_lines and opt_lines >= 2:
            # In topic-wise sections, use "2011-2022" as year for questions
            # that don't have an inline year tag (extract_questions_from_text
            # will override with inline year if it finds one)
            year_for_q = "2011-2022" if in_topic_section else current_year
            # Question page
            qs = extract_questions_from_text(
                text, year_for_q, current_subject, page_num
            )
            for q in qs:
                key = (q["year"], q["q_num"])
                pending_questions[key] = q

        if ans_lines >= 2:
            # Answer page — flush matching questions
            ans = extract_answers_from_text(text, current_year)
            flush_answers(ans, current_year)

            # For topic-wise sections, year in answer text may differ
            # Also try matching with "Unknown" year (topic-wise without year tag)
            if current_year != "Unknown":
                ans_unknown = {k: v for k, v in ans.items()
                               if ("Unknown", k) in pending_questions}
                if ans_unknown:
                    flush_answers(ans_unknown, "Unknown")

    return final_records


# ---------------------------------------------------------------------------
# Step 6: Deduplicate (same question may appear in year-wise AND topic-wise)
# ---------------------------------------------------------------------------
def deduplicate(records: list) -> list:
    seen = set()
    out  = []
    for r in records:
        # Fingerprint: first 80 chars of question text
        fp = (r["q_num"], r["year"], r["question"][:80])
        if fp not in seen:
            seen.add(fp)
            out.append(r)
    return out


# ---------------------------------------------------------------------------
# Step 7: Build SFT conversation dataset
# ---------------------------------------------------------------------------
REQUEST_VARIANTS = {
    "by_subject": [
        "Give me {n} MCQ questions on {subject} from UPSC Prelims.",
        "Generate {n} multiple choice questions on {subject} for UPSC practice.",
        "I want to practice {n} MCQs on {subject}.",
        "Create a {n}-question MCQ set on {subject} for UPSC Prelims preparation.",
        "Give {n} UPSC Prelims practice questions on {subject} with explanations.",
        "Practice {n} MCQs on {subject} from UPSC Civil Services Preliminary exam.",
    ],
    "by_year": [
        "Give me {n} questions from UPSC Prelims {year}.",
        "Show me {n} MCQs from the {year} UPSC Prelims paper.",
        "I want {n} practice questions from Prelims {year}.",
        "Give me {n} questions from the {year} Civil Services Preliminary exam.",
    ],
    "by_year_subject": [
        "Give me {n} {subject} questions from UPSC Prelims {year}.",
        "Show {n} {year} Prelims MCQs on {subject}.",
        "Practice {n} questions on {subject} from the {year} Prelims paper.",
        "I want {n} MCQs from Prelims {year} on {subject}.",
    ],
    "quiz_mode": [
        "Quiz me with {n} MCQ questions on {subject}. Show questions first, answers at end.",
        "Test me on {subject} with {n} MCQs. Give questions first then reveal answers.",
        "Give me {n} {subject} MCQs for practice. Questions only, I will try them first.",
    ],
    "explain": [
        "Explain the answer to this UPSC Prelims question:\n\n{q_text}",
        "What is the correct answer and explanation for this question:\n\n{q_text}",
        "Why is option ({correct}) correct for this UPSC Prelims question:\n\n{q_text}",
    ],
    "single_with_answer": [
        "What is the answer to this UPSC Prelims question:\n\n{q_text}",
        "Solve this UPSC Prelims MCQ:\n\n{q_text}",
        "Answer this question from UPSC Prelims:\n\n{q_text}",
    ],
}


def format_question(r: dict, show_answer: bool = True,
                    show_explanation: bool = True) -> str:
    """Format a single MCQ record as clean readable text."""
    opts  = r["options"]
    lines = [
        f"Q{r['q_num']}. [{r['year']} | {r['subject']}]",
        "",
        r["question"],
        "",
    ]
    for letter in ["a", "b", "c", "d"]:
        if letter in opts:
            lines.append(f"({letter}) {opts[letter]}")

    if show_answer:
        lines.append("")
        lines.append(f"Correct Answer: ({r['correct']})")

    if show_explanation and r.get("explanation"):
        lines.append("")
        lines.append(f"Explanation: {r['explanation'][:600]}")

    return "\n".join(lines)


def format_many(records: list, show_answers: bool = True,
                show_explanations: bool = True) -> str:
    sep   = "\n\n" + "-" * 50 + "\n\n"
    parts = [format_question(r, show_answers, show_explanations)
             for r in records]
    return "\n\n" + sep.join(parts)


def build_sft(records: list) -> list:
    sft    = []
    rec_id = [0]

    def add(user: str, model: str, subject: str, fmt: str):
        rec_id[0] += 1
        sft.append({
            "id"     : f"mcq_sft_{rec_id[0]:05d}",
            "text"   : GEMMA_TEMPLATE.format(question=user, answer=model),
            "subject": subject,
            "format" : fmt,
        })

    by_subj      = defaultdict(list)
    by_yr        = defaultdict(list)
    by_yr_subj   = defaultdict(list)

    for r in records:
        by_subj[r["subject"]].append(r)
        by_yr[r["year"]].append(r)
        by_yr_subj[(r["year"], r["subject"])].append(r)

    # --- Generate by subject ---
    for n in [3, 5, 10]:
        for subj, recs in by_subj.items():
            if len(recs) < n:
                continue
            sample = random.sample(recs, n)
            user   = random.choice(REQUEST_VARIANTS["by_subject"]).format(
                n=n, subject=subj
            )
            model  = (
                f"Here are {n} UPSC Prelims MCQ questions on {subj}:\n"
                + format_many(sample)
            )
            add(user, model, subj, "by_subject")

    # --- Generate by year ---
    for n in [3, 5, 10]:
        for yr, recs in by_yr.items():
            if yr == "Unknown" or len(recs) < n:
                continue
            sample = random.sample(recs, n)
            user   = random.choice(REQUEST_VARIANTS["by_year"]).format(
                n=n, year=yr
            )
            model  = (
                f"Here are {n} questions from UPSC Prelims {yr}:\n"
                + format_many(sample)
            )
            add(user, model, "Mixed", "by_year")

    # --- Generate by year + subject ---
    for n in [3, 5]:
        for (yr, subj), recs in by_yr_subj.items():
            if yr == "Unknown" or len(recs) < n:
                continue
            sample = random.sample(recs, n)
            user   = random.choice(REQUEST_VARIANTS["by_year_subject"]).format(
                n=n, year=yr, subject=subj
            )
            model  = (
                f"Here are {n} {subj} questions from UPSC Prelims {yr}:\n"
                + format_many(sample)
            )
            add(user, model, subj, "by_year_subject")

    # --- Quiz mode (questions first, answers at end) ---
    for n in [3, 5]:
        for subj, recs in by_subj.items():
            if len(recs) < n:
                continue
            sample = random.sample(recs, n)
            user   = random.choice(REQUEST_VARIANTS["quiz_mode"]).format(
                n=n, subject=subj
            )
            q_part   = format_many(sample, show_answers=False,
                                   show_explanations=False)
            ans_part = "\n\nAnswer Key:\n" + "\n".join(
                f"Q{r['q_num']}: ({r['correct']}) -- "
                f"{r['explanation'][:200]}"
                for r in sample
            )
            model = (
                f"Here are {n} {subj} MCQs. Try them first:\n"
                + q_part + ans_part
            )
            add(user, model, subj, "quiz_mode")

    # --- Explain answer (single Q) ---
    explain_pool = random.sample(records, min(len(records), 800))
    for r in explain_pool:
        q_text = format_question(r, show_answer=False, show_explanation=False)

        # explain format
        user  = random.choice(REQUEST_VARIANTS["explain"]).format(
            q_text=q_text, correct=r["correct"]
        )
        model = (
            f"The correct answer is ({r['correct']}).\n\n"
            f"Explanation:\n{r['explanation']}"
        )
        add(user, model, r["subject"], "explain")

        # single answer format
        user2  = random.choice(REQUEST_VARIANTS["single_with_answer"]).format(
            q_text=q_text
        )
        model2 = (
            f"Answer: ({r['correct']})\n\n"
            f"Explanation: {r['explanation'][:400]}"
        )
        add(user2, model2, r["subject"], "single_answer")

    return sft


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf",     default=PDF_PATH)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Reading: {args.pdf}")
    pages = read_pdf(args.pdf)
    print(f"Total pages: {len(pages)}")

    print("Extracting questions and answers...")
    records = extract_all(pages)
    print(f"Records before dedup : {len(records):,}")

    records = deduplicate(records)
    print(f"Records after dedup  : {len(records):,}")

    # Stats
    by_yr   = defaultdict(int)
    by_subj = defaultdict(int)
    for r in records:
        by_yr[r["year"]] += 1
        by_subj[r["subject"]] += 1

    print()
    print("By year:")
    for yr, cnt in sorted(by_yr.items()):
        print(f"  {yr:<10} {cnt:>5}")

    print()
    print("By subject:")
    for subj, cnt in sorted(by_subj.items(), key=lambda x: -x[1]):
        print(f"  {subj:<35} {cnt:>5}")

    if args.dry_run:
        print("\nDry run -- no files written")
        return

    # Write raw
    raw_path = os.path.join(args.out_dir, "mcq_raw.jsonl")
    with open(raw_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nRaw MCQs written: {raw_path}  ({len(records):,} records)")

    # Build and write SFT
    random.seed(42)
    sft = build_sft(records)

    fmt_counts = defaultdict(int)
    for r in sft:
        fmt_counts[r["format"]] += 1

    print(f"SFT records : {len(sft):,}")
    print("Format breakdown:")
    for fmt, cnt in sorted(fmt_counts.items(), key=lambda x: -x[1]):
        print(f"  {fmt:<25} {cnt:>6,}")

    sft_path = os.path.join(args.out_dir, "mcq_sft.jsonl")
    with open(sft_path, "w", encoding="utf-8") as f:
        for r in sft:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"SFT dataset written : {sft_path}")

    # Show one sample
    sample = random.choice([r for r in sft if r["format"] == "by_subject"])
    print()
    print("=" * 60)
    print("Sample SFT record:")
    print("=" * 60)
    print(sample["text"][:1200])
    print("...")


if __name__ == "__main__":
    main()
