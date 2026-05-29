# =============================================================================
# extract_real_mcq_from_pdfs.py
# =============================================================================
# Extracts actual UPSC Prelims MCQs from VisionIAS solution PDFs (2011-2025)
# and builds two training files:
#
#   real_upsc_mcq_raw.jsonl    -- one record per MCQ with full metadata
#   real_upsc_mcq_sft.jsonl    -- instruction-input-output SFT pairs (Alpaca format)
#
# WHY a separate file from synthetic:
#   Synthetic MCQs are generated from NCERT text to build domain knowledge.
#   Real MCQs are ACTUAL UPSC questions — they have a different character:
#   harder, more nuanced traps, real source links, and official answer keys.
#   The SLM must see both: synthetic builds breadth, real MCQs build precision.
#
# VisionIAS PDF structure (consistent 2016-2025):
#   Each question row in a table: Q.No | Subject | Question text | Answer | Explanation
#   Additional inline metadata per question: difficulty (E/M/D), nature (F/FA/CA/CAA/FCA/U),
#   source (EM/EN/RM/RR), motivation text
#   This script extracts all of these fields.
#
# Usage:
#   # Extract all years
#   python3 scripts/extract_real_mcq_from_pdfs.py
#
#   # Single year
#   python3 scripts/extract_real_mcq_from_pdfs.py --year 2025
#
#   # Dry run — print sample records, do not write files
#   python3 scripts/extract_real_mcq_from_pdfs.py --dry-run
#
#   # Only rebuild SFT from existing raw JSONL
#   python3 scripts/extract_real_mcq_from_pdfs.py --rebuild-sft
# =============================================================================

import json
import os
import re
import sys
import argparse
import hashlib
from typing import Optional, List, Dict
from datetime import datetime

try:
    import fitz  # PyMuPDF
except ImportError:
    print("ERROR: PyMuPDF not installed. Run: pip install pymupdf")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = "/Users/satyamurti/Downloads/DataSets"
PDF_DIR    = f"{BASE_DIR}/upsc_pdfs/Prelimns"
OUT_DIR    = f"{BASE_DIR}/dataset_output_final/combined"
RAW_JSONL  = f"{OUT_DIR}/real_upsc_mcq_raw.jsonl"
SFT_JSONL  = f"{OUT_DIR}/real_upsc_mcq_sft.jsonl"

os.makedirs(OUT_DIR, exist_ok=True)

# Gemma 2 chat template
GEMMA_TEMPLATE = (
    "<start_of_turn>user\n{question}<end_of_turn>\n"
    "<start_of_turn>model\n{answer}<end_of_turn>"
)

# ---------------------------------------------------------------------------
# PDF registry — filename, year, page range of questions
# ---------------------------------------------------------------------------
PDF_REGISTRY = [
    {
        "year": 2011,
        "file": "VisionIAS - GS Prelims 2011.pdf",
        "q_pages": (0, 18),   # 0-indexed inclusive; stat tables are on last 2 pages
        "total_pages": 20,
        "format": "old",      # older layout: no structured table
    },
    {
        "year": 2012,
        "file": "VisionIAS - GS Prelims 2012.pdf",
        "q_pages": (0, 29),
        "total_pages": 32,
        "format": "old",
        # 8-col: q_num | topic | question | answer | level | nature | source | explanation
        "col_map": {"q_num": 0, "subject": 1, "question": 2, "q_merge_col": -1,
                    "answer": 3, "level": 4, "nature": 5, "source_type": -1,
                    "explanation": 7, "n_cols": 8},
    },
    {
        "year": 2013,
        "file": "VisionIAS - GS Prelims 2013.pdf",
        "q_pages": (0, 24),
        "total_pages": 28,
        "format": "old",
        # 10-col: q_num | question | subject | answer | level | nature | source_url | source_type | explanation | blank
        "col_map": {"q_num": 0, "subject": 2, "question": 1, "q_merge_col": -1,
                    "answer": 3, "level": 4, "nature": 5, "source_type": 7,
                    "explanation": 8, "n_cols": 10},
    },
    {
        "year": 2014,
        "file": "VisionIAS - GS Prelims 2014.pdf",
        "q_pages": (0, 27),
        "total_pages": 36,
        "format": "old",
        # Same structure as 2013 (10 cols)
        "col_map": {"q_num": 0, "subject": 2, "question": 1, "q_merge_col": -1,
                    "answer": 3, "level": 4, "nature": 5, "source_type": 7,
                    "explanation": 8, "n_cols": 10},
    },
    {
        "year": 2015,
        "file": "VisionIAS - GS Prelims 2015.pdf",
        "q_pages": (0, 35),
        "total_pages": 43,
        "format": "old",
        # 2015 has mixed page layouts (10-col and 12-col).
        # Use 10-col as primary (q=0, subj=1, q_text=2, ans=3, level=4, nature=5, src_type=7, expl=8)
        # Answer scan fallback will handle 12-col pages where answer is at col 5.
        "col_map": {"q_num": 0, "subject": 1, "question": 2, "q_merge_col": -1,
                    "answer": 3, "level": 4, "nature": 5, "source_type": 7,
                    "explanation": 8, "n_cols": 10},
    },
    {
        "year": 2016,
        "file": "VisionIAS - GS Prelims 2016.pdf",
        "q_pages": (0, 72),
        "total_pages": 80,
        "format": "table",
        # 13-col: q_num | subject | q_part1 | q_part2 | answer | level | nature | source_url | source_type | explanation | blank | motivation | vision_ref
        "col_map": {"q_num": 0, "subject": 1, "question": 2, "q_merge_col": 3,
                    "answer": 4, "level": 5, "nature": 6, "source_type": 8,
                    "explanation": 9, "n_cols": 13},
    },
    {
        "year": 2017,
        "file": "VisionIAS - GS Prelims 2017.pdf",
        "q_pages": (0, 59),
        "total_pages": 67,
        "format": "table",
        # Same as 2016
        "col_map": {"q_num": 0, "subject": 1, "question": 2, "q_merge_col": 3,
                    "answer": 4, "level": 5, "nature": 6, "source_type": 8,
                    "explanation": 9, "n_cols": 13},
    },
    {
        "year": 2018,
        "file": "VisionIAS - GS Prelims 2018.pdf",
        "q_pages": (0, 87),
        "total_pages": 94,
        "format": "table",
        # Layout A (11 cols): QN | Section | Question | Answer | Explanation | Level | Nature | Source | SourceType | Motivation | VisionIAS
        "col_map": {"q_num": 0, "subject": 1, "question": 2, "q_merge_col": -1,
                    "answer": 3, "explanation": 4, "level": 5, "nature": 6,
                    "source_type": 8, "n_cols": 11},
    },
    {
        "year": 2019,
        "file": "VisionIAS - GS Prelims 2019.pdf",
        "q_pages": (0, 94),
        "total_pages": 101,
        "format": "table",
        "col_map": {"q_num": 0, "subject": 1, "question": 2, "q_merge_col": -1,
                    "answer": 3, "explanation": 4, "level": 5, "nature": 6,
                    "source_type": 8, "n_cols": 11},
    },
    {
        "year": 2020,
        "file": "VisionIAS - GS Prelims 2020.pdf",
        "q_pages": (0, 132),
        "total_pages": 142,
        "format": "table",
        "col_map": {"q_num": 0, "subject": 1, "question": 2, "q_merge_col": -1,
                    "answer": 3, "explanation": 4, "level": 5, "nature": 6,
                    "source_type": 8, "n_cols": 11},
    },
    {
        "year": 2021,
        "file": "VisionIAS - GS Prelims 2021.pdf",
        "q_pages": (0, 100),
        "total_pages": 109,
        "format": "table",
        "col_map": {"q_num": 0, "subject": 1, "question": 2, "q_merge_col": -1,
                    "answer": 3, "explanation": 4, "level": 5, "nature": 6,
                    "source_type": 8, "n_cols": 11},
    },
    {
        "year": 2022,
        "file": "VisionIAS - GS Prelims 2022.pdf",
        "q_pages": (0, 92),
        "total_pages": 100,
        "format": "table",
        "col_map": {"q_num": 0, "subject": 1, "question": 2, "q_merge_col": -1,
                    "answer": 3, "explanation": 4, "level": 5, "nature": 6,
                    "source_type": 8, "n_cols": 11},
    },
    {
        "year": 2023,
        "file": "VisionIAS - GS Prelims 2023.pdf",
        "q_pages": (0, 107),
        "total_pages": 115,
        "format": "table",
        "col_map": {"q_num": 0, "subject": 1, "question": 2, "q_merge_col": -1,
                    "answer": 3, "explanation": 4, "level": 5, "nature": 6,
                    "source_type": 8, "n_cols": 11},
    },
    {
        "year": 2024,
        "file": "VisionIAS - UPSC GS 2024  Solution, Analysis & Explanation.pdf",
        "q_pages": (0, 108),
        "total_pages": 116,
        "format": "table",
        "col_map": {"q_num": 0, "subject": 1, "question": 2, "q_merge_col": -1,
                    "answer": 3, "explanation": 4, "level": 5, "nature": 6,
                    "source_type": 8, "n_cols": 11},
    },
    {
        "year": 2025,
        "file": "VisionIAS - UPSC GS Paper I Question Paper 2025 with Answer Key.pdf",
        "q_pages": (0, 105),
        "total_pages": 109,
        "format": "table",
        "col_map": {"q_num": 0, "subject": 1, "question": 2, "q_merge_col": -1,
                    "answer": 3, "explanation": 4, "level": 5, "nature": 6,
                    "source_type": 8, "n_cols": 11},
    },
]

# ---------------------------------------------------------------------------
# Subject normalisation map
# ---------------------------------------------------------------------------
SUBJECT_MAP = {
    "polity": "Polity_Governance",
    "polity & governance": "Polity_Governance",
    "polity and governance": "Polity_Governance",
    "polity&governance": "Polity_Governance",
    "governance": "Polity_Governance",
    "environment": "Environment",
    "ecology": "Environment",
    "environment & ecology": "Environment",
    "env": "Environment",
    "geography": "Geography",
    "geo": "Geography",
    "geog": "Geography",
    "economy": "Economy",
    "economics": "Economy",
    "economic": "Economy",
    "ancient india": "Ancient_History",
    "ancient history": "Ancient_History",
    "ancient": "Ancient_History",
    "medieval india": "Medieval_History",
    "medieval history": "Medieval_History",
    "medieval": "Medieval_History",
    "modern india": "Modern_History",
    "modern history": "Modern_History",
    "modern": "Modern_History",
    "post-independence": "Modern_History",
    "art and culture": "Art_Culture",
    "art & culture": "Art_Culture",
    "culture": "Art_Culture",
    "art/culture": "Art_Culture",
    "science and technology": "Science_Technology",
    "science & technology": "Science_Technology",
    "s&t": "Science_Technology",
    "general science and science and technology": "Science_Technology",
    "general science & s&t": "Science_Technology",
    "general science": "Science_Technology",
    "basic science": "Science_Technology",
    "basic science & s&t": "Science_Technology",
    "current affairs": "Current_Affairs",
    "current affair": "Current_Affairs",
    "ca": "Current_Affairs",
    "international relations": "International_Relations",
    "ir": "International_Relations",
    "social issues": "Social_Issues",
    "social issues/schemes": "Social_Issues",
    "miscellaneous": "Miscellaneous",
    "government schemes": "Social_Issues",
    "schemes": "Social_Issues",
    "history": "History",
}


def normalize_subject(raw: str) -> str:
    """Map raw VisionIAS subject label to canonical subject name."""
    if not raw:
        return "Unknown"
    # Collapse all whitespace including newlines (handles line-wrapped cell text)
    raw = re.sub(r'\s+', ' ', raw).strip()
    clean = raw.lower().strip().strip(".")
    # Try exact
    if clean in SUBJECT_MAP:
        return SUBJECT_MAP[clean]
    # Try partial
    for key, val in SUBJECT_MAP.items():
        if key in clean or clean in key:
            return val
    # Current affairs sub-label: "Current Affairs (Economy)" etc.
    if "current" in clean:
        return "Current_Affairs"
    if "internat" in clean:
        return "International_Relations"
    if "social" in clean or "scheme" in clean:
        return "Social_Issues"
    return raw.strip().title().replace(" ", "_")


# ---------------------------------------------------------------------------
# Difficulty / nature / source inline parsers
# ---------------------------------------------------------------------------

DIFF_PATTERN   = re.compile(r'\b([EMD])\b')
NATURE_PATTERN = re.compile(r'\b(FCA|CAA|FA|CA|F|U)\b')
SOURCE_PATTERN = re.compile(r'\b(EM|EN|RM|RR)\b')


def parse_inline_meta(text: str) -> Dict[str, Optional[str]]:
    """
    Parse difficulty / nature / source from the metadata column adjacent to explanation.
    VisionIAS encodes these as single letters: E/M/D, F/FA/CA/CAA/FCA/U, EM/EN/RM/RR
    Returns dict with keys: difficulty, nature, source (all Optional[str])
    """
    result: Dict[str, Optional[str]] = {
        "difficulty": None,
        "nature"    : None,
        "source"    : None,
    }

    # Source: 2-char codes must match first (order matters — EM before E/M)
    src_m = SOURCE_PATTERN.search(text)
    if src_m:
        result["source"] = src_m.group(1)

    # Nature: longer codes first (FCA before CA, CAA before CA)
    nat_m = NATURE_PATTERN.search(text)
    if nat_m:
        result["nature"] = nat_m.group(1)

    # Difficulty: single E/M/D — only if not already captured as source/nature
    diff_m = DIFF_PATTERN.search(text)
    if diff_m:
        d = diff_m.group(1)
        # Sanity: difficulty should not be the same token as a source
        if d not in (result.get("source") or ""):
            result["difficulty"] = {"E": "easy", "M": "medium", "D": "difficult"}.get(d)

    return result


# ---------------------------------------------------------------------------
# Detect question number from line
# ---------------------------------------------------------------------------
def is_question_number_line(text: str) -> Optional[int]:
    """Return question number if line is a standalone Q number, else None."""
    m = re.match(r'^\s*(\d{1,3})\s*$', text.strip())
    if m:
        n = int(m.group(1))
        if 1 <= n <= 100:
            return n
    return None


# ---------------------------------------------------------------------------
# Option parser — extracts (a)(b)(c)(d) options from question text block
# ---------------------------------------------------------------------------
OPT_PATTERN = re.compile(
    r'\(?([a-dA-D])\)?[\.\)]\s*(.+?)(?=\(?[a-dA-D]\)?[\.\)]|$)',
    re.DOTALL
)


def parse_options(text: str) -> Dict[str, str]:
    """Extract option map {a: text, b: text, c: text, d: text} from question block."""
    opts: Dict[str, str] = {}
    for m in OPT_PATTERN.finditer(text):
        key = m.group(1).lower()
        val = m.group(2).strip().rstrip()
        val = re.sub(r'\s+', ' ', val).strip()
        if val:
            opts[key] = val
    return opts


# ---------------------------------------------------------------------------
# Answer normaliser
# ---------------------------------------------------------------------------
def normalize_answer(raw: str) -> str:
    """Extract single letter a/b/c/d from raw answer string like 'B', '(b)', 'Answer: C'"""
    if not raw:
        return ""
    m = re.search(r'\b([a-dA-D])\b', raw)
    return m.group(1).lower() if m else raw.strip().lower()[:1]


# ---------------------------------------------------------------------------
# Table-format extractor (2016-2025)
# ---------------------------------------------------------------------------

def extract_table_format(doc: fitz.Document, year: int,
                         q_pages: tuple,
                         col_map_override: Optional[dict] = None) -> List[dict]:
    """
    Extract MCQs from VisionIAS table-format PDFs (any year).
    col_map_override: if provided, uses this column map instead of auto-detecting.
    """
    records: List[dict] = []

    for pg_idx in range(q_pages[0], min(q_pages[1] + 1, doc.page_count)):
        page = doc[pg_idx]

        # Try table extraction first (PyMuPDF >= 1.23)
        tables_found = False
        try:
            tabs = page.find_tables()
            if tabs and len(tabs.tables) > 0:
                for tab in tabs.tables:
                    tab_data = tab.extract()
                    records.extend(
                        _parse_table_rows(tab_data, year, pg_idx + 1,
                                          col_map_override=col_map_override)
                    )
                    tables_found = True
        except AttributeError:
            pass  # find_tables not available, use text block fallback

        if not tables_found:
            # Text block fallback: extract raw text and parse structurally
            text = page.get_text("text")
            recs = _parse_text_block(text, year, pg_idx + 1)
            records.extend(recs)

    return records


def _sniff_columns(header_row: List) -> dict:
    """
    Detect column indices from table header row.

    VisionIAS PDFs have two main layouts:
      Layout A (2018-2025, 11 cols): QN | Section | Question | Answer | Explanation | Level | Nature | Source | SourceType | ...
      Layout B (2016-2017, 12-13 cols): Q.N. | Section | Question | (blank/cont) | Answer | Level | Nature | Source | SourceType | Explanation | ...
      Layout C (2012-2015, 8 cols): Q.No. | Topics | Question+opts | Answer | Level | Nature | Source | Explanation

    Returns dict: {q_num, subject, question, answer, explanation, level, nature, source_type}
    with integer column indices (or -1 if not found).
    """
    # Flatten header cells to lowercase strings for matching
    h = [str(c or "").lower().replace("\n", " ").strip() for c in header_row]

    def find(keywords: List[str]) -> int:
        for kw in keywords:
            for i, cell in enumerate(h):
                if kw in cell:
                    return i
        return -1

    q_col   = find(["q.n", "qn", "q.no", "q no", "s no", "sl"])
    subj_col = find(["section", "topic", "subject", "area"])
    q_txt_col = find(["question"])
    ans_col  = find(["answer", "ans"])
    expl_col = find(["explanation", "expla"])
    level_col = find(["level", "diffi", "leve"])
    nature_col = find(["nature", "natu"])
    src_type_col = find(["source type", "sourcetype", "srce type"])
    if src_type_col == -1:
        src_type_col = find(["source\ntype", "sou\nrce\ntyp"])

    # For cols not found by keyword, guess by position and number of columns
    n = len(h)

    # If q_col not found but first cell contains a number pattern, use 0
    if q_col == -1:
        q_col = 0
    if subj_col == -1:
        subj_col = 1
    if q_txt_col == -1:
        q_txt_col = 2

    # Answer column heuristic
    if ans_col == -1:
        if n >= 11:
            ans_col = 3   # Layout A
        elif n >= 12:
            ans_col = 4   # Layout B
        else:
            ans_col = 3   # Layout C

    # For layout B (12-13 cols), question may span cols 2+3
    needs_q_merge = (n >= 12 and q_txt_col == 2)

    return {
        "q_num"       : q_col,
        "subject"     : subj_col,
        "question"    : q_txt_col,
        "q_merge_col" : 3 if needs_q_merge else -1,  # second question col to merge
        "answer"      : ans_col,
        "explanation" : expl_col,
        "level"       : level_col,
        "nature"      : nature_col,
        "source_type" : src_type_col,
        "n_cols"      : n,
    }


def _parse_table_rows(tab_data: List[List], year: int, page_num: int,
                      col_map_override: Optional[dict] = None) -> List[dict]:
    """
    Parse rows from a PyMuPDF table extraction result.
    Uses col_map_override if provided (explicit per-year layout), else auto-sniffs.
    """
    records: List[dict] = []
    if not tab_data or len(tab_data) < 1:
        return records

    # If an explicit column map was passed, use it directly
    if col_map_override is not None:
        col_map = col_map_override
        # Skip header row if first cell is non-numeric
        first_cell = str(tab_data[0][0] or "").strip()
        if not re.match(r'^\d+\.?$', first_cell):
            data_rows = tab_data[1:]
        else:
            data_rows = tab_data
    else:
        # Detect column mapping from header row (if present)
        col_map = None
        data_rows = tab_data

        # Check if first row is a header (contains text-like keys, no q-number)
        first_cell = str(tab_data[0][0] or "").strip().lower() if tab_data[0] else ""
        if first_cell in ("", "q.n.", "qn", "q.no.", "s no", "sl", "q.n", "q no"):
            if not re.match(r'^\d+\.?$', first_cell):
                col_map = _sniff_columns(tab_data[0])
                data_rows = tab_data[1:]

    # Default column map if no header detected
    if col_map is None:
        # Guess from number of columns
        n = len(tab_data[0]) if tab_data else 11
        if n >= 12:
            # Layout B: 2016-2017
            col_map = {"q_num": 0, "subject": 1, "question": 2, "q_merge_col": 3,
                       "answer": 4, "level": 5, "nature": 6, "source_type": 8,
                       "explanation": 9, "n_cols": n}
        elif n >= 8:
            # Layout A: 2018+ / Layout C: 2012-2015
            col_map = {"q_num": 0, "subject": 1, "question": 2, "q_merge_col": -1,
                       "answer": 3, "explanation": 4, "level": 5, "nature": 6,
                       "source_type": 8, "n_cols": n}
        else:
            col_map = {"q_num": 0, "subject": 1, "question": 2, "q_merge_col": -1,
                       "answer": 3, "explanation": 7, "level": 4, "nature": 5,
                       "source_type": -1, "n_cols": n}

    def get(row: List, idx: int) -> str:
        if idx < 0 or idx >= len(row):
            return ""
        return str(row[idx] or "").strip()

    for row in data_rows:
        if not row or len(row) < 3:
            continue

        q_num_raw = get(row, col_map["q_num"])
        # Strip trailing dot: "2." -> "2"
        q_num_raw = re.sub(r'\.+$', '', q_num_raw).strip()
        if not q_num_raw or not re.match(r'^\d{1,3}$', q_num_raw):
            continue

        q_num = int(q_num_raw)
        if not (1 <= q_num <= 100):
            continue

        subject_raw = get(row, col_map["subject"])
        subject_raw = re.sub(r'\s+', ' ', subject_raw)

        # Question text — may span two columns (layout B)
        q_part1 = get(row, col_map["question"])
        q_part2 = get(row, col_map["q_merge_col"]) if col_map["q_merge_col"] >= 0 else ""
        q_text_raw = (q_part1 + " " + q_part2).strip()
        q_text_raw = re.sub(r'\s+', ' ', q_text_raw)

        # If question col is empty but there's content elsewhere, look in all cols
        if len(q_text_raw) < 20:
            for ci in range(2, min(5, len(row))):
                candidate = str(row[ci] or "").strip()
                if len(candidate) >= 20 and re.search(r'[A-Za-z]{4,}', candidate):
                    q_text_raw = re.sub(r'\s+', ' ', candidate)
                    break

        if len(q_text_raw) < 15:
            continue

        answer_raw = get(row, col_map["answer"])
        # If answer col is empty or has long text, scan row for a single-letter answer
        if not answer_raw or len(answer_raw) > 5:
            for ci in range(2, len(row)):
                candidate = str(row[ci] or "").strip()
                if candidate.upper() in ("A", "B", "C", "D"):
                    answer_raw = candidate
                    break

        answer = normalize_answer(answer_raw)
        if not answer:
            continue

        expl_raw = get(row, col_map["explanation"]) if col_map["explanation"] >= 0 else ""

        # Parse difficulty / nature / source from dedicated cols or fallback to meta
        level_raw   = get(row, col_map["level"]) if col_map["level"] >= 0 else ""
        nature_raw  = get(row, col_map["nature"]) if col_map["nature"] >= 0 else ""
        src_raw     = get(row, col_map["source_type"]) if col_map["source_type"] >= 0 else ""

        # All remaining columns contribute to meta (use actual row length, not col_map n_cols)
        meta_tail = " ".join(str(row[ci] or "") for ci in range(len(row)))

        # Difficulty
        diff_map = {"E": "easy", "M": "medium", "D": "difficult"}
        difficulty = diff_map.get(level_raw.upper().strip(), None)
        if difficulty is None:
            meta_d = parse_inline_meta(meta_tail)
            difficulty = meta_d["difficulty"] or "medium"

        # Nature
        valid_natures = {"F", "FA", "CA", "CAA", "FCA", "U"}
        nature_val = nature_raw.strip().upper()
        if nature_val not in valid_natures:
            nat_m = NATURE_PATTERN.search(meta_tail)
            nature_val = nat_m.group(1) if nat_m else "F"

        # Source type
        src_val = src_raw.strip().upper()
        if src_val not in ("EM", "EN", "RM", "RR"):
            src_m = SOURCE_PATTERN.search(meta_tail)
            src_val = src_m.group(1) if src_m else None

        # Clean question text — strip option lines
        options = parse_options(q_text_raw)
        q_clean = re.split(r'\(?[a-dA-D]\)?[\.\)]\s', q_text_raw)[0].strip()
        q_clean = re.sub(r'\s+', ' ', q_clean)

        if len(q_clean.split()) < 5:
            continue

        subject_norm = normalize_subject(subject_raw)

        rec = _build_record(
            year=year,
            q_num=q_num,
            page=page_num,
            subject=subject_norm,
            subject_raw=subject_raw,
            question=q_clean,
            options=options,
            answer=answer,
            explanation=_clean_explanation(expl_raw),
            difficulty=difficulty,
            nature=nature_val,
            source=src_val,
        )
        records.append(rec)

    return records


def _parse_text_block(text: str, year: int, page_num: int) -> List[dict]:
    """
    Fallback text-block parser for pages where table extraction fails.
    Looks for patterns: standalone number, question text, (a)/(b)/(c)/(d), answer letter.
    """
    records: List[dict] = []
    lines = [l.rstrip() for l in text.split('\n') if l.strip()]

    i = 0
    while i < len(lines):
        q_num = is_question_number_line(lines[i])
        if q_num is not None:
            # Collect lines until next question number or end of page
            block_lines = []
            j = i + 1
            while j < len(lines):
                next_q = is_question_number_line(lines[j])
                if next_q is not None and next_q != q_num:
                    break
                block_lines.append(lines[j])
                j += 1

            block = "\n".join(block_lines)
            rec = _parse_text_block_question(block, q_num, year, page_num)
            if rec:
                records.append(rec)
            i = j
        else:
            i += 1

    return records


def _parse_text_block_question(block: str, q_num: int,
                                year: int, page_num: int) -> Optional[dict]:
    """
    Parse a single question block from raw text.
    Block contains: subject, question, options (a)(b)(c)(d), answer, explanation.
    """
    if len(block.strip()) < 30:
        return None

    # Split on first occurrence of option pattern
    parts = re.split(r'\n\s*\(a\)', block, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) < 2:
        # Try without newline
        parts = re.split(r'\(a\)', block, maxsplit=1, flags=re.IGNORECASE)

    q_part  = parts[0].strip() if parts else block
    opt_block = "(a)" + parts[1] if len(parts) > 1 else ""

    # Extract subject from first line(s)
    first_lines = q_part.strip().split('\n')
    subject_raw = ""
    q_start_idx = 0
    # Heuristic: if first line is short (< 40 chars) and not a sentence, treat as subject
    if first_lines and len(first_lines[0]) < 50 and '?' not in first_lines[0]:
        subject_raw = first_lines[0].strip()
        q_start_idx = 1

    q_text = " ".join(first_lines[q_start_idx:]).strip()
    q_text = re.sub(r'\s+', ' ', q_text)

    options = parse_options(opt_block) if opt_block else {}

    # Answer: look for standalone letter after options block
    answer_m = re.search(r'\n\s*([a-dA-D])\s*\n', opt_block + "\n")
    answer = normalize_answer(answer_m.group(1)) if answer_m else ""

    # Explanation: text after the answer letter
    expl = ""
    if answer_m:
        expl = opt_block[answer_m.end():].strip()
        expl = re.sub(r'\s+', ' ', expl)[:1000]

    # Inline meta from explanation
    meta = parse_inline_meta(expl + " " + opt_block[-200:])

    if not q_text or len(q_text.split()) < 6:
        return None

    return _build_record(
        year=year,
        q_num=q_num,
        page=page_num,
        subject=normalize_subject(subject_raw),
        subject_raw=subject_raw,
        question=q_text,
        options=options,
        answer=answer,
        explanation=_clean_explanation(expl),
        difficulty=meta["difficulty"],
        nature=meta["nature"],
        source=meta["source"],
    )


# ---------------------------------------------------------------------------
# Old-format extractor (2011-2015) — inline text, no table structure
# ---------------------------------------------------------------------------

def extract_old_format(doc: fitz.Document, year: int,
                       q_pages: tuple,
                       col_map_override: Optional[dict] = None) -> List[dict]:
    """
    Extract MCQs from older VisionIAS PDFs (2011-2015).
    When col_map_override is provided, routes through the unified table extractor.
    """
    if col_map_override is not None:
        # Use unified extractor with explicit column map
        return extract_table_format(doc, year, q_pages,
                                    col_map_override=col_map_override)

    records: List[dict] = []

    for pg_idx in range(q_pages[0], min(q_pages[1] + 1, doc.page_count)):
        page = doc[pg_idx]

        tables_found = False
        try:
            tabs = page.find_tables()
            if tabs and len(tabs.tables) > 0:
                for tab in tabs.tables:
                    tab_data = tab.extract()
                    recs = _parse_old_table_rows(tab_data, year, pg_idx + 1)
                    records.extend(recs)
                    tables_found = True
        except AttributeError:
            pass

        if not tables_found:
            text = page.get_text("text")
            recs = _parse_text_block(text, year, pg_idx + 1)
            records.extend(recs)

    return records


def _parse_old_table_rows(tab_data: List[List], year: int, page_num: int) -> List[dict]:
    """
    Parse rows from old format (2011-2015) table extraction.

    2012-2015 confirmed column layout (8 cols):
      Col 0: Q number (e.g. "2.")
      Col 1: Topic/Subject
      Col 2: Full question text + options (a)(b)(c)(d)
      Col 3: Answer letter (e.g. "A")
      Col 4: Level (E/M/D)
      Col 5: Nature (F/FCA/CA etc.)
      Col 6: Source (URL or book reference)
      Col 7: Explanation

    2011 format is similar but may have 9+ columns with sub-topic at col 2.
    """
    records: List[dict] = []
    if not tab_data or len(tab_data) < 2:
        return records

    # Detect column count from first data row
    first_data = next((r for r in tab_data if r and str(r[0] or "").strip() and
                       re.match(r'^\d+\.?$', str(r[0] or "").strip())), None)
    n_cols = len(first_data) if first_data else 8

    # Column offsets vary by format
    # 8-col (2012-2015): q_num=0, topic=1, question=2, answer=3, level=4, nature=5, source=6, expl=7
    # 9-col (2011):      q_num=0, topic=1, subtopic=2, question=3, answer=4, level=5, nature=6, source=7, expl=8
    if n_cols >= 9:
        q_col, subj_col, q_txt_col, ans_col, lev_col, nat_col, expl_col = 0, 1, 3, 4, 5, 6, 8
    else:
        q_col, subj_col, q_txt_col, ans_col, lev_col, nat_col, expl_col = 0, 1, 2, 3, 4, 5, 7

    def get(row: List, idx: int) -> str:
        if idx < 0 or idx >= len(row):
            return ""
        return str(row[idx] or "").strip()

    for row in tab_data:
        if not row or len(row) < 3:
            continue

        q_num_raw   = get(row, q_col)
        subject_raw = get(row, subj_col)
        q_text_raw  = get(row, q_txt_col)
        answer_raw  = get(row, ans_col)
        # row[5] = level (E/M/D), row[6] = nature, row[7] = source, row[8] = explanation
        level_raw   = get(row, lev_col)
        nature_raw  = get(row, nat_col)
        expl_raw    = get(row, expl_col)

        if not q_num_raw or not re.match(r'^\d{1,3}\.?$', q_num_raw):
            continue
        if not q_text_raw or len(q_text_raw) < 20:
            continue

        q_num = int(re.sub(r'\D', '', q_num_raw))

        options = parse_options(q_text_raw)
        q_clean = re.split(r'\(?[a-dA-D]\)?[\.\)]\s', q_text_raw)[0].strip()
        q_clean = re.sub(r'\s+', ' ', q_clean)

        answer = normalize_answer(answer_raw)

        # Map old level notation
        diff_map = {"E": "easy", "M": "medium", "D": "difficult",
                    "V": "difficult", "EASY": "easy", "MEDIUM": "medium",
                    "DIFFICULT": "difficult", "HARD": "difficult"}
        difficulty = diff_map.get(level_raw.upper(), None)
        if difficulty is None and level_raw:
            d_m = re.search(r'\b([EMD])\b', level_raw)
            difficulty = diff_map.get(d_m.group(1), "medium") if d_m else "medium"

        # Nature normalisation for old format (used "Fundamental"/"Conventional" labels)
        nature = nature_raw.strip().upper()
        valid_natures = {"F", "FA", "CA", "CAA", "FCA", "U"}
        if nature not in valid_natures:
            meta = parse_inline_meta(expl_raw)
            nature = meta["nature"] or "F"

        # Source: col 6 in 8-col layout (index between nat_col and expl_col)
        src_col_idx = nat_col + 1
        source_raw = get(row, src_col_idx) if src_col_idx < expl_col else ""
        source_norm = source_raw.strip().upper()
        if source_norm not in ("EM", "EN", "RM", "RR"):
            src_m = SOURCE_PATTERN.search(source_raw + " " + expl_raw)
            source_norm = src_m.group(1) if src_m else None

        if not q_clean or not answer:
            continue

        rec = _build_record(
            year=year,
            q_num=q_num,
            page=page_num,
            subject=normalize_subject(subject_raw),
            subject_raw=subject_raw,
            question=q_clean,
            options=options,
            answer=answer,
            explanation=_clean_explanation(expl_raw),
            difficulty=difficulty,
            nature=nature,
            source=source_norm,
        )
        records.append(rec)

    return records


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_record(year: int, q_num: int, page: int,
                  subject: str, subject_raw: str,
                  question: str, options: Dict[str, str],
                  answer: str, explanation: str,
                  difficulty: Optional[str], nature: Optional[str],
                  source: Optional[str]) -> dict:
    """Build a standardised raw MCQ record dict."""
    q_hash = hashlib.md5(question.encode()).hexdigest()[:8]
    return {
        "id"           : f"real_{year}_q{q_num:03d}_{q_hash}",
        "year"         : year,
        "q_num"        : q_num,
        "page"         : page,
        "subject"      : subject,
        "subject_raw"  : subject_raw,
        "question"     : question,
        "options"      : options,
        "correct"      : answer,
        "explanation"  : explanation,
        "difficulty"   : difficulty or "medium",
        "nature"       : nature or "F",
        "source"       : source,
        "dataset_type" : "real_upsc",
        "extracted_at" : datetime.now().isoformat(),
    }


def _clean_explanation(text: str) -> str:
    """Clean raw explanation text."""
    if not text:
        return ""
    # Remove URL lines
    text = re.sub(r'https?://\S+', '', text)
    # Remove VisionIAS watermarks
    text = re.sub(r'www\.visionias\.in', '', text, flags=re.IGNORECASE)
    text = re.sub(r'©\s*Vision\s*IAS', '', text, flags=re.IGNORECASE)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:1200]


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate(records: List[dict]) -> List[dict]:
    """Remove duplicate questions (same year + q_num, or identical question text)."""
    seen_year_qnum = set()
    seen_q_hash    = set()
    out: List[dict] = []
    for r in records:
        key1 = (r["year"], r["q_num"])
        key2 = hashlib.md5(r["question"].encode()).hexdigest()
        if key1 in seen_year_qnum or key2 in seen_q_hash:
            continue
        seen_year_qnum.add(key1)
        seen_q_hash.add(key2)
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# Quality filter
# ---------------------------------------------------------------------------

def quality_filter(records: List[dict]) -> List[dict]:
    """
    Keep only records that meet minimum quality bar:
      - question has >= 8 words
      - answer is a valid a/b/c/d
      - options dict is non-empty OR (for old format) at least has an answer
    """
    out: List[dict] = []
    for r in records:
        q_words = len(r["question"].split())
        if q_words < 6:
            continue
        if r["correct"] not in ("a", "b", "c", "d"):
            continue
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# SFT pair builder
# ---------------------------------------------------------------------------

def format_question_text(rec: dict, show_answer: bool = False) -> str:
    """Format a real MCQ as readable text."""
    lines = [rec["question"], ""]
    for letter in ["a", "b", "c", "d"]:
        if letter in rec.get("options", {}):
            lines.append(f"({letter}) {rec['options'][letter]}")
    if show_answer:
        lines.append("")
        lines.append(f"Correct Answer: ({rec['correct']})")
    return "\n".join(lines)


def _real_question_block(rec: dict, show_answer: bool = False) -> str:
    """Format a real UPSC MCQ as a question block string."""
    opts  = rec.get("options", {})
    lines = [rec["question"], ""]
    for letter in ["a", "b", "c", "d"]:
        if letter in opts:
            lines.append(f"({letter}) {opts[letter]}")
    if show_answer:
        lines.append("")
        lines.append(f"Correct Answer: ({rec['correct']})")
    return "\n".join(lines)


def build_sft_pairs(records: List[dict]) -> List[dict]:
    """
    Build instruction-input-output (Alpaca) SFT records from real UPSC MCQ records.

    Output format:
      {
        "instruction": "...",
        "input": "...",
        "question": "<question with options>",
        "answer": "c",
        "explanation": "...",
        "subject": "Environment",
        "question_type": "statement_based",
        "difficulty": "medium",
        "exam": "UPSC",
        "year": 2023,
        "nature": "FCA",
        "format": "<variant name>",
        "source_id": "real_upsc_2023_q042",
        "dataset_type": "real_upsc_sft"
      }

    Variants:
      answer_mcq       -- "Answer this UPSC {year} Prelims question and explain."
      explain_correct  -- "Why is option ({answer}) correct for this UPSC question?"
      context_based    -- "Read the explanation and identify the correct option."
      solve_year       -- "Solve this MCQ from UPSC {year} Prelims on {subject}."
    """
    import random as _random

    ANSWER_VARIANTS = [
        "Answer this UPSC {year} Prelims question with explanation.",
        "Solve this MCQ from UPSC {year} Prelims on {subject} and give the explanation.",
        "What is the correct answer to this UPSC {year} question? Explain your choice.",
        "Identify the correct option for this UPSC {year} Prelims question and justify it.",
    ]

    EXPLAIN_VARIANTS = [
        "For this UPSC {year} question, why is option ({answer}) correct?",
        "Explain why option ({answer}) is the right answer for this UPSC {year} Prelims MCQ.",
        "Why is ({answer}) the correct choice here? Give a detailed explanation.",
        "What makes option ({answer}) correct for this {year} UPSC Prelims question?",
    ]

    CONTEXT_VARIANTS = [
        "Read the following explanation and identify which option is correct.",
        "Based on the explanation below, select the correct option.",
        "Use the explanation provided to determine the correct answer.",
        "Given this explanation, which option is correct?",
    ]

    SOLVE_VARIANTS = [
        "Solve this MCQ from UPSC {year} Prelims on {subject}.",
        "This is a real UPSC {year} Prelims question on {subject}. Answer it with explanation.",
        "From UPSC {year} Prelims — answer this {subject} question.",
    ]

    pairs: List[dict] = []

    def make(instruction: str, inp: str, q_block: str, answer: str,
             explanation: str, fmt: str, rec: dict) -> dict:
        subject_clean = rec["subject"].replace("_", " ")
        return {
            "instruction"  : instruction,
            "input"        : inp,
            "question"     : q_block,
            "answer"       : answer,
            "explanation"  : explanation,
            "subject"      : subject_clean,
            "question_type": rec.get("question_type", "factual"),
            "difficulty"   : rec["difficulty"],
            "exam"         : "UPSC",
            "year"         : rec["year"],
            "nature"       : rec.get("nature", "F"),
            "format"       : fmt,
            "source_id"    : rec["id"],
            "dataset_type" : "real_upsc_sft",
        }

    for rec in records:
        correct      = rec["correct"]
        opts         = rec.get("options", {})
        correct_text = opts.get(correct, "")
        expl         = rec.get("explanation", "").strip()
        subject      = rec["subject"].replace("_", " ")
        year         = rec["year"]

        # Skip records that do not have all 4 options — the question block
        # would be incomplete and confuse the model
        if len(opts) < 4:
            continue

        q_block = _real_question_block(rec, show_answer=False)

        # Build a guaranteed-grounded explanation prefix so the answer letter
        # is always explicitly referenced, even if the VisionIAS explanation
        # does not name it
        grounded_prefix = f"The correct answer is ({correct})"
        if correct_text:
            grounded_prefix += f" — {correct_text}."

        def full_expl(prefix: str, body: str) -> str:
            if body and body.lower().strip() != prefix.lower().strip():
                return f"{prefix}\n\n{body}"
            return prefix

        # --- Variant 1: answer_mcq ---
        instr1 = _random.choice(ANSWER_VARIANTS).format(
            year=year, subject=subject)
        pairs.append(make(
            instr1, q_block, q_block, correct,
            full_expl(grounded_prefix, expl),
            "answer_mcq", rec))

        # --- Variant 2: explain_correct ---
        # Only include if explanation is substantive (>= 20 words)
        if expl and len(expl.split()) >= 20:
            instr2 = _random.choice(EXPLAIN_VARIANTS).format(
                year=year, answer=correct)
            pairs.append(make(
                instr2, q_block, q_block, correct,
                full_expl(grounded_prefix, expl),
                "explain_correct", rec))

        # --- Variant 3: context_based ---
        # Use the VisionIAS explanation itself as the grounding context,
        # then ask the model to identify the answer from it
        if expl and correct_text and len(expl) > 80:
            inp3  = (
                f"Explanation context: {expl[:500]}\n\n"
                f"Question: {rec['question']}\n\n"
                f"Options:\n" +
                "\n".join(f"({k}) {v}" for k, v in sorted(opts.items()))
            )
            instr3 = _random.choice(CONTEXT_VARIANTS)
            exp3   = f"Based on the explanation, the correct answer is ({correct}) — {correct_text}."
            pairs.append(make(instr3, inp3, q_block, correct, exp3, "context_based", rec))

        # --- Variant 4: solve_year ---
        instr4 = _random.choice(SOLVE_VARIANTS).format(
            year=year, subject=subject)
        pairs.append(make(
            instr4, q_block, q_block, correct,
            full_expl(grounded_prefix, expl),
            "solve_year", rec))

    return pairs


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

def extract_pdf(entry: dict, dry_run: bool = False) -> List[dict]:
    """Extract all questions from one PDF entry."""
    path = os.path.join(PDF_DIR, entry["file"])
    if not os.path.exists(path):
        print(f"  WARNING: file not found — {path}")
        return []

    try:
        doc = fitz.open(path)
    except Exception as e:
        print(f"  ERROR opening {path}: {e}")
        return []

    year     = entry["year"]
    q_pages  = entry["q_pages"]
    fmt      = entry["format"]
    col_map  = entry.get("col_map", None)  # explicit per-year column map if defined

    print(f"  Extracting {year} ({fmt} format, pages {q_pages[0]+1}-{q_pages[1]+1}) ...")

    if fmt == "old":
        records = extract_old_format(doc, year, q_pages, col_map_override=col_map)
    else:
        records = extract_table_format(doc, year, q_pages, col_map_override=col_map)

    doc.close()

    records = quality_filter(deduplicate(records))

    if dry_run:
        print(f"  [DRY RUN] {year}: found {len(records)} questions")
        if records:
            sample = records[:2]
            for r in sample:
                print(f"    Q{r['q_num']}: {r['question'][:80]}...")
                print(f"    Ans: ({r['correct']}) | Subj: {r['subject']} | Diff: {r['difficulty']} | Nature: {r['nature']}")
        return []

    print(f"  {year}: extracted {len(records)} valid questions")
    return records


def run(args: argparse.Namespace):
    year_filter = args.year

    registry = PDF_REGISTRY
    if year_filter:
        registry = [e for e in PDF_REGISTRY if e["year"] == year_filter]
        if not registry:
            print(f"ERROR: year {year_filter} not found in registry")
            sys.exit(1)

    all_records: List[dict] = []

    print(f"\n=== Real UPSC MCQ Extractor ===")
    print(f"Processing {len(registry)} PDF(s)...\n")

    for entry in registry:
        year_records = extract_pdf(entry, dry_run=args.dry_run)
        all_records.extend(year_records)

    if args.dry_run:
        print(f"\nDry run complete. No files written.")
        return

    if not all_records:
        print("\nNo records extracted. Check PDF paths and extraction logic.")
        return

    # Global deduplication across years
    all_records = deduplicate(all_records)
    all_records.sort(key=lambda r: (r["year"], r["q_num"]))

    # Write raw JSONL
    with open(RAW_JSONL, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(all_records)} raw MCQ records -> {RAW_JSONL}")

    # Build and write SFT pairs
    if not args.raw_only:
        sft_pairs = build_sft_pairs(all_records)
        with open(SFT_JSONL, "w", encoding="utf-8") as f:
            for p in sft_pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"Wrote {len(sft_pairs)} SFT pairs -> {SFT_JSONL}")

        # Print breakdown
        from collections import Counter
        fmt_counts = Counter(p["format"] for p in sft_pairs)
        yr_counts  = Counter(p["year"] for p in sft_pairs)
        subj_counts = Counter(r["subject"] for r in all_records)

        print(f"\n--- Raw MCQ breakdown ---")
        print(f"Total questions: {len(all_records)}")
        for yr in sorted(yr_counts.keys()):
            yr_recs = [r for r in all_records if r["year"] == yr]
            print(f"  {yr}: {len(yr_recs)} questions")

        print(f"\n--- Subject breakdown (top 12) ---")
        for subj, cnt in subj_counts.most_common(12):
            print(f"  {subj:<30s} {cnt}")

        print(f"\n--- SFT format breakdown ---")
        for fmt_name, cnt in sorted(fmt_counts.items()):
            print(f"  {fmt_name:<25s} {cnt}")

        # Write stats JSON
        stats = {
            "total_raw_questions": len(all_records),
            "total_sft_pairs": len(sft_pairs),
            "years_covered": sorted(set(r["year"] for r in all_records)),
            "format_breakdown": dict(fmt_counts),
            "subject_breakdown": dict(subj_counts),
            "year_breakdown": {str(k): v for k, v in yr_counts.items()},
            "difficulty_breakdown": dict(Counter(r["difficulty"] for r in all_records)),
            "nature_breakdown": dict(Counter(r["nature"] for r in all_records)),
            "generated_at": datetime.now().isoformat(),
        }
        stats_path = os.path.join(OUT_DIR, "real_upsc_mcq_stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\nStats written -> {stats_path}")


def rebuild_sft_only():
    """Rebuild SFT pairs from existing raw JSONL without re-extracting PDFs."""
    if not os.path.exists(RAW_JSONL):
        print(f"ERROR: {RAW_JSONL} not found. Run extraction first.")
        sys.exit(1)

    records = []
    with open(RAW_JSONL) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Loaded {len(records)} raw records from {RAW_JSONL}")
    sft_pairs = build_sft_pairs(records)

    with open(SFT_JSONL, "w", encoding="utf-8") as f:
        for p in sft_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Rebuilt {len(sft_pairs)} SFT pairs -> {SFT_JSONL}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract real UPSC MCQs from VisionIAS PDFs"
    )
    parser.add_argument("--year",        type=int, default=None,
                        help="Process only this year (e.g. 2025)")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Print sample records, do not write files")
    parser.add_argument("--raw-only",    action="store_true",
                        help="Write raw JSONL only, skip SFT generation")
    parser.add_argument("--rebuild-sft", action="store_true",
                        help="Rebuild SFT pairs from existing raw JSONL")
    args = parser.parse_args()

    if args.rebuild_sft:
        rebuild_sft_only()
    else:
        run(args)


if __name__ == "__main__":
    main()
