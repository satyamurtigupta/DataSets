#!/usr/bin/env python3
"""
extract_past_papers.py

Extracts Q&A pairs from official UPSC Mains question paper PDFs (GS1-GS4, 2010-2024).
Outputs JSONL records suitable for Gemma 2 SFT fine-tuning.

Usage:
    # Extract questions from PDFs in upsc_papers/ folder
    python3 scripts/extract_past_papers.py --papers-dir upsc_papers/ --years 2010-2024

    # Test on single PDF
    python3 scripts/extract_past_papers.py --file upsc_papers/GS1_2023.pdf

    # Extract + add basic answers from pretrain data
    python3 scripts/extract_past_papers.py --papers-dir upsc_papers/ --add-answers

    # Show stats only
    python3 scripts/extract_past_papers.py --papers-dir upsc_papers/ --stats-only
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import fitz  # PyMuPDF

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

UPSC_KEYWORDS = {
    "critically_examine":  ["critically examine"],
    "critically_analyse":  ["critically analyse", "critically analyze"],
    "critically_evaluate": ["critically evaluate"],
    "examine":             ["examine"],
    "discuss":             ["discuss"],
    "comment":             ["comment on", "comment"],
    "analyse":             ["analyse", "analyze"],
    "evaluate":            ["evaluate"],
    "assess":              ["assess"],
    "how_far":             ["how far"],
    "do_you_agree":        ["do you agree", "do you think"],
    "throw_light":         ["throw light"],
    "in_light_of":         ["in the light of", "in light of"],
    "what":                ["what is", "what are", "what was", "what were", "what do you"],
    "why":                 ["why"],
    "how":                 ["how"],
    "describe":            ["describe"],
    "explain":             ["explain"],
    "justify":             ["justify"],
    "enumerate":           ["enumerate"],
    "distinguish":         ["distinguish between", "distinguish"],
    "compare":             ["compare"],
    "elucidate":           ["elucidate"],
}

# Ordered so longer/more-specific phrases are checked first
QTYPE_ORDER = [
    "critically_examine", "critically_analyse", "critically_evaluate",
    "how_far", "do_you_agree", "throw_light", "in_light_of",
    "examine", "discuss", "comment", "analyse", "evaluate", "assess",
    "what", "why", "how", "describe", "explain", "justify",
    "enumerate", "distinguish", "compare", "elucidate",
]

SUBJECT_KEYWORDS = {
    "History": [
        "ancient", "medieval", "modern", "colonial", "nationalism", "independence",
        "revolt", "mughal", "british", "maurya", "gupta", "harappa", "vedic",
        "gandhi", "nehru", "ambedkar", "tilak", "partition", "revolution",
        "reform", "renaissance", "swadeshi", "non-cooperation", "civil disobedience",
        "salt march", "quit india", "dynasty", "empire", "governor-general",
        "viceroy", "east india company", "regulating act", "charter act",
        "sepoy mutiny", "1857", "socio-religious",
    ],
    "Geography": [
        "monsoon", "rainfall", "river", "mountain", "plateau", "plain",
        "climate", "weather", "soil", "erosion", "watershed", "basin",
        "earthquake", "volcano", "tsunami", "disaster", "ocean", "sea",
        "coast", "delta", "geography", "topography", "latitude", "longitude",
        "agriculture", "crop", "irrigation", "drought", "flood", "cyclone",
        "biodiversity", "forest", "ecosystem", "wetland", "glacier",
        "urbanisation", "migration", "demographic",
    ],
    "Polity": [
        "constitution", "parliament", "judiciary", "executive", "legislature",
        "fundamental rights", "directive principles", "amendment", "article",
        "federalism", "centre-state", "president", "prime minister", "cabinet",
        "supreme court", "high court", "election", "lok sabha", "rajya sabha",
        "panchayati raj", "municipal", "governor", "comptroller", "upsc",
        "civil services", "bureaucracy", "delegated legislation", "ordinance",
        "emergency", "judicial review", "separation of powers", "basic structure",
    ],
    "Economy": [
        "gdp", "inflation", "fiscal", "monetary", "budget", "taxation",
        "trade", "export", "import", "balance of payments", "foreign exchange",
        "investment", "poverty", "unemployment", "inequality", "growth",
        "liberalisation", "privatisation", "globalisation", "wto", "imf",
        "world bank", "rbi", "sebi", "niti aayog", "planning", "msme",
        "agriculture sector", "industrial", "service sector", "start-up",
        "digital economy", "fintech", "banking", "credit",
    ],
    "Science": [
        "biotechnology", "nanotechnology", "nuclear", "space", "isro",
        "satellite", "missile", "laser", "robotics", "artificial intelligence",
        "machine learning", "internet of things", "blockchain", "genome",
        "dna", "vaccine", "pandemic", "virus", "antibiotic", "crispr",
        "stem cell", "cloning", "renewable energy", "solar", "wind energy",
        "electric vehicle", "5g", "quantum computing",
    ],
    "Environment": [
        "climate change", "global warming", "carbon", "greenhouse gas",
        "ozone", "pollution", "plastic", "waste", "recycling", "sustainability",
        "paris agreement", "cop", "kyoto protocol", "biodiversity",
        "convention on biological diversity", "wildlife", "endangered species",
        "deforestation", "afforestation", "mangrove", "coral reef",
        "environmental impact", "green energy", "clean energy",
    ],
    "International Relations": [
        "foreign policy", "bilateral", "multilateral", "diplomacy", "treaty",
        "united nations", "security council", "nato", "saarc", "asean",
        "brics", "g20", "g7", "shanghai cooperation", "world order",
        "geopolitics", "strategic", "nuclear deal", "terrorism", "refugee",
        "china", "pakistan", "usa", "russia", "eu", "neighbourhood policy",
    ],
    "Social Issues": [
        "women empowerment", "gender", "caste", "discrimination", "reservation",
        "scheduled caste", "scheduled tribe", "obc", "minority", "communalism",
        "secularism", "social justice", "education", "health", "malnutrition",
        "child labour", "human trafficking", "ngo", "civil society",
        "welfare scheme", "social security", "old age", "disability",
    ],
    "Ethics": [
        "ethics", "moral", "integrity", "probity", "governance", "corruption",
        "accountability", "transparency", "whistleblower", "conflict of interest",
        "emotional intelligence", "attitude", "aptitude", "civil servant",
        "public service", "dharma", "virtue", "ethical dilemma", "conscience",
        "corporate governance", "corporate social responsibility",
    ],
    "Governance": [
        "e-governance", "good governance", "right to information", "rti",
        "transparency", "accountability", "decentralisation", "public policy",
        "welfare state", "administrative", "lokpal", "ombudsman",
        "citizen charter", "social audit", "disaster management", "ndma",
    ],
    "Security": [
        "internal security", "naxalism", "left wing extremism", "insurgency",
        "terrorism", "border management", "cyber security", "organised crime",
        "money laundering", "human trafficking", "arms smuggling",
        "critical infrastructure", "paramilitary", "defence", "armed forces",
    ],
}

GEMMA_TEMPLATE = "<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n{answer}<end_of_turn>"

# ---------------------------------------------------------------------------
# PDF filename parser
# ---------------------------------------------------------------------------

def parse_pdf_filename(path: Path) -> tuple[str, int]:
    """
    Return (paper, year) from filename like:
        GS1_2023.pdf, GS-2_2022.pdf, General_Studies_Paper1_2021.pdf
    Returns (None, None) if not parseable.
    """
    name = path.stem.upper()  # e.g. "GS1_2023"

    # Year: four-digit number 2000-2030
    year_match = re.search(r'\b(20[0-2]\d)\b', name)
    year = int(year_match.group(1)) if year_match else None

    # Paper: GS1/GS2/GS3/GS4
    paper = None
    for pat, label in [
        (r'GS[-_\s]?1', 'GS1'),
        (r'GS[-_\s]?2', 'GS2'),
        (r'GS[-_\s]?3', 'GS3'),
        (r'GS[-_\s]?4', 'GS4'),
        (r'PAPER[-_\s]?1', 'GS1'),
        (r'PAPER[-_\s]?2', 'GS2'),
        (r'PAPER[-_\s]?3', 'GS3'),
        (r'PAPER[-_\s]?4', 'GS4'),
        (r'GENERAL[-_\s]STUDIES[-_\s]?1', 'GS1'),
        (r'GENERAL[-_\s]STUDIES[-_\s]?2', 'GS2'),
        (r'GENERAL[-_\s]STUDIES[-_\s]?3', 'GS3'),
        (r'GENERAL[-_\s]STUDIES[-_\s]?4', 'GS4'),
    ]:
        if re.search(pat, name):
            paper = label
            break

    return paper, year


def is_upsc_paper(path: Path) -> bool:
    """Return True if filename suggests it's a GS paper."""
    name = path.stem.upper()
    return bool(
        re.search(r'\bGS[-_\s]?[1-4]\b', name)
        or re.search(r'\bGENERAL[-_\s]STUDIES\b', name)
        or re.search(r'\bPAPER[-_\s]?[1-4]\b', name)
    )


# ---------------------------------------------------------------------------
# Text extraction from PDF
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract all text from PDF using PyMuPDF, concatenating pages."""
    doc = fitz.open(str(pdf_path))
    pages_text = []
    for page in doc:
        text = page.get_text("text")
        if text.strip():
            pages_text.append(text)
    doc.close()
    return "\n".join(pages_text)


# ---------------------------------------------------------------------------
# Question parsing
# ---------------------------------------------------------------------------

# Regex: line starts with number + dot or paren, then a capital letter
Q_START_RE = re.compile(r'^\s*(\d+)[.\)]\s+([A-Z].+)', re.MULTILINE)

# Word limit at end of a question
WORD_LIMIT_RE = re.compile(r'\((\d+)\s+words?\)', re.IGNORECASE)

# Marks
MARKS_RE = re.compile(r'\((\d+)\s+marks?\)', re.IGNORECASE)

# Noise lines to skip (headers, footers, page numbers, watermarks)
NOISE_RE = re.compile(
    r'^(?:'
    r'\s*\d+\s*$'               # bare page number
    r'|.*upsc\.gov\.in.*'       # URL
    r'|.*union public service.*'
    r'|.*civil services.*main.*examination.*'
    r'|.*general studies.*paper\s*[i-iv]+\s*$'
    r'|.*time allowed.*'
    r'|.*maximum marks.*'
    r'|.*question paper.*'
    r'|.*instructions.*'
    r'|.*answer all.*'
    r'|.*do not.*open.*'
    r'|.*candidate.*'
    r')\s*$',
    re.IGNORECASE,
)

UPSC_START_WORDS = set()
for phrases in UPSC_KEYWORDS.values():
    for p in phrases:
        UPSC_START_WORDS.add(p.split()[0].lower())  # first word of each phrase


def clean_text(text: str) -> str:
    """Normalise whitespace, remove zero-width chars."""
    text = re.sub(r'[​‌‍﻿]', '', text)
    text = re.sub(r'\r\n|\r', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def detect_word_limit(text: str) -> int:
    m = WORD_LIMIT_RE.search(text)
    return int(m.group(1)) if m else None


def detect_marks(text: str) -> int:
    m = MARKS_RE.search(text)
    return int(m.group(1)) if m else None


def detect_q_type(text: str) -> str:
    lower = text.lower().lstrip()
    for qtype in QTYPE_ORDER:
        for phrase in UPSC_KEYWORDS[qtype]:
            if lower.startswith(phrase):
                return qtype
    # fallback: check anywhere in first 60 chars
    snippet = lower[:60]
    for qtype in QTYPE_ORDER:
        for phrase in UPSC_KEYWORDS[qtype]:
            if phrase in snippet:
                return qtype
    return "other"


def detect_subject(text: str) -> str:
    lower = text.lower()
    scores: dict[str, int] = defaultdict(int)
    for subject, keywords in SUBJECT_KEYWORDS.items():
        for kw in keywords:
            if kw in lower:
                scores[subject] += 1
    if not scores:
        return "General"
    return max(scores, key=lambda s: scores[s])


def is_valid_question(text: str) -> bool:
    """Filter out garbage lines."""
    words = text.split()
    if len(words) < 15:
        return False
    lower = text.lower().strip()
    # Must either have a question mark OR start with a known UPSC keyword
    has_qmark = '?' in text
    first_word = lower.split()[0] if words else ''
    has_upsc_start = first_word in UPSC_START_WORDS
    if not has_qmark and not has_upsc_start:
        return False
    # Must not be pure noise
    if NOISE_RE.match(text):
        return False
    return True


def make_q_id(paper: str, year: int, q_num: int) -> str:
    paper_slug = (paper or "gs0").lower().replace("-", "")
    return f"{paper_slug}_{year or 0000}_q{q_num:03d}"


def parse_questions_from_text(raw_text: str, paper: str, year: int) -> list:
    """
    Parse numbered questions from raw PDF text.
    Handles multi-line questions by collecting continuation lines.
    """
    raw_text = clean_text(raw_text)
    lines = raw_text.split('\n')

    # Remove noise lines
    cleaned_lines = []
    for line in lines:
        if NOISE_RE.match(line.strip()):
            continue
        cleaned_lines.append(line)

    text = '\n'.join(cleaned_lines)

    # Find all question starts
    matches = list(Q_START_RE.finditer(text))
    if not matches:
        return []

    records = []
    for i, m in enumerate(matches):
        q_num = int(m.group(1))
        # Question body: from current match start to next match start (or end)
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end].strip()

        # Remove the leading "N. " prefix
        chunk = re.sub(r'^\s*\d+[.\)]\s+', '', chunk)

        # Collapse internal newlines that are mid-sentence (continuation lines)
        # Keep paragraph breaks (double newline) but join single-newline continuations
        chunk = re.sub(r'(?<!\n)\n(?!\n)', ' ', chunk)
        chunk = re.sub(r'\n{2,}', '\n', chunk)
        chunk = ' '.join(chunk.split())  # final whitespace normalisation

        # Strip trailing meta like "(150 words)" or "(10 marks)"
        # but keep it for extraction first
        word_limit = detect_word_limit(chunk)
        marks = detect_marks(chunk)

        # Remove trailing parenthetical word/mark instructions for cleaner question text
        question_text = re.sub(r'\s*\(\d+\s+(?:words?|marks?)\)\s*$', '', chunk, flags=re.IGNORECASE).strip()
        question_text = re.sub(r'\s*\(\d+\s+(?:words?|marks?)\)\s*', ' ', question_text, flags=re.IGNORECASE).strip()

        if not is_valid_question(question_text):
            continue

        q_type = detect_q_type(question_text)
        subject = detect_subject(question_text)

        rec = {
            "id": make_q_id(paper, year, q_num),
            "question": question_text,
            "word_limit": word_limit,
            "marks": marks,
            "paper": paper or "Unknown",
            "year": year,
            "q_num": q_num,
            "q_type": q_type,
            "subject": subject,
            "source": "upsc_official",
        }
        records.append(rec)

    return records


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate(records: list) -> list:
    """Keep unique questions by normalised text."""
    seen: set[str] = set()
    unique = []
    for rec in records:
        # Normalise for comparison
        key = re.sub(r'\W+', ' ', rec['question'].lower()).strip()
        if key not in seen:
            seen.add(key)
            unique.append(rec)
    return unique


# ---------------------------------------------------------------------------
# Answer generation from pretrain data
# ---------------------------------------------------------------------------

def load_pretrain_chunks(pretrain_path: Path) -> list:
    """Load unified_pretrain.jsonl into memory."""
    chunks = []
    if not pretrain_path.exists():
        print(f"[WARN] Pretrain file not found: {pretrain_path}", file=sys.stderr)
        return chunks
    with open(pretrain_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    chunks.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return chunks


def find_relevant_chunks(question: str, chunks: list, top_k: int = 3) -> list:
    """
    Simple keyword-overlap retrieval from pretrain chunks.
    Returns up to top_k chunk texts ranked by keyword overlap.
    """
    # Extract meaningful words from question (length >= 5, skip stopwords)
    STOPWORDS = {
        'which', 'what', 'where', 'when', 'while', 'that', 'this', 'with',
        'from', 'have', 'been', 'would', 'could', 'should', 'their', 'there',
        'about', 'after', 'before', 'between', 'through', 'during', 'against',
        'under', 'india', 'indian', 'discuss', 'examine', 'critically', 'analyse',
        'comment', 'assess', 'evaluate', 'explain', 'describe', 'highlight',
        'elucidate', 'enumerate', 'justify', 'compare', 'distinguish',
    }
    words = set(
        w.lower() for w in re.findall(r'[a-zA-Z]{5,}', question)
        if w.lower() not in STOPWORDS
    )
    if not words:
        return []

    scored: list[tuple[int, str]] = []
    for chunk in chunks:
        text_field = chunk.get('text', chunk.get('content', ''))
        if not text_field:
            continue
        chunk_words = set(re.findall(r'[a-zA-Z]{5,}', text_field.lower()))
        score = len(words & chunk_words)
        if score > 0:
            scored.append((score, text_field))

    scored.sort(key=lambda x: -x[0])
    return [t for _, t in scored[:top_k]]


def synthesize_answer(question: str, context_chunks: list, word_limit: int) -> str:
    """
    Build a simple template-based answer from context chunks.
    (For real answers, integrate an LLM API call here.)
    """
    target_words = word_limit or 150
    intro = f"The question pertains to an important aspect of Indian governance, society, and development. "

    body_parts = []
    for chunk in context_chunks:
        # Take first 3 sentences from each chunk
        sentences = re.split(r'(?<=[.!?])\s+', chunk.strip())
        snippet = ' '.join(sentences[:3])
        if snippet:
            body_parts.append(snippet)

    body = ' '.join(body_parts)

    # Trim to approximate word limit
    body_words = body.split()
    if len(body_words) > target_words - 30:
        body_words = body_words[:target_words - 30]
        body = ' '.join(body_words) + '.'

    conclusion = " Therefore, a nuanced and multi-dimensional approach is essential to address this issue in the Indian context."

    answer = intro + body + conclusion
    return answer.strip()


def add_answers_to_records(
    records: list,
    pretrain_path: Path,
    output_path: Path,
) -> None:
    """
    For each record, find relevant pretrain chunks and synthesize an answer.
    Writes full SFT JSONL with Gemma chat template.
    """
    print(f"\n[INFO] Loading pretrain chunks from {pretrain_path} ...")
    chunks = load_pretrain_chunks(pretrain_path)
    print(f"[INFO] Loaded {len(chunks):,} pretrain chunks.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sft_path = output_path.parent / "past_papers_sft.jsonl"

    written = 0
    with open(sft_path, 'w', encoding='utf-8') as out_f:
        for i, rec in enumerate(records):
            ctx = find_relevant_chunks(rec['question'], chunks, top_k=3)
            answer = synthesize_answer(rec['question'], ctx, rec.get('word_limit'))

            sft_rec = {
                **rec,
                "answer": answer,
                "text": GEMMA_TEMPLATE.format(
                    question=rec['question'],
                    answer=answer,
                ),
                "context_chunks_used": len(ctx),
            }
            out_f.write(json.dumps(sft_rec, ensure_ascii=False) + '\n')
            written += 1

            if (i + 1) % 100 == 0:
                print(f"  ... processed {i + 1}/{len(records)} questions", flush=True)

    print(f"[OK] SFT records written: {written:,} → {sft_path}")


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def print_stats(records: list, output_path: Path = None) -> None:
    by_paper: Counter = Counter(r['paper'] for r in records)
    by_year: Counter = Counter(r['year'] for r in records)
    by_type: Counter = Counter(r['q_type'] for r in records)
    by_subject: Counter = Counter(r['subject'] for r in records)

    print("\n" + "=" * 60)
    print(f"  Papers found     : (see below)")
    print(f"  Questions found  : {len(records):,}")
    print()
    paper_str = ', '.join(f"{p}={c}" for p, c in sorted(by_paper.items()))
    print(f"  By paper: {paper_str}")

    year_parts = []
    for y in sorted(by_year.keys(), reverse=True):
        if y:
            year_parts.append(f"{y}={by_year[y]}")
    print(f"  By year:  {', '.join(year_parts)}")

    type_parts = [f"{t}={c}" for t, c in by_type.most_common(8)]
    print(f"  By type:  {', '.join(type_parts)}")

    subj_parts = [f"{s}={c}" for s, c in by_subject.most_common(8)]
    print(f"  By subject: {', '.join(subj_parts)}")

    if output_path:
        print(f"\n  Saved to: {output_path}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_pdf_files(papers_dir: Path, years: tuple = None) -> list[Path]:
    """Recursively find PDFs that look like UPSC Mains papers."""
    all_pdfs = list(papers_dir.rglob("*.pdf")) + list(papers_dir.rglob("*.PDF"))
    result = []
    for p in sorted(all_pdfs):
        name_upper = p.stem.upper()
        # Accept GS, General Studies, or any paper-numbered file
        looks_like_paper = (
            is_upsc_paper(p)
            or 'GS' in name_upper
            or 'GENERAL' in name_upper
            or 'MAINS' in name_upper
        )
        if not looks_like_paper:
            continue
        if years:
            _, year = parse_pdf_filename(p)
            if year and not (years[0] <= year <= years[1]):
                continue
        result.append(p)
    return result


def parse_year_range(s: str) -> tuple[int, int]:
    """Parse '2010-2024' → (2010, 2024)."""
    parts = s.split('-')
    if len(parts) == 2:
        return int(parts[0]), int(parts[1])
    if len(parts) == 1:
        y = int(parts[0])
        return y, y
    raise ValueError(f"Cannot parse year range: {s!r}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_single_pdf(pdf_path: Path) -> list:
    """Extract questions from one PDF file."""
    paper, year = parse_pdf_filename(pdf_path)
    if not paper:
        paper = "Unknown"
    if not year:
        # Try to find year in parent directory name
        for part in pdf_path.parts:
            m = re.search(r'(20[0-2]\d)', part)
            if m:
                year = int(m.group(1))
                break

    print(f"  Reading: {pdf_path.name}  →  paper={paper}, year={year}", flush=True)
    try:
        raw_text = extract_text_from_pdf(pdf_path)
    except Exception as e:
        print(f"  [ERROR] Could not read {pdf_path.name}: {e}", file=sys.stderr)
        return []

    if not raw_text.strip():
        print(f"  [WARN] No text extracted from {pdf_path.name}", file=sys.stderr)
        return []

    records = parse_questions_from_text(raw_text, paper, year)
    print(f"         → {len(records)} questions extracted", flush=True)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract UPSC Mains Q&A pairs from official question paper PDFs."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--papers-dir', type=Path, default=None,
        help="Folder containing UPSC Mains PDFs (searched recursively). Default: upsc_papers/"
    )
    group.add_argument(
        '--file', type=Path, default=None,
        help="Process a single PDF file."
    )
    parser.add_argument(
        '--years', type=str, default=None,
        help="Year range to filter, e.g. 2010-2024."
    )
    parser.add_argument(
        '--output-dir', type=Path, default=None,
        help="Output directory. Default: dataset_output_final/combined/"
    )
    parser.add_argument(
        '--add-answers', action='store_true',
        help="After extracting, synthesize answers from pretrain data and write SFT JSONL."
    )
    parser.add_argument(
        '--pretrain', type=Path, default=None,
        help="Path to unified_pretrain.jsonl (used with --add-answers). "
             "Default: dataset_output_final/combined/unified_pretrain.jsonl"
    )
    parser.add_argument(
        '--stats-only', action='store_true',
        help="Print stats from existing output JSONL (do not re-extract)."
    )
    args = parser.parse_args()

    # Resolve paths relative to script location or cwd
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent  # DataSets/

    default_output = repo_root / "dataset_output_final" / "combined"
    output_dir = args.output_dir or default_output
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "past_papers_questions.jsonl"

    pretrain_path = args.pretrain or (repo_root / "dataset_output_final" / "combined" / "unified_pretrain_clean.jsonl")

    # --stats-only: load existing JSONL and print stats
    if args.stats_only:
        if not output_path.exists():
            print(f"[ERROR] No output file found at {output_path}. Run extraction first.", file=sys.stderr)
            sys.exit(1)
        records = []
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        print(f"[INFO] Loaded {len(records):,} records from {output_path}")
        print_stats(records, output_path)
        return

    # Gather PDFs
    if args.file:
        pdf_files = [args.file.resolve()]
        if not pdf_files[0].exists():
            print(f"[ERROR] File not found: {pdf_files[0]}", file=sys.stderr)
            sys.exit(1)
    else:
        papers_dir = (args.papers_dir or repo_root / "upsc_papers").resolve()
        if not papers_dir.exists():
            # Also try relative to cwd
            papers_dir_cwd = Path.cwd() / (args.papers_dir or "upsc_papers")
            if papers_dir_cwd.exists():
                papers_dir = papers_dir_cwd
            else:
                print(f"[ERROR] Papers directory not found: {papers_dir}", file=sys.stderr)
                print("        Create it and drop UPSC Mains PDFs inside.", file=sys.stderr)
                sys.exit(1)

        year_range = None
        if args.years:
            year_range = parse_year_range(args.years)

        print(f"[INFO] Scanning: {papers_dir}")
        pdf_files = find_pdf_files(papers_dir, year_range)
        print(f"[INFO] Found {len(pdf_files)} PDF file(s) matching criteria.")

        if not pdf_files:
            print("[WARN] No matching PDFs found. Check --papers-dir and --years.")
            sys.exit(0)

    # Extract
    all_records: list = []
    for pdf_path in pdf_files:
        recs = process_single_pdf(pdf_path)
        all_records.extend(recs)

    if not all_records:
        print("[WARN] No questions extracted from any PDF.")
        sys.exit(0)

    # Deduplication
    before = len(all_records)
    all_records = deduplicate(all_records)
    after = len(all_records)
    if before != after:
        print(f"[INFO] Deduplication: {before} → {after} questions ({before - after} duplicates removed)")

    # Write questions JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    print(f"\n[OK] Saved {len(all_records):,} questions → {output_path}")
    print_stats(all_records, output_path)

    # Optionally add answers
    if args.add_answers:
        add_answers_to_records(all_records, pretrain_path, output_path)


if __name__ == '__main__':
    main()
