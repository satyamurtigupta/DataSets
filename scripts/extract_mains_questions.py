#!/usr/bin/env python3
"""
Script 1 of 3: Extract UPSC Mains Questions — PDF-to-LLM Vision approach.

Sends each PDF's pages directly as images to GPT-4o (vision) or Gemini.
No OCR, no text extraction — the LLM reads the actual PDF layout.

This correctly handles:
  - Ethics (GS4) case studies with multiple sub-questions
  - Bilingual PDFs (LLM ignores Hindi automatically)
  - Scanned PDFs (LLM does its own OCR)
  - Multi-line questions that wrap across lines

Output: dataset_output_final/combined/extracted_questions.jsonl

Run:
    python3 scripts/extract_mains_questions.py
    python3 scripts/extract_mains_questions.py --year 2025 --verbose
    python3 scripts/extract_mains_questions.py --year 2024 --append
    python3 scripts/extract_mains_questions.py --api gemini
    python3 scripts/extract_mains_questions.py --year 2019 --dpi 200   (faster OCR)
"""

import re
import gc
import json
import sys
import time
import base64
import argparse
from io import BytesIO
from pathlib import Path
from typing import Optional, List, Tuple

try:
    import fitz  # PyMuPDF — used to render PDF pages as images
except ImportError:
    print("ERROR: PyMuPDF not installed. Run: pip install pymupdf")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow not installed. Run: pip install Pillow")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths & Config
# ---------------------------------------------------------------------------
PAPERS_DIR  = Path("upsc_papers/mains")
OUTPUT_DIR  = Path("dataset_output_final/combined")
OUTPUT_FILE = OUTPUT_DIR / "extracted_questions.jsonl"
ENV_FILE    = Path("env/.env")

# DPI for rendering PDF pages as images (150 = fast, 200 = balanced, 300 = best)
DEFAULT_DPI = 200

# ---------------------------------------------------------------------------
# Load API keys
# ---------------------------------------------------------------------------

def load_env() -> dict:
    env = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, _, v = line.partition('=')
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env

ENV = load_env()
OPENAI_API_KEY   = ENV.get("OPENAI_API_KEY", "")
GEMINI_API_KEY   = ENV.get("GEMINI_API_KEY", "")


# ---------------------------------------------------------------------------
# Paper / Year helpers
# ---------------------------------------------------------------------------

def get_paper_label(pdf_path: Path) -> str:
    fname = pdf_path.stem.lower()

    if 'essay' in fname or 'eassy' in fname:
        return 'Essay'

    prefix_match = re.match(r'^(\d{2})\s', pdf_path.stem)
    if prefix_match:
        mapping = {'00': 'Essay', '02': 'GS1', '03': 'GS2', '04': 'GS3', '05': 'GS4'}
        if prefix_match.group(1) in mapping:
            return mapping[prefix_match.group(1)]

    for gs, n in [('gs4','GS4'),('gs-4','GS4'),('paper-4','GS4'),('paper4','GS4'),
                  ('gs3','GS3'),('gs-3','GS3'),('paper-3','GS3'),('paper3','GS3'),
                  ('gs2','GS2'),('gs-2','GS2'),('paper-2','GS2'),('paper2','GS2'),
                  ('gs1','GS1'),('gs-1','GS1'),('paper-1','GS1'),('paper1','GS1')]:
        if gs in fname:
            return n

    if re.search(r'\bpaper[\s\-]*iv\b|\bgs[\s\-]*iv\b', fname): return 'GS4'
    if re.search(r'\bpaper[\s\-]*iii\b|\bgs[\s\-]*iii\b', fname): return 'GS3'
    if re.search(r'\bpaper[\s\-]*ii\b|\bgs[\s\-]*ii\b', fname): return 'GS2'
    if re.search(r'\bpaper[\s\-]*i\b|\bgs[\s\-]*i\b', fname): return 'GS1'

    if any(w in fname for w in ['polity','constitution','governance']): return 'GS2'
    if any(w in fname for w in ['ethics','integrity','aptitude']): return 'GS4'
    if any(w in fname for w in ['economy','science','environment','security']): return 'GS3'
    if any(w in fname for w in ['history','geography','society','art','culture']): return 'GS1'

    num_match = re.search(r'[\s_\-]([1-4])(?:\s|$)', fname)
    if num_match:
        return f'GS{num_match.group(1)}'

    return 'Unknown'


def get_year_from_path(pdf_path: Path) -> int:
    for part in pdf_path.parts:
        if re.match(r'^20\d\d$', part):
            return int(part)
    m = re.search(r'(20\d\d)', pdf_path.name)
    return int(m.group(1)) if m else 0


# ---------------------------------------------------------------------------
# Question type / subject detection
# ---------------------------------------------------------------------------

QTYPE_PATTERNS = [
    (r'\bcritically\s+examine\b',              'critically_examine'),
    (r'\bcritically\s+analyse\b',              'critically_analyse'),
    (r'\bcritically\s+evaluate\b',             'critically_evaluate'),
    (r'\bcritically\s+comment\b',              'critically_comment'),
    (r'\bcritically\s+assess\b',               'critically_assess'),
    (r'\bhow\s+far\b',                         'how_far'),
    (r'\bto\s+what\s+extent\b',                'to_what_extent'),
    (r'\bdo\s+you\s+(think|agree|believe)\b',  'do_you_agree'),
    (r'\bwhat\s+do\s+you\s+(think|understand)\b', 'what_do_you_think'),
    (r'\bin\s+the\s+light\s+of\b',             'in_light_of'),
    (r'\bthrow\s+light\b',                     'throw_light'),
    (r'\bwith\s+reference\s+to\b',             'with_reference_to'),
    (r'\bbring\s+out\b',                       'bring_out'),
    (r'^\s*comment\b',    'comment'),
    (r'^\s*discuss\b',    'discuss'),
    (r'^\s*explain\b',    'explain'),
    (r'^\s*examine\b',    'examine'),
    (r'^\s*analyse\b',    'analyse'),
    (r'^\s*evaluate\b',   'evaluate'),
    (r'^\s*describe\b',   'describe'),
    (r'^\s*illustrate\b', 'illustrate'),
    (r'^\s*elucidate\b',  'elucidate'),
    (r'^\s*elaborate\b',  'elaborate'),
    (r'^\s*trace\b',      'trace'),
    (r'^\s*compare\b',    'compare'),
    (r'^\s*contrast\b',   'compare'),
    (r'^\s*differentiate\b', 'compare'),
    (r'^\s*highlight\b',  'highlight'),
    (r'^\s*justify\b',    'justify'),
    (r'^\s*assess\b',     'assess'),
    (r'^\s*distinguish\b', 'compare'),
    (r'^\s*write\b',      'write_note'),
    (r'^\s*suggest\b',    'suggest'),
    (r'^\s*(what|why|how|where|who|which)\b',  'factual'),
]

def detect_question_type(text: str) -> str:
    lower = text.lower()
    for pattern, qtype in QTYPE_PATTERNS:
        if re.search(pattern, lower, re.IGNORECASE):
            return qtype
    return 'other'


def detect_subject(text: str, paper: str) -> str:
    lower = text.lower()
    if paper == 'Essay': return 'Essay'
    paper_num = re.search(r'\d', paper)
    gs_num = int(paper_num.group()) if paper_num else 0

    if gs_num == 1:
        if any(w in lower for w in ['history','revolt','independence','colonial','mughal',
                                     'nationalist','partition','freedom','british','ancient',
                                     'medieval','vedic','maurya','gupta','gandhi','nehru',
                                     'rebellion','reform','viceroy']):
            return 'History'
        if any(w in lower for w in ['society','caste','tribe','women','social','culture',
                                     'religion','festival','language','art','architecture',
                                     'sculpture','temple','craft','folk','empowerment']):
            return 'Society_Culture'
        if any(w in lower for w in ['monsoon','river','drought','earthquake','geography',
                                     'climate','soil','mineral','forest','biodiversity',
                                     'ecosystem','watershed','mountain','plateau','ocean']):
            return 'Geography'
        return 'GS1_Other'

    if gs_num == 2:
        if any(w in lower for w in ['constitution','parliament','president','judiciary','court',
                                     'rights','directive','amendment','federal','governor',
                                     'election','fundamental','democratic','legislature']):
            return 'Polity'
        if any(w in lower for w in ['united nations','nato','bilateral','foreign policy',
                                     'geopolitic','border','treaty','china','pakistan',
                                     'international','diaspora','saarc','brics','g20']):
            return 'International_Relations'
        if any(w in lower for w in ['scheme','welfare','health','education','poverty',
                                     'employment','mgnrega','food security','governance',
                                     'policy','administration','e-governance','rti']):
            return 'Governance_Social'
        return 'GS2_Other'

    if gs_num == 3:
        if any(w in lower for w in ['economy','gdp','inflation','budget','fiscal','tax',
                                     'monetary','rbi','banking','growth','agriculture',
                                     'farmer','msp','industry','infrastructure','trade']):
            return 'Economy'
        if any(w in lower for w in ['environment','climate change','pollution','renewable',
                                     'solar','carbon','emission','sustainable','biodiversity',
                                     'wetland','coral','ozone','deforestation']):
            return 'Environment'
        if any(w in lower for w in ['science','technology','space','isro','artificial intelligence',
                                     'machine learning','biotechnology','cyber','digital',
                                     'drone','robotics','quantum']):
            return 'Science_Tech'
        if any(w in lower for w in ['disaster','security','terrorism','naxal','insurgency',
                                     'internal security','left wing','cyber crime']):
            return 'Security'
        return 'GS3_Other'

    if gs_num == 4:
        return 'Ethics'
    return 'Unknown'


# ---------------------------------------------------------------------------
# PDF → Images (using PyMuPDF — no pdf2image needed)
# ---------------------------------------------------------------------------

def pdf_to_images_base64(pdf_path: Path, dpi: int = 200) -> List[str]:
    """
    Render each PDF page as a JPEG image and return list of base64 strings.
    Uses PyMuPDF (fitz) directly — no pdf2image dependency.
    """
    images_b64 = []
    try:
        doc = fitz.open(str(pdf_path))
        mat = fitz.Matrix(dpi / 72, dpi / 72)  # scale factor from 72 DPI base

        for page in doc:
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            # Convert to PIL Image for JPEG compression
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode()
            images_b64.append(b64)
            del pix
        doc.close()
        gc.collect()
    except Exception as e:
        print(f"    ❌ Could not render PDF pages: {e}")
    return images_b64


# ---------------------------------------------------------------------------
# LLM extraction prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a specialist extracting UPSC Civil Services Mains examination questions.
Return ONLY a valid JSON array — no markdown, no explanation, no code fences."""

def build_user_prompt(year: int, paper: str) -> str:
    is_ethics = (paper == 'GS4')

    ethics_note = """
ETHICS PAPER SPECIAL RULES:
- Case studies are long questions with sub-questions (a), (b), (c) etc.
- Treat the ENTIRE case study (including all sub-questions) as ONE question entry.
- Include the full case study text + all sub-questions in the "text" field.
- Do NOT split sub-questions (a), (b), (c) into separate entries.
- Short standalone ethical questions (without a case) are separate entries.
""" if is_ethics else ""

    return f"""Extract all UPSC Mains examination questions from the PDF pages shown.

PDF Info:  Year = {year}  |  Paper = {paper}

RULES:
1. Extract ONLY complete English questions. The PDF may be bilingual (Hindi + English) — ignore all Hindi text.
2. Join multi-line questions into one complete sentence.
3. Include the directive at the end: "Discuss.", "Critically examine.", "Explain with examples." etc.
4. SKIP: section headings, exam instructions, page numbers, word-limit directives like "(Answer in 150 words)".
5. word_limit: extract number from "(Answer in NNN words)" if shown near the question, else null.
6. marks: 10 if word_limit ≤ 150, 15 if word_limit = 250, else null.
7. Number questions sequentially from 1 matching the paper's own numbering.
{ethics_note}
Return a JSON array with this exact schema — every field required:
[
  {{
    "q_num": <integer>,
    "text": "<complete English question text, including all sub-questions if ethics case>",
    "word_limit": <integer or null>,
    "marks": <10 or 15 or null>
  }}
]"""


# ---------------------------------------------------------------------------
# OpenAI vision call
# ---------------------------------------------------------------------------

def parse_retry_after(error_body: str) -> float:
    """Extract wait seconds from OpenAI 429 error message. Returns seconds to wait."""
    # e.g. "Please try again in 548ms" or "Please try again in 1.2s"
    m = re.search(r'try again in ([\d.]+)(ms|s)', error_body)
    if m:
        val  = float(m.group(1))
        unit = m.group(2)
        secs = val / 1000.0 if unit == 'ms' else val
        return max(secs + 3.0, 5.0)   # add 3s buffer, minimum 5s
    return 15.0   # safe default


def call_openai_vision(images_b64: List[str], year: int, paper: str,
                       model: str = "gpt-4o-mini") -> tuple:
    """
    Returns (result, retry_after_seconds).
    result is list of dicts on success, None on failure.
    retry_after_seconds is how long to wait before retrying (from 429 header).
    """
    import urllib.request
    import urllib.error

    content = [{"type": "text", "text": build_user_prompt(year, paper)}]
    for b64 in images_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"},
        })

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": content},
        ],
        "temperature": 0.1,
        "max_tokens": 6000,
        "response_format": {"type": "json_object"},
    }).encode()

    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read())
        content_str = data["choices"][0]["message"]["content"]
        parsed = json.loads(content_str)
        if isinstance(parsed, list):
            return parsed, 0
        for v in parsed.values():
            if isinstance(v, list):
                return v, 0
        return [], 0
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        if e.code == 429:
            wait = parse_retry_after(body)
            print(f"    ⚠  Rate limit — will wait {wait:.0f}s before retry")
            return None, wait
        print(f"    ❌ OpenAI HTTP {e.code}: {body[:200]}")
        return None, 5.0
    except Exception as e:
        print(f"    ❌ OpenAI error: {e}")
        return None, 5.0


# ---------------------------------------------------------------------------
# Gemini vision call (native PDF support)
# ---------------------------------------------------------------------------

def call_gemini_vision(pdf_path: Path, year: int, paper: str,
                       model: str = "gemini-2.0-flash") -> Optional[List[dict]]:
    """
    Upload PDF to Gemini Files API, then extract questions.
    Gemini supports native PDF reading — no image conversion needed.
    """
    import urllib.request
    import urllib.error

    # Step 1: Upload PDF file
    pdf_bytes = pdf_path.read_bytes()
    upload_url = (
        f"https://generativelanguage.googleapis.com/upload/v1beta/files"
        f"?uploadType=media&key={GEMINI_API_KEY}"
    )
    upload_req = urllib.request.Request(
        upload_url, data=pdf_bytes,
        headers={"Content-Type": "application/pdf", "X-Goog-Upload-Protocol": "raw"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(upload_req, timeout=60) as resp:
            file_data = json.loads(resp.read())
        file_uri = file_data["file"]["uri"]
    except Exception as e:
        print(f"    ❌ Gemini file upload failed: {e}")
        return None

    # Step 2: Generate content using the uploaded file
    prompt = SYSTEM_PROMPT + "\n\n" + build_user_prompt(year, paper)
    payload = json.dumps({
        "contents": [{
            "parts": [
                {"text": prompt},
                {"file_data": {"mime_type": "application/pdf", "file_uri": file_uri}},
            ]
        }],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 6000,
            "responseMimeType": "application/json",
        },
    }).encode()

    gen_url = (f"https://generativelanguage.googleapis.com/v1beta/models"
               f"/{model}:generateContent?key={GEMINI_API_KEY}")
    gen_req = urllib.request.Request(
        gen_url, data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(gen_req, timeout=180) as resp:
            data = json.loads(resp.read())
        content_str = data["candidates"][0]["content"]["parts"][0]["text"]
        content_str = re.sub(r'^```(?:json)?\s*', '', content_str.strip())
        content_str = re.sub(r'\s*```$', '', content_str.strip())
        parsed = json.loads(content_str)
        if isinstance(parsed, list):
            return parsed
        for v in parsed.values():
            if isinstance(v, list):
                return v
        return []
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"    ❌ Gemini HTTP {e.code}: {body[:300]}")
        return None
    except Exception as e:
        print(f"    ❌ Gemini error: {e}")
        return None


# ---------------------------------------------------------------------------
# LLM dispatcher with retry
# ---------------------------------------------------------------------------

def call_llm(pdf_path: Path, year: int, paper: str, api: str,
             dpi: int = 150, verbose: bool = False,
             retry: int = 5, delay: float = 10.0) -> Optional[List[dict]]:

    if api == "gemini":
        # Gemini: retry simple — no image caching needed
        for attempt in range(1, retry + 1):
            if attempt > 1:
                wait = delay * attempt
                print(f"    ⏳ Retry {attempt}/{retry} in {wait:.0f}s...")
                time.sleep(wait)
            result = call_gemini_vision(pdf_path, year, paper)
            if result is not None:
                if verbose:
                    print(f"    LLM returned {len(result)} entries")
                return result
        return None

    elif api == "openai":
        # Render images ONCE — reuse across all retries
        print(f"    🖼  Rendering {fitz.open(str(pdf_path)).page_count} pages at {dpi} DPI...")
        images_b64 = pdf_to_images_base64(pdf_path, dpi=dpi)
        if not images_b64:
            print(f"    ❌ Could not render pages")
            return None
        print(f"    📤 Sending {len(images_b64)} page images to GPT-4o...")

        for attempt in range(1, retry + 1):
            if attempt > 1:
                print(f"    📤 Retry {attempt}/{retry} — re-sending cached images...")

            result, retry_after = call_openai_vision(images_b64, year, paper)

            if result is not None:
                if verbose:
                    print(f"    LLM returned {len(result)} entries")
                return result

            # Wait the time the API told us, or the default delay
            wait = max(retry_after, delay)
            print(f"    ⏳ Waiting {wait:.0f}s before retry...")
            time.sleep(wait)

        return None

    else:
        print(f"    ❌ Unknown API: {api}")
        return None


# ---------------------------------------------------------------------------
# Post-LLM quality filtering
# ---------------------------------------------------------------------------

def quality_filter(raw_q: dict, year: int, paper: str, idx: int) -> Optional[dict]:
    text = str(raw_q.get("text", "")).strip()

    if len(text) < 15:
        return None

    # Reject predominantly Hindi (Devanagari)
    deva = sum(1 for c in text if 'ऀ' <= c <= 'ॿ')
    if deva / max(len(text), 1) > 0.10:
        return None

    # Must have enough English words
    english_words = [w for w in text.split() if re.match(r'^[A-Za-z]{2,}', w)]
    if len(english_words) < 5:
        return None

    # Clean up
    text = re.sub(r'\s{2,}', ' ', text).strip()
    text = re.sub(r'\s*\(Answer\s+in\s+\d+\s+words?\)\s*', ' ', text, flags=re.I).strip()
    text = re.sub(r'\s+\b(10|15)\b\s*$', '', text).strip()

    # word_limit
    word_limit = raw_q.get("word_limit")
    try:
        word_limit = int(word_limit) if word_limit is not None else None
    except (ValueError, TypeError):
        word_limit = None

    # marks
    marks = raw_q.get("marks")
    try:
        marks = int(marks) if marks is not None else None
    except (ValueError, TypeError):
        marks = None
    if marks is None and word_limit:
        marks = 10 if word_limit <= 150 else 15 if word_limit == 250 else None

    # q_num
    try:
        q_num = int(raw_q.get("q_num", idx + 1))
    except (ValueError, TypeError):
        q_num = idx + 1

    return {
        "q_num":      q_num,
        "text":       text,
        "word_limit": word_limit,
        "marks":      marks,
        "q_type":     detect_question_type(text),
        "subject":    detect_subject(text, paper),
        "year":       year,
        "paper":      paper,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Extract UPSC Mains questions by sending PDFs directly to LLM vision"
    )
    ap.add_argument("--papers-dir", default=str(PAPERS_DIR))
    ap.add_argument("--year",    type=int, default=None, help="Process only this year")
    ap.add_argument("--paper",   default=None, help="Filter: GS1/GS2/GS3/GS4/Essay")
    ap.add_argument("--output",  default=str(OUTPUT_FILE))
    ap.add_argument("--api",     default=None,
                    help="LLM: 'openai' or 'gemini' (auto-detected from env/.env)")
    ap.add_argument("--dpi",     type=int, default=150,
                    help="Page render DPI (150=fast/default, 200=balanced, 300=best)")
    ap.add_argument("--append",  action="store_true",
                    help="Append to existing output file (for year-by-year runs)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--retry",   type=int, default=5,  help="LLM retry attempts per PDF")
    ap.add_argument("--delay",   type=float, default=10.0,
                    help="Seconds to wait between PDFs / after rate limit (default 10)")
    args = ap.parse_args()

    # ── Determine API ────────────────────────────────────────────────────────
    api = None
    if args.api:
        api = args.api.lower()
    elif OPENAI_API_KEY:
        api = "openai"
    elif GEMINI_API_KEY:
        api = "gemini"
    else:
        print("❌ No API key found in env/.env")
        print("   Add OPENAI_API_KEY or GEMINI_API_KEY to env/.env")
        sys.exit(1)

    print(f"LLM  : {api.upper()}  (vision / PDF-native)")
    print(f"DPI  : {args.dpi}")

    # ── Collect PDFs ─────────────────────────────────────────────────────────
    papers_dir  = Path(args.papers_dir)
    output_file = Path(args.output)

    if not papers_dir.exists():
        print(f"ERROR: papers dir not found: {papers_dir}")
        sys.exit(1)

    pdfs = sorted(papers_dir.rglob("*.pdf"))
    if args.year:
        pdfs = [p for p in pdfs if str(args.year) in str(p)]
    if args.paper:
        pdfs = [p for p in pdfs if get_paper_label(p) == args.paper]

    # Deduplicate: keep the PDF in its own year folder
    seen_names = {}
    deduped = []
    for p in pdfs:
        name = p.name
        yr   = get_year_from_path(p)
        if name not in seen_names:
            seen_names[name] = (p, yr)
            deduped.append(p)
        else:
            prev_p, prev_yr = seen_names[name]
            yr_in_name = re.search(r'(20\d\d)', name)
            yr_in_name = int(yr_in_name.group(1)) if yr_in_name else 0
            if yr == yr_in_name and prev_yr != yr_in_name:
                deduped = [q for q in deduped if q.name != name]
                deduped.append(p)
                seen_names[name] = (p, yr)
    pdfs = sorted(deduped)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # ── Restartability: purge existing data for the years being processed ─────
    # Determine which years this run will touch
    years_in_run = set(get_year_from_path(p) for p in pdfs)
    # Also filter by --paper if set (only purge that paper's entries)
    papers_in_run = {args.paper} if args.paper else None

    existing_kept = []   # questions from OTHER years that we keep
    purged_count  = 0

    if args.append and output_file.exists():
        with open(output_file, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    q = json.loads(line)
                    q_year  = q.get('year')
                    q_paper = q.get('paper')
                    # Keep if it's a different year, or different paper within same year
                    if q_year not in years_in_run:
                        existing_kept.append(q)
                    elif papers_in_run and q_paper not in papers_in_run:
                        existing_kept.append(q)
                    else:
                        purged_count += 1
                except json.JSONDecodeError:
                    pass

        kept_count = len(existing_kept)
        if purged_count:
            print(f"♻️  Purged  {purged_count} old entries for year(s) {sorted(years_in_run)} "
                  f"→ will re-extract fresh")
        if kept_count:
            print(f"📦 Keeping {kept_count} existing entries from other years")

    print("=" * 60)
    print("UPSC Mains Question Extractor — PDF Vision Mode")
    print(f"PDFs   : {len(pdfs)}")
    print(f"Output : {output_file}")
    print("=" * 60)

    all_questions = list(existing_kept)
    by_paper = {}
    by_year  = {}

    # Write back kept questions + new ones (always overwrite at this point)
    outf = open(output_file, 'w', encoding='utf-8')
    for q in existing_kept:
        outf.write(json.dumps(q, ensure_ascii=False) + '\n')
    outf.flush()

    for i, pdf_path in enumerate(pdfs):
        paper = get_paper_label(pdf_path)
        year  = get_year_from_path(pdf_path)

        print(f"\n📄 [{i+1}/{len(pdfs)}] {pdf_path.parent.name}/{pdf_path.name}  [{paper}] [{year}]")

        raw_results = call_llm(
            pdf_path, year, paper, api=api, dpi=args.dpi,
            verbose=args.verbose, retry=args.retry, delay=args.delay,
        )

        if raw_results is None:
            print(f"    ⚠  LLM extraction failed — skipping")
            continue

        print(f"    LLM returned {len(raw_results)} entries")

        accepted = 0
        for j, rq in enumerate(raw_results):
            q = quality_filter(rq, year, paper, j)
            if q is None:
                continue
            if args.verbose:
                print(f"   [{q['q_num']:2d}] [{q['q_type']:20s}] {q['text'][:80]}")
            q['id'] = f"mains_{q['year']}_{q['paper']}_q{q['q_num']:02d}"
            outf.write(json.dumps(q, ensure_ascii=False) + '\n')
            outf.flush()
            all_questions.append(q)
            by_paper[q['paper']] = by_paper.get(q['paper'], 0) + 1
            by_year[q['year']]   = by_year.get(q['year'], 0) + 1
            accepted += 1

        print(f"    ✅ {accepted} questions saved")

        if i < len(pdfs) - 1:
            time.sleep(args.delay)

    outf.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    # Recount from the full file (includes kept + new)
    full_by_year  = {}
    full_by_paper = {}
    for q in all_questions:
        full_by_year[q['year']]   = full_by_year.get(q['year'], 0) + 1
        full_by_paper[q['paper']] = full_by_paper.get(q['paper'], 0) + 1

    print(f"\n{'='*60}")
    print(f"TOTAL IN FILE   : {len(all_questions)}")
    print(f"\nBy year (full file):")
    for yr, cnt in sorted(full_by_year.items()):
        tag = ' ← this run' if yr in years_in_run else ''
        print(f"  {yr}  {cnt:>4}{tag}")
    print(f"\nBy paper (full file):")
    for p, cnt in sorted(full_by_paper.items()):
        print(f"  {p:<10} {cnt:>4}")
    print(f"\n✅ Saved → {output_file}")


if __name__ == "__main__":
    main()
