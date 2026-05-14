#!/usr/bin/env python3
"""
pdf_extractor.py

High-quality PDF → JSONL dataset extractor for Gemma 2 SLM domain adaptation.

Key features:
- Per-page strategy selection: PyMuPDF → PDFPlumber → OCR (Tesseract)
- Quality scoring detects and routes corrupt pages (Type3 fonts, scanned) to OCR
- Handles repeated-char corruption (TThhiiss) and symbol garbage (✎☞✒)
- Header/footer removal across pages
- Sentence-boundary chunking with configurable overlap
- Outputs: JSONL pretrain + Gemma 2 SFT format + HuggingFace Dataset

Usage:
    # Single file
    python scripts/pdf_extractor.py upsc_pdfs/History/NCERT.pdf --subject History

    # Subject folder (all PDFs in folder)
    python scripts/pdf_extractor.py upsc_pdfs/History/ --output dataset_output/

    # Entire tree (auto-detects subjects from sub-folder names)
    python scripts/pdf_extractor.py upsc_pdfs/ --output dataset_output/

    # Dry run - shows OCR plan without extracting
    python scripts/pdf_extractor.py upsc_pdfs/Chemistry/ --dry-run
"""

import argparse
import gc
import json
import logging
import os
import re
import sys
import time
import unicodedata
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import pandas as pd
import pdfplumber
from pdf2image import convert_from_path
from PIL import Image

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False

try:
    from datasets import Dataset, DatasetDict
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PDFExtractorConfig:
    chunk_size_chars: int = 2000
    chunk_overlap_ratio: float = 0.10
    min_chunk_chars: int = 200
    quality_threshold: float = 0.60
    ocr_dpi: int = 300
    ocr_language: str = "eng+hin"
    ocr_psm: int = 3
    ocr_batch_pages: int = 10
    output_dir: str = "./dataset_output"
    recursive: bool = True
    subject_override: Optional[str] = None
    no_ocr: bool = False
    dry_run: bool = False
    resume: bool = False
    hf_test_split: float = 0.10
    hf_shuffle_seed: int = 42


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class PageFontType(Enum):
    CLEAN = "clean"
    TYPE3_CUSTOM = "type3_custom"
    IMAGE_ONLY = "image_only"
    MIXED = "mixed"


@dataclass
class QualityReport:
    word_density: float
    symbol_ratio: float
    repeat_ratio: float
    cid_ratio: float
    overall: float
    verdict: str  # "good", "degraded", "garbage"


@dataclass
class PageResult:
    page_num: int
    text: str
    strategy: str
    quality_score: float
    font_type: PageFontType


@dataclass
class Chunk:
    chunk_id: str
    text: str
    subject: str
    source_file: str
    source_path: str
    page_range: List[int]
    section_heading: Optional[str]
    chunk_index: int
    total_chunks: int
    word_count: int
    char_count: int
    extraction_strategy: str
    quality_score: float
    language: str


@dataclass
class ProcessingResult:
    pdf_path: Path
    subject: str
    chunks: List[Chunk]
    pages_total: int
    pages_ocr: int
    pages_text: int
    pages_failed: int
    mean_quality_score: float
    processing_time_sec: float
    errors: List[str]


# ---------------------------------------------------------------------------
# Unicode helpers
# ---------------------------------------------------------------------------

# Devanagari block (Hindi)
_DEV_START = 0x0900
_DEV_END = 0x097F

# Build symbol-strip character class using non-raw string so \u is interpreted
# Preserves: printable ASCII, Devanagari, Latin Extended, common punctuation
_SYMBOL_STRIP_RE = re.compile(
    '[^\x09\x0A\x0D\x20-\x7E'
    'À-ɏ'        # Latin Extended (accented chars)
    'ऀ-ॿ'        # Devanagari (Hindi)
    '–—'         # en/em dashes
    '‘’'         # curly single quotes
    '“”'         # curly double quotes
    '°²³'  # degree, superscript 2/3
    '×÷'        # multiply, divide
    ']'
)


def _is_real_word(token: str) -> bool:
    """True if token looks like a real English or Devanagari word."""
    if len(token) < 2:
        return False
    if all(_DEV_START <= ord(c) <= _DEV_END for c in token):
        return True  # Devanagari word
    clean = re.sub(r'[^A-Za-z]', '', token)
    return len(clean) >= 2


# ---------------------------------------------------------------------------
# PDF Type Detection (fast pre-flight, no text extraction)
# ---------------------------------------------------------------------------

class PDFTypeDetector:
    """Classifies each page by font type before any text extraction."""

    def classify_page(self, doc: fitz.Document, page_num: int) -> PageFontType:
        page = doc[page_num]

        # No text words at all → could be image-only OR Type3 with no text layer
        words = page.get_text("words")
        if not words:
            return PageFontType.IMAGE_ONLY

        fonts = doc.get_page_fonts(page_num, full=False)
        if not fonts:
            return PageFontType.CLEAN

        has_type3 = False
        has_clean = False
        for font_tuple in fonts:
            # Tuple: (xref, ext, type, basefont, name, encoding, ...)
            if len(font_tuple) >= 3:
                ftype = font_tuple[2]
                if ftype == "Type3":
                    has_type3 = True
                else:
                    has_clean = True

        if has_type3 and not has_clean:
            return PageFontType.TYPE3_CUSTOM
        if has_type3 and has_clean:
            return PageFontType.MIXED
        return PageFontType.CLEAN


# ---------------------------------------------------------------------------
# Text Quality Scoring
# ---------------------------------------------------------------------------

class TextQualityScorer:
    _CID_PAT = re.compile(r'\(cid:\d+\)')
    _REPEAT_PAT = re.compile(r'([A-Za-zऀ-ॿ])\1{3,}')

    def score(self, text: str) -> QualityReport:
        if not text or len(text.strip()) < 20:
            return QualityReport(0.0, 1.0, 0.0, 0.0, 0.0, "garbage")

        symbol_ratio = self._symbol_ratio(text)
        repeat_ratio = self._repeat_ratio(text)
        cid_ratio = self._cid_ratio(text)
        word_density = self._word_density(text)

        word_score = min(1.0, word_density / 0.50)
        sym_score = max(0.0, 1.0 - symbol_ratio / 0.10)
        rep_score = max(0.0, 1.0 - repeat_ratio / 0.05)
        cid_score_val = max(0.0, 1.0 - cid_ratio / 0.02)

        overall = (0.40 * word_score + 0.35 * sym_score
                   + 0.15 * rep_score + 0.10 * cid_score_val)
        overall = max(0.0, min(1.0, overall))

        verdict = "good" if overall >= 0.60 else "degraded" if overall >= 0.30 else "garbage"
        return QualityReport(word_density, symbol_ratio, repeat_ratio,
                             cid_ratio, overall, verdict)

    def _symbol_ratio(self, text: str) -> float:
        non_ws = [c for c in text if not c.isspace()]
        if not non_ws:
            return 0.0
        # Characters that aren't in the "preserved" ranges are symbols
        symbol_count = len(_SYMBOL_STRIP_RE.findall(text))
        return min(1.0, symbol_count / len(non_ws))

    def _repeat_ratio(self, text: str) -> float:
        alpha = [c for c in text if c.isalpha()]
        if not alpha:
            return 0.0
        matches = self._REPEAT_PAT.findall(text)
        # Each match is a char that appeared in a run of 4+; approximate count
        repeat_count = sum(4 for _ in matches)
        return min(1.0, repeat_count / max(1, len(alpha)))

    def _cid_ratio(self, text: str) -> float:
        tokens = text.split()
        if not tokens:
            return 0.0
        cid_count = sum(1 for t in tokens if self._CID_PAT.match(t))
        return cid_count / len(tokens)

    def _word_density(self, text: str) -> float:
        tokens = text.split()
        if not tokens:
            return 0.0
        real = sum(1 for t in tokens if _is_real_word(re.sub(r'[^\wऀ-ॿ]', '', t)))
        return real / len(tokens)


# ---------------------------------------------------------------------------
# Extraction Strategies
# ---------------------------------------------------------------------------

class ExtractionStrategy(ABC):
    @abstractmethod
    def extract_page(self, pdf_path: Path, page_num: int) -> str:
        pass


class PyMuPDFStrategy(ExtractionStrategy):
    _CID_RE = re.compile(r'\(cid:\d+\)')

    def extract_page(self, pdf_path: Path, page_num: int) -> str:
        doc = fitz.open(str(pdf_path))
        try:
            page = doc[page_num]
            text = page.get_text("text")
            return self._CID_RE.sub(' ', text)
        finally:
            doc.close()


class PDFPlumberStrategy(ExtractionStrategy):
    def extract_page(self, pdf_path: Path, page_num: int) -> str:
        with pdfplumber.open(str(pdf_path)) as pdf:
            if page_num >= len(pdf.pages):
                return ""
            page = pdf.pages[page_num]
            text = page.extract_text() or ""
            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    if row:
                        text += " | ".join(str(c) for c in row if c) + "\n"
            return text


class OCRStrategy:
    def __init__(self, config: PDFExtractorConfig):
        self.config = config
        self._available = False
        if HAS_TESSERACT:
            try:
                pytesseract.get_tesseract_version()
                self._available = True
            except Exception:
                pass

    @property
    def available(self) -> bool:
        return self._available

    def extract_pages_batch(self, pdf_path: Path,
                            page_nums: List[int]) -> Dict[int, str]:
        """OCR a batch of pages. Returns {page_num: text}."""
        if not self._available:
            return {n: "" for n in page_nums}

        results: Dict[int, str] = {}
        # pdf2image uses 1-based page indices
        for start_idx in range(0, len(page_nums), self.config.ocr_batch_pages):
            batch = page_nums[start_idx:start_idx + self.config.ocr_batch_pages]
            try:
                images = convert_from_path(
                    str(pdf_path),
                    dpi=self.config.ocr_dpi,
                    first_page=batch[0] + 1,
                    last_page=batch[-1] + 1,
                )
                for img, page_num in zip(images, batch):
                    try:
                        text = pytesseract.image_to_string(
                            img,
                            lang=self.config.ocr_language,
                            config=f"--psm {self.config.ocr_psm}",
                        )
                        results[page_num] = text
                    except Exception as e:
                        log.warning(f"OCR failed page {page_num}: {e}")
                        results[page_num] = ""
                del images
                gc.collect()
            except Exception as e:
                log.warning(f"pdf2image failed pages {batch}: {e}")
                for n in batch:
                    results[n] = ""
        return results


# ---------------------------------------------------------------------------
# Text Cleaning
# ---------------------------------------------------------------------------

# UPSC-specific abbreviations to protect during sentence splitting
_ABBREVS = [
    "Dr", "Mr", "Mrs", "Ms", "Prof", "Sr", "Jr", "vs", "etc",
    "viz", "i.e", "e.g", "Fig", "No", "Vol", "pp", "ca",
    "B.C", "A.D", "B.C.E", "C.E",
    "U.P.S.C", "I.A.S", "I.P.S", "A.S.I", "I.N.C",
    "Lt", "Col", "Capt", "Maj", "Gen", "Sgt",
    "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug",
    "Sep", "Oct", "Nov", "Dec",
]
_ABBREV_RE = re.compile(
    r'\b(' + '|'.join(re.escape(a) for a in _ABBREVS) + r')\.',
    re.IGNORECASE,
)
# Split on sentence boundary: end punctuation followed by space + uppercase/Devanagari
_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Zऀ-ॿ])')

_HEADING_PATTERNS = [
    re.compile(r'^(Chapter|CHAPTER|UNIT|PART)\s+[\dIVXivx]+[^\n]*', re.MULTILINE),
    re.compile(r'^[A-Z][A-Z\s\-]{8,}$', re.MULTILINE),
    re.compile(r'^\d+\.\s+[A-Z][^\n]{3,}', re.MULTILINE),
    re.compile(r'^\d+\.\d+\s+[A-Z][^\n]{3,}', re.MULTILINE),
    re.compile(r'^[IVX]+\.\s+[A-Z][^\n]{3,}', re.MULTILINE),
]

_UPSC_WATERMARKS_RE = re.compile(
    r'(www\.visionias\.in|vision\s*ias|ncert\.nic\.in|'
    r'free\s+distribution|not\s+for\s+sale|'
    r'published\s+by.*ncert|reprinted.*ncert)',
    re.IGNORECASE,
)


class TextCleaner:
    _CID_RE = re.compile(r'\(cid:\d+\)')
    # Detects systematic doubling: at least 3 different chars each doubled in sequence
    _SYSTEMATIC_DOUBLE_RE = re.compile(r'([A-Za-z])\1([A-Za-z])\2([A-Za-z])\3')
    _DOUBLE_CHAR_RE = re.compile(r'(([A-Za-z])\2)+')
    _REPEAT_CHAR_RE = re.compile(r'([A-Za-z])\1{3,}')
    _DOUBLED_PAIR_RE = re.compile(r'([A-Za-z])\1')
    _HYPHEN_NL_RE = re.compile(r'([A-Za-z])-\n([a-z])')
    _PAGE_NUM_RE = re.compile(r'^\s*\d{1,4}\s*$', re.MULTILINE)
    _ROMAN_RE = re.compile(
        r'^\s*(xiv|xiii|xii|xi|ix|viii|vii|vi|iv|iii|ii|i)\s*$',
        re.MULTILINE | re.IGNORECASE,
    )
    _MULTI_SPACE_RE = re.compile(r'[ \t]{2,}')
    _MULTI_NL_RE = re.compile(r'\n{3,}')

    def clean(self, text: str) -> str:
        if not text:
            return ""
        text = unicodedata.normalize('NFKC', text)
        text = self._CID_RE.sub(' ', text)
        text = self._fix_repeated_chars(text)
        text = self._remove_symbol_garbage(text)
        text = self._HYPHEN_NL_RE.sub(r'\1\2', text)
        # Remove watermark lines
        lines = [ln for ln in text.split('\n')
                 if not _UPSC_WATERMARKS_RE.search(ln)]
        text = '\n'.join(lines)
        text = self._PAGE_NUM_RE.sub('', text)
        text = self._ROMAN_RE.sub('', text)
        text = self._MULTI_SPACE_RE.sub(' ', text)
        text = self._MULTI_NL_RE.sub('\n\n', text)
        return text.strip()

    def _fix_repeated_chars(self, text: str) -> str:
        alpha_count = sum(1 for c in text if c.isalpha())
        if alpha_count > 50:
            # Detect systematic doubling: if >15% of alpha chars are in duplicate pairs,
            # it indicates TThhiiss-style corruption (NOT normal English double-letters).
            # Normal English text has ~5% doubled-pair ratio; corrupted text has 80-100%.
            doubled_count = len(self._DOUBLED_PAIR_RE.findall(text)) * 2
            if doubled_count / alpha_count > 0.15:
                # Also require a clear run of at least 3 different doubled chars in sequence
                if self._SYSTEMATIC_DOUBLE_RE.search(text):
                    text = self._DOUBLE_CHAR_RE.sub(lambda m: m.group(2), text)
        # Always collapse runs of 4+ identical chars (IIIIllll → Il)
        text = self._REPEAT_CHAR_RE.sub(r'\1', text)
        return text

    def _remove_symbol_garbage(self, text: str) -> str:
        # Replace characters outside preserved Unicode ranges with space
        result = []
        for c in text:
            cp = ord(c)
            # Always preserve: tab, newline, CR, printable ASCII
            if cp in (0x09, 0x0A, 0x0D) or 0x20 <= cp <= 0x7E:
                result.append(c)
            # Preserve: Devanagari
            elif _DEV_START <= cp <= _DEV_END:
                result.append(c)
            # Preserve: Latin Extended
            elif 0x00C0 <= cp <= 0x024F:
                result.append(c)
            # Preserve: common typographic punctuation
            elif cp in (0x2013, 0x2014, 0x2018, 0x2019, 0x201C, 0x201D,
                        0x00B0, 0x00B2, 0x00B3, 0x00D7, 0x00F7):
                result.append(c)
            else:
                result.append(' ')
        return ''.join(result)


# ---------------------------------------------------------------------------
# Header / Footer Detection
# ---------------------------------------------------------------------------

class HeaderFooterDetector:
    def __init__(self, threshold: float = 0.40, n_lines: int = 3):
        self.threshold = threshold
        self.n_lines = n_lines

    def detect_and_remove(self, pages: List[str]) -> List[str]:
        if len(pages) < 4:
            return pages
        headers = self._candidates(pages, "header")
        footers = self._candidates(pages, "footer")
        return [self._strip(p, headers, footers) for p in pages]

    def _norm(self, line: str) -> str:
        line = line.strip().lower()
        line = re.sub(r'\d+', '#', line)
        return line

    def _candidates(self, pages: List[str], position: str) -> set:
        freq: Counter = Counter()
        for page in pages:
            lines = [ln for ln in page.split('\n') if ln.strip()]
            sample = lines[:self.n_lines] if position == "header" else lines[-self.n_lines:]
            for ln in sample:
                norm = self._norm(ln)
                if len(norm) > 3:
                    freq[norm] += 1
        min_count = max(3, int(self.threshold * len(pages)))
        return {ln for ln, cnt in freq.items() if cnt >= min_count}

    def _strip(self, text: str, headers: set, footers: set) -> str:
        lines = text.split('\n')
        window = self.n_lines * 2
        # Strip from top
        start = 0
        for i, ln in enumerate(lines[:window]):
            if self._norm(ln) in headers and len(self._norm(ln)) > 3:
                start = i + 1
        # Strip from bottom
        end = len(lines)
        for i, ln in enumerate(reversed(lines[-window:])):
            if self._norm(ln) in footers and len(self._norm(ln)) > 3:
                end = len(lines) - i - 1
        return '\n'.join(lines[start:end])


# ---------------------------------------------------------------------------
# Chunk Builder
# ---------------------------------------------------------------------------

class ChunkBuilder:
    def __init__(self, config: PDFExtractorConfig):
        self.config = config

    def build_chunks(self, page_results: List[PageResult],
                     subject: str, source_file: str,
                     source_path: str) -> List[Chunk]:
        if not page_results:
            return []

        # Concatenate pages with sentinel offsets
        full_text = ""
        page_offsets: List[Tuple[int, int]] = []  # (char_offset, page_num)
        for pr in page_results:
            page_offsets.append((len(full_text), pr.page_num))
            full_text += pr.text + "\n\n"

        mean_quality = (sum(pr.quality_score for pr in page_results)
                        / max(1, len(page_results)))
        dominant_strategy = Counter(pr.strategy for pr in page_results).most_common(1)[0][0]

        # Split into sections at detected headings
        sections = self._split_at_headings(full_text)

        raw_chunks: List[Tuple[str, Optional[str], int]] = []
        for heading, body, start_pos in sections:
            for sub in self._chunk_text(body):
                # Approximate character position
                pos = start_pos + max(0, body.find(sub[:40]))
                raw_chunks.append((sub, heading, pos))

        # Build Chunk objects
        base_id = (re.sub(r'[^a-z0-9]', '_', source_file.lower().replace('.pdf', ''))[:25])
        subj_slug = re.sub(r'[^a-z0-9]', '_', subject.lower())

        chunks: List[Chunk] = []
        for i, (text, heading, pos) in enumerate(raw_chunks):
            text = text.strip()
            if len(text) < self.config.min_chunk_chars:
                continue
            page_range = self._resolve_page_range(pos, pos + len(text), page_offsets)
            lang = self._detect_language(text)
            chunks.append(Chunk(
                chunk_id=f"{subj_slug}_{base_id}_{i:04d}",
                text=text,
                subject=subject,
                source_file=source_file,
                source_path=source_path,
                page_range=page_range,
                section_heading=heading,
                chunk_index=i,
                total_chunks=0,
                word_count=len(text.split()),
                char_count=len(text),
                extraction_strategy=dominant_strategy,
                quality_score=round(mean_quality, 4),
                language=lang,
            ))

        total = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total
        return chunks

    def _split_at_headings(self, text: str) -> List[Tuple[Optional[str], str, int]]:
        positions: List[Tuple[int, str]] = []
        for pat in _HEADING_PATTERNS:
            for m in pat.finditer(text):
                positions.append((m.start(), m.group().strip()))
        if not positions:
            return [(None, text, 0)]
        positions.sort(key=lambda x: x[0])

        sections: List[Tuple[Optional[str], str, int]] = []
        if positions[0][0] > 200:
            sections.append((None, text[:positions[0][0]], 0))

        for idx, (pos, heading) in enumerate(positions):
            body_start = pos + len(heading)
            body_end = positions[idx + 1][0] if idx + 1 < len(positions) else len(text)
            sections.append((heading, text[body_start:body_end], pos))
        return sections

    def _chunk_text(self, text: str) -> List[str]:
        if len(text) <= self.config.chunk_size_chars:
            return [text] if text.strip() else []
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        chunks: List[str] = []
        current: List[str] = []
        current_chars = 0
        overlap_buf: List[str] = []

        for sent in sentences:
            s_len = len(sent)
            if current_chars + s_len > self.config.chunk_size_chars and current:
                prefix = (" ".join(overlap_buf) + " ") if overlap_buf else ""
                chunks.append(prefix + " ".join(current))
                # Build overlap from tail of current
                ov_chars = 0
                overlap_buf = []
                for s in reversed(current):
                    ov_chars += len(s)
                    overlap_buf.insert(0, s)
                    if ov_chars >= self.config.chunk_size_chars * self.config.chunk_overlap_ratio:
                        break
                current = [sent]
                current_chars = s_len
            else:
                current.append(sent)
                current_chars += s_len

        if current:
            prefix = (" ".join(overlap_buf) + " ") if overlap_buf else ""
            chunks.append(prefix + " ".join(current))

        return [c for c in chunks if len(c.strip()) >= self.config.min_chunk_chars]

    def _split_sentences(self, text: str) -> List[str]:
        protected = _ABBREV_RE.sub(lambda m: m.group().replace('.', '\x00'), text)
        parts = _SENT_SPLIT_RE.split(protected)
        return [p.replace('\x00', '.').strip() for p in parts if p.strip()]

    def _resolve_page_range(self, start: int, end: int,
                             offsets: List[Tuple[int, int]]) -> List[int]:
        pages = []
        for i, (off, pnum) in enumerate(offsets):
            next_off = offsets[i + 1][0] if i + 1 < len(offsets) else float('inf')
            if off <= end and next_off > start:
                pages.append(pnum)
        if not pages:
            return [1, 1]
        return [min(pages) + 1, max(pages) + 1]  # convert to 1-based

    def _detect_language(self, text: str) -> str:
        dev_chars = sum(1 for c in text if _DEV_START <= ord(c) <= _DEV_END)
        if len(text) > 0 and dev_chars / len(text) > 0.20:
            return "hi"
        if HAS_LANGDETECT:
            try:
                return detect(text[:500])
            except Exception:
                pass
        return "en"


# ---------------------------------------------------------------------------
# PDF Processor (single PDF)
# ---------------------------------------------------------------------------

class PDFProcessor:
    def __init__(self, config: PDFExtractorConfig):
        self.config = config
        self.type_detector = PDFTypeDetector()
        self.scorer = TextQualityScorer()
        self.cleaner = TextCleaner()
        self.hf_detector = HeaderFooterDetector()
        self.chunker = ChunkBuilder(config)
        self.pymupdf_strat = PyMuPDFStrategy()
        self.plumber_strat = PDFPlumberStrategy()
        self.ocr_strat = OCRStrategy(config)

    def process(self, pdf_path: Path, subject: str,
                subject_root: Optional[Path] = None) -> ProcessingResult:
        t0 = time.time()
        errors: List[str] = []
        source_path = (str(pdf_path.relative_to(subject_root))
                       if subject_root else pdf_path.name)

        log.info(f"  Processing: {pdf_path.name}")

        try:
            doc = fitz.open(str(pdf_path))
            num_pages = len(doc)
        except Exception as e:
            return ProcessingResult(
                pdf_path=pdf_path, subject=subject, chunks=[],
                pages_total=0, pages_ocr=0, pages_text=0, pages_failed=1,
                mean_quality_score=0.0, processing_time_sec=time.time() - t0,
                errors=[f"Cannot open: {e}"],
            )

        # Classify all pages by font type
        page_types: Dict[int, PageFontType] = {}
        for i in range(num_pages):
            try:
                page_types[i] = self.type_detector.classify_page(doc, i)
            except Exception:
                page_types[i] = PageFontType.MIXED
        doc.close()

        if self.config.dry_run:
            self._print_dry_run(pdf_path, num_pages, page_types)
            return ProcessingResult(
                pdf_path=pdf_path, subject=subject, chunks=[],
                pages_total=num_pages, pages_ocr=0, pages_text=0, pages_failed=0,
                mean_quality_score=0.0, processing_time_sec=time.time() - t0, errors=[],
            )

        # Phase 1 — Batch OCR pages definitively needing it (Type3 / image-only)
        definite_ocr = [i for i, t in page_types.items()
                        if t in (PageFontType.TYPE3_CUSTOM, PageFontType.IMAGE_ONLY)]
        ocr_results: Dict[int, str] = {}

        if definite_ocr and not self.config.no_ocr and self.ocr_strat.available:
            log.info(f"    Pre-OCR: {len(definite_ocr)}/{num_pages} pages (Type3/image)")
            ocr_results = self.ocr_strat.extract_pages_batch(pdf_path, definite_ocr)
        elif definite_ocr:
            reason = "disabled" if self.config.no_ocr else "tesseract not available"
            log.warning(f"    {len(definite_ocr)} pages need OCR but OCR is {reason}")

        # Phase 2 — Text extraction for CLEAN / MIXED pages (single pass)
        # Collect results; mark pages below quality threshold for batch OCR fallback
        text_candidates: Dict[int, Tuple[str, str, float]] = {}
        needs_ocr_fallback: List[int] = []

        for page_num in range(num_pages):
            p_type = page_types[page_num]
            if page_num in ocr_results:
                continue
            if p_type in (PageFontType.TYPE3_CUSTOM, PageFontType.IMAGE_ONLY):
                continue  # OCR was attempted above; skip here

            txt, strategy, quality = "", "unknown", 0.0
            try:
                txt = self.pymupdf_strat.extract_page(pdf_path, page_num)
                quality = self.scorer.score(txt).overall
                strategy = "pymupdf"
            except Exception as e:
                errors.append(f"PyMuPDF p{page_num}: {e}")

            if quality < self.config.quality_threshold:
                try:
                    txt2 = self.plumber_strat.extract_page(pdf_path, page_num)
                    q2 = self.scorer.score(txt2).overall
                    if q2 > quality:
                        txt, quality, strategy = txt2, q2, "pdfplumber"
                except Exception as e:
                    errors.append(f"pdfplumber p{page_num}: {e}")

            text_candidates[page_num] = (txt, strategy, quality)
            if quality < self.config.quality_threshold:
                needs_ocr_fallback.append(page_num)

        # Phase 3 — Single batch OCR for all text pages that scored below threshold
        if needs_ocr_fallback and not self.config.no_ocr and self.ocr_strat.available:
            log.info(f"    OCR fallback: {len(needs_ocr_fallback)} low-quality pages")
            fallback = self.ocr_strat.extract_pages_batch(pdf_path, needs_ocr_fallback)
            for page_num, ocr_txt in fallback.items():
                q_ocr = self.scorer.score(ocr_txt).overall
                if q_ocr > text_candidates[page_num][2]:
                    text_candidates[page_num] = (ocr_txt, "ocr", q_ocr)

        # Phase 4 — Assemble final page results
        page_results: List[PageResult] = []
        pages_text = pages_ocr = pages_failed = 0

        for page_num in range(num_pages):
            p_type = page_types[page_num]

            if page_num in ocr_results:
                txt = ocr_results[page_num]
                q = self.scorer.score(txt).overall
                page_results.append(PageResult(page_num, txt, "ocr", q, p_type))
                pages_ocr += 1
            elif page_num in text_candidates:
                txt, strategy, quality = text_candidates[page_num]
                if txt.strip():
                    page_results.append(PageResult(page_num, txt, strategy, quality, p_type))
                    if strategy == "ocr":
                        pages_ocr += 1
                    else:
                        pages_text += 1
                else:
                    pages_failed += 1
            else:
                pages_failed += 1

        if not page_results:
            return ProcessingResult(
                pdf_path=pdf_path, subject=subject, chunks=[],
                pages_total=num_pages, pages_ocr=pages_ocr,
                pages_text=pages_text, pages_failed=pages_failed,
                mean_quality_score=0.0, processing_time_sec=round(time.time() - t0, 2),
                errors=errors,
            )

        # Header/footer removal then cleaning
        raw_texts = [pr.text for pr in page_results]
        cleaned = self.hf_detector.detect_and_remove(raw_texts)
        for pr, ct in zip(page_results, cleaned):
            pr.text = self.cleaner.clean(ct)

        chunks = self.chunker.build_chunks(
            page_results, subject, pdf_path.name, source_path
        )

        mean_q = (sum(pr.quality_score for pr in page_results)
                  / max(1, len(page_results)))

        log.info(f"    → {len(chunks)} chunks | quality={mean_q:.2f} | "
                 f"text={max(0, pages_text)} ocr={pages_ocr} failed={pages_failed}")

        return ProcessingResult(
            pdf_path=pdf_path, subject=subject, chunks=chunks,
            pages_total=num_pages, pages_ocr=pages_ocr,
            pages_text=max(0, pages_text), pages_failed=pages_failed,
            mean_quality_score=round(mean_q, 4),
            processing_time_sec=round(time.time() - t0, 2),
            errors=errors,
        )

    def _print_dry_run(self, pdf_path: Path, num_pages: int,
                       page_types: Dict[int, PageFontType]):
        counts: Counter = Counter(t.value for t in page_types.values())
        print(f"\n[DRY RUN] {pdf_path.name} ({num_pages} pages)")
        icons = {"clean": "✓", "mixed": "~", "type3_custom": "✗", "image_only": "📷"}
        for k, v in sorted(counts.items()):
            print(f"  {icons.get(k, '?')} {k}: {v} pages")
        ocr_count = counts.get("type3_custom", 0) + counts.get("image_only", 0)
        print(f"  → OCR needed: {ocr_count} pages")


# ---------------------------------------------------------------------------
# Dataset Writer
# ---------------------------------------------------------------------------

_SFT_TEMPLATE = (
    "<start_of_turn>user\n"
    "The following is an excerpt from {subject} study material for UPSC examination "
    "preparation. Read it carefully.\n\n"
    "{text}\n\n"
    "Identify and list the key facts from this passage.<end_of_turn>\n"
    "<start_of_turn>model\n"
    "{summary}<end_of_turn>"
)


def _extractive_summary(text: str, n: int = 3) -> str:
    """First N sentences as a simple extractive summary."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return " ".join(sentences[:n])


class DatasetWriter:
    def write_subject(self, chunks: List[Chunk], output_dir: Path,
                      subject: str, append: bool = False) -> None:
        subj_dir = output_dir / subject
        subj_dir.mkdir(parents=True, exist_ok=True)
        out = subj_dir / f"{subject}_chunks.jsonl"
        mode = 'a' if append and out.exists() else 'w'
        with open(out, mode, encoding='utf-8') as f:
            for chunk in chunks:
                f.write(json.dumps(self._pretrain_record(chunk), ensure_ascii=False) + '\n')
        action = "Appended" if mode == 'a' else "Saved"
        log.info(f"    {action} {len(chunks)} chunks → {out}")

    def write_metadata(self, result: ProcessingResult, output_dir: Path,
                       subject: str) -> None:
        subj_dir = output_dir / subject
        subj_dir.mkdir(parents=True, exist_ok=True)
        meta_file = subj_dir / f"{result.pdf_path.stem}_metadata.json"
        meta = {
            "source_file": result.pdf_path.name,
            "subject": result.subject,
            "pages_total": result.pages_total,
            "pages_text": result.pages_text,
            "pages_ocr": result.pages_ocr,
            "pages_failed": result.pages_failed,
            "chunks_generated": len(result.chunks),
            "mean_quality_score": result.mean_quality_score,
            "processing_time_sec": result.processing_time_sec,
            "errors": result.errors,
        }
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    def _pretrain_record(self, chunk: Chunk) -> dict:
        return {
            "id": chunk.chunk_id,
            "text": chunk.text,
            "subject": chunk.subject,
            "source_file": chunk.source_file,
            "source_path": chunk.source_path,
            "page_range": chunk.page_range,
            "section_heading": chunk.section_heading,
            "chunk_index": chunk.chunk_index,
            "total_chunks": chunk.total_chunks,
            "word_count": chunk.word_count,
            "char_count": chunk.char_count,
            "extraction_strategy": chunk.extraction_strategy,
            "quality_score": chunk.quality_score,
            "language": chunk.language,
        }

    def _sft_record(self, chunk: Chunk) -> dict:
        summary = _extractive_summary(chunk.text)
        sft_text = _SFT_TEMPLATE.format(
            subject=chunk.subject,
            text=chunk.text,
            summary=summary,
        )
        return {
            "id": chunk.chunk_id + "_sft",
            "text": sft_text,
            "subject": chunk.subject,
            "source_file": chunk.source_file,
            "chunk_index": chunk.chunk_index,
            "language": chunk.language,
        }


# ---------------------------------------------------------------------------
# Folder Processor
# ---------------------------------------------------------------------------

class FolderProcessor:
    def __init__(self, config: PDFExtractorConfig):
        self.config = config
        self.processor = PDFProcessor(config)
        self.writer = DatasetWriter()

    def process_folder(self, folder: Path, subject: Optional[str],
                       output_dir: Path) -> List[Chunk]:
        subj = subject or folder.name
        pattern = "**/*.pdf" if self.config.recursive else "*.pdf"
        pdfs = sorted(folder.glob(pattern))

        if not pdfs:
            log.warning(f"No PDFs found in {folder}")
            return []

        subj_dir = output_dir / subj

        # Resume: filter out already-processed PDFs
        if self.config.resume:
            pending, skipped = [], []
            for pdf in pdfs:
                meta_file = subj_dir / f"{pdf.stem}_metadata.json"
                if meta_file.exists():
                    skipped.append(pdf.name)
                else:
                    pending.append(pdf)
            if skipped:
                log.info(f"  [resume] Skipping {len(skipped)} already done: "
                         + ", ".join(skipped[:3]) + ("..." if len(skipped) > 3 else ""))
            pdfs = pending

        log.info(f"\n{'='*60}")
        log.info(f"Subject: {subj}  ({len(pdfs)} PDF{'s' if len(pdfs) != 1 else ''} to process)")
        log.info(f"{'='*60}")

        if not pdfs:
            log.info(f"  All PDFs already processed — skipping subject '{subj}'")
            return []

        all_chunks: List[Chunk] = []
        for pdf in pdfs:
            result = self.processor.process(pdf, subj, folder)
            all_chunks.extend(result.chunks)
            if not self.config.dry_run:
                self.writer.write_metadata(result, output_dir, subj)

        if all_chunks and not self.config.dry_run:
            # Append to existing JSONL if resuming, otherwise overwrite
            self.writer.write_subject(all_chunks, output_dir, subj,
                                      append=self.config.resume)

        log.info(f"Subject '{subj}': {len(all_chunks)} new chunks")
        return all_chunks


# ---------------------------------------------------------------------------
# Dataset Merger (unified output + HuggingFace Dataset)
# ---------------------------------------------------------------------------

class DatasetMerger:
    def __init__(self, config: PDFExtractorConfig):
        self.config = config
        self.writer = DatasetWriter()

    def merge(self, all_chunks: List[Chunk], output_dir: Path) -> None:
        # When resuming, also load chunks from already-processed subject JSONLs
        existing_records: List[dict] = []
        for jsonl_file in sorted(output_dir.glob("*/*_chunks.jsonl")):
            if jsonl_file.parent.name == "combined":
                continue
            try:
                with open(jsonl_file, encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            existing_records.append(json.loads(line))
            except Exception as e:
                log.warning(f"Could not read {jsonl_file}: {e}")

        # Merge: existing records from disk + new chunks from this run
        new_ids = {c.chunk_id for c in all_chunks}
        deduped_existing = [r for r in existing_records if r.get("id") not in new_ids]
        new_records = [self.writer._pretrain_record(c) for c in all_chunks]
        all_records = deduped_existing + new_records

        if not all_records:
            log.warning("No chunks to merge.")
            return

        combined_dir = output_dir / "combined"
        combined_dir.mkdir(parents=True, exist_ok=True)

        import random
        rng = random.Random(self.config.hf_shuffle_seed)
        rng.shuffle(all_records)

        # Stage 1 — pretrain JSONL (plain text, no special tokens)
        pretrain_path = combined_dir / "unified_pretrain.jsonl"
        with open(pretrain_path, 'w', encoding='utf-8') as f:
            for rec in all_records:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        log.info(f"Pretrain JSONL: {pretrain_path}  ({len(all_records)} records)")

        # Stage 2 — SFT JSONL (Gemma 2 chat format)
        sft_path = combined_dir / "unified_sft.jsonl"
        sft_count = 0
        with open(sft_path, 'w', encoding='utf-8') as f:
            for rec in all_records:
                if rec.get("word_count", 0) >= 50:
                    sft_rec = {
                        "id": rec["id"] + "_sft",
                        "text": _SFT_TEMPLATE.format(
                            subject=rec.get("subject", ""),
                            text=rec["text"],
                            summary=_extractive_summary(rec["text"]),
                        ),
                        "subject": rec.get("subject", ""),
                        "source_file": rec.get("source_file", ""),
                        "chunk_index": rec.get("chunk_index", 0),
                        "language": rec.get("language", "en"),
                    }
                    f.write(json.dumps(sft_rec, ensure_ascii=False) + '\n')
                    sft_count += 1
        log.info(f"SFT JSONL:     {sft_path}  ({sft_count} records)")

        # HuggingFace Dataset
        if HAS_DATASETS:
            self._save_hf_dataset(all_records, combined_dir)
        else:
            log.warning("datasets package not found — skipping HuggingFace Dataset output")

        self._print_summary(all_records, combined_dir)

    def _save_hf_dataset(self, records: List[dict], combined_dir: Path) -> None:
        ds = Dataset.from_list(records)
        split_idx = int(len(ds) * (1.0 - self.config.hf_test_split))
        dd = DatasetDict({
            "train": ds.select(range(split_idx)),
            "test": ds.select(range(split_idx, len(ds))),
        })
        hf_path = combined_dir / "hf_dataset"
        dd.save_to_disk(str(hf_path))
        log.info(f"HF Dataset:    {hf_path}  "
                 f"(train={len(dd['train'])}, test={len(dd['test'])})")

    def _print_summary(self, records: List[dict], combined_dir: Path) -> None:
        by_subject = Counter(r.get("subject", "unknown") for r in records)
        by_lang = Counter(r.get("language", "en") for r in records)
        by_strategy = Counter(r.get("extraction_strategy", "unknown") for r in records)
        mean_q = sum(r.get("quality_score", 0) for r in records) / max(1, len(records))

        summary = {
            "total_chunks": len(records),
            "total_words": sum(r.get("word_count", 0) for r in records),
            "mean_quality_score": round(mean_q, 4),
            "by_subject": dict(sorted(by_subject.items())),
            "by_language": dict(sorted(by_lang.items())),
            "by_strategy": dict(sorted(by_strategy.items())),
        }
        with open(combined_dir / "dataset_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 60)
        print("DATASET SUMMARY")
        print("=" * 60)
        print(f"Total chunks : {summary['total_chunks']:,}")
        print(f"Total words  : {summary['total_words']:,}")
        print(f"Mean quality : {summary['mean_quality_score']:.3f}")
        print("\nBy subject:")
        for subj, cnt in sorted(by_subject.items()):
            bar = "█" * min(30, cnt // max(1, max(by_subject.values()) // 30))
            print(f"  {subj:<20} {cnt:>6}  {bar}")
        print("\nBy strategy:")
        for strat, cnt in sorted(by_strategy.items()):
            print(f"  {strat:<20} {cnt:>6}")
        print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="High-quality PDF → dataset extractor for Gemma 2 SLM fine-tuning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single PDF file
  python scripts/pdf_extractor.py upsc_pdfs/History/NCERT.pdf --subject History

  # Subject folder (all PDFs)
  python scripts/pdf_extractor.py upsc_pdfs/History/ --output dataset_output/

  # Entire tree — auto-detects subject names from sub-folders
  python scripts/pdf_extractor.py upsc_pdfs/ --output dataset_output/

  # Dry run — shows OCR plan without extracting
  python scripts/pdf_extractor.py upsc_pdfs/Chemistry/ --dry-run

  # Disable OCR (fast, text-only PDFs)
  python scripts/pdf_extractor.py upsc_pdfs/Polity/ --no-ocr
        """,
    )
    p.add_argument("inputs", nargs="+", metavar="input",
                   help="PDF file(s) or folder(s)")
    p.add_argument("--output", default="./dataset_output",
                   help="Output directory (default: ./dataset_output)")
    p.add_argument("--subject",
                   help="Subject label override (default: folder name)")
    p.add_argument("--recursive", action="store_true", default=True,
                   help="Search sub-folders for PDFs (default: on)")
    p.add_argument("--no-recursive", action="store_false", dest="recursive")
    p.add_argument("--chunk-size", type=int, default=2000, metavar="N",
                   help="Target chunk size in chars (default: 2000 ≈ 512 tokens)")
    p.add_argument("--chunk-overlap", type=float, default=0.10, metavar="R",
                   help="Overlap ratio 0.0–0.5 (default: 0.10)")
    p.add_argument("--quality-threshold", type=float, default=0.60, metavar="T",
                   help="Min quality score to accept text strategy (default: 0.60)")
    p.add_argument("--ocr-dpi", type=int, default=300,
                   help="DPI for PDF→image conversion (default: 300)")
    p.add_argument("--ocr-lang", default="eng+hin",
                   help="Tesseract language string (default: eng+hin)")
    p.add_argument("--no-ocr", action="store_true",
                   help="Disable OCR — fast but Type3/scanned PDFs are skipped")
    p.add_argument("--no-merge", action="store_true",
                   help="Skip creating unified combined dataset")
    p.add_argument("--resume", action="store_true",
                   help="Skip PDFs that already have a metadata file (safe restart)")
    p.add_argument("--dry-run", action="store_true",
                   help="Classify pages and show OCR plan, do not extract")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    config = PDFExtractorConfig(
        chunk_size_chars=args.chunk_size,
        chunk_overlap_ratio=args.chunk_overlap,
        quality_threshold=args.quality_threshold,
        ocr_dpi=args.ocr_dpi,
        ocr_language=args.ocr_lang,
        no_ocr=args.no_ocr,
        dry_run=args.dry_run,
        resume=args.resume,
        recursive=args.recursive,
        subject_override=args.subject,
        output_dir=args.output,
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    folder_proc = FolderProcessor(config)
    merger = DatasetMerger(config)
    writer = DatasetWriter()
    proc = PDFProcessor(config)

    all_chunks: List[Chunk] = []

    for inp in args.inputs:
        inp_path = Path(inp)
        if not inp_path.exists():
            log.error(f"Path does not exist: {inp_path}")
            continue

        if inp_path.is_file() and inp_path.suffix.lower() == ".pdf":
            subject = args.subject or inp_path.parent.name or "Unknown"
            result = proc.process(inp_path, subject)
            all_chunks.extend(result.chunks)
            if not config.dry_run:
                writer.write_subject(result.chunks, output_dir, subject)
                writer.write_metadata(result, output_dir, subject)

        elif inp_path.is_dir():
            direct_pdfs = list(inp_path.glob("*.pdf"))
            sub_dirs = [d for d in sorted(inp_path.iterdir())
                        if d.is_dir() and not d.name.startswith('.')]

            if args.subject or direct_pdfs:
                # Single subject folder
                chunks = folder_proc.process_folder(inp_path, args.subject, output_dir)
                all_chunks.extend(chunks)
            else:
                # Top-level folder containing subject sub-folders
                for sub in sub_dirs:
                    sub_pdfs = list(sub.glob(
                        "**/*.pdf" if config.recursive else "*.pdf"
                    ))
                    if sub_pdfs:
                        chunks = folder_proc.process_folder(sub, None, output_dir)
                        all_chunks.extend(chunks)
        else:
            log.warning(f"Skipping (not a PDF or directory): {inp_path}")

    if not args.no_merge and not config.dry_run and all_chunks:
        log.info("\nMerging into unified dataset...")
        merger.merge(all_chunks, output_dir)

    log.info("Done.")


if __name__ == "__main__":
    main()
