# UPSC PDF Dataset Extractor

High-quality PDF → JSONL dataset extractor for **Gemma 2 SLM domain adaptation** on UPSC exam content. Handles all PDF types including scanned pages, Type3 symbol fonts, and corrupt font encodings, producing clean training data in two formats: continued pre-training (Stage 1) and instruction fine-tuning (Stage 2).

---

## The Problem This Solves

Standard PDF extractors fail on UPSC/NCERT PDFs in two ways:

| Corruption Type | Example | Root Cause |
|----------------|---------|------------|
| Repeated-char corruption | `IIIIlllluuuussstttrrraaa` | Bad ToUnicode font maps |
| Symbol garbage | `✎ ☞ ✒ ♦ ☛` | Type3 custom symbol fonts |

This script detects corruption **per page** and automatically routes affected pages to OCR (Tesseract), while extracting clean pages directly — giving the best quality with the least OCR overhead.

---

## Installation

### Python Dependencies
```bash
cd $HOME/Downloads/DataSets
python3 -m venv .venv
source .venv/bin/activate
pip install pymupdf pdfplumber pdf2image pillow pytesseract langdetect datasets
```

### System Dependencies (macOS)
```bash
brew install tesseract tesseract-lang poppler
```

### Verify Setup
```bash
tesseract --version       # should show 5.x.x
python3 -c "import fitz, pdfplumber, pytesseract; print('OK')"
```

---

## Quick Start

```bash
cd $HOME/Downloads/DataSets
source .venv/bin/activate

# Single subject folder
python3 scripts/pdf_extractor.py upsc_pdfs/History/ --output dataset_output_final/

# All subjects at once
python3 scripts/pdf_extractor.py upsc_pdfs/ --output dataset_output_final/

# Single file
python3 scripts/pdf_extractor.py upsc_pdfs/Biology/NCERT-Class-11-Biology.pdf \
  --subject Biology --output dataset_output_final/
```

---

## CLI Reference

```
python3 scripts/pdf_extractor.py [inputs...] [options]
```

### Input
| Argument | Description |
|----------|-------------|
| `input` | PDF file, subject folder, or top-level folder containing subject sub-folders |

### Core Options
| Flag | Default | Description |
|------|---------|-------------|
| `--output DIR` | `./dataset_output` | Output directory |
| `--subject NAME` | folder name | Subject label override |
| `--no-merge` | off | Skip creating unified combined files |
| `--merge-only` | off | Skip extraction; just rebuild unified files from existing subject JSONLs |
| `--resume` | off | Skip PDFs that already have a `_metadata.json` checkpoint |

### OCR Options
| Flag | Default | Description |
|------|---------|-------------|
| `--force-ocr` | off | Force OCR on every page, bypassing quality scoring. Use for PDFs with character-substitution corruption (e.g. NCERT Class-12 Biology) |
| `--no-ocr` | off | Disable OCR entirely — fast, but Type3/scanned pages are skipped |
| `--ocr-dpi N` | `300` | DPI for PDF→image conversion |
| `--ocr-lang LANG` | `eng+hin` | Tesseract language string |

### Chunking Options
| Flag | Default | Description |
|------|---------|-------------|
| `--chunk-size N` | `2000` | Target chunk size in characters (~512 tokens) |
| `--chunk-overlap R` | `0.10` | Overlap ratio between consecutive chunks (0.0–0.5) |
| `--quality-threshold T` | `0.60` | Minimum quality score to accept text extraction; below this triggers OCR fallback |

### Utilities
| Flag | Description |
|------|-------------|
| `--dry-run` | Classify pages and show OCR plan without extracting anything |
| `--log-level` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

---

## Common Workflows

### 1. Process subjects one at a time (recommended for large datasets)
```bash
# Process each subject separately — skip unified merge each time
python3 scripts/pdf_extractor.py upsc_pdfs/History/   --output dataset_output_final/ --no-merge
python3 scripts/pdf_extractor.py upsc_pdfs/Geography/ --output dataset_output_final/ --no-merge
python3 scripts/pdf_extractor.py upsc_pdfs/Polity/    --output dataset_output_final/ --no-merge

# When all subjects are done, merge everything into unified files
python3 scripts/pdf_extractor.py --merge-only --output dataset_output_final/
```

### 2. Resume an interrupted run
```bash
# If a run crashes, re-run with --resume — already-completed PDFs are skipped
python3 scripts/pdf_extractor.py upsc_pdfs/History/ \
  --output dataset_output_final/ --no-merge --resume
```

### 3. Fix a PDF with character-substitution corruption
```bash
# Delete its checkpoint so it re-processes
rm dataset_output_final/Biology/NCERT-Class-12-Biology_metadata.json

# Re-extract with forced OCR
python3 scripts/pdf_extractor.py upsc_pdfs/Biology/NCERT-Class-12-Biology.pdf \
  --subject Biology --output dataset_output_final/ --no-merge --force-ocr
```

### 4. Dry run — check OCR strategy before committing time
```bash
python3 scripts/pdf_extractor.py upsc_pdfs/Chemistry/ --dry-run
```

---

## How It Works — Per-Page Strategy

Every page is classified independently before any text extraction:

```
For each page:
  1. PDFTypeDetector → classify font type
     ├── Type3 / image-only → OCR directly (batch all such pages first)
     └── Clean / Mixed → PyMuPDF extraction → quality score
                              ├── score ≥ 0.60 → accept
                              └── score < 0.60 → try PDFPlumber
                                                    ├── score ≥ 0.60 → accept
                                                    └── score < 0.60 → OCR fallback (batched)
```

OCR fallback pages are collected across the whole PDF and sent to Tesseract in a **single batch**, avoiding the 20+ minute per-page stall that naive per-page OCR causes.

### Quality Scoring Formula
```
overall = 0.40 × word_density   (are tokens real words?)
        + 0.35 × symbol_score   (low dingbat/PUA ratio?)
        + 0.15 × repeat_score   (no repeated-char runs?)
        + 0.10 × cid_score      (no (cid:N) leakage?)
```

> **Note:** Character-substitution corruption (e.g. `lliese` for "these", `orgmisms` for "organisms") scores falsely high because chars are still alphabetic. Use `--force-ocr` for such PDFs.

---

## Output Structure

```
dataset_output_final/
├── History/
│   ├── History_chunks.jsonl              ← extracted chunks for this subject
│   ├── NCERT-Class-11-History_metadata.json   ← per-PDF stats (checkpoint)
│   └── NCERT-Class-12-History_metadata.json
├── Geography/
│   └── ...
└── combined/
    ├── unified_pretrain.jsonl            ← Stage 1: plain text for continued pre-training
    ├── unified_sft.jsonl                 ← Stage 2: Gemma 2 chat format for SFT
    ├── hf_dataset/                       ← HuggingFace Dataset (90/10 train/test split)
    │   ├── train/
    │   └── test/
    └── dataset_summary.json             ← chunk counts by subject, language, strategy
```

### Chunk Schema (JSONL record)
```json
{
  "id": "history_ncert_class_11_history_0042",
  "text": "The Mughal Empire was founded in 1526...",
  "subject": "History",
  "source_file": "NCERT-Class-11-History.pdf",
  "source_path": "NCERT-Class-11-History.pdf",
  "page_range": [45, 47],
  "section_heading": "THE MUGHAL EMPIRE",
  "chunk_index": 42,
  "total_chunks": 317,
  "word_count": 289,
  "char_count": 1874,
  "extraction_strategy": "pymupdf",
  "quality_score": 0.9982,
  "language": "en"
}
```

### Metadata Schema (per-PDF checkpoint)
```json
{
  "source_file": "NCERT-Class-11-History.pdf",
  "subject": "History",
  "pages_total": 260,
  "pages_text": 208,
  "pages_ocr": 52,
  "pages_failed": 0,
  "chunks_generated": 317,
  "mean_quality_score": 0.9971,
  "raw_chars": 892341,
  "cleaned_chars": 887124,
  "char_retention_pct": 99.42,
  "processing_time_sec": 583.2,
  "errors": []
}
```

---

## Understanding the Log Output

```
============================================================
EXTRACTING: HISTORY  (9 PDFs)
============================================================
  Processing: NCERT-Class-11-History.pdf
    OCR: 52/260 pages              ← 52 pages sent directly to OCR (Type3/image)
    OCR fallback: 3 low-quality pages  ← 3 more pages failed quality check, OCR'd
    → 317 chunks | quality=0.97 | text=35 ocr=225 failed=0
    chars: 892,341 raw → 887,124 clean (99.4% retained)

  ──────────────────────────────────────────────────────
  SUBJECT SUMMARY: History
  ──────────────────────────────────────────────────────
  PDFs processed : 9
  Total pages    : 1431  (failed: 0)
  Total chunks   : 1971
  Avg chunk len  : 1,842 chars
  Raw chars      : 4,213,445
  Clean chars    : 4,198,320
  Char retention : 99.6%       ← should be >95%; if <80%, investigate
  ──────────────────────────────────────────────────────
```

| Field | What to look for |
|-------|-----------------|
| `quality` | Should be ≥ 0.90; below 0.80 means extraction issues |
| `failed` | Should be 0; any failures = pages with no usable text |
| `char retention` | >95% = normal; <80% = check for over-aggressive cleaning |

---

## Training Pipeline

### Stage 1 — Continued Pre-training (domain knowledge)
```python
# Use unified_pretrain.jsonl
# Each record: {"id": "...", "text": "clean domain text", ...}

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
# Tokenizer adds BOS/EOS automatically — do NOT add them manually in the JSONL
encoded = tokenizer(text, add_special_tokens=True)
```

### Stage 2 — Instruction Fine-tuning (Q&A behaviour)
```python
# Use unified_sft.jsonl
# Each record already formatted with Gemma 2 chat tokens:
# <start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n...<end_of_turn>

# When tokenizing SFT records, set add_special_tokens=False
# (BOS/EOS are already embedded in the template)
encoded = tokenizer(text, add_special_tokens=False)
```

> **Training order:** Always run Stage 1 first. Stage 2 teaches conversational behaviour but relies on domain knowledge injected in Stage 1.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `tesseract not available` | Tesseract not installed | `brew install tesseract tesseract-lang` |
| `pdf2image failed` | Poppler not installed | `brew install poppler` |
| Corrupt text in output despite high quality score | Character-substitution font encoding | Re-run with `--force-ocr` |
| Duplicate chunks in JSONL | Single-file re-run appended instead of replaced | Delete the subject JSONL and re-run |
| Very low char retention (<70%) | Over-cleaning or blank PDF pages | Check `errors` field in `_metadata.json` |
| OCR taking 20+ min per PDF | Old per-page OCR path hit | Update to latest script version (uses batch OCR) |

---

## PDF Subjects

| Subject | Folder | Notes |
|---------|--------|-------|
| Art | `upsc_pdfs/Art/` | Mix of clean and Type3 pages |
| Biology | `upsc_pdfs/Biology/` | Class-12 needs `--force-ocr` |
| Chemistry | `upsc_pdfs/Chemistry/` | Mostly Type3 — always OCR |
| Economics | `upsc_pdfs/Economics/` | Generally clean |
| Geography | `upsc_pdfs/Geography/` | Generally clean |
| History | `upsc_pdfs/History/` | Mix; some scanned pages |
| Physics | `upsc_pdfs/Physics/` | Mostly Type3 — always OCR |
| Polity | `upsc_pdfs/Polity/` | Generally clean |
| Science | `upsc_pdfs/Science/` | Mix |
| Sociology | `upsc_pdfs/Sociology/` | Generally clean |
