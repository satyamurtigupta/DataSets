# UPSC SLM Project — Complete Handoff Plan

> Use this document to brief another AI assistant (Copilot, ChatGPT, Gemini, etc.)
> on everything built so far, what the current state is, and what to do next.
> All paths are absolute on the local machine.

---

## 1. Project Goal

Fine-tune **Gemma 2 2B-IT** as an **offline UPSC Civil Services exam preparation assistant**.

- Works without internet (on-device)
- Covers both Prelims (MCQ) and Mains (essay answers)
- Deploys on Android, iOS, laptop, and Raspberry Pi
- A research paper is planned alongside the product

---

## 2. Working Directories

| Path | Purpose |
|------|---------|
| `/Users/satyamurti/Downloads/DataSets/` | All data, scripts, output |
| `/Users/satyamurti/Downloads/DataSets/scripts/` | All Python scripts |
| `/Users/satyamurti/Downloads/DataSets/dataset_output_final/combined/` | All output JSONL files |
| `/Users/satyamurti/Downloads/DataSets/upsc_pdfs/Prelimns/` | VisionIAS PDF files 2011-2025 |
| `/Users/satyamurti/Downloads/DataSets/env/.env` | API keys (OpenAI, HF) |
| `/Users/satyamurti/Downloads/SLM_Train_Competetive.ipynb` | Google Colab training notebook |

Python version: **3.9** (use `Optional[X]` not `X | None` — strict requirement).

---

## 3. Model Architecture

- **Base model**: `google/gemma-2-2b-it`
- **Fine-tuning method**: LoRA (via `peft` + `trl`)
- **Training framework**: Hugging Face `SFTTrainer` with `SFTConfig`
- **Quantization during training**: 4-bit NF4 (`BitsAndBytesConfig`)
- **Dataset format**: **Alpaca (instruction-input-output)** — see Section 6

---

## 4. Training Pipeline — 2 Stages

### Stage 1: Continued Pre-Training (CPT) — COMPLETE
- **Goal**: Teach the model NCERT domain knowledge (History, Polity, Geography, Economy, Science, Art, etc.)
- **Data**: `unified_pretrain.jsonl` — 11,912 chunks of NCERT text, 23MB
- **Notebook cell**: `SFTConfig` with `dataset_text_field="text"`, `max_length=512`, `packing=True`
- **Result**: `lora_adapter_epoch3` — saved in Google Drive folder `1zkuey1exTUqw29h0jJSZVZYqKvk4Syhy`
- **Status**: DONE. Adapter confirmed 83MB, 3 epochs trained.

### Stage 2: Supervised Fine-Tuning (SFT) — IN PROGRESS

#### Stage 2A: UPSC Mains Q&A (DONE — partially)
- **Goal**: Teach the model to write UPSC Mains-style essay answers
- **Data**: `unified_sft_v3.jsonl` — 34,734 records, 58MB
- **Format**: Gemma 2 chat template (`<start_of_turn>user...<end_of_turn>`)
- **Note**: This was trained with `dataset_text_field="text"` — the template is pre-baked in the data. SFTTrainer does NOT apply the template again. This is correct.
- **Content**: Question types: discuss, comment, explain, critically_examine, analyse, evaluate
- **Status**: Training run completed on Colab (T4 GPU)

#### Stage 2B: Prelims MCQ Training — NEXT
- **Goal**: Teach the model to generate, answer, and explain UPSC Prelims MCQs
- **Data**: `unified_mcq_training_clean.jsonl` — 2,496 records, 100% quality-checked
- **Format**: **Alpaca format** (instruction + input + question + answer + explanation)
- **Status**: Dataset ready. Training NOT yet run.

---

## 5. Data Files — Current State

| File | Records | Format | Status |
|------|---------|--------|--------|
| `unified_pretrain.jsonl` | 11,912 | Raw NCERT text chunks | Used in Stage 1 ✓ |
| `unified_sft_v3.jsonl` | 34,734 | Gemma template Q&A | Used in Stage 2A ✓ |
| `real_upsc_mcq_raw.jsonl` | 1,323 | Raw extracted MCQ records | Source data |
| `real_upsc_mcq_sft_clean.jsonl` | 2,442 | Alpaca, quality-checked | Ready for training |
| `synthetic_mcq_raw.jsonl` | 21 | Raw synthetic MCQ records | Only 21 — full gen not run yet |
| `synthetic_mcq_sft_clean.jsonl` | 54 | Alpaca, quality-checked | Ready for training |
| `unified_mcq_training_clean.jsonl` | 2,496 | Alpaca, 100% pass quality | **Use this for Stage 2B** |

---

## 6. Dataset Format — ALPACA (Critical)

Every MCQ training record must follow this structure exactly:

```json
{
  "instruction": "Generate a UPSC-style MCQ on Environment, Biodiversity Hotspots.",
  "input": "<NCERT or VisionIAS passage about the topic>",
  "question": "Consider the following statements:\n1. ...\n2. ...\nWhich is/are correct?\n\n(a) 1 only\n(b) 2 only\n(c) Both 1 and 2\n(d) Neither 1 nor 2",
  "answer": "c",
  "explanation": "The correct answer is (c). Statement 1 is correct because...",
  "subject": "Environment",
  "question_type": "statement_based",
  "difficulty": "medium",
  "exam": "UPSC"
}
```

**Key rules:**
- `answer` must be exactly one of: `a`, `b`, `c`, `d`
- `explanation` must explicitly say "The correct answer is (X)" — not just the explanation body
- `question` field must contain all 4 options `(a)`, `(b)`, `(c)`, `(d)` embedded inline
- `question_type` must be one of: `statement_based`, `assertion_reason`, `match_pairs`, `factual`, `conceptual`, `how_many_correct`
- `difficulty` must be one of: `easy`, `medium`, `difficult`

**Why Alpaca and NOT Gemma template in data:**
When using `SFTTrainer` with a `messages` list or `formatting_func`, the trainer applies the chat template automatically. You do NOT pre-bake `<start_of_turn>user` into the dataset. The Alpaca fields are model-agnostic — they work with Gemma, Llama, Mistral, or any model.

---

## 7. Scripts — What Each Does

| Script | Purpose | Command |
|--------|---------|---------|
| `extract_real_mcq_from_pdfs.py` | Extract real UPSC MCQs from VisionIAS PDFs (2011-2025) | `python3 scripts/extract_real_mcq_from_pdfs.py` |
| `generate_mcq_synthetic.py` | Generate synthetic MCQs from NCERT chunks via GPT-4o | `python3 scripts/generate_mcq_synthetic.py --delay 1.5 --resume` |
| `dataset_quality_check.py` | Run 3 quality checks on any MCQ JSONL | `python3 scripts/dataset_quality_check.py --input file.jsonl --clean-output clean.jsonl` |
| `generate_sft_dataset.py` | Generate Mains Q&A pairs (Stage 2A) | `python3 scripts/generate_sft_dataset.py --api openai --resume` |

---

## 8. Two MCQ Data Sources — Why Both

### Synthetic MCQs (from NCERT)
- **How**: GPT-4o reads NCERT text chunks → generates UPSC-pattern MCQs
- **Why**: Builds broad domain knowledge coverage across all NCERT subjects
- **Status**: Only 21 records exist. Full generation NOT run yet.
- **Command to run**: `python3 scripts/generate_mcq_synthetic.py --delay 1.5 --resume`
- **Expected output**: ~30,000-40,000 MCQs from 6,591 usable NCERT chunks

### Real MCQs (from VisionIAS PDFs 2011-2025)
- **How**: PyMuPDF extracts tables from solution PDFs → parsed per year layout
- **Why**: Real exam questions — harder, more nuanced, authentic patterns
- **Records**: 1,323 questions extracted (88% of 1,500 possible across 15 years)
- **Poor years**: 2016 (31/100), 2017 (55/100) — complex table structure in PDFs
- **Status**: Done. Clean SFT file ready at `real_upsc_mcq_sft_clean.jsonl`

---

## 9. UPSC Question Pattern (2022-2025 Consensus)

The GPT-4o generation prompt uses this distribution — any AI helping with generation must follow it:

| Format | Share |
|--------|-------|
| "Consider the following statements... which is/are correct?" | 40% |
| Assertion-Reason: "Statement I/II" | 18% |
| Match Pairs | 12% |
| Direct factual | 15% |
| "How many of the above are correct?" | 8% |
| Conceptual/analytical | 7% |

**Difficulty** (equal thirds since 2025): 34% easy, 33% medium, 33% difficult

**Subjects** (most to least frequent): Environment, Economy, Polity, History, Geography, Science & Tech, Art & Culture, International Relations

---

## 10. Dataset Quality Check — 3 Rules

Run `dataset_quality_check.py` on any JSONL before training. It checks:

### Check 1 — Structural Integrity
- All Alpaca fields present and non-empty
- `answer` is one of a/b/c/d
- All 4 options `(a)(b)(c)(d)` appear inside the `question` field
- Explanation is at least 15 words

### Check 2 — Instruction Sanity
- Instruction is not a NCERT exercise question number (e.g. "6. Compare the effects of...")
- No OCR artifacts: `|)`, `|(`, hex bytes
- Context (`input` field) does not start mid-sentence with a lowercase letter
- No chunk ID leaked into instruction

### Check 3 — Content Grounding (synthetic only)
- Correct answer letter explicitly mentioned in explanation ("The correct answer is (c)")
- Explanation does not introduce more than 4 proper nouns absent from the source passage (hallucination signal)

**Current pass rates:**
- `unified_mcq_training_clean.jsonl`: 100% (2,496 records)
- `real_upsc_mcq_sft.jsonl` before fix: 39.7%
- `real_upsc_mcq_sft_clean.jsonl` after fix: 72.1% → filtered to 100%

---

## 11. Key Problems Found and Fixed

| Problem | Root Cause | Fix Applied |
|---------|-----------|-------------|
| Topics like "6. Compare the effects of railways..." | NCERT chunk came from exercise section, not chapter body | `_is_usable_chunk()` filter rejects mid-sentence chunks and equation-heading chunks |
| Context starting with "countries in the world." (mid-sentence) | Chunk boundary was cut wrong | Same filter — rejects chunks where `text[0].islower()` |
| `\|)` OCR artifact in training data | PyMuPDF table extraction noise | `_clean_context()` strips known artifact patterns before sending to GPT-4o |
| GPT-4o explaining "Ohio, Mississippi, Seattle" from passage that doesn't mention them | Hallucination — GPT-4o used its own knowledge | `_is_grounded_mcq()` rejects MCQs where explanation introduces > 4 proper nouns not in context |
| Training data in Gemma template format (`<start_of_turn>user...`) | Early design decision | Rebuilt `build_sft_pairs()` in both scripts to output Alpaca format |
| Real MCQ records with <4 options | PDF extraction failure for some years | `build_sft_pairs()` skips records with fewer than 4 options |
| Explanation not referencing answer letter | VisionIAS PDF explanations skip "answer is (X)" | Always prepend `"The correct answer is ({correct}) — {correct_text}."` |

---

## 12. Instruction Format Variants (8 types)

The SFT dataset uses 8 different instruction phrasings for the same MCQ so the model generalises across phrasing:

1. `"Generate a UPSC-style MCQ on {subject}, {topic}."`
2. `"Generate 1 {difficulty}-level UPSC Prelims question on {topic}."`
3. `"Answer this UPSC Prelims MCQ and explain the correct option."`
4. `"Why is option ({answer}) the correct answer for this question?"`
5. `"Solve this MCQ from UPSC {year} Prelims on {subject}."`
6. `"Based on the following passage, answer the MCQ."`
7. `"I want to practice UPSC questions on {topic}. Give me {n} questions."`
8. `"Give me {n} MCQ questions on {subject} for UPSC Prelims."`

**Why this matters**: The model never memorises exact instruction phrases. It learns the PATTERN: instruction (intent) + input (context) → structured MCQ output. When a user says "Generate 2 MCQ on Environment," the model applies the same learned pattern to a new subject.

---

## 13. Immediate Next Steps (Priority Order)

### Step 1 — Run full synthetic MCQ generation
```bash
cd /Users/satyamurti/Downloads/DataSets
python3 scripts/generate_mcq_synthetic.py --delay 1.5 --resume
```
- Will process ~6,591 usable NCERT chunks (55.3% of 11,912 total)
- Expected: ~25,000-35,000 synthetic MCQ raw records
- Cost: ~$15-20 in OpenAI API credits at GPT-4o pricing
- After completion, run: `python3 scripts/generate_mcq_synthetic.py --rebuild-sft`

### Step 2 — Quality check and merge
```bash
python3 scripts/dataset_quality_check.py \
  --input dataset_output_final/combined/synthetic_mcq_sft.jsonl \
  --clean-output dataset_output_final/combined/synthetic_mcq_sft_clean.jsonl

# Then merge with real MCQ clean file:
# (python one-liner — merge + shuffle synthetic_mcq_sft_clean.jsonl and real_upsc_mcq_sft_clean.jsonl)
```

### Step 3 — Train Stage 2B (MCQ SFT)
In `SLM_Train_Competetive.ipynb` on Google Colab:
- Load Stage 1 adapter: `lora_adapter_epoch3` from Google Drive
- Load dataset: `unified_mcq_training_clean.jsonl`
- Use `dataset_text_field`: switch to formatting_func approach for Alpaca format
- Suggested config: LR=2e-4, epochs=2, batch=2, grad_accum=8

### Step 4 — Rotate HF token (security)
The token <REDACTED_HF_TOKEN> was committed in the notebook and has been removed here.
Go to: https://huggingface.co/settings/tokens → revoke and create a new token.
Always use `userdata.get('HF_TOKEN')` in Colab, never hardcode.

---

## 14. Pending Learning Sessions (user requested these)

### Session A — Code walkthrough (layman terms)
Explain both training notebooks cell by cell in simple English:
- What each line does
- Why it is needed
- What would break if removed
Goal: user can explain the code to others and build ML/NLP foundation.

### Session B — Build SLM from scratch (full template)
Cover end-to-end pipeline with template code:
1. What is an LLM/SLM — how it works conceptually
2. Dataset collection and formats
3. Data cleaning using NLP
4. HuggingFace Datasets — why over plain JSONL
5. Tokenization — padding, truncation, special tokens
6. LoRA — what it is, why, rank/alpha parameters
7. Continued Pre-Training (CPT) vs SFT — how they differ
8. Training config — epochs, batch size, gradient accumulation, LR, schedulers
9. Evaluation — loss, ROUGE, BERTScore, human eval
10. Quantization — 4-bit NF4, GGUF Q4_K_M, trade-offs
11. Deployment — Ollama, llama.cpp, MediaPipe, Core ML

---

## 15. Full Product Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | NCERT domain knowledge (Stage 1 CPT) | DONE |
| Phase 2A | UPSC Mains answer structure SFT | DONE (partially) |
| Phase 2B | Prelims MCQ training | NEXT — dataset ready |
| Phase 2C | Coaching institute answers (VisionIAS/Drishti) | Not started |
| Phase 2D | Real UPSC topper answers calibration | Not started |
| Phase 3 | Multi-turn chat (3-5 turn tutoring) | Not started |
| Phase 4 | Answer checking — student uploads answer, model marks + feedback | Not started |
| Phase 5 | Personalisation — weak area tracking, RAG for current affairs | Not started |
| Phase 6 | On-device deployment — GGUF Q4_K_M, Flutter app, llama.cpp | Not started |
| Phase 7 | Research paper — arXiv preprint → AIED 2027 | Not started |

---

## 16. Deployment Target

| Platform | Method | RAM Needed |
|----------|--------|-----------|
| Android | MediaPipe/Google AI Edge, GGUF Q4_K_M ~1.5GB | 4GB+ |
| iOS/iPad | llama.cpp Swift or Core ML | 4GB+ |
| Laptop (Mac/Windows) | Ollama or llama.cpp | 8GB |
| Raspberry Pi | GGUF Q3_K_M (smaller) | 4GB |

Fully offline — no internet required after download.

---

## 17. Environment Setup

```bash
# Python environment
cd /Users/satyamurti/Downloads/DataSets
source .venv/bin/activate       # always activate before running scripts

# API keys location
cat env/.env                    # OPENAI_API_KEY and HF_TOKEN are here

# Install dependencies (if needed)
pip install pymupdf openai python-dotenv

# Key dependencies
# pymupdf   — PDF table extraction for VisionIAS PDFs
# openai    — GPT-4o calls for synthetic MCQ generation
# trl       — SFTTrainer for fine-tuning (Colab only)
# peft      — LoRA adapters (Colab only)
# bitsandbytes — 4-bit quantization (Colab only)
```

---

## 18. Important Technical Decisions

**Why Gemma 2 2B-IT and not larger?**
Runs on mobile devices. 2B parameters quantized to Q4 = ~1.5GB. Fits in 4GB phone RAM.

**Why LoRA and not full fine-tuning?**
Full fine-tuning on 2B model needs ~32GB VRAM. LoRA trains ~1% of parameters, fits in T4 (16GB) on free Colab.

**Why two data types (synthetic + real)?**
Synthetic MCQs (from NCERT) build breadth — model learns all subjects.
Real UPSC MCQs (from VisionIAS PDFs) build precision — model learns the actual exam's difficulty, traps, and nuance.

**Why Alpaca format instead of Gemma chat template in dataset?**
Alpaca format is model-agnostic. The SFT trainer applies the chat template at training time. If you pre-bake the Gemma template into every dataset record, you cannot reuse the same dataset with Llama or Mistral without regenerating it.

**Why a quality check script?**
GPT-4o sometimes hallucinates facts not in the source passage. VisionIAS PDFs have incomplete options and short explanations. Without quality filtering, garbage records enter training and degrade the model.
