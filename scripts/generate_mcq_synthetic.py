# =============================================================================
# generate_mcq_synthetic.py
# =============================================================================
# Generates synthetic UPSC Prelims-style MCQs from pretrain context chunks
# using GPT-4o. This creates a high-quality MCQ training dataset for the SLM
# so it learns to both ANSWER and GENERATE MCQ questions.
#
# WHY this matters for training:
#   The model must do two things in production:
#     1. Answer a given MCQ (pick correct option + explain)
#     2. Generate MCQs on demand (e.g. "Give me 5 MCQs on Mughal Empire")
#   Both behaviors need to be in the SFT dataset.
#
# GPT-4o reads a context chunk -> generates 3-5 UPSC Prelims-style MCQs
# with 4 options, correct answer, and explanation.
#
# SFT formats generated:
#   answer_mcq        -- user gives Q+options, model answers with explanation
#   explain_correct   -- user asks why (x) is correct, model explains
#   generate_by_topic -- user asks for N MCQs on a topic, model generates them
#   quiz_mode         -- questions first, answer key at end
#   context_based     -- given this passage + question + options, what is answer?
#
# Output:
#   synthetic_mcq_raw.jsonl   -- raw extracted MCQs (one per line)
#   synthetic_mcq_sft.jsonl   -- SFT training pairs (instruction-input-output / Alpaca format)
#
# Usage:
#   python3 scripts/generate_mcq_synthetic.py --api openai
#   python3 scripts/generate_mcq_synthetic.py --dry-run --limit 5
#   python3 scripts/generate_mcq_synthetic.py --resume
#   python3 scripts/generate_mcq_synthetic.py --subject History --limit 100
# =============================================================================

import json
import os
import re
import sys
import time
import random
import argparse
import hashlib
from collections import defaultdict
from datetime import datetime
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR      = "/Users/satyamurti/Downloads/DataSets"
PRETRAIN_PATH = f"{BASE_DIR}/dataset_output_final/combined/unified_pretrain_clean.jsonl"
OUT_DIR       = f"{BASE_DIR}/dataset_output_final/combined"
RAW_JSONL     = f"{OUT_DIR}/synthetic_mcq_raw.jsonl"
SFT_JSONL     = f"{OUT_DIR}/synthetic_mcq_sft.jsonl"
PROGRESS_FILE = f"{OUT_DIR}/synthetic_mcq_progress.json"

GEMMA_TEMPLATE = (
    "<start_of_turn>user\n{question}<end_of_turn>\n"
    "<start_of_turn>model\n{answer}<end_of_turn>"
)

# ---------------------------------------------------------------------------
# GPT-4o Prompt
# ---------------------------------------------------------------------------
# This is the most important part. The prompt defines the quality of MCQs.
# UPSC Prelims MCQs follow very specific patterns — the prompt enforces them.

SYSTEM_PROMPT = """You are an expert UPSC Civil Services Prelims question setter with 15 years of experience.
You create high-quality multiple choice questions that mirror the actual UPSC Prelims exam pattern.

UPSC PRELIMS PATTERN (2022-2025 CONSENSUS — apply these patterns):

A. QUESTION FORMAT DISTRIBUTION (use this exact mix per batch):
   - 40% "Consider the following statements" (2-4 statements, which is/are correct)
   - 18% Assertion-Reason: "Statement I: ... Statement II: ..." with explanation-link test
   - 12% Match Pairs: "Match the following. How many pairs are correctly matched?"
   - 15% Direct factual: single-line question, 4 distinct options
   - 8%  "How many of the above are correct?" (options: Only one / Only two / All / None)
   - 7%  Conceptual/analytical

B. CRITICAL RULES FOR STATEMENT-BASED QUESTIONS:
   - NEVER make all statements correct. At minimum one must be wrong.
   - The wrong statement must be SUBTLY wrong — one word, one number, one attribution error
   - Options pattern: "1 only", "2 only", "1 and 2 only", "1 and 3 only", "1, 2 and 3", "2 and 3 only"
   - Common UPSC traps (use these): swap constitutional bodies, invert statistics, wrong article number,
     confuse mandatory vs discretionary, wrong treaty signatory, invert direction of effect

C. CRITICAL RULES FOR ASSERTION-REASON FORMAT:
   - Statement I: a factual claim about the topic
   - Statement II: a claim that may or may not correctly explain Statement I
   - Four fixed options:
     (a) Both Statement I and Statement II are correct and Statement II explains Statement I
     (b) Both Statement I and Statement II are correct but Statement II does NOT explain Statement I
     (c) Statement I is correct but Statement II is incorrect
     (d) Statement I is incorrect but Statement II is correct

D. CRITICAL RULES FOR MATCH PAIRS:
   - 3-4 pairs. At least one pair must be INCORRECTLY matched.
   - Options: "Only one pair", "Only two pairs", "All three pairs", "None of the pairs"
   - Never make all pairs correct — that is not UPSC style.

E. DIFFICULTY CALIBRATION (EQUAL THIRDS — 2025 UPSC POLICY):
   - Easy (33%): direct recall of NCERT/standard textbook fact, no inference needed
   - Medium (33%): links two concepts, or tests a specific provision/exception
   - Difficult (33%): tests precision — exact article number, exact count, subtle legal distinction,
     or requires distinguishing near-identical correct/incorrect facts

F. NATURE OF QUESTION (mirror UPSC 2022-2025 mix):
   - 41% Pure Fundamental (F): static knowledge from standard books
   - 24% Current Affairs (CA): linked to a recent event or development
   - 14% Fundamental + Current Affairs (FCA): static concept triggered by a current event
   - 12% Fundamental Applied (FA): analytical question, answer derived from applying a textbook concept
   - 9%  Current Affairs Applied (CAA): applied reasoning on a current event
   - For each question, add a "nature" field: "F", "FA", "CA", "CAA", or "FCA"

G. EXPLANATION REQUIREMENTS:
   - Must identify WHICH specific statement is wrong and state the CORRECT fact
   - Must explain why the correct option is right
   - Must cite the specific fact from the context
   - Format: "Statement X is correct because [reason]. Statement Y is NOT correct — the correct fact is [fact]. Hence option (Z) is the answer."

H. QUESTION QUALITY RULES:
   - All 4 options must be plausible — wrong options must not be obviously absurd
   - Only ONE option is correct — no ambiguity
   - Questions must be answerable from the provided context
   - Minimum 8 words in question, minimum 30 words in explanation
   - Do NOT use phrases like "according to the passage" or "based on the context" in questions
   - Do NOT make all MCQs about the same sub-topic within a batch

OUTPUT FORMAT — return a JSON array only, no other text:
[
  {
    "question": "Consider the following statements about [topic]:\\n1. [statement]\\n2. [statement]\\n3. [statement]\\nHow many of the above statements are correct?",
    "options": {
      "a": "Only one",
      "b": "Only two",
      "c": "All three",
      "d": "None"
    },
    "correct": "b",
    "explanation": "Statement 1 is correct because [specific reason]. Statement 2 is correct because [specific reason]. Statement 3 is NOT correct — [correct fact]. Since two statements are correct, option (b) is the answer.",
    "question_type": "statement_based",
    "difficulty": "medium",
    "nature": "F"
  }
]

question_type must be one of: statement_based, assertion_reason, match_pairs, factual, conceptual, how_many_correct
difficulty must be one of: easy, medium, difficult
nature must be one of: F, FA, CA, CAA, FCA"""

USER_PROMPT_TEMPLATE = """Generate {n} UPSC Prelims MCQ questions based on the following context passage.

Subject: {subject}
Context:
{context}

MANDATORY RULES FOR THIS BATCH:
1. Use a MIX of formats — do not make all questions the same type. Target: 2 statement_based, 1 assertion_reason, and 1 factual per 4 questions.
2. For EVERY statement-based question: at minimum ONE statement MUST be factually wrong (never all statements correct).
3. Vary difficulty: target equal thirds (Easy / Medium / Difficult). For 3 questions: one each. For 4 questions: 1E + 1M + 1D + 1M or 1D.
4. For match_pairs: at least one pair must be INCORRECTLY matched. Never all correct.
5. Explanations must name the EXACT fact that makes a statement wrong and state the correct fact.
6. Questions must be directly answerable from the context above.
7. Make wrong options PLAUSIBLE — a well-prepared candidate must think carefully.
8. Add a "nature" field to each question: F, FA, CA, CAA, or FCA. Aim for mostly F and FA from NCERT context.

Return only the JSON array, no preamble or explanation outside the array."""


# ---------------------------------------------------------------------------
# GPT-4o client
# ---------------------------------------------------------------------------
def load_api_key() -> str:
    """Load OpenAI API key from env/.env file."""
    env_path = os.path.join(BASE_DIR, "env", ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip().strip('"').strip("'")
                if line.startswith("OPENAI_API_KEY"):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if key:
                        return key
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise ValueError("OPENAI_API_KEY not found in env/.env or environment")
    return key


def call_gpt4o(context: str, subject: str, n: int, api_key: str,
               max_retries: int = 3) -> Optional[list]:
    """
    Call GPT-4o to generate MCQs from context.
    Returns list of MCQ dicts, or None on failure.
    """
    import urllib.request

    user_msg = USER_PROMPT_TEMPLATE.format(
        n=n, subject=subject, context=context[:3000]
    )

    payload = json.dumps({
        "model"      : "gpt-4o",
        "temperature": 0.7,
        "max_tokens" : 2000,
        "messages"   : [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
    }).encode("utf-8")

    headers = {
        "Content-Type" : "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    for attempt in range(max_retries):
        try:
            req  = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=payload, headers=headers, method="POST"
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data    = json.loads(resp.read().decode("utf-8"))
                content = data["choices"][0]["message"]["content"].strip()

            # Strip markdown code fences if present
            content = re.sub(r'^```(?:json)?\s*', '', content, flags=re.MULTILINE)
            content = re.sub(r'\s*```$', '', content, flags=re.MULTILINE)
            content = content.strip()

            # Fix literal newlines inside strings (common GPT-4o issue)
            def fix_json_literals(raw: str) -> str:
                result = []
                in_string = False
                i = 0
                while i < len(raw):
                    ch = raw[i]
                    if ch == '"' and (i == 0 or raw[i-1] != '\\'):
                        in_string = not in_string
                        result.append(ch)
                    elif in_string and ch == '\n':
                        result.append('\\n')
                    elif in_string and ch == '\t':
                        result.append('\\t')
                    else:
                        result.append(ch)
                    i += 1
                return ''.join(result)

            content = fix_json_literals(content)
            mcqs = json.loads(content)

            if not isinstance(mcqs, list):
                raise ValueError("Response is not a JSON array")

            return mcqs

        except Exception as e:
            wait = 2 ** attempt
            print(f"  Attempt {attempt+1} failed: {e}. Retrying in {wait}s...")
            time.sleep(wait)

    return None


# ---------------------------------------------------------------------------
# Validate a single MCQ from GPT-4o
# ---------------------------------------------------------------------------
VALID_QUESTION_TYPES = {
    "statement_based", "assertion_reason", "match_pairs",
    "factual", "conceptual", "analytical", "how_many_correct",
}

def validate_mcq(mcq: dict) -> tuple:
    """
    Returns (is_valid: bool, reason: str).
    Checks required fields, option completeness, correct answer validity.
    """
    required = ["question", "options", "correct", "explanation"]
    for field in required:
        if not mcq.get(field):
            return False, f"missing field: {field}"

    opts = mcq["options"]
    if not isinstance(opts, dict) or len(opts) < 4:
        return False, f"insufficient options: {len(opts) if isinstance(opts, dict) else 0}"

    correct = mcq["correct"].lower().strip()
    if correct not in opts:
        return False, f"correct='{correct}' not in options {list(opts.keys())}"

    q = mcq["question"].strip()
    if len(q.split()) < 8:
        return False, f"question too short: {len(q.split())} words"

    expl = mcq["explanation"].strip()
    if len(expl.split()) < 15:
        return False, f"explanation too short: {len(expl.split())} words"

    # Normalise question_type to a known value
    qt = mcq.get("question_type", "factual").lower().strip()
    if qt not in VALID_QUESTION_TYPES:
        mcq["question_type"] = "factual"  # coerce unknown types

    # Normalise nature field (new field from updated prompt)
    valid_natures = {"F", "FA", "CA", "CAA", "FCA"}
    nature = mcq.get("nature", "F").upper().strip()
    if nature not in valid_natures:
        mcq["nature"] = "F"
    else:
        mcq["nature"] = nature

    # Check for passage references — invalid because at inference time there's no passage
    passage_phrases = ["this passage", "the passage", "as mentioned in the passage",
                       "according to the passage", "the above passage",
                       "as stated in the context", "according to the context"]
    q_lower = q.lower() + expl.lower()
    for phrase in passage_phrases:
        if phrase in q_lower:
            return False, f"passage reference: '{phrase}'"

    return True, ""


# ---------------------------------------------------------------------------
# Load progress
# ---------------------------------------------------------------------------
def load_progress(path: str) -> set:
    """Load set of already-processed chunk IDs."""
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        data = json.load(f)
    return set(data.get("processed_chunk_ids", []))


def save_progress(path: str, processed_ids: set):
    with open(path, "w") as f:
        json.dump({
            "processed_chunk_ids": list(processed_ids),
            "count"              : len(processed_ids),
            "updated_at"         : datetime.now().isoformat(),
        }, f, indent=2)


# ---------------------------------------------------------------------------
# Build SFT pairs from raw MCQs
# ---------------------------------------------------------------------------

# Request variants for the "generate MCQs on topic" format
GENERATE_REQUESTS = [
    "Give me {n} UPSC Prelims MCQ questions on {topic}.",
    "Generate {n} multiple choice questions on {topic} for UPSC Prelims practice.",
    "I want to practice UPSC Prelims MCQs on {topic}. Give me {n} questions.",
    "Create {n} Prelims-style MCQs on {topic} with explanations.",
    "Give {n} UPSC practice MCQs on {topic} with correct answers.",
    "Test me on {topic} with {n} Prelims-style MCQs.",
    "Prepare {n} UPSC Prelims questions from the topic {topic}.",
    "What are {n} important MCQ questions on {topic} for UPSC Prelims?",
]

ANSWER_REQUESTS = [
    "What is the answer to this UPSC Prelims question?\n\n{q_text}",
    "Solve this MCQ from UPSC Prelims:\n\n{q_text}",
    "Answer this UPSC question with explanation:\n\n{q_text}",
    "Which option is correct and why?\n\n{q_text}",
    "For this UPSC Prelims MCQ, what is the correct answer?\n\n{q_text}",
]

EXPLAIN_REQUESTS = [
    "Explain why option ({correct}) is correct for this question:\n\n{q_text}",
    "Why is ({correct}) the right answer for this UPSC question?\n\n{q_text}",
    "For the following MCQ, explain the correct answer in detail:\n\n{q_text}",
    "This UPSC question has answer ({correct}). Explain why:\n\n{q_text}",
]

CONTEXT_REQUESTS = [
    "Based on the following passage, answer the MCQ:\n\nPassage: {context}\n\n{q_text}",
    "Read this passage and answer the question:\n\nPassage: {context}\n\nQuestion: {q_text}",
    "Use the context below to answer the MCQ.\n\nContext: {context}\n\n{q_text}",
]

QUIZ_REQUESTS = [
    "Quiz me with {n} MCQs on {topic}. Show questions first, reveal answers at the end.",
    "Give me {n} {topic} MCQs for practice. Questions only first, then the answer key.",
    "Test my knowledge on {topic} — {n} MCQs, answers at the end.",
]


def format_question_text(mcq: dict, show_answer: bool = False) -> str:
    """Format a single MCQ as readable text for the SFT prompt."""
    opts = mcq["options"]
    lines = [mcq["question"], ""]
    for letter in ["a", "b", "c", "d"]:
        if letter in opts:
            lines.append(f"({letter}) {opts[letter]}")
    if show_answer:
        lines.append("")
        lines.append(f"Correct Answer: ({mcq['correct']})")
    return "\n".join(lines)


def format_many_questions(mcqs: list, show_answers: bool = True,
                          show_explanations: bool = True) -> str:
    """Format a list of MCQs as a block of text."""
    sep   = "\n\n" + "-" * 50 + "\n\n"
    parts = []
    for i, mcq in enumerate(mcqs, 1):
        opts = mcq["options"]
        lines = [f"Q{i}. {mcq['question']}", ""]
        for letter in ["a", "b", "c", "d"]:
            if letter in opts:
                lines.append(f"({letter}) {opts[letter]}")
        if show_answers:
            lines.append("")
            lines.append(f"Correct Answer: ({mcq['correct']})")
        if show_explanations and mcq.get("explanation"):
            lines.append("")
            lines.append(f"Explanation: {mcq['explanation']}")
        parts.append("\n".join(lines))
    return sep.join(parts)


def _make_question_block(r: dict) -> str:
    """Format question + options as a string (no answer shown)."""
    opts  = r["options"]
    lines = [r["question"], ""]
    for letter in ["a", "b", "c", "d"]:
        if letter in opts:
            lines.append(f"({letter}) {opts[letter]}")
    return "\n".join(lines)


def _make_answer_block(r: dict) -> str:
    """Format question + options + correct answer + explanation."""
    opts  = r["options"]
    lines = [r["question"], ""]
    for letter in ["a", "b", "c", "d"]:
        if letter in opts:
            lines.append(f"({letter}) {opts[letter]}")
    lines.append("")
    lines.append(f"Correct Answer: ({r['correct']})")
    if r.get("explanation"):
        lines.append(f"\nExplanation: {r['explanation']}")
    return "\n".join(lines)


def build_sft_pairs(raw_records: list) -> list:
    """
    Build instruction-input-output (Alpaca) SFT records from raw MCQ records.

    Output format:
      {
        "instruction": "Generate a UPSC-style MCQ on {subject}, {topic}.",
        "input": "<context passage or question block>",
        "question": "<full question with options>",
        "answer": "c",
        "explanation": "...",
        "subject": "History",
        "question_type": "statement_based",
        "difficulty": "medium",
        "exam": "UPSC",
        "format": "<variant name>",
        "source_ids": [...]
      }

    Variants generated per MCQ:
      generate_by_topic  -- "Generate a UPSC MCQ on {subject}, {topic}."
      answer_mcq         -- "Answer this UPSC Prelims MCQ and explain."
      explain_correct    -- "Why is option ({answer}) correct?"
      context_based      -- "Based on this passage, answer the MCQ."
      practice_request   -- "Give me {n} MCQs on {topic}." (batch per topic)
      quiz_by_subject    -- "Give me {n} MCQs on {subject}." (batch per subject)
    """
    sft    = []
    sft_id = [0]

    GENERATE_VARIANTS = [
        "Generate a UPSC-style MCQ on {subject}, {topic}.",
        "Generate 1 {difficulty}-level UPSC Prelims question on {topic}.",
        "Create a UPSC Prelims MCQ on {topic} from the subject {subject}.",
        "Give me one UPSC-style question on {topic}.",
    ]

    ANSWER_VARIANTS = [
        "Answer this UPSC Prelims MCQ and explain the correct option.",
        "Solve this MCQ from UPSC Prelims on {subject} and give the explanation.",
        "What is the correct answer to this UPSC Prelims question? Explain.",
        "Identify the correct option and justify your answer.",
    ]

    EXPLAIN_VARIANTS = [
        "Why is option ({answer}) the correct answer for this question?",
        "Explain why ({answer}) is correct for this UPSC Prelims question.",
        "For this UPSC question, justify why option ({answer}) is right.",
        "What makes option ({answer}) the best answer here?",
    ]

    CONTEXT_VARIANTS = [
        "Based on the following passage, answer the MCQ.",
        "Read the context and answer the question.",
        "Use the passage below to answer this UPSC Prelims question.",
        "Given this reading, identify the correct MCQ option.",
    ]

    PRACTICE_VARIANTS = [
        "I want to practice UPSC questions on {topic}. Give me {n} questions.",
        "Quiz me with {n} MCQs on {topic}. Show answers at the end.",
        "Give me {n} MCQ questions on {topic} for UPSC Prelims.",
        "Generate {n} UPSC Prelims MCQs on {topic}.",
    ]

    SUBJ_VARIANTS = [
        "Give me {n} MCQ questions on {subject} for UPSC Prelims.",
        "I want {n} Prelims practice questions on {subject}.",
        "Generate {n} UPSC Prelims MCQs from the subject {subject}.",
        "What are {n} important MCQs on {subject} for Civil Services exam?",
    ]

    def add(instruction: str, inp: str, q_block: str, answer: str,
            explanation: str, subject: str, q_type: str, difficulty: str,
            fmt: str, source_ids: list):
        sft_id[0] += 1
        sft.append({
            "id"           : f"synth_mcq_sft_{sft_id[0]:06d}",
            "instruction"  : instruction,
            "input"        : inp,
            "question"     : q_block,
            "answer"       : answer,
            "explanation"  : explanation,
            "subject"      : subject,
            "question_type": q_type,
            "difficulty"   : difficulty,
            "exam"         : "UPSC",
            "format"       : fmt,
            "source_ids"   : source_ids,
        })

    by_subject = defaultdict(list)
    by_topic   = defaultdict(list)

    for r in raw_records:
        by_subject[r["subject"]].append(r)
        topic = r.get("topic", r["subject"])
        by_topic[topic].append(r)

    # ── Variant 1: generate_by_topic ──────────────────────────────────────
    for r in raw_records:
        topic   = r.get("topic", r["subject"])
        subject = r["subject"]
        diff    = r.get("difficulty", "medium")
        instr   = random.choice(GENERATE_VARIANTS).format(
            subject=subject, topic=topic, difficulty=diff)
        context = r.get("source_context", "")[:800]
        q_block = _make_question_block(r)
        add(
            instruction=instr,
            inp=context,
            q_block=q_block,
            answer=r["correct"],
            explanation=r.get("explanation", ""),
            subject=subject,
            q_type=r.get("question_type", "factual"),
            difficulty=diff,
            fmt="generate_by_topic",
            source_ids=[r["id"]],
        )

    # ── Variant 2: answer_mcq ─────────────────────────────────────────────
    for r in raw_records:
        subject = r["subject"]
        diff    = r.get("difficulty", "medium")
        instr   = random.choice(ANSWER_VARIANTS).format(subject=subject)
        q_block = _make_question_block(r)
        add(
            instruction=instr,
            inp=q_block,
            q_block=q_block,
            answer=r["correct"],
            explanation=r.get("explanation", ""),
            subject=subject,
            q_type=r.get("question_type", "factual"),
            difficulty=diff,
            fmt="answer_mcq",
            source_ids=[r["id"]],
        )

    # ── Variant 3: explain_correct ────────────────────────────────────────
    for r in raw_records:
        if not r.get("explanation") or len(r["explanation"].split()) < 15:
            continue
        subject = r["subject"]
        diff    = r.get("difficulty", "medium")
        instr   = random.choice(EXPLAIN_VARIANTS).format(answer=r["correct"])
        q_block = _make_question_block(r)
        add(
            instruction=instr,
            inp=q_block,
            q_block=q_block,
            answer=r["correct"],
            explanation=r["explanation"],
            subject=subject,
            q_type=r.get("question_type", "factual"),
            difficulty=diff,
            fmt="explain_correct",
            source_ids=[r["id"]],
        )

    # ── Variant 4: context_based ──────────────────────────────────────────
    for r in raw_records:
        if not r.get("source_context"):
            continue
        subject = r["subject"]
        diff    = r.get("difficulty", "medium")
        instr   = random.choice(CONTEXT_VARIANTS)
        context = r["source_context"][:800]
        q_block = _make_question_block(r)
        inp     = f"Passage:\n{context}\n\nQuestion:\n{q_block}"
        add(
            instruction=instr,
            inp=inp,
            q_block=q_block,
            answer=r["correct"],
            explanation=r.get("explanation", ""),
            subject=subject,
            q_type=r.get("question_type", "factual"),
            difficulty=diff,
            fmt="context_based",
            source_ids=[r["id"]],
        )

    # ── Variant 5: practice_request (batch, 3-5 per topic) ────────────────
    for n in [3, 5]:
        for topic, recs in by_topic.items():
            if len(recs) < n:
                continue
            sample  = random.sample(recs, n)
            subject = sample[0]["subject"]
            diff    = sample[0].get("difficulty", "medium")
            instr   = random.choice(PRACTICE_VARIANTS).format(
                n=n, topic=topic, subject=subject)
            q_block = "\n\n".join(
                f"Q{i+1}.\n{_make_answer_block(r)}"
                for i, r in enumerate(sample)
            )
            answer  = ", ".join(
                f"Q{i+1}: ({r['correct']})" for i, r in enumerate(sample))
            expl    = "\n\n".join(
                f"Q{i+1}: {r.get('explanation', '')[:300]}"
                for i, r in enumerate(sample)
            )
            add(
                instruction=instr,
                inp="",
                q_block=q_block,
                answer=answer,
                explanation=expl,
                subject=subject,
                q_type="batch",
                difficulty=diff,
                fmt="practice_request",
                source_ids=[r["id"] for r in sample],
            )

    # ── Variant 6: quiz_by_subject (batch, 5 per subject) ─────────────────
    for n in [5]:
        for subj, recs in by_subject.items():
            if len(recs) < n:
                continue
            sample = random.sample(recs, n)
            diff   = sample[0].get("difficulty", "medium")
            instr  = random.choice(SUBJ_VARIANTS).format(n=n, subject=subj)
            q_block = "\n\n".join(
                f"Q{i+1}.\n{_make_answer_block(r)}"
                for i, r in enumerate(sample)
            )
            answer  = ", ".join(
                f"Q{i+1}: ({r['correct']})" for i, r in enumerate(sample))
            expl    = "\n\n".join(
                f"Q{i+1}: {r.get('explanation', '')[:300]}"
                for i, r in enumerate(sample)
            )
            add(
                instruction=instr,
                inp="",
                q_block=q_block,
                answer=answer,
                explanation=expl,
                subject=subj,
                q_type="batch",
                difficulty=diff,
                fmt="quiz_by_subject",
                source_ids=[r["id"] for r in sample],
            )

    return sft


# ---------------------------------------------------------------------------
# Chunk quality filters  (applied before sending to GPT-4o)
# ---------------------------------------------------------------------------

# PDF/OCR artifact patterns to strip from context before sending to GPT-4o
_ARTIFACT_RE = re.compile(
    r'\|\)|'           # |) — common OCR table artifact
    r'\|\(|'           # |(
    r'<[A-Z\']{2,}>|'  # <MCRO>, <BI'> — PyMuPDF box artifacts
    r'[~]{2,}|'        # ~~~~ — separator noise
    r'\x00|'           # null bytes
    r'\\x[0-9a-f]{2}'  # hex-encoded bytes
)

# Exercise-question pattern: "6. Compare..." "3) Explain..."
_EXERCISE_RE = re.compile(r'^\s*\d{1,2}[\.\)]\s+[A-Za-z]')

# Equation/formula heading: "14.7 A particle...", "10.6 MICRO..."
_EQUATION_RE = re.compile(r'^\d+[\.\d]+\s')


def _is_usable_chunk(chunk: dict) -> bool:
    """
    Return True only if the chunk is suitable for MCQ generation.

    Rejects:
      - Too short (< 80 words)
      - Starts mid-sentence (chunk boundary error)
      - Section heading is a formula reference or single char
      - Repeated-word OCR artifacts (full garbage pages)
    """
    text    = chunk.get("text", "").strip()
    heading = chunk.get("section_heading") or chunk.get("topic") or ""

    # Minimum content
    if len(text.split()) < 80:
        return False

    # Starts mid-sentence — chunk was cut at a wrong boundary
    if text and text[0].islower():
        return False

    # Heading is a single letter (OCR dropped the rest)
    if re.match(r'^[A-Z]\s*$', heading.strip()):
        return False

    # Heading looks like an equation reference ("10.6 MICRO...")
    if _EQUATION_RE.match(heading.strip()):
        return False

    # Repeated-word garbage (e.g. "the the the the")
    if re.search(r'\b(\w{3,})\s+\1\s+\1', text):
        return False

    return True


def _clean_context(text: str) -> str:
    """
    Strip PDF extraction artifacts from context before sending to GPT-4o.
    Collapses excessive whitespace and removes known OCR noise.
    """
    text = _ARTIFACT_RE.sub(' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)   # max 2 consecutive newlines
    text = re.sub(r'[ \t]{2,}', ' ', text)   # collapse inline spaces
    text = text.strip()
    return text


def _extract_clean_topic(chunk: dict) -> str:
    """
    Extract a human-readable topic string from a chunk.

    Priority: section_heading > source_file stem > subject.
    Rejects headings that look like IDs, equations, or single chars.
    """
    raw = chunk.get("section_heading") or chunk.get("topic") or ""

    # Strip underscores (chunk IDs use them)
    topic = re.sub(r'[_]+', ' ', raw).strip()

    # Remove leading digit sequences (like "14 7" after underscore removal)
    topic = re.sub(r'^\d[\d\s]+', '', topic).strip()

    # Collapse excessive whitespace / newlines
    topic = re.sub(r'\s+', ' ', topic).strip()

    # Reject if still looks like a formula or is too short
    if (not topic
            or len(topic) < 4
            or _EQUATION_RE.match(topic)
            or re.match(r'^[A-Z]\s*$', topic)
            or re.search(r'\d{4,}', topic)):
        # Fall back to source file name, humanised
        src = chunk.get("source_file", "")
        stem = re.sub(r'\.(pdf|jsonl|txt)$', '', src, flags=re.IGNORECASE)
        stem = re.sub(r'[_\-]+', ' ', stem).strip()
        stem = re.sub(r'\b(NCERT|Class|Part)\b', '', stem, flags=re.IGNORECASE).strip()
        stem = re.sub(r'\s+', ' ', stem).strip()
        if stem and len(stem) >= 4:
            return stem[:60]
        return chunk.get("subject", "General Studies")

    return topic[:80]


def _is_grounded_mcq(mcq: dict, context: str) -> bool:
    """
    Return False if the MCQ explanation introduces proper nouns that are not
    present in the source context or in the question text itself.

    Threshold: > 4 ungrounded proper nouns = likely hallucination.
    This catches the case where GPT-4o uses its own knowledge instead of
    the passage (e.g. explaining "Ohio, Mississippi, Seattle" from a chunk
    that only mentions "colonies").
    """
    STOPWORDS = {
        "Statement", "Option", "Hence", "Therefore", "However",
        "Correct", "Incorrect", "Because", "According", "Consider",
        "Which", "These", "This", "Thus", "Both", "Neither", "Only",
        "Based", "Note", "Also", "While", "Since", "Although",
        "India", "Indian",  # almost always present in UPSC context
    }

    explanation = mcq.get("explanation", "")
    question    = mcq.get("question", "")

    expl_proper = set(re.findall(r'\b([A-Z][a-z]{3,})\b', explanation))
    expl_proper -= STOPWORDS

    ref_text    = context + " " + question
    ref_words   = set(re.findall(r'\b([A-Z][a-z]{3,})\b', ref_text))
    ref_lower   = {w.lower() for w in ref_words}

    ungrounded  = [
        w for w in expl_proper
        if w not in ref_words and w.lower() not in ref_lower
    ]

    return len(ungrounded) <= 4


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------
def load_pretrain_chunks(path: str, subject_filter: Optional[str] = None,
                         limit: Optional[int] = None) -> list:
    """
    Load pretrain chunks, applying quality filters.

    Only chunks that pass _is_usable_chunk() are returned.
    Prints a summary of how many were rejected and why.
    """
    total    = 0
    rejected = 0
    chunks   = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            total += 1

            if subject_filter:
                if r.get("subject", "").lower() != subject_filter.lower():
                    continue

            if not _is_usable_chunk(r):
                rejected += 1
                continue

            chunks.append(r)
            if limit and len(chunks) >= limit:
                break

    print(f"Chunk filter    : {total:,} total, {rejected:,} rejected, "
          f"{len(chunks):,} usable ({100*len(chunks)/max(total,1):.1f}%)")
    return chunks


def run_generation(args):
    """Main generation loop — process chunks, call GPT-4o, save output."""

    api_key = load_api_key()
    print(f"API key loaded (last 6 chars: ...{api_key[-6:]})")

    chunks = load_pretrain_chunks(
        PRETRAIN_PATH,
        subject_filter=args.subject,
        limit=args.chunk_limit,
    )
    print(f"Chunks loaded   : {len(chunks):,}")

    # Load progress
    processed_ids = load_progress(PROGRESS_FILE)
    chunks_to_do  = [c for c in chunks if c["id"] not in processed_ids]
    print(f"Already done    : {len(processed_ids):,}")
    print(f"Remaining       : {len(chunks_to_do):,}")
    print()

    if not chunks_to_do:
        print("All chunks processed. Nothing to do.")
        return

    # Load existing raw records to avoid duplicates
    existing_raw_ids = set()
    if os.path.exists(RAW_JSONL):
        with open(RAW_JSONL) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    existing_raw_ids.add(r.get("source_chunk_id", ""))

    raw_file = open(RAW_JSONL, "a", encoding="utf-8")

    total_mcqs    = 0
    total_skipped = 0
    start_time    = time.time()

    for i, chunk in enumerate(chunks_to_do):
        chunk_id = chunk["id"]
        subject  = chunk.get("subject", "General Studies")

        # Clean context before sending to GPT-4o
        context  = _clean_context(chunk["text"])

        # Extract a human-readable topic (not an exercise number or chunk ID)
        topic    = _extract_clean_topic(chunk)

        # How many MCQs to generate from this chunk
        # Longer chunks -> more MCQs (2025 pattern: ~4 per 300-word chunk)
        word_count = chunk.get("word_count", len(context.split()))
        n_mcqs     = 3 if word_count < 150 else 4

        if args.dry_run:
            print(f"[{i+1}/{len(chunks_to_do)}] DRY RUN | {chunk_id} | {subject} | {word_count} words | would generate {n_mcqs} MCQs")
            processed_ids.add(chunk_id)
            continue

        print(f"[{i+1}/{len(chunks_to_do)}] {chunk_id} | {subject} | {word_count}w | requesting {n_mcqs} MCQs...")

        mcqs = call_gpt4o(context, subject, n_mcqs, api_key,
                          max_retries=3)

        if mcqs is None:
            print(f"  FAILED after retries — skipping")
            total_skipped += 1
            processed_ids.add(chunk_id)
            save_progress(PROGRESS_FILE, processed_ids)
            continue

        valid_count = 0
        for j, mcq in enumerate(mcqs):
            is_valid, reason = validate_mcq(mcq)
            if not is_valid:
                print(f"  MCQ {j+1} invalid: {reason}")
                continue

            # Grounding check — reject MCQs whose explanation introduces
            # proper nouns absent from the source context (hallucination signal)
            if not _is_grounded_mcq(mcq, context):
                print(f"  MCQ {j+1} rejected: explanation not grounded in context")
                continue

            # Normalize options to lowercase keys
            mcq["options"] = {k.lower(): v for k, v in mcq["options"].items()}
            mcq["correct"] = mcq["correct"].lower().strip()

            # Build unique ID from chunk + hash of question
            q_hash = hashlib.md5(mcq["question"].encode()).hexdigest()[:6]
            mcq_id = f"synth_{chunk_id}_{j+1}_{q_hash}"

            # Normalise difficulty label: "hard" -> "difficult", "easy/medium/difficult" only
            raw_diff = mcq.get("difficulty", "medium").lower().strip()
            if raw_diff in ("hard", "difficult", "d"):
                norm_diff = "difficult"
            elif raw_diff in ("easy", "simple", "e"):
                norm_diff = "easy"
            else:
                norm_diff = "medium"

            # Normalise question_type
            raw_qt = mcq.get("question_type", "factual").lower().strip().replace(" ", "_")
            norm_qt = raw_qt if raw_qt in VALID_QUESTION_TYPES else "factual"

            # Normalise nature field (F/FA/CA/CAA/FCA)
            valid_natures = {"F", "FA", "CA", "CAA", "FCA"}
            norm_nature = mcq.get("nature", "F").upper().strip()
            if norm_nature not in valid_natures:
                norm_nature = "F"

            raw_record = {
                "id"             : mcq_id,
                "subject"        : subject,
                "topic"          : topic,
                "source_chunk_id": chunk_id,
                "source_context" : context[:600],
                "question"       : mcq["question"],
                "options"        : mcq["options"],
                "correct"        : mcq["correct"],
                "explanation"    : mcq["explanation"],
                "question_type"  : norm_qt,
                "difficulty"     : norm_diff,
                "nature"         : norm_nature,
                "generated_at"   : datetime.now().isoformat(),
            }

            raw_file.write(json.dumps(raw_record, ensure_ascii=False) + "\n")
            valid_count  += 1
            total_mcqs   += 1

        print(f"  Generated {valid_count} valid MCQs  (total so far: {total_mcqs:,})")

        processed_ids.add(chunk_id)

        # Save progress every 10 chunks
        if (i + 1) % 10 == 0:
            raw_file.flush()
            save_progress(PROGRESS_FILE, processed_ids)
            elapsed = time.time() - start_time
            rate    = (i + 1) / elapsed * 60
            eta_min = (len(chunks_to_do) - i - 1) / max(rate, 0.1)
            print(f"  Progress: {i+1}/{len(chunks_to_do)} | "
                  f"{rate:.1f} chunks/min | ETA ~{eta_min:.0f} min")

        time.sleep(args.delay)

    raw_file.flush()
    raw_file.close()
    save_progress(PROGRESS_FILE, processed_ids)

    if args.dry_run:
        print("\nDry run complete — no files written")
        return

    print(f"\nGeneration complete.")
    print(f"Total MCQs generated : {total_mcqs:,}")
    print(f"Chunks skipped       : {total_skipped:,}")

    # Build SFT pairs from ALL raw records accumulated so far
    print("\nBuilding SFT pairs...")
    all_raw = []
    with open(RAW_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_raw.append(json.loads(line))

    random.seed(42)
    sft = build_sft_pairs(all_raw)

    with open(SFT_JSONL, "w", encoding="utf-8") as f:
        for r in sft:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    fmt_counts = Counter(r["format"] for r in sft)
    print(f"\nSFT pairs written : {len(sft):,}")
    print(f"Raw MCQs total    : {len(all_raw):,}")
    print("Format breakdown:")
    for fmt, cnt in fmt_counts.most_common():
        print(f"  {fmt:<25} {cnt:>6,}")
    print(f"\nRaw  : {RAW_JSONL}")
    print(f"SFT  : {SFT_JSONL}")


# ---------------------------------------------------------------------------
# Rebuild-only mode: regenerate SFT from existing raw MCQs without API calls
# ---------------------------------------------------------------------------
def rebuild_sft_only():
    """Rebuild SFT pairs from already-generated raw MCQs. No API calls needed."""
    if not os.path.exists(RAW_JSONL):
        print(f"No raw MCQ file found at {RAW_JSONL}")
        print("Run without --rebuild-sft first to generate raw MCQs.")
        return

    all_raw = []
    with open(RAW_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_raw.append(json.loads(line))

    print(f"Raw MCQs loaded : {len(all_raw):,}")
    print("Building SFT pairs...")

    random.seed(42)
    sft = build_sft_pairs(all_raw)

    with open(SFT_JSONL, "w", encoding="utf-8") as f:
        for r in sft:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    fmt_counts = Counter(r["format"] for r in sft)
    print(f"SFT pairs written : {len(sft):,}")
    print("Format breakdown:")
    for fmt, cnt in fmt_counts.most_common():
        print(f"  {fmt:<25} {cnt:>6,}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic UPSC Prelims MCQs from pretrain context chunks"
    )
    parser.add_argument(
        "--api", default="openai",
        help="LLM API to use (default: openai)"
    )
    parser.add_argument(
        "--subject", default=None,
        help="Process only this subject (e.g. 'History')"
    )
    parser.add_argument(
        "--chunk-limit", type=int, default=None,
        help="Max pretrain chunks to process (for testing)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be done without calling the API"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last saved progress (default behavior)"
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Ignore progress and start fresh"
    )
    parser.add_argument(
        "--rebuild-sft", action="store_true",
        help="Rebuild SFT pairs from existing raw MCQs only (no API calls)"
    )
    parser.add_argument(
        "--delay", type=float, default=1.5,
        help="Seconds to wait between API calls (default: 1.5)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Chunks processed per batch (default: 1)"
    )
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    if args.rebuild_sft:
        rebuild_sft_only()
        return

    if args.reset and os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
        print("Progress reset.")

    print("=" * 60)
    print("Synthetic MCQ Generator")
    print("=" * 60)
    print(f"Pretrain source : {PRETRAIN_PATH}")
    print(f"Raw output      : {RAW_JSONL}")
    print(f"SFT output      : {SFT_JSONL}")
    print(f"Subject filter  : {args.subject or 'All'}")
    print(f"Chunk limit     : {args.chunk_limit or 'All'}")
    print(f"Dry run         : {args.dry_run}")
    print(f"API delay       : {args.delay}s")
    print("=" * 60)
    print()

    run_generation(args)


if __name__ == "__main__":
    main()
