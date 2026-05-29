#!/usr/bin/env python3
"""
Script 3 of 3: Generate UPSC-style SFT Q+A pairs from pretrain chunks.

For each pretrain chunk → LLM generates 5 Q+A pairs using real UPSC patterns.
Answers are grounded ONLY in the provided context (no hallucination).

Input:
  - dataset_output_final/combined/unified_pretrain_clean.jsonl  (9,569 quality-filtered chunks)
  - dataset_output_final/combined/question_patterns.json  (from analyse_question_patterns.py)

Output:
  - dataset_output_final/combined/unified_sft_v3.jsonl   (~70K records)

Cost estimate:
  - Gemini 1.5 Flash: FREE (1500 req/day, 37 days for all chunks)
  - GPT-4o-mini: ~$15-20 total
  - Claude Haiku: ~$12 total

Run:
    python3 scripts/generate_sft_dataset.py
    python3 scripts/generate_sft_dataset.py --api gemini --limit 100 --dry-run
    python3 scripts/generate_sft_dataset.py --api openai --limit 500 --resume
    python3 scripts/generate_sft_dataset.py --workers 1 --delay 0.5
    python3 scripts/generate_sft_dataset.py --batch-size 5   # 5 QA per chunk (default)
"""

import json
import re
import sys
import os
import time
import random
import argparse
import urllib.request
import urllib.error
from pathlib import Path
from collections import Counter

random.seed(42)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PRETRAIN_FILE  = Path("dataset_output_final/combined/unified_pretrain_clean.jsonl")
PATTERNS_FILE  = Path("dataset_output_final/combined/question_patterns.json")
OUTPUT_FILE    = Path("dataset_output_final/combined/unified_sft_v3.jsonl")
ENV_FILE       = Path("env/.env")
PROGRESS_FILE  = Path("dataset_output_final/combined/sft_v3_progress.json")

# ---------------------------------------------------------------------------
# Subject → best question types (curated from UPSC analysis)
# ---------------------------------------------------------------------------
SUBJECT_QTYPES = {
    "History": [
        "comment", "critically_examine", "how_far", "discuss", "factual",
        "to_what_extent", "explain", "do_you_agree"
    ],
    "Society_Culture": [
        "comment", "discuss", "explain", "evaluate", "factual"
    ],
    "Geography": [
        "explain", "discuss", "factual", "comment", "analyse"
    ],
    "Polity": [
        "critically_examine", "discuss", "evaluate", "comment", "to_what_extent",
        "examine", "do_you_agree"
    ],
    "International_Relations": [
        "critically_examine", "discuss", "comment", "analyse", "evaluate"
    ],
    "Governance_Social": [
        "critically_examine", "discuss", "evaluate", "comment", "examine"
    ],
    "Economy": [
        "discuss", "critically_examine", "evaluate", "analyse", "examine",
        "comment", "factual"
    ],
    "Environment": [
        "discuss", "comment", "critically_examine", "explain", "evaluate"
    ],
    "Science_Tech": [
        "discuss", "explain", "factual", "comment", "evaluate"
    ],
    "Ethics": [
        "discuss", "comment", "evaluate", "do_you_agree", "examine"
    ],
    "Essay": [
        "discuss", "comment", "critically_examine", "evaluate"
    ],
    "default": [
        "discuss", "comment", "explain", "critically_examine", "factual"
    ],
}

# ---------------------------------------------------------------------------
# Question prompts per type (what the model should ask)
# ---------------------------------------------------------------------------
QTYPE_PROMPT_STYLE = {
    "comment":             "Comment on {topic}.",
    "critically_examine":  "Critically examine {topic}.",
    "discuss":             "Discuss {topic}.",
    "explain":             "Explain {topic}.",
    "analyse":             "Analyse {topic}.",
    "evaluate":            "Evaluate {topic}.",
    "examine":             "Examine {topic}.",
    "how_far":             "How far do you agree that {topic} was significant? Justify.",
    "to_what_extent":      "To what extent has {topic} shaped its domain? Discuss.",
    "do_you_agree":        'Do you agree that {topic} is a critical issue? Give reasons.',
    "factual":             "What are the key aspects of {topic}? Explain briefly.",
    "bring_out":           "Bring out the significance of {topic}.",
    "highlight":           "Highlight the main features of {topic}.",
    "justify":             "Justify the importance of {topic}.",
    "assess":              "Assess the role and significance of {topic}.",
    "compare":             "Compare and contrast the key dimensions of {topic}.",
    "write_note":          "Write a short note on {topic}.",
    "elucidate":           "Elucidate the concept of {topic} with examples.",
    "illustrate":          "Illustrate how {topic} impacts its broader context.",
    "trace":               "Trace the evolution and significance of {topic}.",
    "suggest":             "Suggest measures to address the challenges related to {topic}.",
    "in_light_of":         "In the light of recent developments, analyse {topic}.",
    "throw_light":         "Throw light on the key dimensions of {topic}.",
    "with_reference_to":   "With reference to India, explain the significance of {topic}.",
    "critically_analyse":  "Critically analyse the significance and limitations of {topic}.",
    "critically_evaluate": "Critically evaluate the impact of {topic}.",
    "critically_comment":  "Critically comment on {topic}.",
    "critically_assess":   "Critically assess the role of {topic}.",
    "other":               "Discuss the key aspects of {topic}.",
}

# ---------------------------------------------------------------------------
# Gemma 2 chat format
# ---------------------------------------------------------------------------
GEMMA_TEMPLATE = "<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n{answer}<end_of_turn>"

# ---------------------------------------------------------------------------
# Env / API
# ---------------------------------------------------------------------------

def load_env():
    env = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def auto_detect_api(env):
    if env.get("GEMINI_API_KEY"):    return "gemini"
    if env.get("OPENAI_API_KEY"):    return "openai"
    if env.get("ANTHROPIC_API_KEY"): return "claude"
    return None


def call_gemini(prompt, api_key):
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-1.5-flash:generateContent?key={api_key}"
    )
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 4096,
            "topP": 0.9,
        },
    }).encode()
    req = urllib.request.Request(url, data=body,
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        resp = json.loads(r.read())
    return resp["candidates"][0]["content"]["parts"][0]["text"]


def call_openai(prompt, api_key):
    body = json.dumps({
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 4096,
    }).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        resp = json.loads(r.read())
    return resp["choices"][0]["message"]["content"]


def call_claude(prompt, api_key):
    body = json.dumps({
        "model": "claude-haiku-20240307",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        resp = json.loads(r.read())
    return resp["content"][0]["text"]


def call_llm(prompt, api, env, retries=3):
    for attempt in range(retries):
        try:
            if api == "gemini":
                return call_gemini(prompt, env["GEMINI_API_KEY"])
            if api == "openai":
                return call_openai(prompt, env["OPENAI_API_KEY"])
            if api == "claude":
                return call_claude(prompt, env["ANTHROPIC_API_KEY"])
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 60 * (attempt + 1)
                print(f"    Rate limited — waiting {wait}s...")
                time.sleep(wait)
            elif e.code in (500, 503):
                wait = 10 * (attempt + 1)
                print(f"    Server error {e.code} — retry in {wait}s...")
                time.sleep(wait)
            else:
                raise
        except Exception as e:
            if attempt < retries - 1:
                print(f"    Error: {e} — retry {attempt+1}/{retries}")
                time.sleep(5)
            else:
                raise
    raise RuntimeError(f"LLM call failed after {retries} retries")


# ---------------------------------------------------------------------------
# Pattern-aware prompt builder
# Loads question type details + per-paper answer frameworks from patterns JSON
# ---------------------------------------------------------------------------

MAX_CONTEXT_CHARS = 2000   # ~500 tokens of passage context

# Full question type library — key phrases + answer structure + word limit
# Populated from patterns JSON at startup, with hardcoded fallbacks
QTYPE_LIBRARY = {
    "discuss": {
        "key_phrases": [
            "Discuss the significance of", "Discuss the role of",
            "Discuss the impact of", "Discuss with suitable examples",
            "Discuss the challenges and prospects of",
        ],
        "answer_structure": (
            "**Introduction** (2 lines: define/contextualize the topic from the passage)\n"
            "**Key Dimension 1** (with evidence from passage)\n"
            "**Key Dimension 2** (with evidence from passage)\n"
            "**Key Dimension 3** (if applicable)\n"
            "**Conclusion** (1 balanced line tying back to the question)"
        ),
        "word_limits": [150, 250],
        "marks_map": {150: 10, 250: 15},
    },
    "critically_examine": {
        "key_phrases": [
            "Critically examine", "Critically examine the statement that",
            "Critically examine the role of", "Critically examine the factors behind",
        ],
        "answer_structure": (
            "**Introduction** (2-3 lines: background and scope from passage)\n"
            "**Positive Aspects / Significance** (2-3 points from passage)\n"
            "**Critical Analysis / Limitations** (2-3 points from passage)\n"
            "**Way Forward / Conclusion** (1-2 lines: balanced assessment)"
        ),
        "word_limits": [250],
        "marks_map": {250: 15},
    },
    "comment": {
        "key_phrases": [
            "Comment on", "Comment on the significance of",
            "Comment on the role of", "Comment on the statement that",
        ],
        "answer_structure": (
            "**Introduction** (2 lines: set context using passage)\n"
            "**Supporting Points** (2-3 bullets with passage evidence)\n"
            "**Critical Perspective** (1-2 lines: limitation or counter-view)\n"
            "**Conclusion** (1 balanced line)"
        ),
        "word_limits": [150],
        "marks_map": {150: 10},
    },
    "explain": {
        "key_phrases": [
            "Explain the concept of", "Explain the significance of",
            "Explain the causes and effects of", "Explain briefly",
            "Explain with suitable examples",
        ],
        "answer_structure": (
            "**Introduction** (1-2 lines: one-line definition from passage)\n"
            "**Key Aspects** (3-4 numbered points from passage)\n"
            "**Examples / Evidence** (from passage)\n"
            "**Conclusion** (significance or implication)"
        ),
        "word_limits": [150],
        "marks_map": {150: 10},
    },
    "examine": {
        "key_phrases": [
            "Examine the role of", "Examine the factors responsible for",
            "Examine the significance of", "Examine the challenges of",
        ],
        "answer_structure": (
            "**Introduction** (scope of examination)\n"
            "**Key Aspects Examined** (3-4 points from passage)\n"
            "**Evidence and Examples** (from passage)\n"
            "**Conclusion** (overall assessment)"
        ),
        "word_limits": [150, 250],
        "marks_map": {150: 10, 250: 15},
    },
    "evaluate": {
        "key_phrases": [
            "Evaluate the effectiveness of", "Evaluate the significance of",
            "Evaluate the role of", "Evaluate the impact of",
        ],
        "answer_structure": (
            "**Introduction** (why evaluation is relevant)\n"
            "**Strengths / Positive Impact** (2-3 points from passage)\n"
            "**Weaknesses / Gaps** (2-3 points from passage)\n"
            "**Overall Assessment** (1-2 lines: is it effective?)"
        ),
        "word_limits": [250],
        "marks_map": {250: 15},
    },
    "how_far": {
        "key_phrases": [
            "How far do you agree that", "How far is the statement correct that",
            "How far has", "How far is it true that",
        ],
        "answer_structure": (
            "**Position Statement** (agree / partially agree / disagree — 1 line)\n"
            "**Evidence Supporting** (2-3 points from passage)\n"
            "**Qualifications / Counter-evidence** (1-2 points from passage)\n"
            "**Conclusion** (balanced stance with reasoned judgment)"
        ),
        "word_limits": [150, 250],
        "marks_map": {150: 10, 250: 15},
    },
    "to_what_extent": {
        "key_phrases": [
            "To what extent has", "To what extent is it true that",
            "To what extent has the following been achieved",
        ],
        "answer_structure": (
            "**Extent of Agreement** (partial/full — 1 line)\n"
            "**Factors Supporting the Statement** (2-3 from passage)\n"
            "**Factors Limiting the Statement** (1-2 from passage)\n"
            "**Conclusion** (nuanced judgment)"
        ),
        "word_limits": [150, 250],
        "marks_map": {150: 10, 250: 15},
    },
    "do_you_agree": {
        "key_phrases": [
            "Do you agree that", "Do you think that",
            "Do you agree with the view that",
        ],
        "answer_structure": (
            "**My Position** (agree/disagree/partially — 1 line)\n"
            "**Arguments in Favour** (2-3 points from passage)\n"
            "**Counter-Arguments** (1-2 from passage)\n"
            "**Reasoned Conclusion** (own stance with justification)"
        ),
        "word_limits": [150, 250],
        "marks_map": {150: 10, 250: 15},
    },
    "factual": {
        "key_phrases": [
            "What are the main features of", "What are the causes of",
            "What is the significance of", "What are the challenges faced by",
            "What are the key aspects of",
        ],
        "answer_structure": (
            "**Introduction** (one-line answer to the 'what')\n"
            "**Key Facts / Features** (3-4 numbered points from passage)\n"
            "**Relevance / Significance** (1-2 lines from passage)\n"
            "**Conclusion** (brief summary)"
        ),
        "word_limits": [150],
        "marks_map": {150: 10},
    },
    "critically_analyse": {
        "key_phrases": [
            "Critically analyse the significance of",
            "Critically analyse the factors behind",
            "Critically analyse the role of",
        ],
        "answer_structure": (
            "**Introduction** (2-3 lines: scope of analysis)\n"
            "**Positive Dimensions** (2-3 from passage)\n"
            "**Critical Limitations** (2-3 from passage)\n"
            "**Conclusion** (balanced long-term assessment)"
        ),
        "word_limits": [250],
        "marks_map": {250: 15},
    },
    "elucidate": {
        "key_phrases": [
            "Elucidate the concept of", "Elucidate with examples",
            "Elucidate the significance of",
        ],
        "answer_structure": (
            "**Introduction** (definition from passage)\n"
            "**Elaboration** (3-4 points with examples from passage)\n"
            "**Significance** (why it matters)\n"
            "**Conclusion**"
        ),
        "word_limits": [150],
        "marks_map": {150: 10},
    },
    "bring_out": {
        "key_phrases": [
            "Bring out the significance of", "Bring out the key features of",
            "Bring out the relationship between",
        ],
        "answer_structure": (
            "**Introduction** (2 lines)\n"
            "**Key Features / Significance** (3-4 points from passage)\n"
            "**Conclusion**"
        ),
        "word_limits": [150],
        "marks_map": {150: 10},
    },
    "in_light_of": {
        "key_phrases": [
            "In the light of the above, discuss",
            "In the light of recent developments, analyse",
        ],
        "answer_structure": (
            "**Context** (what the light/development refers to)\n"
            "**Analysis** (3-4 points from passage)\n"
            "**Implications** (from passage)\n"
            "**Conclusion**"
        ),
        "word_limits": [150, 250],
        "marks_map": {150: 10, 250: 15},
    },
}

# Fallback for types not in library
_DEFAULT_QTYPE = {
    "key_phrases": ["Discuss", "Examine", "Analyse"],
    "answer_structure": (
        "**Introduction** (2 lines)\n"
        "**Main Points** (3-4 bullets from passage)\n"
        "**Conclusion** (1 line)"
    ),
    "word_limits": [150],
    "marks_map": {150: 10},
}

# Subject → GS paper mapping (for selecting paper's answer framework)
SUBJECT_TO_PAPER = {
    "History": "GS1", "Society_Culture": "GS1", "Geography": "GS1",
    "Art_Culture": "GS1", "GS1_Other": "GS1",
    "Polity": "GS2", "International_Relations": "GS2",
    "Governance_Social": "GS2", "GS2_Other": "GS2",
    "Economy": "GS3", "Environment": "GS3", "Science_Tech": "GS3",
    "Security": "GS3", "GS3_Other": "GS3",
    "Ethics": "GS4",
    "Essay": "Essay",
}


def load_patterns(patterns_file: Path) -> dict:
    """
    Load patterns JSON and enrich QTYPE_LIBRARY with LLM-analysed data.
    Also returns per-paper answer frameworks.
    """
    if not patterns_file.exists():
        return {"paper_frameworks": {}}

    with open(patterns_file, encoding="utf-8") as f:
        p = json.load(f)

    # Enrich QTYPE_LIBRARY from synthesis question_types
    for qt_data in p.get("question_types", []):
        qtype = qt_data.get("type", "").replace(" ", "_").lower()
        if qtype and qtype not in QTYPE_LIBRARY:
            QTYPE_LIBRARY[qtype] = {
                "key_phrases":      qt_data.get("key_phrases", []),
                "answer_structure": qt_data.get("answer_structure", ""),
                "word_limits":      qt_data.get("word_limits", [150]),
                "marks_map":        {150: 10, 250: 15},
            }

    # Build per-paper frameworks and top question types
    paper_frameworks = {}
    for paper, analysis in p.get("paper_analyses", {}).items():
        af = analysis.get("answer_framework", {})
        paper_frameworks[paper] = {
            "150_word": af.get("150_word_structure", ""),
            "250_word": af.get("250_word_structure", ""),
            "tips":     af.get("tips", []),
            "top_topics": analysis.get("top_20_must_prepare_topics", [])[:10],
        }

    return {"paper_frameworks": paper_frameworks}


def get_qtype_details(qtype: str) -> dict:
    """Return key phrases and answer structure for a question type."""
    return QTYPE_LIBRARY.get(qtype, _DEFAULT_QTYPE)


def build_pair_instructions(qtypes_to_use: list) -> str:
    """
    Build per-pair instructions block — each pair gets:
    - question type name
    - which key phrase to start with
    - exact answer structure to follow
    - word limit + marks
    """
    blocks = []
    for i, qtype in enumerate(qtypes_to_use, 1):
        details    = get_qtype_details(qtype)
        word_limit = random.choice(details["word_limits"])
        marks      = details["marks_map"].get(word_limit, 10)
        phrase     = random.choice(details["key_phrases"]) if details["key_phrases"] else "Discuss"

        block = (
            f"Q+A PAIR {i}:\n"
            f"  Question Type : {qtype}\n"
            f"  Start question with : \"{phrase} ...\"\n"
            f"  Word Limit    : {word_limit} words  ({marks} marks)\n"
            f"  Answer Structure:\n"
        )
        for line in details["answer_structure"].split("\n"):
            block += f"    {line}\n"
        blocks.append(block)

    return "\n".join(blocks)


MASTER_PROMPT = """You are an expert UPSC Mains answer writer coaching IAS aspirants for the Civil Services examination.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REFERENCE MATERIAL (extract facts ONLY — never mention this in questions or answers)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Subject : {subject}
Paper   : {paper}
Topic   : {heading}

{context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
UPSC MAINS ANSWER STYLE — STUDY THESE EXAMPLES CAREFULLY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Example 1 — "Discuss" type (10 marks, 150 words)
Q: Discuss the significance of the Revolt of 1857 in India's freedom struggle.

A: The Revolt of 1857, India's first large-scale armed uprising against British rule, marked a
decisive turning point in the country's history and laid the foundation for the independence movement.

**Political Significance**
1. End of East India Company rule — British Crown assumed direct control through the Government
   of India Act 1858, centralising administration.
2. Rise of nationalist consciousness — the revolt exposed the exploitative nature of colonial
   rule and united Indians across caste and religion for the first time.

**Military and Administrative Impact**
1. Reorganisation of the Indian Army — ratio of British to Indian soldiers increased sharply.
2. Divide and rule policy intensified — British deliberately exploited religious differences to
   prevent future unified revolts.

**Conclusion**: Though suppressed, the revolt planted the seeds of nationalism that eventually
culminated in independence in 1947. Historians like V.D. Savarkar called it the First War of Independence.

---

Example 2 — "Explain" type (10 marks, 150 words)
Q: Explain the structure and functions of the Election Commission of India.

A: The Election Commission of India (ECI), established under Article 324 of the Constitution,
is an autonomous constitutional body responsible for administering all elections to Parliament,
State Legislatures, and the offices of President and Vice-President.

**Structure**
1. Originally a single-member body, it became a multi-member commission in 1989 with the
   appointment of two Election Commissioners alongside the Chief Election Commissioner.
2. The Chief Election Commissioner can only be removed through a process similar to that of
   a Supreme Court judge, ensuring independence from executive pressure.

**Functions**
1. Delimitation of constituencies and preparation of electoral rolls.
2. Recognition of political parties and allotment of election symbols.
3. Enforcement of the Model Code of Conduct during elections.
4. Superintendence, direction, and control of the entire election process.

**Conclusion**: The ECI's independence and authority are crucial to India's democratic health.
Recent reforms like VVPAT machines have further strengthened electoral credibility.

---

Example 3 — "Critically examine" type (15 marks, 250 words)
Q: Critically examine the role of NITI Aayog in India's development planning.

A: NITI Aayog (National Institution for Transforming India), established in 2015 by replacing
the Planning Commission, represents a fundamental shift from centralised directive planning
to cooperative federalism and indicative planning.

**Positive Contributions**
1. Cooperative federalism — unlike the Planning Commission, states are active participants
   in policy formulation through the Governing Council.
2. SDG localisation — NITI Aayog has mapped all 17 Sustainable Development Goals to
   government schemes and tracks district-level progress through the SDG India Index.
3. Innovation ecosystem — Atal Innovation Mission has established 10,000+ Atal Tinkering
   Labs and supported startups through ARISE and SAMRIDH programmes.
4. Evidence-based policy — publishes indices (Health Index, Water Management Index) that
   create competitive federalism among states.

**Critical Limitations**
1. Lacks financial powers — cannot allocate funds to states unlike the Planning Commission,
   reducing its leverage over policy implementation.
2. Advisory role only — recommendations are not binding, leading to inconsistent adoption
   by states and ministries.
3. Centre-dominated — despite cooperative federalism rhetoric, the PM chairs the body,
   limiting genuine state autonomy.
4. Accountability deficit — no clear mechanism to evaluate whether its policy recommendations
   produce measurable outcomes.

**Way Forward**: Granting NITI Aayog limited financial devolution powers and establishing
a statutory basis for its recommendations would strengthen its effectiveness without reverting
to the centralised planning model.

**Conclusion**: NITI Aayog's consultative approach is a positive departure, but institutional
gaps limit its transformative impact. Structural reforms are needed to fulfil its stated mandate
of cooperative federalism.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT MAKES THESE ANSWERS UPSC-QUALITY — RULES FOR YOU TO FOLLOW:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ANSWER STYLE RULES:
1. First sentence — assertive factual statement, not "In this answer I will discuss..."
   ✗ WRONG: "Article 32 is very important for various reasons."
   ✓ CORRECT: "Article 32, described by Dr. B.R. Ambedkar as the heart and soul of the
     Constitution, guarantees citizens the right to directly approach the Supreme Court
     for enforcement of Fundamental Rights."

2. Each numbered point — one bold sub-heading + 1-2 sentences of specific content.
   ✗ WRONG: "It is very significant and plays a major role."
   ✓ CORRECT: "Checks executive power — courts can strike down any law or action that
     violates Fundamental Rights, making the judiciary a guardian of the Constitution."

3. Use specific facts — names, dates, article numbers, act names, statistics.
   ✗ WRONG: "Many people were affected by this policy."
   ✓ CORRECT: "The Green Revolution (1965-1970) increased wheat production from 12 MT
     to 20 MT, achieving food self-sufficiency within five years."

4. Conclusion must connect to current relevance or way forward — not just summarise.
   ✗ WRONG: "Thus, X is important and has many features as discussed above."
   ✓ CORRECT: "In the context of rising authoritarian trends globally, Article 32
     remains India's most critical safeguard for democratic freedoms."

5. Never say "as mentioned above", "as we discussed", "as explained", "in conclusion I
   have shown" — these are filler phrases that examiners penalise.

6. Never reference any passage, text, or study material in the answer.

QUESTION RULES:
1. Questions must be COMPLETELY STANDALONE — exactly as they appear in a real UPSC exam paper.
   ✗ WRONG: "Explain sneezing as described in the passage."
   ✗ WRONG: "According to the passage, discuss..."
   ✓ CORRECT: "Explain the physiological mechanism of sneezing."
   ✓ CORRECT: "Discuss the significance of X in the context of Y."
2. Questions must be about something ACTUALLY covered in the reference material.
3. Do NOT start multiple questions with the same word.
4. Word limit is a TARGET — write within ±15 words of target.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TASK: Generate {n} UPSC Mains Q+A pairs from the reference material above.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PAIR SPECIFICATIONS (follow exactly):
{pair_instructions}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT — respond ONLY with this JSON array, no other text:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[
  {{
    "q_num"     : 1,
    "q_type"    : "question_type_here",
    "word_limit": 150,
    "marks"     : 10,
    "question"  : "Complete standalone UPSC-style question.",
    "answer"    : "Assertive first sentence with key fact.\\n\\n**Section Header**\\n1. Bold sub-point — specific content with facts.\\n2. Bold sub-point — specific content.\\n\\n**Conclusion**: Current relevance or way forward."
  }}
]"""


def _clean_heading(raw: str) -> str:
    """Collapse multi-line / single-char OCR artifact headings into one clean line."""
    # Collapse whitespace and newlines
    h = " ".join(raw.split())
    # If result is a single char or pure whitespace → discard
    if len(h) <= 2:
        return ""
    return h


def build_generation_prompt(chunk: dict, qtypes_to_use: list,
                            paper_frameworks: dict, n: int = 5) -> str:
    subject = chunk.get("subject", "General Studies")
    raw_heading = chunk.get("section_heading", "") or ""
    heading = _clean_heading(raw_heading) or subject
    text    = chunk.get("text", "")[:MAX_CONTEXT_CHARS]
    paper   = SUBJECT_TO_PAPER.get(subject, "GS1")

    pair_instructions = build_pair_instructions(qtypes_to_use)

    return MASTER_PROMPT.format(
        subject=subject,
        paper=paper,
        heading=heading,
        context=text,
        n=n,
        pair_instructions=pair_instructions,
    )


def pick_qtypes_for_chunk(chunk, n=5):
    subject = chunk.get("subject", "default")
    pool = SUBJECT_QTYPES.get(subject, SUBJECT_QTYPES["default"])
    # Shuffle and pick n unique types
    shuffled = list(pool)
    random.shuffle(shuffled)
    selected = shuffled[:n]
    # Pad if pool smaller than n
    if len(selected) < n:
        extras = [t for t in SUBJECT_QTYPES["default"] if t not in selected]
        selected += extras[:n - len(selected)]
    return selected[:n]


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def _fix_json_literals(s: str) -> str:
    """
    Escape literal newlines / tabs that appear INSIDE JSON string values.
    JSON is valid only if strings contain \\n not a real newline character.
    The LLM often puts real newlines in multi-line "answer" fields.
    This fixes that without touching structural whitespace between keys.
    """
    result = []
    in_string = False
    escape_next = False
    for ch in s:
        if escape_next:
            result.append(ch)
            escape_next = False
        elif ch == '\\':
            result.append(ch)
            escape_next = True
        elif ch == '"':
            result.append(ch)
            in_string = not in_string
        elif in_string and ch == '\n':
            result.append('\\n')
        elif in_string and ch == '\r':
            result.append('\\r')
        elif in_string and ch == '\t':
            result.append('\\t')
        else:
            result.append(ch)
    return ''.join(result)


def _try_parse_json(text: str):
    """Try multiple strategies to parse a JSON array from LLM output."""
    # Strategy 1: direct parse of full response
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except Exception:
        pass

    # Strategy 2: extract [...] block (handles markdown code fences etc.)
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if not match:
        return None
    raw = match.group()

    # Strategy 3: direct parse of extracted block
    try:
        return json.loads(raw)
    except Exception:
        pass

    # Strategy 4: fix literal newlines/tabs inside string values, then parse
    try:
        return json.loads(_fix_json_literals(raw))
    except Exception:
        pass

    # Strategy 5: strip markdown fences and retry
    stripped = re.sub(r'```(?:json)?', '', raw).strip()
    try:
        return json.loads(stripped)
    except Exception:
        pass

    return None


def parse_qa_response(response_text, chunk, qtypes_used):
    """Parse LLM JSON response into list of QA dicts."""
    pairs = _try_parse_json(response_text)
    if not pairs:
        return []

    records = []
    for i, pair in enumerate(pairs):
        if not isinstance(pair, dict):
            continue
        question = str(pair.get("question", "")).strip()
        answer   = str(pair.get("answer", "")).strip()
        q_type   = str(pair.get("q_type", qtypes_used[i] if i < len(qtypes_used) else "discuss"))
        word_limit = int(pair.get("word_limit", 150))

        # Quality checks
        if len(question) < 20 or len(answer) < 100:
            continue
        if question == answer:
            continue

        # Reject questions that reference "the passage" / "the text" / "above"
        # These are broken — UPSC questions are always standalone
        _q_lower = question.lower()
        _passage_refs = [
            "the passage", "in the passage", "as described in", "according to the",
            "the text above", "the above passage", "based on the passage",
            "mentioned in the", "given passage", "from the passage",
            "the extract", "the material above",
        ]
        if any(ref in _q_lower for ref in _passage_refs):
            continue

        # Check answer isn't just copy-paste of the context
        context_words = set(chunk.get("text", "").lower().split()[:20])
        answer_start_words = set(answer.lower().split()[:10])
        if len(context_words & answer_start_words) > 8:
            # Answer literally starts with context text — skip
            continue

        records.append({
            "id":         f"{chunk['id']}_sft{i+1}",
            "text":       GEMMA_TEMPLATE.format(question=question, answer=answer),
            "subject":    chunk.get("subject", "General"),
            "q_type":     q_type,
            "word_limit": word_limit,
            "question":   question,
            "answer_preview": answer[:200],
            "source_chunk": chunk["id"],
            "source_file":  chunk.get("source_file", ""),
        })

    return records


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return set(json.load(f))
    return set()


def save_progress(processed_ids):
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(list(processed_ids), f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Generate UPSC-style SFT Q+A dataset")
    ap.add_argument("--pretrain",    default=str(PRETRAIN_FILE))
    ap.add_argument("--patterns",    default=str(PATTERNS_FILE))
    ap.add_argument("--output",      default=str(OUTPUT_FILE))
    ap.add_argument("--api",         choices=["gemini", "openai", "claude"], default=None)
    ap.add_argument("--limit",       type=int, default=None, help="Process only N chunks")
    ap.add_argument("--batch-size",  type=int, default=5,   help="QA pairs per chunk (default 5)")
    ap.add_argument("--delay",       type=float, default=1.0, help="Seconds between API calls")
    ap.add_argument("--resume",      action="store_true", help="Skip already-processed chunks")
    ap.add_argument("--dry-run",     action="store_true", help="Print prompts without calling API")
    ap.add_argument("--min-words",   type=int, default=80, help="Min word count for chunks")
    ap.add_argument("--subject",     default=None, help="Process only this subject")
    args = ap.parse_args()

    pretrain_file = Path(args.pretrain)
    output_file   = Path(args.output)

    if not pretrain_file.exists():
        print(f"ERROR: Pretrain file not found: {pretrain_file}")
        sys.exit(1)

    # Load patterns — enriches QTYPE_LIBRARY and extracts paper frameworks
    patterns_file = Path(args.patterns)
    patterns_data = load_patterns(patterns_file)
    paper_frameworks = patterns_data["paper_frameworks"]
    if patterns_file.exists():
        print(f"✅ Loaded patterns from {patterns_file}")
        print(f"   Paper frameworks available: {list(paper_frameworks.keys()) or 'none'}")
    else:
        print("⚠  No patterns file found — using built-in subject→qtype mapping")

    # Load env and detect API
    env = load_env()
    api = args.api or auto_detect_api(env)

    if not api and not args.dry_run:
        print("ERROR: No API key found in env/.env")
        print("  Add GEMINI_API_KEY=... for free tier (1500 req/day)")
        print("  Or run with --dry-run to preview prompts")
        sys.exit(1)

    if api:
        print(f"API: {api}")

    # Load chunks
    chunks = []
    with open(pretrain_file, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    print(f"Loaded {len(chunks):,} pretrain chunks")

    # Filter by subject
    if args.subject:
        chunks = [c for c in chunks if c.get("subject") == args.subject]
        print(f"Filtered to subject '{args.subject}': {len(chunks):,} chunks")

    # Filter by word count
    chunks = [c for c in chunks if c.get("word_count", 0) >= args.min_words]
    print(f"After word filter (>={args.min_words}): {len(chunks):,} chunks")

    # Resume: skip already processed
    processed_ids = set()
    if args.resume and PROGRESS_FILE.exists():
        processed_ids = load_progress()
        before = len(chunks)
        chunks = [c for c in chunks if c['id'] not in processed_ids]
        print(f"Resume: skipping {before - len(chunks):,} already done")

    # Apply limit
    if args.limit:
        chunks = chunks[:args.limit]
        print(f"Limited to {len(chunks):,} chunks")

    # Shuffle for variety
    random.shuffle(chunks)

    print(f"\n{'='*60}")
    print(f"Generating {args.batch_size} Q+A pairs per chunk")
    print(f"Estimated output: ~{len(chunks) * args.batch_size:,} records")
    print(f"Output: {output_file}")
    if args.dry_run:
        print("DRY RUN — no API calls will be made")
    print(f"{'='*60}\n")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    total_generated = 0
    total_failed    = 0
    q_type_counter  = Counter()
    subject_counter = Counter()

    # Open output in append mode (supports resume)
    mode = 'a' if args.resume else 'w'
    with open(output_file, mode, encoding='utf-8') as out_f:
        for idx, chunk in enumerate(chunks):
            chunk_id = chunk.get("id", f"chunk_{idx}")
            subject  = chunk.get("subject", "General")
            heading  = chunk.get("section_heading", "") or ""

            if idx % 50 == 0 and idx > 0:
                pct = idx / len(chunks) * 100
                print(f"\n[{idx:>5}/{len(chunks)}] ({pct:.1f}%) "
                      f"Generated: {total_generated:,} | Failed: {total_failed}")

            # Pick question types for this chunk
            qtypes = pick_qtypes_for_chunk(chunk, n=args.batch_size)

            # Build prompt
            prompt = build_generation_prompt(chunk, qtypes, paper_frameworks, n=args.batch_size)

            if args.dry_run:
                print(f"\n{'─'*50}")
                print(f"Chunk: {chunk_id} [{subject}] {heading[:50]}")
                print(f"Q-types: {', '.join(qtypes)}")
                print(f"Prompt ({len(prompt)} chars):\n{prompt[:400]}...")
                if idx >= 2:
                    print("\n[dry-run] Showing first 3 chunks only")
                    break
                continue

            # Call LLM
            try:
                response = call_llm(prompt, api, env)
            except Exception as e:
                print(f"  ❌ [{chunk_id}] LLM failed: {e}")
                total_failed += 1
                processed_ids.add(chunk_id)
                if idx % 100 == 0:
                    save_progress(processed_ids)
                time.sleep(args.delay)
                continue

            # Parse response
            records = parse_qa_response(response, chunk, qtypes)

            if not records:
                print(f"  ⚠  [{chunk_id}] No valid Q+A parsed (response: {response[:80]}...)")
                total_failed += 1
            else:
                for rec in records:
                    out_f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                    q_type_counter[rec['q_type']] += 1
                    subject_counter[rec['subject']] += 1
                total_generated += len(records)

                if idx % 10 == 0:
                    out_f.flush()

                if args.batch_size <= 3:
                    # verbose preview for small batches
                    print(f"  ✅ [{chunk_id}] {len(records)} pairs | {subject} | {heading[:40]}")

            processed_ids.add(chunk_id)

            # Save progress every 50 chunks
            if idx % 50 == 0:
                save_progress(processed_ids)

            time.sleep(args.delay)

    # Final progress save
    save_progress(processed_ids)

    # Summary
    print(f"\n{'='*60}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Chunks processed : {len(chunks):,}")
    print(f"Records generated: {total_generated:,}")
    print(f"Failed chunks    : {total_failed:,}")
    print(f"Output file      : {output_file}")

    print(f"\nBy question type:")
    for qt, cnt in q_type_counter.most_common(15):
        print(f"  {qt:<25} {cnt:>6,}")

    print(f"\nBy subject:")
    for subj, cnt in subject_counter.most_common():
        print(f"  {subj:<25} {cnt:>6,}")

    # Verify output
    if output_file.exists():
        line_count = sum(1 for _ in open(output_file))
        size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"\n✅ Output: {line_count:,} lines, {size_mb:.1f} MB")

    print(f"\nNext step: python3 scripts/merge_sft.py")


if __name__ == "__main__":
    main()
