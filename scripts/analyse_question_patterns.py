#!/usr/bin/env python3
"""
Script 2 of 3: Deep UPSC Mains Question Pattern Analysis using LLM.

Sends questions paper-by-paper (GS1, GS2, GS3, GS4, Essay) to LLM.
For each paper the LLM identifies:
  - Recurring topics / subtopics (what keeps repeating year after year)
  - Trending topics (new themes in 2022-2025)
  - Year-wise topic heatmap
  - Most important themes to prepare
  - Answer structure guidance per paper

Output: dataset_output_final/combined/question_patterns.json

Run:
    python3 scripts/analyse_question_patterns.py --api openai
    python3 scripts/analyse_question_patterns.py --api gemini
    python3 scripts/analyse_question_patterns.py --no-llm   (stats only, no API)
"""

import json
import re
import sys
import time
import argparse
import urllib.request
import urllib.error
from pathlib import Path
from collections import Counter, defaultdict

INPUT_FILE  = Path("dataset_output_final/combined/extracted_questions.jsonl")
OUTPUT_FILE = Path("dataset_output_final/combined/question_patterns.json")
ENV_FILE    = Path("env/.env")

PAPERS = ["GS1", "GS2", "GS3", "GS4", "Essay"]

PAPER_CONTEXT = {
    "GS1":   "Indian Heritage & Culture, History, Geography, Indian Society",
    "GS2":   "Governance, Constitution, Polity, Social Justice, International Relations",
    "GS3":   "Economy, Agriculture, Science & Technology, Environment, Security & Disaster",
    "GS4":   "Ethics, Integrity, Aptitude — including case studies",
    "Essay": "Essay paper — tests breadth of knowledge and quality of writing on any topic",
}

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def load_env() -> dict:
    env = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def call_openai(prompt: str, api_key: str) -> str:
    payload = json.dumps({
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are an expert UPSC Mains examiner and coaching faculty. Return only valid JSON."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 4000,
        "response_format": {"type": "json_object"},
    }).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        resp = json.loads(r.read())
    return resp["choices"][0]["message"]["content"]


def call_gemini(prompt: str, api_key: str) -> str:
    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2, "maxOutputTokens": 4000,
            "responseMimeType": "application/json",
        },
    }).encode()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    req = urllib.request.Request(url, data=payload,
                                  headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        resp = json.loads(r.read())
    return resp["candidates"][0]["content"]["parts"][0]["text"]


def call_llm(prompt: str, api: str, env: dict, retry: int = 3) -> str:
    for attempt in range(1, retry + 1):
        try:
            if api == "openai":
                return call_openai(prompt, env["OPENAI_API_KEY"])
            elif api == "gemini":
                return call_gemini(prompt, env["GEMINI_API_KEY"])
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            if e.code == 429:
                wait = 20 * attempt
                print(f"    ⏳ Rate limit — waiting {wait}s (attempt {attempt}/{retry})")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("All retries exhausted")


def parse_json(text: str) -> dict:
    """Extract JSON from LLM response."""
    text = re.sub(r'^```(?:json)?\s*', '', text.strip())
    text = re.sub(r'\s*```$', '', text.strip())
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        return json.loads(m.group())
    return json.loads(text)


# ---------------------------------------------------------------------------
# Build per-paper question list for prompting
# ---------------------------------------------------------------------------

def group_by_paper(questions: list) -> dict:
    grouped = defaultdict(list)
    for q in questions:
        grouped[q.get("paper", "Unknown")].append(q)
    return grouped


def format_questions_for_prompt(qs: list) -> str:
    """Format questions as numbered list for LLM."""
    lines = []
    for q in qs:
        yr   = q.get("year", "?")
        wl   = q.get("word_limit", "?")
        text = q.get("text", "").strip()
        lines.append(f"[{yr}] ({wl}w) {text}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Paper-level deep analysis prompt
# ---------------------------------------------------------------------------

PAPER_ANALYSIS_PROMPT = """You are an expert UPSC Mains examiner with 15+ years experience.

Below are ALL extracted UPSC Mains questions from the **{paper}** paper ({context}).
Years covered: {years}
Total questions: {total}

QUESTIONS:
{questions}

---

Analyse these questions DEEPLY and return a JSON object with these exact keys:

{{
  "paper": "{paper}",
  "total_questions": {total},
  "recurring_topics": [
    {{
      "topic": "topic name",
      "frequency": "high/medium/low",
      "years_appeared": [2016, 2018, 2022],
      "sample_questions": ["question text 1", "question text 2"],
      "importance": "why this topic keeps appearing — 1 sentence"
    }}
  ],
  "trending_topics": [
    {{
      "topic": "emerging topic name",
      "first_appeared": 2020,
      "why_trending": "1 sentence explanation",
      "sample_question": "example question"
    }}
  ],
  "year_wise_themes": {{
    "2014": ["theme1", "theme2"],
    "2015": ["theme1", "theme2"],
    "2016": ["theme1", "theme2"],
    "2017": ["theme1", "theme2"],
    "2018": ["theme1", "theme2"],
    "2019": ["theme1", "theme2"],
    "2020": ["theme1", "theme2"],
    "2021": ["theme1", "theme2"],
    "2022": ["theme1", "theme2"],
    "2023": ["theme1", "theme2"],
    "2024": ["theme1", "theme2"],
    "2025": ["theme1", "theme2"]
  }},
  "top_20_must_prepare_topics": [
    "topic 1", "topic 2", "topic 3"
  ],
  "question_type_breakdown": {{
    "type_name": count
  }},
  "answer_framework": {{
    "150_word_structure": "How to structure a 10-mark 150-word answer for this paper",
    "250_word_structure": "How to structure a 15-mark 250-word answer for this paper",
    "common_mistakes": ["mistake 1", "mistake 2", "mistake 3"],
    "tips": ["tip 1", "tip 2", "tip 3"]
  }},
  "subject_clusters": {{
    "cluster_name": ["topic1", "topic2", "topic3"]
  }}
}}

Rules:
- recurring_topics: identify 10-15 topics that appeared in multiple years
- trending_topics: identify 5-8 topics that are new/growing since 2020
- top_20_must_prepare_topics: the most important topics a student MUST prepare
- Be specific with topic names (e.g. "Federalism & Centre-State Relations" not just "Polity")
- year_wise_themes: only include years that actually appear in the questions list above
- Return ONLY valid JSON, nothing else
"""


# ---------------------------------------------------------------------------
# Overall synthesis prompt — runs after all papers are analysed
# ---------------------------------------------------------------------------

SYNTHESIS_PROMPT = """You are an expert UPSC Mains examiner.

I have analysed all 5 UPSC Mains papers (GS1, GS2, GS3, GS4, Essay).
Below is a summary of recurring topics per paper:

{paper_summaries}

Now produce a SYNTHESIS JSON with these keys:

{{
  "cross_paper_themes": [
    {{
      "theme": "theme that cuts across multiple papers",
      "papers": ["GS1", "GS3"],
      "how_to_link": "how an answer in GS1 connects to GS3 on this theme"
    }}
  ],
  "question_types": [
    {{
      "type": "discuss/critically_examine/factual/etc",
      "description": "what UPSC tests with this type",
      "answer_structure": "intro → body points → conclusion format",
      "key_phrases": ["opening phrase 1", "opening phrase 2"],
      "word_limits": [150, 250],
      "typical_papers": ["GS1", "GS2"]
    }}
  ],
  "generation_guidelines": {{
    "dos": ["do 1", "do 2", "do 3", "do 4", "do 5"],
    "donts": ["dont 1", "dont 2", "dont 3", "dont 4", "dont 5"],
    "word_limit_rules": "when to use 150 vs 250 words",
    "format_template": "reusable template for model UPSC answers"
  }},
  "high_priority_topics_all_papers": [
    "most important topic 1 to train the model on",
    "topic 2",
    "topic 3"
  ]
}}

Return ONLY valid JSON.
"""


# ---------------------------------------------------------------------------
# Stats-only analysis (no LLM)
# ---------------------------------------------------------------------------

def build_stats(questions: list) -> dict:
    by_type    = Counter(q.get("q_type", "?") for q in questions)
    by_subject = Counter(q.get("subject", "?") for q in questions)
    by_paper   = Counter(q.get("paper", "?")   for q in questions)
    by_year    = Counter(q.get("year", "?")     for q in questions)
    by_wl      = Counter(q.get("word_limit")    for q in questions)
    return {
        "total_questions": len(questions),
        "by_q_type":    dict(by_type.most_common()),
        "by_subject":   dict(by_subject.most_common()),
        "by_paper":     dict(by_paper.most_common()),
        "by_year":      dict(sorted(by_year.items())),
        "word_limits":  {str(k): v for k, v in by_wl.most_common()},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Deep UPSC Mains pattern analysis")
    ap.add_argument("--input",   default=str(INPUT_FILE))
    ap.add_argument("--output",  default=str(OUTPUT_FILE))
    ap.add_argument("--api",     choices=["openai", "gemini"], default=None)
    ap.add_argument("--no-llm",  action="store_true", help="Stats only — no LLM call")
    ap.add_argument("--delay",   type=float, default=5.0, help="Delay between paper API calls")
    args = ap.parse_args()

    # ── Load questions ────────────────────────────────────────────────────────
    input_file = Path(args.input)
    if not input_file.exists():
        print(f"ERROR: {input_file} not found. Run extract_mains_questions.py first.")
        sys.exit(1)

    questions = []
    with open(input_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    questions.append(json.loads(line))
                except Exception:
                    pass

    print(f"Loaded {len(questions)} questions from {input_file}")
    stats = build_stats(questions)

    print(f"\nBy paper:")
    for paper, cnt in sorted(stats["by_paper"].items()):
        print(f"  {paper:<10} {cnt:>4} questions")
    print(f"\nBy year:")
    for yr, cnt in sorted(stats["by_year"].items()):
        print(f"  {yr}  {cnt:>4} questions")

    if args.no_llm:
        print("\n--no-llm: saving stats only")
        result = {"stats": stats, "paper_analyses": {}, "synthesis": {}}
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved → {args.output}")
        return

    # ── Setup API ─────────────────────────────────────────────────────────────
    env = load_env()
    api = args.api
    if not api:
        if env.get("OPENAI_API_KEY"):  api = "openai"
        elif env.get("GEMINI_API_KEY"): api = "gemini"
        else:
            print("❌ No API key in env/.env. Use --no-llm or add a key.")
            sys.exit(1)
    print(f"\nAPI: {api.upper()}")

    # ── Group by paper ────────────────────────────────────────────────────────
    by_paper = group_by_paper(questions)
    paper_analyses = {}

    for paper in PAPERS:
        qs = by_paper.get(paper, [])
        if not qs:
            print(f"\n⏭  {paper}: no questions found — skipping")
            continue

        years = sorted(set(q.get("year") for q in qs if q.get("year")))
        print(f"\n{'─'*60}")
        print(f"📄 Analysing {paper} — {len(qs)} questions across {len(years)} years")
        print(f"   Years: {years}")

        prompt = PAPER_ANALYSIS_PROMPT.format(
            paper=paper,
            context=PAPER_CONTEXT.get(paper, ""),
            years=years,
            total=len(qs),
            questions=format_questions_for_prompt(qs),
        )

        try:
            raw = call_llm(prompt, api, env)
            analysis = parse_json(raw)
            paper_analyses[paper] = analysis
            topics = analysis.get("recurring_topics", [])
            trending = analysis.get("trending_topics", [])
            top20 = analysis.get("top_20_must_prepare_topics", [])
            print(f"   ✅ {len(topics)} recurring topics | {len(trending)} trending | {len(top20)} must-prepare")
            if top20:
                print(f"   Top 5: {', '.join(top20[:5])}")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            paper_analyses[paper] = {"error": str(e), "questions_count": len(qs)}

        if paper != PAPERS[-1]:
            print(f"   ⏳ Waiting {args.delay}s before next paper...")
            time.sleep(args.delay)

    # ── Synthesis across all papers ───────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("🔗 Running cross-paper synthesis...")

    paper_summaries = ""
    for paper, analysis in paper_analyses.items():
        topics = [t.get("topic", "") for t in analysis.get("recurring_topics", [])[:10]]
        trending = [t.get("topic", "") for t in analysis.get("trending_topics", [])[:5]]
        top20 = analysis.get("top_20_must_prepare_topics", [])[:10]
        paper_summaries += f"\n{paper}:\n"
        paper_summaries += f"  Recurring: {', '.join(topics)}\n"
        paper_summaries += f"  Trending: {', '.join(trending)}\n"
        paper_summaries += f"  Must-prepare: {', '.join(top20)}\n"

    synthesis = {}
    try:
        raw = call_llm(
            SYNTHESIS_PROMPT.format(paper_summaries=paper_summaries),
            api, env
        )
        synthesis = parse_json(raw)
        print(f"   ✅ Cross-paper synthesis complete")
        q_types = synthesis.get("question_types", [])
        cross   = synthesis.get("cross_paper_themes", [])
        print(f"   {len(q_types)} question types | {len(cross)} cross-paper themes")
    except Exception as e:
        print(f"   ❌ Synthesis failed: {e}")

    # ── Save ─────────────────────────────────────────────────────────────────
    result = {
        "stats":          stats,
        "paper_analyses": paper_analyses,
        "synthesis":      synthesis,
        # Flatten for backward compatibility with generate_sft_dataset.py
        "question_types":          synthesis.get("question_types", []),
        "generation_guidelines":   synthesis.get("generation_guidelines", {}),
        "cross_paper_themes":      synthesis.get("cross_paper_themes", []),
        "high_priority_topics":    synthesis.get("high_priority_topics_all_papers", []),
    }

    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"✅ Patterns saved → {output_file}")
    print(f"   Papers analysed : {len(paper_analyses)}")
    for p, a in paper_analyses.items():
        n = len(a.get("recurring_topics", []))
        m = len(a.get("top_20_must_prepare_topics", []))
        print(f"   {p}: {n} recurring topics, {m} must-prepare topics")


if __name__ == "__main__":
    main()
