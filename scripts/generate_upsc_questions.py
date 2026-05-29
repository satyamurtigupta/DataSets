#!/usr/bin/env python3
"""
Generate high-quality UPSC Mains-style Q&A pairs.
Supports Gemini, OpenAI (ChatGPT) and Anthropic (Claude).

API keys are read from env/.env — just paste your key there, no flags needed.

Run:
    # Auto-detect API from env/.env (recommended):
    python3 scripts/generate_upsc_questions.py

    # Force a specific API:
    python3 scripts/generate_upsc_questions.py --api gemini
    python3 scripts/generate_upsc_questions.py --api openai
    python3 scripts/generate_upsc_questions.py --api claude

    # Dry run (no API call, test prompts):
    python3 scripts/generate_upsc_questions.py --dry-run --limit 5

    # Resume interrupted run:
    python3 scripts/generate_upsc_questions.py --resume
"""

import json
import time
import random
import argparse
import re
import os
from pathlib import Path
from collections import Counter

random.seed(42)

# ---------------------------------------------------------------------------
# Load API keys from env/.env
# ---------------------------------------------------------------------------

ENV_FILE = Path(__file__).parent.parent / "env" / ".env"

def load_env():
    """Read env/.env and return dict of key=value pairs."""
    keys = {}
    if not ENV_FILE.exists():
        return keys
    with open(ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                v = v.strip()
                if v:
                    keys[k.strip()] = v
    return keys

ENV = load_env()


def get_api_key(api: str) -> str:
    """Get API key for chosen provider from env/.env."""
    key_names = {
        "gemini": "GEMINI_API_KEY",
        "openai": "OPENAI_API_KEY",
        "claude": "ANTHROPIC_API_KEY",
    }
    key = ENV.get(key_names.get(api, ""), "")
    if not key:
        raise ValueError(
            f"\n❌ No {key_names[api]} found in env/.env\n"
            f"   Open: {ENV_FILE}\n"
            f"   Paste your {api} API key after the = sign\n"
            f"   Get Gemini key (free): https://aistudio.google.com\n"
            f"   Get OpenAI key:        https://platform.openai.com/api-keys\n"
            f"   Get Claude key:        https://console.anthropic.com\n"
        )
    return key


def auto_detect_api() -> str:
    """Pick whichever API key is present in .env (Gemini preferred = free)."""
    if ENV.get("GEMINI_API_KEY"):
        return "gemini"
    if ENV.get("OPENAI_API_KEY"):
        return "openai"
    if ENV.get("ANTHROPIC_API_KEY"):
        return "claude"
    raise ValueError(
        f"\n❌ No API key found in {ENV_FILE}\n"
        f"   Open that file and paste at least one key.\n"
        f"   Gemini is free (1500 req/day): https://aistudio.google.com\n"
    )

INPUT_JSONL  = Path("dataset_output_final/combined/unified_pretrain_clean.jsonl")
OUTPUT_JSONL = Path("dataset_output_final/combined/unified_sft_v3.jsonl")
FAILED_LOG   = Path("dataset_output_final/combined/failed_chunks.jsonl")

GEMMA_TEMPLATE = "<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n{answer}<end_of_turn>"

# ---------------------------------------------------------------------------
# 15-Year UPSC Mains Question Patterns (GS1, GS2, GS3, GS4)
# ---------------------------------------------------------------------------

UPSC_PATTERNS = {
    "History": [
        ("critically_examine", 'Critically examine {topic} and its impact on the trajectory of Indian nationalism.'),
        ("critically_analyse", 'Critically analyse the role of {topic} in shaping modern India.'),
        ("comment",            'Comment on the significance of {topic} in the context of India\'s struggle for independence.'),
        ("discuss",            'Discuss the causes and consequences of {topic}. How did it alter the course of Indian history?'),
        ("how_far",            'How far is it correct to say that {topic} was a turning point in Indian history? Justify.'),
        ("examine",            'Examine the socio-economic and political dimensions of {topic}.'),
        ("throw_light",        'Throw light on the legacy of {topic} and its relevance to contemporary India.'),
        ("do_you_agree",       'Do you agree that {topic} fundamentally changed the relationship between Indians and the colonial state? Give reasons.'),
        ("assess",             'Assess the contribution of {topic} to India\'s cultural and political development.'),
        ("in_light_of",        'In the light of historical evidence, discuss the long-term significance of {topic}.'),
    ],
    "Polity": [
        ("critically_examine", 'Critically examine the constitutional provisions relating to {topic} and their practical implications.'),
        ("critically_analyse", 'Critically analyse the role of {topic} in strengthening Indian democracy.'),
        ("comment",            'Comment on the relevance of {topic} in the context of Indian federalism.'),
        ("discuss",            'Discuss the significance of {topic} in protecting the rights of citizens.'),
        ("examine",            'Examine the challenges faced in implementing {topic} in India.'),
        ("how_far",            'How far has {topic} succeeded in achieving its constitutional objectives? Discuss.'),
        ("do_you_agree",       'Do you think {topic} adequately balances individual rights with national interest? Justify.'),
        ("evaluate",           'Evaluate the effectiveness of {topic} as a mechanism for social justice.'),
        ("in_light_of",        'In the light of recent judicial pronouncements, discuss the evolving interpretation of {topic}.'),
        ("assess",             'Assess the role of {topic} in upholding the basic structure of the Indian Constitution.'),
    ],
    "Geography": [
        ("critically_examine", 'Critically examine the geographical factors that influence {topic} in India.'),
        ("analyse",            'Analyse the impact of {topic} on India\'s agricultural productivity and food security.'),
        ("comment",            'Comment on the significance of {topic} for India\'s sustainable development.'),
        ("discuss",            'Discuss how {topic} affects the distribution of natural resources in India.'),
        ("examine",            'Examine the relationship between {topic} and India\'s climate variability.'),
        ("assess",             'Assess the environmental and economic consequences of {topic} in the Indian subcontinent.'),
        ("in_light_of",        'In the light of climate change, discuss the changing dynamics of {topic} in India.'),
        ("how_far",            'How far has India been successful in managing the challenges posed by {topic}?'),
    ],
    "Economy": [
        ("critically_examine", 'Critically examine the role of {topic} in India\'s economic transformation since liberalisation.'),
        ("critically_analyse", 'Critically analyse the impact of {topic} on poverty alleviation and inclusive growth in India.'),
        ("comment",            'Comment on the effectiveness of {topic} as an instrument of India\'s fiscal policy.'),
        ("discuss",            'Discuss the challenges and opportunities associated with {topic} in the Indian context.'),
        ("examine",            'Examine the significance of {topic} for India\'s integration into the global economy.'),
        ("evaluate",           'Evaluate the contribution of {topic} to India\'s GDP and employment generation.'),
        ("in_light_of",        'In the light of recent economic developments, assess the relevance of {topic} for India.'),
        ("do_you_agree",       'Do you agree that {topic} is the key to achieving India\'s $5 trillion economy target? Justify.'),
        ("assess",             'Assess the impact of {topic} on India\'s balance of payments and currency stability.'),
        ("how_far",            'How far has {topic} helped in reducing regional economic disparities in India?'),
    ],
    "Science": [
        ("critically_examine", 'Critically examine the applications of {topic} and their implications for society.'),
        ("comment",            'Comment on the role of {topic} in advancing India\'s technological capabilities.'),
        ("discuss",            'Discuss the importance of {topic} in the context of India\'s science and technology policy.'),
        ("examine",            'Examine the ethical concerns associated with {topic} and suggest a regulatory framework.'),
        ("analyse",            'Analyse the potential of {topic} in addressing India\'s developmental challenges.'),
        ("how_far",            'How far has India progressed in the field of {topic}? Discuss the challenges ahead.'),
        ("assess",             'Assess the impact of {topic} on India\'s strategic and economic interests.'),
    ],
    "Biology": [
        ("critically_examine", 'Critically examine the role of {topic} in maintaining ecological balance.'),
        ("comment",            'Comment on the significance of {topic} for biodiversity conservation in India.'),
        ("discuss",            'Discuss how {topic} affects human health and environmental sustainability.'),
        ("examine",            'Examine the challenges posed by {topic} to India\'s natural ecosystems.'),
        ("analyse",            'Analyse the socio-economic implications of {topic} for rural communities in India.'),
        ("assess",             'Assess India\'s policy response to the challenges associated with {topic}.'),
    ],
    "Chemistry": [
        ("critically_examine", 'Critically examine the applications of {topic} in industrial processes and their environmental impact.'),
        ("comment",            'Comment on the role of {topic} in India\'s pharmaceutical and chemical industry.'),
        ("discuss",            'Discuss the relevance of {topic} for green chemistry and sustainable development.'),
        ("examine",            'Examine the health and environmental risks associated with {topic} and suggest mitigation measures.'),
    ],
    "Physics": [
        ("critically_examine", 'Critically examine the role of {topic} in modern technological development.'),
        ("comment",            'Comment on the significance of {topic} for India\'s space and defence programmes.'),
        ("discuss",            'Discuss the applications of {topic} in renewable energy and sustainable development.'),
        ("examine",            'Examine how {topic} has transformed communication and information technology in India.'),
    ],
    "Sociology": [
        ("critically_examine", 'Critically examine the impact of {topic} on the traditional social structure of India.'),
        ("critically_analyse", 'Critically analyse how {topic} has influenced gender relations in contemporary India.'),
        ("comment",            'Comment on the role of {topic} in shaping Indian social identity.'),
        ("discuss",            'Discuss the relationship between {topic} and social mobility in India.'),
        ("examine",            'Examine the challenges posed by {topic} to social cohesion and national integration.'),
        ("in_light_of",        'In the light of recent social changes, discuss the relevance of {topic} in India.'),
        ("do_you_agree",       'Do you agree that {topic} is a major obstacle to achieving social equality in India? Justify.'),
    ],
    "Art": [
        ("critically_examine", 'Critically examine the contribution of {topic} to India\'s intangible cultural heritage.'),
        ("comment",            'Comment on the significance of preserving {topic} in the age of globalisation.'),
        ("discuss",            'Discuss how {topic} reflects the diversity and unity of Indian culture.'),
        ("examine",            'Examine the role of {topic} in promoting India\'s soft power globally.'),
        ("assess",             'Assess the threats faced by {topic} and suggest measures for its conservation.'),
    ],
    "Economics": [
        ("critically_examine", 'Critically examine the role of {topic} in India\'s financial sector reforms.'),
        ("critically_analyse", 'Critically analyse the impact of {topic} on financial inclusion in rural India.'),
        ("comment",            'Comment on the effectiveness of {topic} as a tool for economic stabilisation.'),
        ("discuss",            'Discuss the challenges in implementing {topic} and suggest corrective measures.'),
        ("evaluate",           'Evaluate the role of {topic} in reducing income inequality in India.'),
        ("in_light_of",        'In the light of global economic trends, assess the significance of {topic} for India.'),
    ],
}

DEFAULT_PATTERNS = [
    ("critically_examine", 'Critically examine {topic} and its broader implications for India.'),
    ("comment",            'Comment on the significance of {topic} in the context of India\'s development.'),
    ("discuss",            'Discuss the key dimensions of {topic} and its relevance for UPSC examination.'),
    ("examine",            'Examine the challenges and opportunities associated with {topic}.'),
    ("assess",             'Assess the impact of {topic} on India\'s socio-economic and political landscape.'),
    ("in_light_of",        'In the light of recent developments, critically analyse the role of {topic}.'),
]

# Word limits by question type (UPSC standard)
WORD_LIMITS = {
    "critically_examine": 150,
    "critically_analyse": 150,
    "comment":            100,
    "discuss":            150,
    "examine":            150,
    "how_far":            150,
    "do_you_agree":       150,
    "evaluate":           150,
    "in_light_of":        150,
    "assess":             150,
    "throw_light":        100,
    "analyse":            150,
}

# ---------------------------------------------------------------------------
# API Clients
# ---------------------------------------------------------------------------

def call_gemini(prompt: str, api_key: str) -> str:
    """Call Gemini 1.5 Flash API."""
    import urllib.request
    import json as json_lib

    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"gemini-1.5-flash:generateContent?key={api_key}")

    payload = json_lib.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 600,
        }
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json_lib.loads(resp.read().decode("utf-8"))

    return result["candidates"][0]["content"]["parts"][0]["text"].strip()


def call_openai(prompt: str, api_key: str) -> str:
    """Call OpenAI GPT-4o-mini API."""
    import urllib.request
    import json as json_lib

    payload = json_lib.dumps({
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 600,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST"
    )

    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json_lib.loads(resp.read().decode("utf-8"))

    return result["choices"][0]["message"]["content"].strip()


def call_claude(prompt: str, api_key: str) -> str:
    """Call Anthropic Claude Haiku API."""
    import urllib.request
    import json as json_lib

    payload = json_lib.dumps({
        "model": "claude-haiku-20240307",
        "max_tokens": 600,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type":      "application/json",
            "x-api-key":         api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST"
    )

    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json_lib.loads(resp.read().decode("utf-8"))

    return result["content"][0]["text"].strip()


# ---------------------------------------------------------------------------
# Prompt Builder
# ---------------------------------------------------------------------------

def build_prompt(chunk: dict, q_type: str, question: str, word_limit: int) -> str:
    """Build the API prompt for generating a UPSC-quality answer."""

    subject  = chunk.get("subject", "General Studies")
    text     = chunk.get("text", "")[:1200]   # limit context to ~300 tokens

    return f"""You are an expert UPSC Mains answer writer. Write a model answer for the following question.

Subject: {subject}
Question: {question}
Word limit: {word_limit} words

Reference passage (use relevant facts from this):
\"\"\"{text}\"\"\"

Write the answer in proper UPSC Mains format:
1. Start with a brief introduction (2-3 lines giving context)
2. Body: Use numbered points OR subheadings for clarity
   - Cover multiple dimensions (political, economic, social, historical as relevant)
   - Include specific facts, examples, dates where available
   - For "critically" questions: present BOTH positive aspects AND limitations
   - For "comment" questions: give a balanced multi-dimensional view
3. End with a forward-looking conclusion (2-3 lines)

Rules:
- Stay within {word_limit} words
- Do NOT copy the passage directly — synthesise and present analytically
- Use formal but clear language
- No bullet points in introduction or conclusion
- Underline key terms (use **bold** for key terms)

Write the answer now:"""


# ---------------------------------------------------------------------------
# Main Generator
# ---------------------------------------------------------------------------

def make_record(chunk: dict, api_func, dry_run: bool = False) :
    subject  = chunk.get("subject", "General")
    text     = chunk.get("text", "").strip()
    heading  = (chunk.get("section_heading") or "").strip()
    word_cnt = chunk.get("word_count", 0)

    # Quality filter
    if word_cnt < 80:
        return None
    if len(text) < 250:
        return None

    # Clean heading
    if heading:
        if re.match(r'^\d', heading):
            heading = ""
        elif sum(1 for c in heading if c.isupper()) / max(len(heading), 1) > 0.7:
            heading = ""
        elif re.search(r'[^A-Za-z0-9\s,.\'\-]{3,}', heading):
            heading = ""

    topic = heading if heading and len(heading) > 3 else subject

    # Pick question pattern
    patterns = UPSC_PATTERNS.get(subject, DEFAULT_PATTERNS)
    q_type, q_template = random.choice(patterns)
    question   = q_template.format(topic=topic)
    word_limit = WORD_LIMITS.get(q_type, 150)

    if dry_run:
        print(f"\n{'─'*60}")
        print(f"Subject  : {subject}")
        print(f"Q-Type   : {q_type}")
        print(f"Question : {question}")
        print(f"[DRY RUN — no API call]")
        return {
            "id":       chunk["id"] + "_v3",
            "text":     GEMMA_TEMPLATE.format(question=question, answer="[DRY RUN]"),
            "subject":  subject,
            "question": question,
            "q_type":   q_type,
        }

    # Call API
    prompt = build_prompt(chunk, q_type, question, word_limit)
    answer = api_func(prompt)

    if not answer or len(answer) < 50:
        return None

    return {
        "id":       chunk["id"] + "_v3",
        "text":     GEMMA_TEMPLATE.format(question=question, answer=answer),
        "subject":  subject,
        "question": question,
        "q_type":   q_type,
        "word_limit": word_limit,
    }


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate UPSC-quality SFT dataset via API")
    parser.add_argument("--api",     choices=["gemini", "openai", "claude"],
                        default=None, help="API to use (auto-detected from env/.env if not set)")
    parser.add_argument("--limit",   type=int, default=None, help="Max records to process")
    parser.add_argument("--dry-run", action="store_true", help="Test prompts without API calls")
    parser.add_argument("--delay",   type=float, default=0.5, help="Seconds between API calls")
    parser.add_argument("--resume",  action="store_true", help="Skip already processed IDs")
    args = parser.parse_args()

    # API function — auto-detect from env/.env if not specified
    if args.dry_run:
        api_func = None
        print("Mode: DRY RUN (no API calls)")
    else:
        chosen_api = args.api or auto_detect_api()
        api_key    = get_api_key(chosen_api)

        if chosen_api == "gemini":
            api_func = lambda p: call_gemini(p, api_key)
            days = 12512 / 1500
            print(f"✅ Using: Gemini 1.5 Flash (FREE — 1500 req/day)")
            print(f"   Est. time on free tier: ~{days:.0f} days")
            print(f"   Key loaded from: env/.env")
        elif chosen_api == "openai":
            api_func = lambda p: call_openai(p, api_key)
            print(f"✅ Using: OpenAI GPT-4o-mini (~$6 for 12K records)")
            print(f"   Key loaded from: env/.env")
        elif chosen_api == "claude":
            api_func = lambda p: call_claude(p, api_key)
            print(f"✅ Using: Anthropic Claude Haiku (~$9 for 12K records)")
            print(f"   Key loaded from: env/.env")

    # Load pretrain chunks
    print(f"\nReading: {INPUT_JSONL}")
    with open(INPUT_JSONL, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded : {len(chunks):,} chunks")

    # Resume: skip already done IDs
    done_ids = set()
    if args.resume and OUTPUT_JSONL.exists():
        with open(OUTPUT_JSONL, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    done_ids.add(rec["id"])
        print(f"Resuming: {len(done_ids):,} already done")

    # Filter
    chunks = [c for c in chunks if c["id"] + "_v3" not in done_ids]
    if args.limit:
        chunks = chunks[:args.limit]

    random.shuffle(chunks)
    print(f"To process: {len(chunks):,} chunks")

    if not args.dry_run:
        est_cost = len(chunks) * 900 / 1_000_000
        if args.api == "gemini":
            days = len(chunks) / 1500
            print(f"\nGemini free tier: 1500 req/day → ~{days:.1f} days to complete")
            print(f"Or get paid tier for faster completion")
        else:
            usd = est_cost * 0.75  # gpt-4o-mini blended rate
            inr = usd * 84
            print(f"\nEst. cost: ~${usd:.2f} (~₹{inr:.0f})")

    # Process
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    failed = []
    success = 0
    skipped = 0

    mode = "a" if args.resume else "w"

    with open(OUTPUT_JSONL, mode, encoding="utf-8") as out_f:
        for i, chunk in enumerate(chunks):
            try:
                rec = make_record(chunk, api_func, dry_run=args.dry_run)

                if rec:
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out_f.flush()
                    success += 1

                    if success % 50 == 0:
                        print(f"  [{i+1}/{len(chunks)}] ✅ {success} done, {skipped} skipped")
                else:
                    skipped += 1

            except Exception as e:
                print(f"  [{i+1}] ❌ Error: {e}")
                failed.append({"id": chunk.get("id"), "error": str(e)})
                time.sleep(2)   # back off on error
                continue

            if not args.dry_run:
                time.sleep(args.delay)

    # Stats
    print(f"\n{'='*60}")
    print(f"✅ Generated : {success:,} records")
    print(f"⏭️  Skipped   : {skipped} (too short)")
    print(f"❌ Failed    : {len(failed)}")
    print(f"📁 Saved to  : {OUTPUT_JSONL}")

    if failed:
        with open(FAILED_LOG, "w") as f:
            for r in failed:
                f.write(json.dumps(r) + "\n")
        print(f"❌ Failed IDs: {FAILED_LOG}")

    # Show samples
    if success > 0 and not args.dry_run:
        print(f"\n{'='*60}")
        print("SAMPLE RECORDS")
        print(f"{'='*60}")
        with open(OUTPUT_JSONL, encoding="utf-8") as f:
            samples = [json.loads(f.readline()) for _ in range(min(2, success))]
        for rec in samples:
            print(f"\nSubject  : {rec['subject']}")
            print(f"Q-Type   : {rec['q_type']}")
            print(f"Question : {rec['question']}")
            answer = rec["text"].split("<start_of_turn>model\n")[-1][:400]
            print(f"Answer   : {answer}")
            print("─" * 60)


if __name__ == "__main__":
    main()
