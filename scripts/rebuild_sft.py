#!/usr/bin/env python3
"""
Rebuild SFT dataset with proper UPSC-style Q&A pairs.
Fixes the copy-paste answer problem in unified_sft.jsonl.

Run:
    python3 scripts/rebuild_sft.py
"""

import json
import random
import re
from pathlib import Path
from collections import Counter

random.seed(42)

INPUT_JSONL  = Path("dataset_output_final/combined/unified_pretrain_clean.jsonl")
OUTPUT_JSONL = Path("dataset_output_final/combined/unified_sft_v2.jsonl")

# ---------------------------------------------------------------------------
# UPSC question templates per subject
# ---------------------------------------------------------------------------

TEMPLATES = {
    "History": [
        ("explain",  "Explain the causes and significance of {topic}."),
        ("discuss",  "Discuss the impact of {topic} on Indian history."),
        ("comment",  "Comment on the role of {topic} in shaping modern India."),
        ("analyse",  "Analyse the long-term consequences of {topic}."),
        ("opinion",  "Give your opinion on whether {topic} was a turning point in Indian history."),
        ("how",      "How did {topic} influence the course of Indian independence?"),
        ("compare",  "Compare and contrast the causes and effects of {topic}."),
    ],
    "Polity": [
        ("explain",  "Explain the significance of {topic} in the Indian Constitution."),
        ("discuss",  "Discuss the importance of {topic} for Indian democracy."),
        ("analyse",  "Analyse the role of {topic} in protecting citizens' rights."),
        ("comment",  "Comment on the relevance of {topic} in contemporary India."),
        ("how",      "How does {topic} strengthen the federal structure of India?"),
        ("evaluate", "Evaluate the effectiveness of {topic} in ensuring social justice."),
    ],
    "Geography": [
        ("explain",  "Explain the importance of {topic} for India's physical environment."),
        ("discuss",  "Discuss how {topic} affects agricultural patterns in India."),
        ("analyse",  "Analyse the impact of {topic} on India's economy."),
        ("comment",  "Comment on the significance of {topic} for India's climate."),
        ("how",      "How does {topic} influence the distribution of natural resources?"),
    ],
    "Economy": [
        ("explain",  "Explain the role of {topic} in India's economic development."),
        ("discuss",  "Discuss the impact of {topic} on poverty and inequality in India."),
        ("analyse",  "Analyse the significance of {topic} for India's fiscal policy."),
        ("comment",  "Comment on the effectiveness of {topic} in promoting growth."),
        ("evaluate", "Evaluate the contribution of {topic} to India's GDP."),
        ("how",      "How has {topic} shaped India's economic policy?"),
    ],
    "Science": [
        ("explain",  "Explain the concept of {topic} and its applications."),
        ("discuss",  "Discuss the importance of {topic} in everyday life."),
        ("analyse",  "Analyse the impact of {topic} on modern science."),
        ("how",      "How does {topic} contribute to technological development?"),
    ],
    "Biology": [
        ("explain",  "Explain the significance of {topic} in living organisms."),
        ("discuss",  "Discuss the role of {topic} in maintaining ecological balance."),
        ("analyse",  "Analyse the impact of {topic} on biodiversity conservation."),
        ("how",      "How does {topic} affect human health and environment?"),
    ],
    "Chemistry": [
        ("explain",  "Explain the principles behind {topic} and its applications."),
        ("discuss",  "Discuss the importance of {topic} in industrial processes."),
        ("how",      "How does {topic} contribute to environmental chemistry?"),
    ],
    "Physics": [
        ("explain",  "Explain the concept of {topic} and its real-world applications."),
        ("discuss",  "Discuss the significance of {topic} in modern technology."),
        ("how",      "How does {topic} apply to everyday phenomena?"),
    ],
    "Sociology": [
        ("explain",  "Explain the concept of {topic} in Indian social context."),
        ("discuss",  "Discuss the impact of {topic} on Indian society."),
        ("analyse",  "Analyse how {topic} has changed Indian social structure."),
        ("comment",  "Comment on the relevance of {topic} in contemporary India."),
    ],
    "Art": [
        ("explain",  "Explain the significance of {topic} in Indian cultural heritage."),
        ("discuss",  "Discuss the evolution of {topic} through Indian history."),
        ("comment",  "Comment on the importance of preserving {topic} for future generations."),
        ("how",      "How does {topic} reflect India's diverse cultural traditions?"),
    ],
    "Economics": [
        ("explain",  "Explain the role of {topic} in India's economic development."),
        ("discuss",  "Discuss the impact of {topic} on India's growth story."),
        ("analyse",  "Analyse the significance of {topic} for financial inclusion."),
        ("evaluate", "Evaluate the effectiveness of {topic} in reducing poverty."),
    ],
}

DEFAULT_TEMPLATES = [
    ("explain",  "Explain the concept of {topic} in detail."),
    ("discuss",  "Discuss the importance of {topic} for UPSC examination."),
    ("analyse",  "Analyse the significance of {topic}."),
    ("comment",  "Comment on {topic} and its broader implications."),
    ("how",      "How is {topic} relevant to India's development?"),
]

# ---------------------------------------------------------------------------
# Answer builders — one per question type
# ---------------------------------------------------------------------------

def build_answer(q_type: str, topic: str, text: str, subject: str) -> str:
    """Build a structured answer based on question type."""

    # Extract key sentences (not just copy first 3)
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 30]
    key_sentences = sentences[:6] if len(sentences) >= 6 else sentences

    if q_type == "explain":
        return f"""{topic} is an important concept in {subject} for UPSC preparation.

Key aspects to understand:

{chr(10).join(f"• {s}" for s in key_sentences[:4])}

This topic is significant because it forms the foundation for understanding broader themes in {subject}. UPSC aspirants should focus on the factual details, dates, and cause-effect relationships associated with this topic."""

    elif q_type == "discuss":
        return f"""The topic of {topic} has far-reaching implications in the context of {subject}.

Background and context:
{key_sentences[0] if key_sentences else ""}

Key dimensions to discuss:

{chr(10).join(f"{i+1}. {s}" for i, s in enumerate(key_sentences[1:4]))}

Critical analysis:
{key_sentences[4] if len(key_sentences) > 4 else "This requires multi-dimensional analysis considering political, economic and social factors."}

Conclusion:
Understanding {topic} helps in appreciating the complexity of {subject} and its relevance to contemporary India."""

    elif q_type == "comment":
        return f"""Comment on {topic}:

{topic} represents a significant aspect of {subject} that deserves careful examination.

{chr(10).join(f"• {s}" for s in key_sentences[:3])}

Critical perspective:
{key_sentences[3] if len(key_sentences) > 3 else f"The significance of {topic} cannot be understated in its impact on India's development."}

Overall assessment:
{key_sentences[4] if len(key_sentences) > 4 else f"{topic} continues to be relevant in understanding contemporary India."}

For UPSC purposes, one must appreciate both the historical context and the present-day implications of {topic}."""

    elif q_type == "analyse":
        return f"""Analysis of {topic}:

Introduction:
{key_sentences[0] if key_sentences else f"{topic} is a multifaceted concept in {subject}."}

Key factors to analyse:

1. Causes/Origins:
{key_sentences[1] if len(key_sentences) > 1 else f"The origins of {topic} are rooted in historical and social circumstances."}

2. Process/Development:
{key_sentences[2] if len(key_sentences) > 2 else f"The development of {topic} involved multiple stakeholders and complex processes."}

3. Impact/Consequences:
{key_sentences[3] if len(key_sentences) > 3 else f"The impact of {topic} was felt across political, economic and social domains."}

4. Long-term significance:
{key_sentences[4] if len(key_sentences) > 4 else f"{topic} continues to shape India's trajectory in important ways."}

Conclusion:
A comprehensive analysis of {topic} reveals its centrality to understanding {subject} in the UPSC context."""

    elif q_type in ("opinion", "evaluate"):
        return f"""Opinion/Evaluation on {topic}:

Yes, {topic} was indeed a significant development because:

Supporting arguments:
{chr(10).join(f"• {s}" for s in key_sentences[:3])}

Counter perspective:
However, it is also important to note the limitations and challenges associated with {topic}.
{key_sentences[3] if len(key_sentences) > 3 else "Not all aspects were uniformly positive or impactful."}

Balanced assessment:
{key_sentences[4] if len(key_sentences) > 4 else f"Overall, {topic} must be evaluated in its historical and social context."}

Conclusion:
While acknowledging both strengths and weaknesses, {topic} undeniably left a lasting mark on {subject} and India's development narrative."""

    elif q_type in ("how", "compare"):
        return f"""How {topic} shaped its domain:

{topic} influenced {subject} in several important ways:

{chr(10).join(f"{i+1}. {s}" for i, s in enumerate(key_sentences[:4]))}

Mechanism of impact:
{key_sentences[4] if len(key_sentences) > 4 else f"The influence of {topic} operated through multiple channels and over extended periods."}

Legacy:
{key_sentences[5] if len(key_sentences) > 5 else f"The legacy of {topic} continues to be felt in contemporary India."}

For UPSC Mains, it is important to provide specific examples and dates when discussing {topic}."""

    else:
        return f"""{topic} — Key Points for UPSC:

{chr(10).join(f"• {s}" for s in key_sentences[:5])}

This topic falls under {subject} and is important for both Prelims and Mains examination."""


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

GEMMA_TEMPLATE = "<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n{answer}<end_of_turn>"


def make_sft_record(chunk: dict, idx: int):
    subject  = chunk.get("subject", "General")
    text     = chunk.get("text", "").strip()
    heading  = (chunk.get("section_heading") or "").strip()
    word_cnt = chunk.get("word_count", 0)

    # Skip very short or corrupt chunks
    if word_cnt < 60:
        return None
    if len(text) < 200:
        return None

    # Use heading as topic, fall back to subject
    # Filter out headings that are figure refs, pure numbers, or OCR garbage
    heading_clean = heading
    if heading_clean:
        # Reject if starts with digit (figure/table refs like "3.22 Figure...")
        if re.match(r'^\d', heading_clean):
            heading_clean = ""
        # Reject if too many uppercase-only or symbol chars (OCR garbage)
        elif sum(1 for c in heading_clean if c.isupper()) / max(len(heading_clean), 1) > 0.7:
            heading_clean = ""
        # Reject if contains 3+ consecutive non-alpha chars (encoding junk)
        elif re.search(r'[^A-Za-z0-9\s,.\'\-]{3,}', heading_clean):
            heading_clean = ""

    topic = heading_clean if heading_clean and len(heading_clean) > 3 else subject

    # Pick template
    templates = TEMPLATES.get(subject, DEFAULT_TEMPLATES)
    q_type, q_template = random.choice(templates)
    question = q_template.format(topic=topic)

    # Build structured answer
    answer = build_answer(q_type, topic, text, subject)

    return {
        "id":       chunk["id"] + "_v2",
        "text":     GEMMA_TEMPLATE.format(question=question, answer=answer),
        "subject":  subject,
        "question": question,
        "q_type":   q_type,
    }


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def main():
    print(f"Reading : {INPUT_JSONL}")
    chunks = []
    with open(INPUT_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    print(f"Loaded  : {len(chunks):,} chunks")

    records = []
    skipped = 0
    for i, chunk in enumerate(chunks):
        rec = make_sft_record(chunk, i)
        if rec:
            records.append(rec)
        else:
            skipped += 1

    random.shuffle(records)

    # Stats
    q_type_counts = Counter(r["q_type"] for r in records)
    subj_counts   = Counter(r["subject"] for r in records)

    print(f"\nGenerated : {len(records):,} SFT records")
    print(f"Skipped   : {skipped} (too short/corrupt)")
    print(f"\nQuestion types:")
    for qt, cnt in sorted(q_type_counts.items(), key=lambda x: -x[1]):
        print(f"  {qt:<10} {cnt:>5,}")
    print(f"\nBy subject:")
    for subj, cnt in sorted(subj_counts.items(), key=lambda x: -x[1]):
        print(f"  {subj:<20} {cnt:>5,}")

    # Save
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n✓ Saved → {OUTPUT_JSONL}")

    # Show 2 samples
    print("\n" + "=" * 60)
    print("SAMPLE RECORDS")
    print("=" * 60)
    for rec in records[:2]:
        print(f"\nSubject  : {rec['subject']}")
        print(f"Q-Type   : {rec['q_type']}")
        print(f"Question : {rec['question']}")
        print(f"\nAnswer preview:")
        answer_part = rec['text'].split('<start_of_turn>model')[-1]
        print(answer_part[:400])
        print("-" * 60)


if __name__ == "__main__":
    main()
