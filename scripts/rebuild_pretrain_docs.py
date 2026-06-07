#!/usr/bin/env python3
"""
rebuild_pretrain_docs.py
========================
Step 1: Merge chunks back into coherent section-level documents.
Step 2: Normalize subject naming to title case.

Groups by (subject, source_file, section_heading), sorts by NNNN in ID,
merges text. Output: one JSONL record per topic section.

Usage:
    python3 scripts/rebuild_pretrain_docs.py
    python3 scripts/rebuild_pretrain_docs.py --dry-run
    python3 scripts/rebuild_pretrain_docs.py --stats
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "dataset_output_final" / "combined"
INPUT_FILE = OUTPUT_DIR / "unified_pretrain_v2.jsonl"
OUTPUT_FILE = OUTPUT_DIR / "pretrain_docs.jsonl"

# ---------------------------------------------------------------------------
# Subject name normalization
# ---------------------------------------------------------------------------

SUBJECT_MAP = {
    "art":                    "Art",
    "biology":                "Biology",
    "chemistry":              "Chemistry",
    "economics":              "Economics",
    "geography":              "Geography",
    "history":                "History",
    "physics":                "Physics",
    "polity":                 "Polity",
    "science":                "Science",
    "sociology":              "Sociology",
    "vision/art":             "Vision/Art",
    "vision/dm":              "Vision/DM",
    "vision/economics":       "Vision/Economics",
    "vision/env":             "Vision/Environment",
    "vision/essay":           "Vision/Essay",
    "vision/ethics":          "Vision/Ethics",
    "vision/geography":       "Vision/Geography",
    "vision/governance":      "Vision/Governance",
    "vision/history":         "Vision/History",
    "vision/ir":              "Vision/IR",
    "vision/polity":          "Vision/Polity",
    "vision/post independence": "Vision/Post Independence",
    "vision/security":        "Vision/Security",
    "vision/social justice":  "Vision/Social Justice",
    "vision/society":         "Vision/Society",
}

def normalize_subject(subj: str) -> str:
    return SUBJECT_MAP.get(subj.lower().strip(), subj.title())


# ---------------------------------------------------------------------------
# Section heading cleanup — strip leading number prefixes
# e.g. "1.1 A SIMPLE ECONOMY" -> "A SIMPLE ECONOMY"
#      "13.3 WHERE DOES PHOTOSYNTHESIS" -> "WHERE DOES PHOTOSYNTHESIS"
# ---------------------------------------------------------------------------

def clean_heading(heading: str) -> str:
    heading = heading.strip()
    # Strip patterns like: 1. / 1.1 / 1.1.2 / 2.2.1 at start
    heading = re.sub(r'^\d+(\.\d+)*[\.\s]+', '', heading).strip()
    return heading


# ---------------------------------------------------------------------------
# Extract NNNN from ID for sort order
# e.g. "biology_ncert_class_11_biology_0161" -> 161
# ---------------------------------------------------------------------------

def id_to_seq(chunk_id: str) -> int:
    m = re.search(r'_(\d+)$', chunk_id)
    return int(m.group(1)) if m else 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true',
                        help='Show stats without writing output')
    parser.add_argument('--stats', action='store_true',
                        help='Print per-subject doc counts after build')
    parser.add_argument('--input',  type=Path, default=INPUT_FILE)
    parser.add_argument('--output', type=Path, default=OUTPUT_FILE)
    args = parser.parse_args()

    # Load
    chunks = []
    with open(args.input, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    chunks.append(json.loads(line))
                except Exception:
                    pass

    print('Chunks loaded    : %d' % len(chunks))

    # Normalize subject names (Step 2)
    for c in chunks:
        c['subject'] = normalize_subject(c.get('subject', ''))

    # Group by (subject, source_file, section_heading)
    groups = defaultdict(list)
    for c in chunks:
        key = (
            c['subject'],
            c.get('source_file', ''),
            c.get('section_heading', '').strip(),
        )
        groups[key].append(c)

    print('Unique sections  : %d' % len(groups))

    # Build docs — sort chunks within each group by ID sequence number
    docs = []
    for (subject, source_file, heading), group_chunks in groups.items():
        group_chunks.sort(key=lambda c: id_to_seq(c.get('id', '')))

        merged_text = ' '.join(c['text'].strip() for c in group_chunks if c.get('text', '').strip())
        merged_text = re.sub(r'\s+', ' ', merged_text).strip()

        if not merged_text:
            continue

        clean_head = clean_heading(heading)
        word_count = len(merged_text.split())
        char_count = len(merged_text)

        docs.append({
            'subject':         subject,
            'source_file':     source_file,
            'section_heading': clean_head,
            'text':            merged_text,
            'word_count':      word_count,
            'char_count':      char_count,
            'chunk_count':     len(group_chunks),
        })

    # Sort docs: subject → source_file → first chunk ID (preserves chapter order)
    def doc_sort_key(d):
        return (d['subject'], d['source_file'], d['section_heading'])

    docs.sort(key=doc_sort_key)

    total_words = sum(d['word_count'] for d in docs)
    print('Docs built       : %d' % len(docs))
    print('Total words      : %d (~%.1fM)' % (total_words, total_words / 1_000_000))

    # Stats
    if args.stats or args.dry_run:
        from collections import Counter
        by_subj = Counter(d['subject'] for d in docs)
        print()
        print('%-30s %6s %8s' % ('Subject', 'Docs', 'Words'))
        print('-' * 48)
        for subj, count in sorted(by_subj.items()):
            wc = sum(d['word_count'] for d in docs if d['subject'] == subj)
            print('%-30s %6d %8d' % (subj, count, wc))

        # Word count distribution of merged docs
        print()
        wcs = [d['word_count'] for d in docs]
        buckets = [
            ('<100',   sum(1 for w in wcs if w < 100)),
            ('100-300', sum(1 for w in wcs if 100 <= w < 300)),
            ('300-600', sum(1 for w in wcs if 300 <= w < 600)),
            ('600-1000', sum(1 for w in wcs if 600 <= w < 1000)),
            ('1000+',  sum(1 for w in wcs if w >= 1000)),
        ]
        print('Merged doc word count distribution:')
        for label, count in buckets:
            print('  %-12s %d' % (label, count))

    if args.dry_run:
        print()
        print('DRY RUN — no file written.')
        return

    # Write
    with open(args.output, 'w', encoding='utf-8') as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

    print()
    print('Output written   : %s' % args.output)
    size_mb = args.output.stat().st_size / (1024 * 1024)
    print('File size        : %.1f MB' % size_mb)
    print()
    print('Next: push pretrain_docs.jsonl to HuggingFace for Colab training.')


if __name__ == '__main__':
    main()
