#!/usr/bin/env python3
"""
Vision IAS QRM PDF Downloader
Parses the download links file, classifies each topic under UPSC syllabus,
creates the folder structure, and downloads PDFs into the correct category.

Usage:
    python3 scripts/visionias_downloader.py \
        --links ~/Downloads/Vision_IAS_Prelims_Only_Download_Links.txt \
        --output ~/Downloads/VisionIAS_PDFs \
        --cookie "PHPSESSID=<your_session_id>"

    # Dry-run (classify only, no downloads):
    python3 scripts/visionias_downloader.py --links ... --dry-run

    # Resume (skip already-downloaded files):
    python3 scripts/visionias_downloader.py --links ... --cookie "..." --resume

How to get your session cookie:
    1. Log in to visionias.in in Chrome/Firefox
    2. Open DevTools → Application → Cookies → visionias.in
    3. Copy the value of the PHPSESSID cookie
    4. Pass it as: --cookie "PHPSESSID=abc123..."
"""

from __future__ import annotations

import argparse
import re
import time
import logging
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests

log = logging.getLogger("visionias_dl")

# ---------------------------------------------------------------------------
# UPSC syllabus category definitions
# ---------------------------------------------------------------------------

CATEGORIES = [
    "Geography",
    "Environment_and_Ecology",
    "Economy",
    "Polity",
    "History_Modern",
    "History_Ancient_Medieval",
    "Art_and_Culture",
    "Miscellaneous",
]

# Keyword rules — checked in order; first match wins.
# Each rule: (category, [keywords_any_match])
# Keywords are matched against lowercased title using substring search.
_KEYWORD_RULES: list[tuple[str, list[str]]] = [
    # --- Art & Culture (very distinctive terms, check first) ---
    ("Art_and_Culture", [
        "music of india", "music- classical", "dances of india", "dance of india",
        "indian theatre", "indian puppetry", "indian paintings", "paintings",
        "language and literature", "science and technology through the ages",
        "science through the ages", "folk", "puppetry",
    ]),

    # --- Polity ---
    ("Polity", [
        "preamble", "citizenship", "fundamental rights", "fundamental duties",
        "dpsp", "directive principles of state policy",
        "parliament", "supreme court", "high court", "subordinate court",
        "union executive", "state executive", "state legislature",
        "constitutional bodies", "non-constitutional bodies",
        "electoral process", "local self government",
        "centre state", "centre-state", "inter-state",
        "emergency provision", "union and its territory",
        "different states", "administrative system",
        "cooperative societies", "cabinet committees",
        "basic polity", "evolution of constitution",
        "historical background of indian constitution",
        "judiciary part", "judiciary -",
        "judiciary part_3", "judiciary part_2", "judiciary part_1",
    ]),

    # --- History: Modern India ---
    ("History_Modern", [
        "revolt of 1857", "governor general", "viceroy",
        "settlement policy", "east india company", "consolidation",
        "freedom movement", "towards freedom and partition",
        "mahatma gandhi", "early political organizations",
        "socio religious reform", "revolutionary nationalism",
        "peasant", "civil rebellion",
        "development of education",
        "important outcomes",         # outcomes of freedom movement
        "capitalists",
        "1939-1947", "1919-1938",
        "anglo french", "marathas and maysore",
        "acts_judiciary_police",      # colonial administration
        "acts", "police", "army", "development)",  # entry 110 / 155
    ]),

    # Gandhi / Congress checks (avoid partial matches with Polity "centre")
    ("History_Modern", ["mahatma"]),

    # --- History: Ancient & Medieval ---
    ("History_Ancient_Medieval", [
        "ancient india", "mahajanapada", "mahajanpada",
        "gupta and harsha", "gupta period", "harsha period",
        "delhi sultanate",
        "great mughals", "mughal empire", "of the mughal",
        "important kingdom", "important kingdoms",
        "regional powers", "emergence of regional",
        "architecture and sculpture",
        "religion and philosophy",
        "including marathas",         # regional powers / decline of mughals era
    ]),

    # --- Environment & Ecology ---
    ("Environment_and_Ecology", [
        "biodiversity", "of biodiversity",
        "ecosystem", "ecology",
        "wetland", "mangrove",
        "conservation of biodiversity",
        "climate change", "global warming", "combating climate",
        "pollution",
        "waste management",
        "renewable energy", "energy conservation",
        "land and water degradation",
        "prominent protected areas", "protected areas",
        "prominent threatened species", "threatened species",
        "sustainable development",
        "sustainable agriculture",
        "initiatives",               # "Initiatives" in env block (entry 170)
        "etc change",                # OCR artifact → "Climate Change" (entry 007)
    ]),

    # --- Economy ---
    ("Economy", [
        "national income",
        "banking",
        "monetary policy",
        "money market", "money\n",    # entry 194 "Money"
        "inflation",
        "taxation",
        "union budget",
        "external sector",
        "the wto", "wto",
        "poverty and inequality",
        "employment", "unemployment",
        "infrastructure part",
        "food processing",
        "planning in india",
        "industry part", "industrial policies",
        "india and international institutions",
        "miscellaneous",             # entry 196 sits in economy block
    ]),

    # "Agriculture" alone (no "Indian") → Economy (entries 049, 179 are economic policy)
    ("Economy", ["agriculture"]),

    # --- Geography (catch-all for physical & economic geography topics) ---
    ("Geography", [
        "water transport", "air transport", "land transport",
        "volcanism",
        "universe and solar system", "universe",
        "solar system",
        "precipitation",
        "ocean basics", "ocean",
        "mountain building", "island formation", "hotspot",
        "landform",
        "international trade",
        "interior of the earth", "interior of earth",
        "insolation",
        "indian climate",
        "indian agriculture",
        "geography location", "physiography",
        "earthquake", "tsunami",
        "drainage pattern",
        "distribution of key natural resources", "natural resources",
        "coral reef",
        "continental drift", "plate tectonic",
        "composition of atmosphere", "composition of the atmosphere",
        "climate and global climate",
        "basics of soils", "soils",
        "air mass",
        "mineral resources",
        "manufacturing industries",
        "location of industries",
        "systems",                   # "Monsoon Systems" entries 022, 210
        "indian agriculture",        # specifically "Indian Agriculture" geo entries 017, 197
        "distribution of key",
    ]),
]

# Index-range fallback (QRM 2024 section layout, also mirrors QRM 2021)
_INDEX_RANGE_MAP: list[tuple[tuple[int, int], str]] = [
    ((1, 29),   "Geography"),
    ((30, 45),  "Environment_and_Ecology"),
    ((46, 65),  "Economy"),
    ((66, 97),  "Polity"),
    ((98, 113), "History_Modern"),
    ((114, 120),"Art_and_Culture"),
    ((121, 129),"History_Ancient_Medieval"),
    ((130, 135),"History_Ancient_Medieval"),
    ((136, 143),"Art_and_Culture"),
    ((144, 159),"History_Modern"),
    ((160, 175),"Environment_and_Ecology"),
    ((176, 196),"Economy"),
    ((197, 224),"Geography"),
    ((225, 256),"Polity"),
]


def classify_topic(index: int, title: str) -> str:
    t = title.lower()

    for category, keywords in _KEYWORD_RULES:
        for kw in keywords:
            if kw in t:
                return category

    # Congress — avoid matching "centre-state" which contains no "congress"
    if "congress" in t:
        return "History_Modern"

    # Index-range fallback
    for (lo, hi), cat in _INDEX_RANGE_MAP:
        if lo <= index <= hi:
            return cat

    return "Miscellaneous"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Entry:
    index: int
    title: str
    pdf_id: str
    url: str
    category: str = ""
    filename: str = ""

    def __post_init__(self) -> None:
        self.category = classify_topic(self.index, self.title)
        safe = re.sub(r'[^\w\s\-]', '', self.title).strip()
        safe = re.sub(r'\s+', '_', safe)
        self.filename = f"{self.index:03d}_{safe}.pdf"


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

_ENTRY_RE = re.compile(
    r'^(\d{3,})\.\s+(.+?)\s*\n'      # "001. Title\n"
    r'\s+ID:\s+(\d+)\s*\n'            # "    ID: 3043\n"
    r'\s+URL:\s+(https?://\S+)',       # "    URL: https://..."
    re.MULTILINE,
)


def parse_links_file(path: Path) -> list[Entry]:
    text = path.read_text(encoding="utf-8")
    entries: list[Entry] = []
    for m in _ENTRY_RE.finditer(text):
        idx   = int(m.group(1))
        title = m.group(2).strip()
        pid   = m.group(3).strip()
        url   = m.group(4).strip()
        entries.append(Entry(index=idx, title=title, pdf_id=pid, url=url))
    return entries


# ---------------------------------------------------------------------------
# Downloader
# ---------------------------------------------------------------------------

def build_session(cookie_str: Optional[str]) -> requests.Session:
    sess = requests.Session()
    sess.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Referer": "https://www.visionias.in/student/study_material_dashboard.php",
        "Accept": "application/pdf,*/*",
    })
    if cookie_str:
        for pair in cookie_str.split(";"):
            pair = pair.strip()
            if "=" in pair:
                k, _, v = pair.partition("=")
                sess.cookies.set(k.strip(), v.strip(), domain="www.visionias.in")
    return sess


def download_entry(
    entry: Entry,
    dest_dir: Path,
    sess: requests.Session,
    delay: float,
    max_retries: int,
) -> bool:
    dest_path = dest_dir / entry.filename
    if dest_path.exists() and dest_path.stat().st_size > 1024:
        log.info(f"  [skip]  {entry.filename} (already exists)")
        return True

    for attempt in range(1, max_retries + 1):
        try:
            resp = sess.get(entry.url, timeout=60, stream=True)
            if resp.status_code == 302 or "login" in resp.url.lower():
                log.error(
                    f"  [auth]  {entry.filename} — redirected to login. "
                    "Check your --cookie value."
                )
                return False
            if resp.status_code != 200:
                log.warning(f"  [HTTP {resp.status_code}]  {entry.filename} (attempt {attempt})")
                time.sleep(delay * attempt)
                continue

            ct = resp.headers.get("Content-Type", "")
            if "pdf" not in ct and "octet" not in ct:
                log.warning(
                    f"  [warn]  {entry.filename} — unexpected Content-Type: {ct!r}. "
                    "May not be a PDF (check auth)."
                )

            dest_path.write_bytes(resp.content)
            size_kb = dest_path.stat().st_size // 1024
            log.info(f"  [ok]    {entry.filename}  ({size_kb} KB)")
            time.sleep(delay)
            return True

        except requests.RequestException as exc:
            log.warning(f"  [err]   {entry.filename} attempt {attempt}: {exc}")
            time.sleep(delay * attempt)

    log.error(f"  [fail]  {entry.filename} — gave up after {max_retries} attempts")
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download Vision IAS QRM PDFs organised by UPSC syllabus category",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--links", required=True, type=Path,
        help="Path to Vision_IAS_Prelims_Only_Download_Links.txt",
    )
    ap.add_argument(
        "--output", default=Path.home() / "Downloads" / "VisionIAS_PDFs",
        type=Path, help="Root output directory (default: ~/Downloads/VisionIAS_PDFs)",
    )
    ap.add_argument(
        "--cookie", default="",
        help='Session cookie string, e.g. "PHPSESSID=abc123"',
    )
    ap.add_argument(
        "--cookie-file", type=Path,
        help="File containing the cookie string (alternative to --cookie)",
    )
    ap.add_argument(
        "--delay", type=float, default=1.5,
        help="Seconds to wait between downloads (default: 1.5)",
    )
    ap.add_argument(
        "--retries", type=int, default=3,
        help="Max retry attempts per file (default: 3)",
    )
    ap.add_argument(
        "--resume", action="store_true",
        help="Skip files that already exist in the output directory",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Parse and classify only — print plan without downloading",
    )
    ap.add_argument(
        "--category", metavar="CAT",
        help="Download only this category (e.g. Geography)",
    )
    ap.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Parse
    if not args.links.exists():
        log.error(f"Links file not found: {args.links}")
        sys.exit(1)

    entries = parse_links_file(args.links)
    if not entries:
        log.error("No entries parsed from the links file.")
        sys.exit(1)

    log.info(f"Parsed {len(entries)} entries from {args.links.name}")

    # Filter by category if requested
    if args.category:
        entries = [e for e in entries if e.category == args.category]
        log.info(f"Filtered to {len(entries)} entries in category '{args.category}'")

    # Summary by category
    from collections import Counter
    counts = Counter(e.category for e in entries)
    print("\nClassification summary:")
    for cat in CATEGORIES:
        if counts[cat]:
            print(f"  {cat:<30} {counts[cat]:>3} PDFs")
    print()

    if args.dry_run:
        print("DRY RUN — no files will be downloaded.\n")
        for e in entries:
            print(f"  [{e.index:03d}] {e.category:<30} {e.title}")
        return

    # Cookie
    cookie_str = args.cookie
    if args.cookie_file and args.cookie_file.exists():
        cookie_str = args.cookie_file.read_text().strip()

    if not cookie_str:
        log.warning(
            "No --cookie provided. Downloads will likely fail (redirected to login). "
            "Run with --dry-run to verify classification without needing auth."
        )

    # Create folder structure
    args.output.mkdir(parents=True, exist_ok=True)
    for cat in CATEGORIES:
        (args.output / cat).mkdir(exist_ok=True)

    # Save classification manifest
    manifest = [
        {"index": e.index, "title": e.title, "category": e.category,
         "filename": e.filename, "url": e.url}
        for e in entries
    ]
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    log.info(f"Manifest saved → {manifest_path}")

    # Download
    sess = build_session(cookie_str)
    ok = failed = skipped = 0

    current_cat = None
    for e in entries:
        if e.category != current_cat:
            current_cat = e.category
            print(f"\n{'='*60}")
            print(f"  {current_cat}  ({counts[current_cat]} PDFs)")
            print(f"{'='*60}")

        dest_dir = args.output / e.category

        if args.resume:
            dest_path = dest_dir / e.filename
            if dest_path.exists() and dest_path.stat().st_size > 1024:
                log.info(f"  [skip]  {e.filename} (resume)")
                skipped += 1
                continue

        success = download_entry(e, dest_dir, sess, args.delay, args.retries)
        if success:
            ok += 1
        else:
            failed += 1

    print(f"\n{'='*60}")
    print(f"  Done.  OK={ok}  Failed={failed}  Skipped={skipped}")
    print(f"  Output: {args.output}")
    print(f"{'='*60}\n")

    if failed:
        log.warning(
            f"{failed} file(s) failed. Re-run with --resume to retry only failed files."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
