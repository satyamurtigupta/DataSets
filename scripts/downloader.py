#!/usr/bin/env python3
"""
Universal File Downloader
Downloads files from any website, organised into folders by category.

INPUT FORMATS (auto-detected):
  CSV          — must have a 'url' column; optional 'title', 'category' columns
  JSON         — array of {"url":"...", "title":"...", "category":"..."} objects
  Plain text   — one URL per line  OR  "title<TAB>url" per line
  Vision IAS   — the "001. Title / ID / URL" format (auto-detected)

CLASSIFICATION (optional):
  Pass --classify rules.json to auto-assign categories from title keywords.
  Format:
    {
      "Geography":  ["ocean", "climate", "landform"],
      "Economy":    ["banking", "inflation", "wto"],
      "Polity":     ["parliament", "judiciary"]
    }
  If input already has a 'category' column/field, that takes priority.
  If no classify config and no category column, all files go into one flat folder.

AUTH OPTIONS:
  --cookie "PHPSESSID=abc123"        session cookie (from browser DevTools)
  --cookie-file cookies.txt          file containing the cookie string
  --header "Authorization: Bearer X" add any request header (repeatable)
  --basic-auth user:password         HTTP Basic auth

EXAMPLES:
  # Plain list of URLs
  python3 downloader.py urls.txt --output ./downloads

  # CSV with title + url columns, auto-classify
  python3 downloader.py data.csv --classify rules.json --output ./downloads

  # JSON input, authenticated
  python3 downloader.py files.json --cookie "session=xyz" --output ./downloads

  # Vision IAS format
  python3 downloader.py Vision_IAS_Links.txt --classify upsc_rules.json \\
      --cookie "PHPSESSID=abc" --output ./VisionIAS_PDFs

  # Dry-run to preview without downloading
  python3 downloader.py data.csv --classify rules.json --dry-run

  # Download only one category
  python3 downloader.py data.csv --classify rules.json --category Geography

  # Resume interrupted download
  python3 downloader.py data.csv --cookie "..." --output ./out --resume
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import mimetypes
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests

log = logging.getLogger("downloader")

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Item:
    url: str
    title: str = ""
    category: str = ""
    index: int = 0

    def safe_filename(self, ext_hint: str = "") -> str:
        if self.title:
            name = re.sub(r'[^\w\s\-]', '', self.title).strip()
            name = re.sub(r'\s+', '_', name)[:120]
        else:
            name = Path(urlparse(self.url).path).name or f"file_{self.index:04d}"
            name = re.sub(r'[^\w.\-]', '_', name)

        if self.index:
            name = f"{self.index:03d}_{name}"

        if ext_hint and not name.lower().endswith(ext_hint.lower()):
            name += ext_hint
        return name


# ---------------------------------------------------------------------------
# Input parsers
# ---------------------------------------------------------------------------

def _is_visionias_format(text: str) -> bool:
    return bool(re.search(r'^\d{3}\.\s+.+\n\s+ID:\s+\d+\n\s+URL:', text, re.MULTILINE))


def parse_visionias(text: str) -> list[Item]:
    pattern = re.compile(
        r'^(\d{3,})\.\s+(.+?)\s*\n'
        r'\s+ID:\s+\d+\s*\n'
        r'\s+URL:\s+(https?://\S+)',
        re.MULTILINE,
    )
    return [
        Item(url=m.group(3).strip(), title=m.group(2).strip(), index=int(m.group(1)))
        for m in pattern.finditer(text)
    ]


def parse_csv(path: Path) -> list[Item]:
    items: list[Item] = []
    with path.open(encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = [h.lower().strip() for h in (reader.fieldnames or [])]
        url_col   = _pick(headers, ["url", "link", "href", "download_url", "file_url"])
        title_col = _pick(headers, ["title", "name", "topic", "subject", "label"])
        cat_col   = _pick(headers, ["category", "cat", "folder", "subject", "section"])

        if not url_col:
            raise ValueError("CSV has no recognisable URL column (tried: url, link, href, download_url, file_url)")

        for i, row in enumerate(reader, 1):
            row_lower = {k.lower().strip(): v for k, v in row.items()}
            url = row_lower.get(url_col, "").strip()
            if not url:
                continue
            items.append(Item(
                url=url,
                title=row_lower.get(title_col, "").strip() if title_col else "",
                category=row_lower.get(cat_col, "").strip() if cat_col else "",
                index=i,
            ))
    return items


def parse_json(path: Path) -> list[Item]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("items") or data.get("data") or data.get("results") or list(data.values())
    items: list[Item] = []
    for i, obj in enumerate(data, 1):
        if isinstance(obj, str):
            items.append(Item(url=obj, index=i))
        elif isinstance(obj, dict):
            url = (obj.get("url") or obj.get("link") or obj.get("href") or "").strip()
            if not url:
                continue
            items.append(Item(
                url=url,
                title=(obj.get("title") or obj.get("name") or obj.get("topic") or "").strip(),
                category=(obj.get("category") or obj.get("folder") or obj.get("section") or "").strip(),
                index=i,
            ))
    return items


def parse_text(text: str) -> list[Item]:
    items: list[Item] = []
    for i, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "\t" in line:
            parts = line.split("\t", 1)
            title, url = (parts[0].strip(), parts[1].strip()) if parts[1].startswith("http") \
                         else (parts[1].strip(), parts[0].strip())
        else:
            url = line
            title = ""
        if url.startswith("http"):
            items.append(Item(url=url, title=title, index=i))
    return items


def load_items(path: Path) -> list[Item]:
    text = path.read_text(encoding="utf-8", errors="replace")
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return parse_csv(path)
    if suffix == ".json":
        return parse_json(path)
    if _is_visionias_format(text):
        log.debug("Detected Vision IAS format")
        return parse_visionias(text)
    return parse_text(text)


def _pick(headers: list[str], candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in headers:
            return c
    return None


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def load_rules(path: Path) -> dict[str, list[str]]:
    return json.loads(path.read_text(encoding="utf-8"))


def classify(title: str, rules: dict[str, list[str]]) -> str:
    t = title.lower()
    for category, keywords in rules.items():
        for kw in keywords:
            if kw.lower() in t:
                return category
    return "Uncategorised"


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

def build_session(
    cookie_str: str,
    headers: list[str],
    basic_auth: Optional[str],
    url_hint: str,
) -> requests.Session:
    sess = requests.Session()
    domain = urlparse(url_hint).netloc if url_hint else ""

    sess.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "*/*",
    })

    for h in headers:
        if ":" in h:
            k, _, v = h.partition(":")
            sess.headers[k.strip()] = v.strip()

    if cookie_str:
        for pair in cookie_str.split(";"):
            pair = pair.strip()
            if "=" in pair:
                k, _, v = pair.partition("=")
                sess.cookies.set(k.strip(), v.strip(), domain=domain or None)

    if basic_auth and ":" in basic_auth:
        user, _, pwd = basic_auth.partition(":")
        sess.auth = (user, pwd)

    return sess


# ---------------------------------------------------------------------------
# Downloader
# ---------------------------------------------------------------------------

def _infer_ext(resp: requests.Response, url: str) -> str:
    ct = resp.headers.get("Content-Type", "").split(";")[0].strip()
    ext = mimetypes.guess_extension(ct) or ""
    if ext in (".jpe", ".jpeg"):
        ext = ".jpg"
    if not ext:
        ext = Path(urlparse(url).path).suffix or ""
    return ext


# Magic bytes for common file types → used to catch HTML masquerading as downloads
_HTML_SIGNATURES = (b"<!DOCTYPE", b"<html", b"<HTML", b"<?xml")


def _is_html(content: bytes) -> bool:
    return content[:512].lstrip().startswith(_HTML_SIGNATURES)


def _is_login_page(content: bytes) -> bool:
    snippet = content[:4096].lower()
    return any(kw in snippet for kw in (b"login", b"sign in", b"signin", b"password", b"username"))


def download_item(
    item: Item,
    dest_dir: Path,
    sess: requests.Session,
    delay: float,
    max_retries: int,
    resume: bool,
    ext_override: str,
) -> tuple[bool, str]:
    """Returns (success, filename)."""
    base_name = item.safe_filename(ext_override)

    # Resume check
    if resume:
        matches = list(dest_dir.glob(f"{base_name}*")) if not ext_override \
                  else ([dest_dir / base_name] if (dest_dir / base_name).exists() else [])
        if matches and matches[0].stat().st_size > 512:
            log.info(f"  [skip]  {matches[0].name}")
            return True, matches[0].name

    for attempt in range(1, max_retries + 1):
        try:
            # allow_redirects=True (default) — requests follows to final URL
            resp = sess.get(item.url, timeout=60)

            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", delay * attempt * 2))
                log.warning(f"  [rate-limited] waiting {wait}s ...")
                time.sleep(wait)
                continue

            if resp.status_code != 200:
                log.warning(f"  [HTTP {resp.status_code}]  {item.url}  (attempt {attempt})")
                time.sleep(delay * attempt)
                continue

            content = resp.content

            # Detect HTML response — means auth failed or we landed on an error page
            if _is_html(content):
                final_url = resp.url  # the URL after all redirects
                if _is_login_page(content):
                    log.error(
                        f"  [auth]  Got login page for: {item.title or item.url}\n"
                        f"          Final URL: {final_url}\n"
                        f"          Your session cookie has expired or is wrong.\n"
                        f"          Re-login to the site and copy a fresh PHPSESSID."
                    )
                else:
                    log.error(
                        f"  [html]  Got HTML instead of a file for: {item.title or item.url}\n"
                        f"          Final URL: {final_url}\n"
                        f"          Content-Type: {resp.headers.get('Content-Type', 'unknown')}"
                    )
                return False, ""

            # Determine final filename with extension
            if not ext_override:
                ext = _infer_ext(resp, resp.url)
                if ext and not base_name.lower().endswith(ext.lower()):
                    base_name = base_name + ext
            dest_path = dest_dir / base_name

            dest_path.write_bytes(content)
            size_kb = dest_path.stat().st_size // 1024
            log.info(f"  [ok]    {base_name}  ({size_kb} KB)")
            time.sleep(delay)
            return True, base_name

        except requests.RequestException as exc:
            log.warning(f"  [err]   {item.url}  attempt {attempt}: {exc}")
            time.sleep(delay * attempt)

    log.error(f"  [fail]  {item.url} — gave up after {max_retries} attempts")
    return False, ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Universal file downloader — works with any website",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input
    ap.add_argument("input", type=Path,
                    help="Input file: CSV, JSON, plain-text URL list, or Vision IAS format")

    # Output
    ap.add_argument("--output", "-o", default="./downloads", type=Path,
                    help="Root output directory (default: ./downloads)")
    ap.add_argument("--flat", action="store_true",
                    help="Dump all files into --output directly, no category subfolders")
    ap.add_argument("--ext", default="",
                    help="Force file extension, e.g. .pdf  (default: auto-detect from Content-Type)")

    # Classification
    ap.add_argument("--classify", metavar="RULES_JSON", type=Path,
                    help="JSON file mapping category names to keyword lists")
    ap.add_argument("--category", metavar="CAT",
                    help="Download only this category (after classification)")

    # Auth
    ap.add_argument("--cookie", default="",
                    help='Cookie string, e.g. "PHPSESSID=abc; other=xyz"')
    ap.add_argument("--cookie-file", type=Path,
                    help="File containing the cookie string")
    ap.add_argument("--header", action="append", default=[], metavar="NAME:VALUE",
                    help="Extra request header (repeatable)")
    ap.add_argument("--basic-auth", metavar="USER:PASS",
                    help="HTTP Basic authentication")

    # Behaviour
    ap.add_argument("--delay", type=float, default=1.5,
                    help="Seconds between requests (default: 1.5)")
    ap.add_argument("--retries", type=int, default=3,
                    help="Max retries per file (default: 3)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip files that already exist")
    ap.add_argument("--dry-run", action="store_true",
                    help="Parse and classify only — no downloads")
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load input
    if not args.input.exists():
        log.error(f"Input file not found: {args.input}")
        sys.exit(1)

    items = load_items(args.input)
    if not items:
        log.error("No items found in the input file.")
        sys.exit(1)
    log.info(f"Loaded {len(items)} items from {args.input.name}")

    # Classify
    rules: dict[str, list[str]] = {}
    if args.classify:
        if not args.classify.exists():
            log.error(f"Rules file not found: {args.classify}")
            sys.exit(1)
        rules = load_rules(args.classify)
        log.info(f"Loaded {len(rules)} category rules from {args.classify.name}")

    for item in items:
        if not item.category and rules:
            item.category = classify(item.title, rules)
        if not item.category:
            item.category = "Downloads"

    # Filter by category
    if args.category:
        items = [it for it in items if it.category == args.category]
        log.info(f"Filtered to {len(items)} items in '{args.category}'")

    # Summary
    counts = Counter(it.category for it in items)
    print(f"\n{'─'*50}")
    print(f"  Total items: {len(items)}")
    if len(counts) > 1 or (len(counts) == 1 and list(counts)[0] != "Downloads"):
        print("  By category:")
        for cat, n in sorted(counts.items()):
            print(f"    {cat:<35} {n:>4}")
    print(f"{'─'*50}\n")

    if args.dry_run:
        print("DRY RUN — no files will be downloaded.\n")
        for it in items:
            cat_label = f"[{it.category}]" if it.category != "Downloads" else ""
            print(f"  [{it.index:03d}] {cat_label:<35} {it.title or it.url}")
        return

    # Cookie
    cookie_str = args.cookie
    if args.cookie_file and args.cookie_file.exists():
        cookie_str = args.cookie_file.read_text().strip()

    # Session
    url_hint = items[0].url if items else ""
    sess = build_session(cookie_str, args.header, args.basic_auth, url_hint)

    # Create folders
    if args.flat:
        args.output.mkdir(parents=True, exist_ok=True)
    else:
        for cat in counts:
            (args.output / cat).mkdir(parents=True, exist_ok=True)

    # Save manifest
    manifest = [
        {"index": it.index, "title": it.title, "category": it.category, "url": it.url}
        for it in items
    ]
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )

    # Download
    ok = failed = skipped = 0
    current_cat = None

    for it in items:
        if not args.flat and it.category != current_cat:
            current_cat = it.category
            print(f"\n{'='*50}")
            print(f"  {current_cat}  ({counts[current_cat]} files)")
            print(f"{'='*50}")

        dest_dir = args.output if args.flat else (args.output / it.category)
        success, fname = download_item(
            it, dest_dir, sess, args.delay, args.retries, args.resume, args.ext
        )
        if fname == "" and not success:
            failed += 1
        elif args.resume and not fname:
            skipped += 1
        elif success:
            ok += 1
        else:
            failed += 1

    print(f"\n{'='*50}")
    print(f"  Done.  Downloaded={ok}  Failed={failed}  Skipped={skipped}")
    print(f"  Output: {args.output.resolve()}")
    print(f"{'='*50}\n")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
