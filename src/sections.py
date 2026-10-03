"""Page extraction (cached) and 10-K Item/section detection."""
import re
import json
from pathlib import Path
import pymupdf

REPO = Path(__file__).resolve().parent.parent          # repo root, wherever you run from
DATA = REPO / "finbench_github" / "financebench"
PAGES_DIR = REPO / "scratch" / "pages"                 # page-text cache from Day 1


def get_pages(doc_name: str) -> list[str]:
    """Text of every page, index = 0-based page number. Extracts once, then reads the cache."""
    cache = PAGES_DIR / f"{doc_name}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(DATA / "pdfs" / f"{doc_name}.pdf") as pdf:
        pages = [page.get_text() for page in pdf]
    cache.write_text(json.dumps(pages), encoding="utf-8")
    return pages


# Real Item heading: start of line, "Item", number (+ optional A/B/C), then punctuation.
HEADING = re.compile(r"^\s*item\s*(\d{1,2}[a-c]?)\s*[.:\-–—]", re.IGNORECASE)
TOC_SCAN_PAGES = 5
TOC_MIN_LABELS = 3


def find_item_starts(pages: list[str]) -> dict[str, int]:
    """{item_label: start_page}, 0-based. A missing Item was not found."""
    hits = []
    for page_num, text in enumerate(pages):
        for line in text.splitlines():
            m = HEADING.match(line)
            if m:
                hits.append((page_num, m.group(1).upper()))
    labels_per_page = {}
    for p, label in hits:
        labels_per_page.setdefault(p, set()).add(label)
    toc_pages = {p for p, labels in labels_per_page.items()
                 if p < TOC_SCAN_PAGES and len(labels) >= TOC_MIN_LABELS}
    starts = {}
    for p, label in hits:
        if p not in toc_pages:
            starts.setdefault(label, p)
    return starts


def item_order(label: str) -> tuple[int, str]:
    """Sort key: "9B" -> (9, "B"), "14" -> (14, "")."""
    m = re.match(r"(\d+)([A-C]?)", label)
    return int(m.group(1)), m.group(2)


LABELS = {"1": "Item 1 Business", "1A": "Item 1A Risk Factors", "7": "Item 7 MD&A",
          "7A": "Item 7A Market Risk", "8": "Item 8 Financial Statements"}


def section_of(starts: dict[str, int], page: int) -> str:
    """Section label for one page."""
    before = [(p, item_order(item), item) for item, p in starts.items() if p <= page]
    if not before:
        return "Front Matter"
    item = max(before)[2]
    if item_order(item) >= (15, ""):
        return "Financial Statements & Exhibits"
    return LABELS.get(item, f"Item {item}")
