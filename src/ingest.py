# src/ingest.py
# This ingests all 10K documents from the downloaded finance bench repo

"""
Page extraction (cached) and 10-K Item/section detection.

Split each 10-K into ~400-token chunks with metadata. Writes data/chunks.jsonl.

Rules (see decisions-log.md, Day 1):
- A chunk never crosses a section (Item) boundary, but may cross page boundaries.
- Recursive splitting: paragraphs, then lines, then sentences, then words, until each
  piece is <= 400 tokens; pieces are then packed into chunks with ~50 tokens of overlap.
- page_start / page_end record which pages a chunk touches (0-based, like FinanceBench).
"""

import re
import json
import pymupdf
import statistics
from bisect import bisect_right
from collections import Counter
import tiktoken
import config

MAX_TOKENS = config.CHUNK_MAX_TOKENS
OVERLAP_TOKENS = config.CHUNK_OVERLAP_TOKENS
OUT = config.CHUNKS_PATH
PAGES_DIR = config.PAGES_DIR
PDF_DIR = config.PDF_DIR
QUESTIONS_PATH = config.QUESTIONS_PATH



# Real Item heading: start of line, "Item", number (+ optional A/B/C), then punctuation.
HEADING = re.compile(r"^\s*item\s*(\d{1,2}[a-c]?)\s*[.:\-–—]", re.IGNORECASE)
TOC_SCAN_PAGES = 5
TOC_MIN_LABELS = 3


LABELS = {"1": "Item 1 Business", "1A": "Item 1A Risk Factors", "7": "Item 7 MD&A",
          "7A": "Item 7A Market Risk", "8": "Item 8 Financial Statements"}


enc = tiktoken.get_encoding("cl100k_base")   # tokenizer used by text-embedding-3-small




def get_pages(doc_name: str) -> list[str]:
    """Text of every page, index = 0-based page number. Extracts once, then reads the cache."""
    cache = PAGES_DIR / f"{doc_name}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(PDF_DIR / f"{doc_name}.pdf") as pdf:
        pages = [page.get_text() for page in pdf]
    cache.write_text(json.dumps(pages), encoding="utf-8")
    return pages

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


def n_tokens(s: str) -> int:
    return len(enc.encode(s))


# ---------------------------------------------------------------- 1. whole document

def join_document(pages: list[str]) -> tuple[str, list[int]]:
    """Join all pages into one string. page_offsets[p] = where page p starts in it."""
    text, page_offsets = "", []
    for page in pages:
        page_offsets.append(len(text))
        page = page.replace("\xa0", " ")              # non-breaking spaces -> normal spaces
        page = re.sub(r"[ \t]+", " ", page)          # collapse runs of spaces
        page = "\n".join(l.strip() for l in page.splitlines())  # whitespace-only lines -> empty
        page = re.sub(r"\n{3,}", "\n\n", page).strip()           # one blank line = one paragraph break
        text += page + "\n\n"                       # page break is also a paragraph break
    return text, page_offsets


def page_of(pos: int, page_offsets: list[int]) -> int:
    """Which page a character position falls on."""
    return bisect_right(page_offsets, pos) - 1


# ---------------------------------------------------------------- 2. sections

def label_for(item: str) -> str:
    if item_order(item) >= (15, ""):
        return "Financial Statements & Exhibits"     # 9 filings keep statements after Item 15
    return LABELS.get(item, f"Item {item}")


def section_bounds(starts: dict[str, int], n_pages: int) -> list[tuple[str, int, int]]:
    """[(label, first_page, last_page)]. A section ends where the next one starts."""
    # Sort by page; for Items sharing a start page, the last one wins (3M p11: 1B,2,3,4 -> 4)
    by_page = {}
    for item in sorted(starts, key=lambda it: (starts[it], item_order(it))):
        by_page[starts[item]] = label_for(item)

    firsts = []                                       # [(first_page, label)], merged
    for page, label in sorted(by_page.items()):
        if not firsts or firsts[-1][1] != label:
            firsts.append((page, label))

    if not firsts or firsts[0][0] > 0:
        firsts.insert(0, (0, "Front Matter"))         # cover page etc.

    bounds = []
    for i, (first, label) in enumerate(firsts):
        last = firsts[i + 1][0] - 1 if i + 1 < len(firsts) else n_pages - 1
        bounds.append((label, first, last))
    return bounds


# ---------------------------------------------------------------- 3. splitter (recursive)

SEPARATORS = ["\n\n", "\n", ". ", " "]          # paragraph -> line -> sentence -> word


def split_units(text: str, start: int, end: int, level: int = 0) -> list[tuple[int, int, int]]:
    """Recursively cut text[start:end] into pieces of <= MAX_TOKENS -> [(s, e, tokens)].
    Try the coarsest separator first; only pieces still too big go down a level.
    Positions are in the whole document, so pages can be looked up later."""
    n = n_tokens(text[start:end])
    if n <= MAX_TOKENS or level == len(SEPARATORS):
        return [(start, end, n)]
    sep, pieces, pos = SEPARATORS[level], [], start
    while pos < end:
        cut = text.find(sep, pos, end)
        nxt = end if cut == -1 else cut + len(sep)   # separator stays with the piece before it
        pieces += split_units(text, pos, nxt, level + 1)
        pos = nxt
    return pieces


def split_section(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Pack the recursive pieces into chunks of <= MAX_TOKENS, in order.
    Each chunk starts ~OVERLAP_TOKENS before the previous one ended (whole pieces only)."""
    units = split_units(text, start, end)
    spans, i = [], 0
    while i < len(units):
        j, total = i, 0
        while j < len(units) and total + units[j][2] <= MAX_TOKENS:   # fill the chunk with pieces
            total += units[j][2]
            j += 1
        j = max(j, i + 1)                                             # always make progress
        spans.append((units[i][0], units[j - 1][1]))
        if j >= len(units):
            break
        k, back = j, 0                                                # step back for overlap
        while k - 1 > i and back + units[k - 1][2] <= OVERLAP_TOKENS:
            k -= 1
            back += units[k][2]
        i = k
    return spans


# ---------------------------------------------------------------- 4. records

def build_chunks(doc_name: str, company: str) -> list[dict]:
    pages = get_pages(doc_name)
    text, page_offsets = join_document(pages)
    fiscal_year = int(doc_name.split("_")[-2])       # "<NAME>_<YEAR>_10K"; NAME may contain "_" (JOHNSON_JOHNSON)

    records = []
    for section, first, last in section_bounds(find_item_starts(pages), len(pages)):
        sec_start = page_offsets[first]
        sec_end = page_offsets[last + 1] if last + 1 < len(pages) else len(text)
        for s, e in split_section(text, sec_start, sec_end):
            chunk = text[s:e].strip()
            if not chunk:
                continue
            idx = len(records)
            records.append({
                "chunk_id": f"{doc_name}_{idx:05d}",
                "doc_id": doc_name,
                "company": company,
                "fiscal_year": fiscal_year,
                "form_type": "10-K",
                "section": section,
                "page_start": page_of(s, page_offsets),
                "page_end": page_of(e - 1, page_offsets),
                "chunk_index": idx,
                # prefix kept separate: indexed with the text, shown to the LLM only as a source label
                "prefix": f"{company} | fiscal year {fiscal_year} FY{fiscal_year} | 10-K | {section}",
                "text": chunk,
            })
    return records

def save_filings(chunks):
    filings = {}
    for chunk in chunks:
        filings.setdefault(chunk['company'], set()).add(int(chunk['fiscal_year']))
    filings = {company: sorted(years) for company, years in filings.items()}
    with open(config.FILINGS_PATH, "w") as f:
        json.dump(filings, f, indent=2)

# ---------------------------------------------------------------- 5. run + checks

def main():
    questions = [json.loads(l) for l in open(QUESTIONS_PATH, encoding="utf-8")]
    tenk = [q for q in questions if q["doc_name"].endswith("_10K")]
    company_of = {q["doc_name"]: q["company"] for q in tenk}
    docs = sorted(company_of)

    all_chunks = []
    for n, d in enumerate(docs, 1):
        print(f"\r{n}/{len(docs)} {d:<30}", end="")
        all_chunks += build_chunks(d, company_of[d])
    print()

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c) + "\n")
    print(f"wrote {len(all_chunks)} chunks to {OUT}")

    # Check 1: counts
    print("\n[1] chunks per section:")
    for sec, n in Counter(c["section"] for c in all_chunks).most_common():
        print(f"    {sec:<34} {n}")

    # Check 2: token lengths (prefix excluded; it adds ~20 tokens)
    lens = [n_tokens(c["text"]) for c in all_chunks]
    print(f"\n[2] tokens: min {min(lens)}  median {statistics.median(lens):.0f}  max {max(lens)}"
          f"   (<50 tokens: {sum(l < 50 for l in lens)})")

    # Check 3: every evidence page is inside at least one chunk of its document
    by_doc = {}
    for c in all_chunks:
        by_doc.setdefault(c["doc_id"], []).append(c)
    missing = [(q["doc_name"], e["evidence_page_num"]) for q in tenk for e in q["evidence"]
               if not any(c["page_start"] <= e["evidence_page_num"] <= c["page_end"]
                          for c in by_doc.get(q["doc_name"], []))]
    print(f"\n[3] evidence pages not covered by any chunk: {len(missing)} {missing[:5]}")

    # Check 4: the known answer — 3M FY2018 capex 1,577 on page 59
    hit = [c["chunk_id"] for c in by_doc.get("3M_2018_10K", [])
           if c["page_start"] <= 59 <= c["page_end"] and "1,577" in c["text"]]
    print(f"\n[4] 3M p59 chunks containing '1,577': {hit}")

    save_filings(all_chunks)

if __name__ == "__main__":
    main()