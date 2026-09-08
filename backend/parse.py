"""
PDF → page text → candidate detection → contextual chunks.

Pipeline:
    pages  (1 per PDF page)
      ↓
    candidate detection (numeric/financial signal filter)
      ↓
    windows of WINDOW_SIZE consecutive candidate pages
      ↓
    chunks under CHUNK_CHAR_LIMIT characters

Only pages containing quantitative/financial signals are forwarded to the LLM.
This avoids sending cover pages, table-of-contents, disclaimers, etc. to the LLM.

Candidate signals (all generic — no document-specific rules):
  - numbers, percentages, currency symbols (₹ $ £ € ¥)
  - financial keywords (revenue, profit, EBITDA, margin, growth, etc.)
  - temporal keywords (FY, Q1-Q4, fiscal year, year-over-year, 20XX)
  - table-like structures (pipe-separated or tab-separated rows)
"""
import re
import fitz  # PyMuPDF
import logging
from pathlib import Path

logger = logging.getLogger("factloom.parse")

WINDOW_SIZE = 8           # pages per context window sent to the LLM
CHUNK_CHAR_LIMIT = 12000  # ~3-4k tokens; safe for extraction prompts

# --- Candidate detection patterns ---

_NUMERIC_RE = re.compile(r"\b\d[\d,]*\.?\d*\s*(%|percent|bps|basis\s+points)?\b", re.IGNORECASE)
_CURRENCY_RE = re.compile(r"[₹\$£€¥]|(?:INR|USD|GBP|EUR|JPY|Rs\.?)\s*[\d,]", re.IGNORECASE)
_PERIOD_RE = re.compile(
    r"\bFY\s*'?\d{2,4}\b"              # FY2024, FY'24
    r"|\bQ[1-4]\b"                      # Q1, Q2, Q3, Q4
    r"|\b(fiscal\s+year|year[-\s]over[-\s]year|YoY|QoQ)\b"
    r"|\b20\d{2}\b",                    # 2020, 2023, etc.
    re.IGNORECASE,
)
_FINANCIAL_KEYWORDS_RE = re.compile(
    r"\b(revenue|profit|loss|margin|ebitda|earnings|growth|decline|income"
    r"|turnover|sales|expenditure|expense|cost|debt|equity|dividend|yield"
    r"|cagr|roi|roa|roe|eps|pat|pbt|ebit|capex|opex|npa|gdp|inflation"
    r"|repo\s+rate|current\s+account|fiscal\s+deficit|forex|reserves"
    r"|shipment|volume|metric|tonne|kg|km)\b",
    re.IGNORECASE,
)
_TABLE_RE = re.compile(r"(\|.+\|)|(\t.+\t)", re.MULTILINE)


def _is_candidate_page(text: str) -> bool:
    """Return True if the page likely contains extractable quantitative claims."""
    if not text or len(text.strip()) < 30:
        return False
    # At least one numeric signal AND one contextual signal (keyword/period/currency)
    has_number = bool(_NUMERIC_RE.search(text))
    has_context = (
        bool(_CURRENCY_RE.search(text))
        or bool(_PERIOD_RE.search(text))
        or bool(_FINANCIAL_KEYWORDS_RE.search(text))
        or bool(_TABLE_RE.search(text))
    )
    return has_number and has_context


def _extract_pages(pdf_path: str) -> list[dict]:
    pdf = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(pdf)):
        text = pdf[page_num].get_text("text")
        cleaned = " ".join(text.split())
        pages.append({"page_no": page_num + 1, "text": cleaned})
    pdf.close()
    return [p for p in pages if p["text"]]


def _windows(pages: list[dict], size: int) -> list[list[dict]]:
    return [pages[i:i + size] for i in range(0, len(pages), size)]


def _split_window_to_chunks(document_id: str, window: list[dict], chunk_char_limit: int) -> list[dict]:
    """Split a page window into char-safe chunks, never splitting mid-page."""
    chunks = []
    current_pages: list[dict] = []
    current_len = 0

    def flush():
        if not current_pages:
            return
        text_parts = [f"[page {p['page_no']}]\n{p['text']}" for p in current_pages]
        chunks.append({
            "document_id": document_id,
            "start_page": current_pages[0]["page_no"],
            "end_page": current_pages[-1]["page_no"],
            "page_map": [p["page_no"] for p in current_pages],
            "text": "\n\n".join(text_parts),
        })

    for page in window:
        page_len = len(page["text"])
        if page_len > chunk_char_limit:
            flush()
            current_pages, current_len = [], 0
            truncated = page["text"][:chunk_char_limit]
            chunks.append({
                "document_id": document_id,
                "start_page": page["page_no"],
                "end_page": page["page_no"],
                "page_map": [page["page_no"]],
                "text": f"[page {page['page_no']}]\n{truncated}",
            })
            continue

        if current_len + page_len > chunk_char_limit:
            flush()
            current_pages, current_len = [], 0

        current_pages.append(page)
        current_len += page_len

    flush()
    return chunks


def parse_pdf_to_chunks(pdf_path: str) -> list[dict]:
    """Return candidate chunks for LLM extraction.

    Only pages with numeric/financial signals are included.
    Each chunk dict has: document_id, chunk_id, start_page, end_page, page_map, text.
    """
    doc_id = Path(pdf_path).stem
    all_pages = _extract_pages(pdf_path)
    if not all_pages:
        return []

    # Candidate detection: filter to pages with quantitative signals
    candidate_pages = [p for p in all_pages if _is_candidate_page(p["text"])]
    dropped = len(all_pages) - len(candidate_pages)
    logger.info(
        "doc=%s  pages=%d  candidate_pages=%d  dropped_by_filter=%d",
        doc_id, len(all_pages), len(candidate_pages), dropped,
    )

    if not candidate_pages:
        logger.warning("doc=%s: no candidate pages detected — falling back to all pages", doc_id)
        candidate_pages = all_pages

    all_chunks = []
    for window in _windows(candidate_pages, WINDOW_SIZE):
        all_chunks.extend(_split_window_to_chunks(doc_id, window, CHUNK_CHAR_LIMIT))

    for i, chunk in enumerate(all_chunks):
        chunk["chunk_id"] = f"{doc_id}_c{i:03d}"

    logger.info("doc=%s  chunks_to_llm=%d", doc_id, len(all_chunks))
    return all_chunks
