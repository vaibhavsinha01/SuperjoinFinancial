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

WINDOW_SIZE = 1           # Focused chunking: 1-2 pages per context window for high extraction fidelity
CHUNK_CHAR_LIMIT = 5000   # Safe, fast, and exhaustive for extraction prompts
PAGE_MERGE_LIMIT = 3500   # Merge consecutive short pages only if combined text stays under this

# --- Candidate detection patterns ---

_NUMERIC_RE = re.compile(r"\b\d[\d,]*\.?\d*\s*(%|percent|bps|basis\s+points)?\b", re.IGNORECASE)
_CURRENCY_RE = re.compile(r"[₹\$£€¥]|(?:INR|USD|GBP|EUR|JPY|Rs\.?)\s*[\d,]", re.IGNORECASE)
_PERIOD_RE = re.compile(
    r"\bFY\s*'?\d{2,4}\b"              # FY2024, FY'24
    r"|\bCY\s*'?\d{2,4}\b"              # CY14, CY15E
    r"|\bQ[1-4]\b"                      # Q1, Q2, Q3, Q4
    r"|\b(fiscal\s+year|year[-\s]over[-\s]year|YoY|QoQ)\b"
    r"|\b20\d{2}\b"                     # 2020, 2023, etc.
    r"|\b\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+20\d{2}\b",
    re.IGNORECASE,
)
_FINANCIAL_KEYWORDS_RE = re.compile(
    r"\b(revenue|profit|loss|margin|ebitda|earnings|growth|decline|income"
    r"|turnover|sales|expenditure|expense|cost|debt|equity|dividend|yield"
    r"|cagr|roi|roa|roe|eps|pat|pbt|ebit|capex|opex|npa|gnpa|gdp|inflation"
    r"|repo\s+rate|current\s+account|fiscal\s+deficit|forex|reserves"
    r"|shipment|volume|metric|tonne|kg|km"
    r"|market\s+cap|target\s+price|cmp|current\s+market\s+price|52[-\s]week"
    r"|shareholding|promoter|promoters|institutional|aum|nim|asset|assets"
    r"|liability|liabilities|borrowing|borrowings|cash\s+flow|valuation"
    r"|stop\s+loss|upside|contraire|financials|balance\s+sheet|p&l|ratio|pe|pb)\b",
    re.IGNORECASE,
)
_TABLE_RE = re.compile(r"(\|.+\|)|(\t.+\t)|(\b(CY|FY)\d{2}.*\b(Revenue|EBITDA|PAT|Sales|EPS)\b)", re.IGNORECASE)


def _clean_page_text(raw_text: str) -> str:
    """Clean extracted page text while preserving line breaks and table layout."""
    if not raw_text:
        return ""
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    cleaned_lines = []
    blank_count = 0
    for line in lines:
        if not line:
            blank_count += 1
            if blank_count <= 1:
                cleaned_lines.append("")
        else:
            blank_count = 0
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def _is_candidate_page(text: str) -> bool:
    """Return True if the page likely contains extractable quantitative claims."""
    if not text or len(text.strip()) < 30:
        return False
    # At least one numeric signal AND one contextual signal (keyword/period/currency/table)
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
        cleaned = _clean_page_text(text)
        pages.append({"page_no": page_num + 1, "text": cleaned})
    pdf.close()
    return [p for p in pages if p["text"]]


def _split_long_page(document_id: str, page: dict, chunk_char_limit: int) -> list[dict]:
    """Split an oversized single page along paragraph boundaries without losing text."""
    text = page["text"]
    page_no = page["page_no"]
    if len(text) <= chunk_char_limit:
        return [{
            "document_id": document_id,
            "start_page": page_no,
            "end_page": page_no,
            "page_map": [page_no],
            "text": f"[page {page_no}]\n{text}",
        }]

    paragraphs = text.split("\n\n")
    chunks = []
    curr_parts = []
    curr_len = 0

    for para in paragraphs:
        p_len = len(para)
        if curr_len + p_len > chunk_char_limit and curr_parts:
            chunk_text = "\n\n".join(curr_parts)
            chunks.append({
                "document_id": document_id,
                "start_page": page_no,
                "end_page": page_no,
                "page_map": [page_no],
                "text": f"[page {page_no}]\n{chunk_text}",
            })
            curr_parts = []
            curr_len = 0
        curr_parts.append(para)
        curr_len += p_len + 2

    if curr_parts:
        chunk_text = "\n\n".join(curr_parts)
        chunks.append({
            "document_id": document_id,
            "start_page": page_no,
            "end_page": page_no,
            "page_map": [page_no],
            "text": f"[page {page_no}]\n{chunk_text}",
        })
    return chunks


def _pages_to_chunks(document_id: str, pages: list[dict], chunk_char_limit: int = CHUNK_CHAR_LIMIT) -> list[dict]:
    """Build chunks from candidate pages. Merges consecutive short pages up to PAGE_MERGE_LIMIT;
    otherwise 1 chunk per page for complete attention and accurate provenance."""
    chunks = []
    idx = 0
    while idx < len(pages):
        page = pages[idx]
        page_len = len(page["text"])

        # If page is extra long, split it safely along paragraph boundaries
        if page_len > chunk_char_limit:
            chunks.extend(_split_long_page(document_id, page, chunk_char_limit))
            idx += 1
            continue

        # Try to merge with adjacent consecutive page if both are short
        merged_pages = [page]
        curr_len = page_len
        while (
            idx + 1 < len(pages)
            and pages[idx + 1]["page_no"] == merged_pages[-1]["page_no"] + 1
            and curr_len + len(pages[idx + 1]["text"]) <= PAGE_MERGE_LIMIT
            and len(merged_pages) < 2
        ):
            next_p = pages[idx + 1]
            merged_pages.append(next_p)
            curr_len += len(next_p["text"])
            idx += 1

        text_parts = [f"[page {p['page_no']}]\n{p['text']}" for p in merged_pages]
        chunks.append({
            "document_id": document_id,
            "start_page": merged_pages[0]["page_no"],
            "end_page": merged_pages[-1]["page_no"],
            "page_map": [p["page_no"] for p in merged_pages],
            "text": "\n\n".join(text_parts),
        })
        idx += 1

    return chunks


# PDFs with more pages than this go through embedding-based chunk retrieval before
# extraction, instead of sending every candidate chunk to the LLM. Small PDFs use the
# normal pipeline (all candidate chunks) unchanged.
LARGE_PDF_PAGE_THRESHOLD = 7

# Generic (not document-specific) queries used to retrieve the most relevant chunks of a
# large PDF via local embeddings before extraction. These describe the *kind* of content
# FactLoom extracts facts about, not any particular company/document.
_RETRIEVAL_QUERIES = [
    "revenue, profit, and earnings figures",
    "financial results and key performance metrics",
    "growth rates, margins, and ratios",
    "balance sheet, assets, liabilities, and debt",
    "macroeconomic indicators such as GDP, inflation, and rates",
    "reporting period, fiscal year, and quarterly figures",
]

# How many chunks (max) to keep for a large PDF after retrieval. Kept generous relative
# to a typical LLM context window since each chunk is already page-scoped and filtered
# to candidate (numeric/financial) pages only.
MAX_RETRIEVED_CHUNKS = 25


def _filter_chunks_by_retrieval(chunks: list[dict], doc_id: str) -> list[dict]:
    """For large PDFs: rank candidate chunks by embedding similarity to a set of generic
    financial/quantitative queries and keep only the top-scoring ones, so the LLM sees a
    retrieved subset of chunks rather than the entire document. Page numbers (page_map,
    start_page, end_page) are untouched, so evidence stays traceable to exact pages.

    Falls back to returning all chunks unchanged if local embeddings are unavailable for
    any reason (e.g. model failed to load) — large-PDF handling should degrade gracefully,
    never crash the upload.
    """
    if len(chunks) <= MAX_RETRIEVED_CHUNKS:
        return chunks

    try:
        from backend import embed as embed_mod

        # Embed each chunk's text (truncate very long chunk text for embedding speed only;
        # the full text is still sent to the LLM for any chunk that's kept).
        chunk_vectors = []
        for c in chunks:
            rec = embed_mod.embed_text(c["text"][:2000], fact_id=None)
            chunk_vectors.append(rec.vector if rec.status.value == "success" else None)

        if not any(chunk_vectors):
            logger.warning("doc=%s: no chunk embeddings available, skipping retrieval filter", doc_id)
            return chunks

        # Score each chunk as its best similarity across all retrieval queries.
        import numpy as np
        query_vectors = []
        for q in _RETRIEVAL_QUERIES:
            rec = embed_mod.embed_text(q, fact_id=None)
            if rec.status.value == "success":
                query_vectors.append(np.array(rec.vector, dtype=float))
        if not query_vectors:
            logger.warning("doc=%s: no query embeddings available, skipping retrieval filter", doc_id)
            return chunks

        scored = []
        for chunk, vec in zip(chunks, chunk_vectors):
            if vec is None:
                score = -1.0  # keep unembeddable chunks out unless we're short on chunks
            else:
                v = np.array(vec, dtype=float)
                score = max(embed_mod.cosine_sim(v.tolist(), q.tolist()) for q in query_vectors)
            scored.append((score, chunk))

        scored.sort(key=lambda x: -x[0])
        kept = [c for _, c in scored[:MAX_RETRIEVED_CHUNKS]]
        # Preserve original document order so multi-page context reads naturally.
        kept_ids = {c["chunk_id"] if "chunk_id" in c else id(c) for c in kept}
        ordered_kept = [c for c in chunks if (c.get("chunk_id") if "chunk_id" in c else id(c)) in kept_ids]

        logger.info(
            "doc=%s: large-PDF retrieval filter kept %d/%d chunks (threshold=%d pages)",
            doc_id, len(ordered_kept), len(chunks), LARGE_PDF_PAGE_THRESHOLD,
        )
        return ordered_kept
    except Exception as e:
        logger.warning("doc=%s: retrieval filter failed (%s), falling back to all chunks", doc_id, e)
        return chunks


def parse_pdf_to_chunks(pdf_path: str) -> list[dict]:
    """Return candidate chunks for LLM extraction.

    Only pages with numeric/financial signals are included.
    Each chunk dict has: document_id, chunk_id, start_page, end_page, page_map, text.

    Small PDFs (<= LARGE_PDF_PAGE_THRESHOLD pages): normal pipeline — every candidate
    chunk is sent to the LLM, as before.

    Large PDFs (> LARGE_PDF_PAGE_THRESHOLD pages): candidate chunks are additionally
    ranked and filtered via local embeddings + retrieval before being sent to the LLM,
    so the whole document is never dumped into a single extraction pass. Page numbers
    remain attached to every retrieved chunk (start_page/end_page/page_map), so evidence
    stays traceable back to the exact source page. This never changes which LLM
    provider is used — only how much text reaches it.
    """
    doc_id = Path(pdf_path).stem
    all_pages = _extract_pages(pdf_path)
    if not all_pages:
        return []

    total_pages = len(all_pages)
    is_large_pdf = total_pages > LARGE_PDF_PAGE_THRESHOLD

    # Candidate detection: filter to pages with quantitative signals
    candidate_pages = [p for p in all_pages if _is_candidate_page(p["text"])]
    dropped = len(all_pages) - len(candidate_pages)
    logger.info(
        "doc=%s  pages=%d  candidate_pages=%d  dropped_by_filter=%d  large_pdf=%s",
        doc_id, total_pages, len(candidate_pages), dropped, is_large_pdf,
    )

    if not candidate_pages:
        logger.warning("doc=%s: no candidate pages detected — falling back to all pages", doc_id)
        candidate_pages = all_pages

    all_chunks = _pages_to_chunks(doc_id, candidate_pages, CHUNK_CHAR_LIMIT)

    for i, chunk in enumerate(all_chunks):
        chunk["chunk_id"] = f"{doc_id}_c{i:03d}"

    if is_large_pdf:
        all_chunks = _filter_chunks_by_retrieval(all_chunks, doc_id)

    logger.info("doc=%s  chunks_to_llm=%d  large_pdf=%s", doc_id, len(all_chunks), is_large_pdf)
    return all_chunks
