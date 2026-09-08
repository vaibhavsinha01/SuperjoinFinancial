# FactLoom

A fact knowledge layer: upload PDFs, extract grounded facts, and see how facts across
documents corroborate, contradict, or reconcile through context.

## Setup and Run Instructions

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# put your GEMINI_API_KEY in .env

uvicorn backend.main:app --reload
```

Open http://localhost:8000 — upload a PDF, browse extracted facts, click a fact to see
its source quote and any related facts from other documents.

Sample PDFs are in `data/starter-datasets/` (Delhivery filings + India macroeconomy reports).

## Approach

**Pipeline:** PDF → page-level text (PyMuPDF) → per-page LLM extraction (Gemini) into
structured facts (`entity, metric, value, unit, period, scope, quote, confidence`) →
embed each fact (`text-embedding-004`) → for each new fact, find similar existing facts
via cosine similarity → LLM classifies each candidate pair as `corroborates`,
`contradicts`, `reconcilable`, or `unrelated` with an explanation.

**Why embeddings before LLM comparison:** comparing every fact against every other fact
with an LLM call is O(n²) and slow. Embedding each fact and taking top-k cosine-similar
candidates turns this into a cheap retrieval step before the expensive reasoning step —
the same retrieve-then-reason pattern as RAG, applied to fact matching instead of
document Q&A. At this scale (tens–low hundreds of facts) plain numpy cosine similarity
is fast enough that a vector DB (FAISS etc.) isn't warranted.

**Storage:** SQLite, three tables — `documents`, `facts`, `relations`. No graph database;
a graph layer alone doesn't do the reasoning the assignment asks for, so relations are
computed and explained, then just stored as rows.

**Schema stays loose:** `scope` and `period` are free-text, LLM-populated fields rather
than a fixed enum, so the extractor isn't hardcoded to Delhivery/India-macro concepts and
should generalize to other PDFs.

**AI tools used:** Gemini 2.0 Flash for extraction and comparison reasoning,
`text-embedding-004` for fact embeddings.

## The Four Required Cases

*(fill in after running the demo PDFs — pick one clear example of each from the UI)*

1. **Corroborated fact:** e.g. Delhivery FY24 revenue stated similarly in the annual
   report and the earnings presentation, worded differently.
2. **Contradiction:** e.g. a figure that differs between two documents with no
   time/scope/unit difference to explain it.
3. **Reconciled via context:** e.g. two revenue figures differing because one is
   consolidated and one standalone, or different fiscal periods.
4. **Failure case:** *(document whatever extraction/reasoning miss you observe —
   e.g. a fact misattributed to the wrong entity, or a real relationship the
   similarity threshold missed because the two facts were phrased too differently
   to embed close together)*.

## Limitations and Next Steps

- Extraction confidence isn't yet surfaced/filterable in the UI.
- Cosine similarity threshold (0.72) is a heuristic — may miss true matches phrased very
  differently, or over-match on generic financial boilerplate.
- No incremental re-embedding cache — re-uploading the same PDF reprocesses it fully.
- Next: incremental ingestion (only compare new facts against existing embeddings index,
  already partially true), table-aware extraction (current parsing is plain text, so
  facts locked in complex tables may be lost or garbled).

## Additional Notes

Starter datasets included under `data/starter-datasets/` for convenience.

## Architecture v2 (production-hardening pass)

The core pipeline is now:

```
PDF → parse (page-window chunking, 8 pages/window, char-safe splits)
    → LLM extraction (Gemini → Groq fallback, retry+backoff, cached)
    → validate
    → normalize (canonical entity/metric/unit/period)
    → validate
    → local embeddings (sentence-transformers, status-tracked)
    → validate
    → SQLite storage (facts + normalized fields + embedding status)
    → metadata filtering (by normalized metric)
    → embedding similarity top-k
    → LLM relationship classification (corroborates/contradicts/reconcilable/unrelated + reason)
    → validate
    → stored relation with confidence, reason, evidence
```

### Key files

- `backend/providers/{base,gemini,groq}.py` — provider interface + implementations
- `backend/llm.py` — retry (exponential backoff + jitter, ~4 attempts/provider) →
  Gemini → Groq fallback → graceful `LLMUnavailableError`. Hard quota exhaustion
  (daily limit) skips retries and falls through to the next provider immediately;
  transient errors (429/500/502/503/504, network issues) are retried.
- `backend/cache.py` — SQLite cache of LLM responses keyed by `sha256(prompt)`.
- `backend/models.py` — Pydantic schemas for every stage (`Fact`, `NormalizedFact`,
  `EmbeddingRecord` with `pending/success/failed` status, `Relation`).
- `backend/validate.py` — validation gate called after each stage; drops/logs bad
  items instead of crashing the whole document.
- `backend/normalize.py` — canonical entity/metric/unit/period mapping; preserves
  `original_value`/`original_unit` for provenance.
- `backend/parse.py` — pools pages into ~8-page windows, splits into char-safe chunks
  without ever crossing a page boundary mid-page; every chunk keeps `start_page`,
  `end_page`, `page_map` so the LLM (and the stored fact) can cite an exact page.
- `backend/embed.py` — local embeddings (`BAAI/bge-small-en-v1.5` by default),
  explicit status instead of `[]`, safe parsing of `None`/malformed JSON/zero vectors.
  Set `EMBEDDING_OFFLINE_FALLBACK=true` to use a lexical hashed-bag-of-words vector
  instead, for environments without internet access to huggingface.co (used only for
  development/testing in this sandbox — leave unset on a normal machine).
- `backend/retrieve.py` — metadata filter (by normalized metric) before embedding
  similarity, so unrelated metrics never reach the LLM comparison step.
- `backend/store.py` — schema migration via `ALTER TABLE` (checks `PRAGMA table_info`
  and adds missing columns), so an existing `factloom.db` upgrades in place.
- `backend/eval.py` — runs the 6 starter documents through the full pipeline; use to
  manually verify the 4 relationship cases.

### Why not LlamaIndex

The pipeline here is a single, fairly linear flow (parse → extract → normalize →
embed → retrieve → compare) over a small, fixed corpus (a handful of PDFs). LlamaIndex's
value is mainly in managing many heterogeneous data connectors, complex multi-index
retrieval, and agentic query routing — none of which this project needs. Its chunking
utilities also don't preserve the page-exact provenance this project requires without
extra wrapper code, which would end up being roughly the same amount of code as
`parse.py` already is. A custom ~100-line parser stays easier to reason about, debug,
and extend than adopting a general-purpose framework for a narrowly-scoped pipeline.

### Known limitations / TODOs

- `backend/main.py` (`/upload`) still calls `process_pdf` synchronously in the request
  handler — fine for the eval script, but for the API a background task queue would
  stop a large PDF from blocking the HTTP request.
- Normalization's metric/unit alias tables (`normalize.py`) are hand-curated and cover
  the metrics likely to appear in the starter dataset; a larger deployment would want a
  more systematic canonicalization approach (e.g. LLM-assisted metric clustering).
- No automated test suite (pytest) yet — validation was done via a mocked pipeline run
  (see conversation) and is documented in this README, but should be turned into
  `tests/test_normalize.py`, `tests/test_parse.py`, `tests/test_embed.py` etc.
- `backend/eval.py` has not been run against live Gemini/Groq APIs in this environment
  because network egress here doesn't allow `generativelanguage.googleapis.com` /
  `api.groq.com`. Run it on your machine with a real `.env` to get the actual 4-case
  results against the Delhivery + macro documents.
- Groq's `response_format=json_object` requires a JSON *object*, so the extraction
  prompt (which wants a JSON array) is asked to wrap arrays as `{"items": [...]}` when
  Groq is used as fallback for extraction — this is unwrapped in `providers/groq.py`
  but hasn't been tested against a live Groq response for the extraction prompt
  specifically (only implemented via code review).
