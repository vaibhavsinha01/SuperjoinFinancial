"""
Retrieval pipeline: metadata filtering → FAISS semantic search → top-k.

Flow:
    all facts
     ↓
    exclude same-document facts
     ↓
    metadata filter (normalized metric match)
     ↓
    FAISS inner-product search on L2-normalized embeddings
     ↓
    validate results (similarity range, no self-candidates)
     ↓
    top-k candidates for relationship reasoning

cosine similarity ≈ inner product when vectors are L2-normalized (see embed.py).
"""
import logging
from backend import embed as embed_mod

logger = logging.getLogger("factloom.retrieve")


def _metadata_filter(fact: dict, candidates: list[dict]) -> list[dict]:
    """Keep only candidates whose normalized metric matches the query fact.
    Falls back to no filtering when the fact has no normalized metric
    (non-numeric facts) so we don't lose recall unnecessarily."""
    fact_metric = fact.get("norm_metric")
    if not fact_metric:
        return candidates  # no metric → can't filter, rely on embeddings alone
    return [c for c in candidates if c.get("norm_metric") == fact_metric]


def retrieve_candidates(fact: dict, all_facts: list[dict], k: int = 5, threshold: float = 0.55) -> list[dict]:
    """Full retrieval: exclude same-document → metadata filter → FAISS top-k.

    Validation:
    - Excludes self (same fact_id)
    - Excludes same-document facts
    - Requires similarity >= threshold
    - Only returns facts with valid IDs
    """
    fact_id = fact.get("id")
    doc_id = fact.get("document_id")

    # Exclude same-document facts (cross-document comparison only)
    cross_doc = [f for f in all_facts if f.get("document_id") != doc_id and f.get("id") != fact_id]
    if not cross_doc:
        logger.info("fact_id=%s: no cross-document candidates available", fact_id)
        return []

    # Metadata filter
    metadata_filtered = _metadata_filter(fact, cross_doc)
    if not metadata_filtered:
        logger.info("fact_id=%s: no metadata-matching candidates (norm_metric=%s)", fact_id, fact.get("norm_metric"))
        return []

    # FAISS search via top_similar (which uses the module-level FAISS index)
    results = embed_mod.top_similar(fact, metadata_filtered, k=k, threshold=threshold)

    # Validate results: must have valid IDs and not be self
    validated = [
        r for r in results
        if r.get("id") is not None and r.get("id") != fact_id
    ]

    if len(validated) != len(results):
        logger.warning("fact_id=%s: dropped %d invalid candidates", fact_id, len(results) - len(validated))

    logger.debug("fact_id=%s: retrieved %d candidates", fact_id, len(validated))
    return validated
