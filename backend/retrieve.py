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

from backend.normalize import clean_entity

logger = logging.getLogger("factloom.retrieve")


def retrieve_candidates(fact: dict, all_facts: list[dict], k: int = 5, threshold: float = 0.50) -> list[dict]:
    """Retrieve cross-document candidate facts for relation comparison.

    Pipeline:
    1. Filter out same-document facts and self.
    2. Entity compatibility: If fact has a company entity, prioritize facts from the same company.
    3. Semantic and metric ranking via top_similar.
    """
    fact_id = fact.get("id")
    doc_id = fact.get("document_id")

    # Exclude same-document facts (cross-document comparison only)
    cross_doc = [f for f in all_facts if f.get("document_id") != doc_id and f.get("id") != fact_id]
    if not cross_doc:
        logger.info("fact_id=%s: no cross-document candidates available", fact_id)
        return []

    # Entity compatibility:
    # If the fact has an entity, prioritize candidates with the same normalized entity.
    fact_entity = clean_entity(fact.get("entity"))
    if fact_entity:
        same_entity = [
            c for c in cross_doc
            if clean_entity(c.get("entity")) == fact_entity
        ]
        candidates = same_entity if same_entity else cross_doc
    else:
        candidates = cross_doc

    # Search top similar candidates
    results = embed_mod.top_similar(fact, candidates, k=k, threshold=threshold)

    # Validate results: must have valid IDs and not be self
    validated = [
        r for r in results
        if r.get("id") is not None and r.get("id") != fact_id
    ]

    if len(validated) != len(results):
        logger.warning("fact_id=%s: dropped %d invalid candidates", fact_id, len(results) - len(validated))

    logger.debug("fact_id=%s: retrieved %d candidates", fact_id, len(validated))
    return validated
