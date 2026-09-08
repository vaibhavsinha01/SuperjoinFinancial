"""
Explicit validation gates between pipeline stages. Each function takes a raw dict
(or list of dicts) and returns validated Pydantic objects, dropping/logging anything
malformed instead of letting it propagate. Nothing here raises on bad input from an
individual item — a single bad fact should not abort processing of a whole document.
"""
import logging
from pydantic import ValidationError

from backend.models import Fact, NormalizedFact, EmbeddingRecord, Relation

logger = logging.getLogger("factloom.validate")


def validate_facts(raw_facts: list[dict]) -> list[Fact]:
    """Post-extraction gate. Drops facts missing required fields or with an out-of-range page."""
    valid = []
    for rf in raw_facts:
        try:
            valid.append(Fact(**rf))
        except ValidationError as e:
            logger.warning("dropped invalid fact: %s | errors=%s", rf, e.errors())
    return valid


def validate_normalized(raw: list[dict]) -> list[NormalizedFact]:
    """Post-normalization gate."""
    valid = []
    for rn in raw:
        try:
            valid.append(NormalizedFact(**rn))
        except ValidationError as e:
            logger.warning("dropped invalid normalized fact: %s | errors=%s", rn, e.errors())
    return valid


def validate_embedding(raw: dict) -> EmbeddingRecord | None:
    """Post-embedding gate for a single record. Returns None (and logs) if malformed."""
    try:
        return EmbeddingRecord(**raw)
    except ValidationError as e:
        logger.warning("invalid embedding record: fact_id=%s errors=%s", raw.get("fact_id"), e.errors())
        return None


def validate_relation(raw: dict) -> Relation | None:
    """Post-comparison gate. Returns None (and logs) if the LLM's structured output is malformed."""
    try:
        return Relation(**raw)
    except ValidationError as e:
        logger.warning("dropped invalid relation: %s | errors=%s", raw, e.errors())
        return None


def validate_chunk(chunk: dict) -> bool:
    """Pre-extraction sanity check on a parsed chunk before it's sent to the LLM."""
    required = ("document_id", "chunk_id", "start_page", "end_page", "text")
    if not all(k in chunk for k in required):
        logger.warning("dropped malformed chunk (missing keys): %s", list(chunk.keys()))
        return False
    if chunk["start_page"] < 1 or chunk["end_page"] < chunk["start_page"]:
        logger.warning("dropped chunk with invalid page range: %s-%s", chunk["start_page"], chunk["end_page"])
        return False
    if not chunk["text"].strip():
        logger.warning("dropped empty chunk: %s", chunk.get("chunk_id"))
        return False
    return True
