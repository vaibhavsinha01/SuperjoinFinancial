"""
Full per-document pipeline:

    parse -> chunks
    for each chunk: LLM extract -> validate -> normalize -> validate -> embed -> validate -> store
"""
import os
import json
import logging

from backend import store, llm, embed as embed_mod, normalize as norm_mod, validate as val
from backend.parse import parse_pdf_to_chunks
from backend.models import EmbeddingStatus

logger = logging.getLogger("factloom.extract")

PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "extract_facts.txt")
with open(PROMPT_PATH) as f:
    EXTRACT_PROMPT = f.read()


def _extract_chunk_facts(chunk: dict) -> list[dict]:
    """Call the LLM on one chunk and return raw fact dicts (page_no filled in per-fact
    by the model itself, since a chunk spans multiple pages)."""
    prompt = EXTRACT_PROMPT.format(
        document_id=chunk["document_id"],
        start_page=chunk["start_page"],
        end_page=chunk["end_page"],
        chunk_text=chunk["text"],
    )
    try:
        raw = llm.generate_json(prompt, call_type="extraction")
    except llm.LLMUnavailableError as e:
        logger.error("chunk %s extraction failed, all providers exhausted: %s", chunk["chunk_id"], e)
        return []

    if not isinstance(raw, list):
        logger.warning("chunk %s: expected list, got %s", chunk["chunk_id"], type(raw))
        return []

    out = []
    for rf in raw:
        if not isinstance(rf, dict):
            continue
        page_no = rf.get("page_no")
        # guard against the model returning a page outside this chunk's actual range
        if not isinstance(page_no, int) or page_no not in chunk["page_map"]:
            page_no = chunk["start_page"]
        out.append({
            "document_id": chunk["document_id"],
            "chunk_id": chunk["chunk_id"],
            "page_no": page_no,
            "entity": str(rf.get("entity", "")).strip(),
            "metric": str(rf.get("metric", "")).strip(),
            "value": str(rf.get("value", "")).strip(),
            "unit": rf.get("unit"),
            "period": rf.get("period"),
            "scope": rf.get("scope"),
            "quote": str(rf.get("quote", "")).strip(),
            "confidence": rf.get("confidence", "medium"),
            "is_reported_value": rf.get("is_reported_value", True),
        })
    return out


def process_pdf(pdf_path: str) -> list[dict]:
    """Run the full pipeline for one PDF. Returns stored facts (as dicts, DB rows)."""
    import fitz

    chunks_raw = parse_pdf_to_chunks(pdf_path)
    chunks = [c for c in chunks_raw if val.validate_chunk(c)]
    if not chunks:
        return []

    document_id = chunks[0]["document_id"]
    num_pages = fitz.open(pdf_path).page_count
    store.add_document(document_id, os.path.basename(pdf_path), num_pages)

    stored_facts = []
    try:
        for chunk in chunks:
            raw_facts = _extract_chunk_facts(chunk)
            # ---- validate (post-extraction) ----
            valid_facts = val.validate_facts(raw_facts)

            for fact in valid_facts:
                # ---- normalize ----
                normalized = norm_mod.normalize_fact(fact)
                # ---- validate (post-normalization); normalized may be None for non-numeric facts ----
                norm_fields = {}
                if normalized is not None:
                    validated_norm = val.validate_normalized([normalized.model_dump(exclude={"id"})])
                    if validated_norm:
                        n = validated_norm[0]
                        norm_fields = {
                            "norm_metric": n.metric, "norm_value": n.value,
                            "norm_unit": n.unit, "norm_period": n.period, "norm_scope": n.scope,
                        }

                # ---- embed ----
                embed_input = f"{fact.entity} | {fact.metric} | {fact.value} {fact.unit or ''} | {fact.period or ''} | {fact.scope or ''}"
                emb_record = embed_mod.embed_text(embed_input)
                # ---- validate (post-embedding) ----
                validated_emb = val.validate_embedding(emb_record.model_dump())

                embedding_fields = {
                    "embedding_status": (validated_emb.status.value if validated_emb else EmbeddingStatus.FAILED.value),
                    "embedding_model": validated_emb.model if validated_emb else None,
                    "embedding_error": validated_emb.error if validated_emb and validated_emb.status == EmbeddingStatus.FAILED else emb_record.error,
                    "embedding_json": json.dumps(validated_emb.vector) if (validated_emb and validated_emb.vector) else None,
                }

                fact_dict = fact.model_dump(exclude={"id"})
                fact_dict.update(norm_fields)
                fact_dict.update(embedding_fields)

                # ---- store ----
                fact_id = store.add_fact(fact_dict)
                fact_dict["id"] = fact_id
                fact_dict["embedding"] = embedding_fields["embedding_json"]
                stored_facts.append(fact_dict)

        store.set_document_status(document_id, "complete")
    except Exception as e:
        logger.exception("document %s processing failed", document_id)
        store.set_document_status(document_id, "failed", str(e))

    return stored_facts
