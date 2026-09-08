"""
Relationship classification:
    FAISS retrieval → deterministic pre-check → LLM comparison → validate → store

Architecture:
    candidate pair
         ↓
    metadata comparison (same entity/metric/unit/period?)
         ↓
    numeric comparison (values within tolerance?)
         ↓
    obvious deterministic result?
          /           \\
        YES             NO
         ↓               ↓
    store result    LLM (Groq primary → Gemini fallback)
                         ↓
                    validate relation
                         ↓
                    store result

Deterministic logic conserves LLM calls for cases that can be resolved structurally.
LLM is used for semantic/contextual ambiguity only.
"""
import os
import logging

from backend import store, llm, validate as val
from backend.retrieve import retrieve_candidates
from backend.models import Relation, RelationType, ReconcileReason

logger = logging.getLogger("factloom.compare")

PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "compare_facts.txt")
with open(PROMPT_PATH) as f:
    COMPARE_PROMPT = f.read()

# Numeric tolerance for corroboration (1% relative difference)
_CORROBORATE_TOLERANCE = 0.01


def _deterministic_compare(fact_a: dict, fact_b: dict) -> Relation | None:
    """Attempt to classify the relationship between two facts without an LLM call.

    Returns a Relation if the classification can be made deterministically,
    or None if the LLM should handle it.

    Deterministic cases:
    - Same entity + metric + unit + scope (if stated), numerically within 1% → CORROBORATES
    - Same entity + metric + unit + scope (if stated), different period only → RECONCILABLE (different_reporting_period)
    - Same entity + metric + unit + scope (if stated), same period, values differ >1% → RECONCILABLE (updated_information)
    - Scope stated on both sides and differs → always deferred to the LLM (needs semantic judgment)
    """
    from backend.normalize import clean_entity
    na, nb = fact_a.get("norm_metric"), fact_b.get("norm_metric")
    ua, ub = fact_a.get("norm_unit"), fact_b.get("norm_unit")
    va, vb = fact_a.get("norm_value"), fact_b.get("norm_value")
    pa, pb = fact_a.get("norm_period"), fact_b.get("norm_period")
    ea, eb = clean_entity(fact_a.get("entity")), clean_entity(fact_b.get("entity"))

    # All required normalized fields must be present
    if not all([na, ua, va is not None, vb is not None, ea, eb]):
        return None

    # Entities must match (canonical cleaned form)
    if ea != eb:
        return None

    # Metrics must match
    if na != nb:
        return None

    # Units must match (or one is unitless, or equivalent currencies)
    units_match = (
        (ua == ub)
        or ua == "unitless"
        or ub == "unitless"
        or {ua, ub} <= {"inr", "rs"}
        or {ua, ub} <= {"usd", "$"}
    )
    if not units_match:
        return None

    # Scope must match when both facts explicitly state one. Different scope (e.g.
    # consolidated vs standalone, India vs global) is a common source of false
    # "corroborates" calls — if both sides state a real scope and it differs, this needs
    # semantic judgment (is a small value difference expected given the scope difference,
    # or not?), so hand off to the LLM rather than deciding deterministically. Facts with
    # no stated scope are normalized to "unspecified" (see normalize.py) and are not
    # treated as a real scope value here, so two same-value facts where neither document
    # mentions scope can still be resolved deterministically.
    sa = (fact_a.get("norm_scope") or "").strip().lower()
    sb = (fact_b.get("norm_scope") or "").strip().lower()
    sa = "" if sa == "unspecified" else sa
    sb = "" if sb == "unspecified" else sb
    if sa and sb and sa != sb:
        return None

    fid_a = fact_a.get("id", 0)
    fid_b = fact_b.get("id", 0)

    # Numeric comparison
    try:
        av, bv = float(va), float(vb)
    except (TypeError, ValueError):
        return None

    if av == 0 and bv == 0:
        return None  # both zero — let LLM evaluate

    max_val = max(abs(av), abs(bv))
    rel_diff = abs(av - bv) / max_val if max_val > 0 else 0.0

    periods_match = (pa == pb) or (not pa and not pb)

    if rel_diff <= _CORROBORATE_TOLERANCE:
        # Values are effectively the same
        explanation = (
            f"Both facts report {ea} {na} as {av} {ua}"
            + (f" for {pa}" if pa else "")
            + f" (relative difference: {rel_diff*100:.2f}%)."
        )
        return Relation(
            fact_a_id=fid_a, fact_b_id=fid_b,
            relation_type=RelationType.CORROBORATES,
            confidence=0.95,
            reason=ReconcileReason.NONE,
            explanation=explanation,
        )

    if not periods_match and pa and pb:
        # Same metric/unit, different periods → reconcilable by period
        explanation = (
            f"Both facts report {ea} {na} but for different periods: {pa} vs {pb}."
            f" Values: {av} vs {bv} {ua}."
        )
        return Relation(
            fact_a_id=fid_a, fact_b_id=fid_b,
            relation_type=RelationType.RECONCILABLE,
            confidence=0.85,
            reason=ReconcileReason.DIFFERENT_PERIOD,
            explanation=explanation,
        )

    if periods_match and rel_diff > _CORROBORATE_TOLERANCE:
        # Same period, same metric, values differ → updated information or definition difference
        explanation = (
            f"Both facts report {ea} {na} for {pa} but values differ: {av} vs {bv} {ua}"
            f" (relative difference: {rel_diff*100:.1f}%). Likely updated or restated figures."
        )
        return Relation(
            fact_a_id=fid_a, fact_b_id=fid_b,
            relation_type=RelationType.RECONCILABLE,
            confidence=0.75,
            reason=ReconcileReason.UPDATED_INFORMATION,
            explanation=explanation,
        )

    return None  # ambiguous — hand off to LLM


def compare_new_facts(new_facts: list[dict]):
    """For each newly stored fact, retrieve candidates and classify relationships.
    Stores validated relations only; skips 'unrelated' to avoid cluttering storage."""
    if not new_facts:
        return

    all_facts = store.get_all_facts()

    det_count = 0
    llm_count = 0

    for fact in new_facts:
        candidates = retrieve_candidates(fact, all_facts, k=5)

        for cand in candidates:
            # Skip if this pair already has a stored relation
            existing = store.get_relations_for_fact(fact["id"])
            if any({r["fact_a_id"], r["fact_b_id"]} == {fact["id"], cand["id"]} for r in existing):
                continue

            # --- Deterministic pre-check ---
            det_relation = _deterministic_compare(fact, cand)
            if det_relation is not None:
                det_count += 1
                llm.increment_deterministic()
                if det_relation.relation_type != RelationType.UNRELATED:
                    store.add_relation(
                        fact["id"], cand["id"],
                        det_relation.relation_type.value, det_relation.explanation,
                        confidence=det_relation.confidence, reason=det_relation.reason.value,
                    )
                logger.debug(
                    "deterministic: fact %s vs %s → %s",
                    fact["id"], cand["id"], det_relation.relation_type.value,
                )
                continue

            # --- LLM comparison (only for ambiguous cases) ---
            llm_count += 1
            prompt = COMPARE_PROMPT.format(
                doc_a=fact["document_id"], page_a=fact["page_no"],
                entity_a=fact["entity"], metric_a=fact["metric"], value_a=fact["value"],
                unit_a=fact["unit"] or "", period_a=fact["period"] or "unspecified",
                scope_a=fact["scope"] or "unspecified", quote_a=fact["quote"],
                doc_b=cand["document_id"], page_b=cand["page_no"],
                entity_b=cand["entity"], metric_b=cand["metric"], value_b=cand["value"],
                unit_b=cand["unit"] or "", period_b=cand["period"] or "unspecified",
                scope_b=cand["scope"] or "unspecified", quote_b=cand["quote"],
            )
            try:
                result = llm.generate_json(prompt, call_type="relation")
            except llm.LLMUnavailableError as e:
                logger.error("compare failed for fact %s vs %s: %s", fact["id"], cand["id"], e)
                continue

            if not isinstance(result, dict):
                logger.warning("compare: expected dict, got %s", type(result))
                continue

            result["fact_a_id"] = fact["id"]
            result["fact_b_id"] = cand["id"]
            relation = val.validate_relation(result)
            if relation is None:
                continue
            if relation.relation_type.value == "unrelated":
                continue  # don't clutter storage with noise

            store.add_relation(
                fact["id"], cand["id"], relation.relation_type.value, relation.explanation,
                confidence=relation.confidence, reason=relation.reason.value,
                fact_a_evidence=relation.fact_a_evidence, fact_b_evidence=relation.fact_b_evidence,
            )

    logger.info(
        "compare_new_facts: processed=%d  deterministic=%d  llm_calls=%d",
        len(new_facts), det_count, llm_count,
    )
