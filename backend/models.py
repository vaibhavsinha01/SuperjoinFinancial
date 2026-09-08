"""
Pydantic models used to validate objects as they move between pipeline stages:

    raw extraction -> Fact
    Fact -> NormalizedFact
    NormalizedFact + embedding -> EmbeddingRecord
    comparison result -> Relation
"""
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ---------- Extraction ----------

class Fact(BaseModel):
    id: Optional[int] = None
    document_id: str
    chunk_id: Optional[str] = None
    page_no: int = Field(ge=1)
    entity: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    value: str = Field(min_length=1)
    unit: Optional[str] = None
    period: Optional[str] = None
    scope: Optional[str] = None
    quote: str = Field(min_length=1)
    confidence: str = "medium"
    numeric_confidence: float = 0.6

    @field_validator("confidence", mode="before")
    @classmethod
    def _valid_confidence(cls, v):
        s = str(v).strip().lower() if v is not None else "medium"
        if s not in ("low", "medium", "high"):
            return "medium"
        return s

    @field_validator("numeric_confidence", mode="before")
    @classmethod
    def _normalize_numeric_confidence(cls, v):
        """Indicative 0-1 score, not a calibrated probability. Falls back to a
        mapping from the qualitative label if no usable numeric value is present."""
        try:
            f = float(v)
            if f != f:
                raise ValueError
            if 5.0 < f <= 100.0:
                f = f / 100.0
            return max(0.0, min(1.0, f))
        except (TypeError, ValueError):
            return 0.6


# ---------- Normalization ----------

class NormalizedFact(BaseModel):
    id: Optional[int] = None
    fact_id: Optional[int] = None  # filled in once the parent Fact is stored and has a DB id
    entity: str = Field(min_length=1)
    metric: str = Field(min_length=1)          # canonical snake_case metric, e.g. "revenue_growth"
    value: float
    unit: str                                   # canonical unit, e.g. "percent", "inr_crore", "usd"
    period: Optional[str] = None                # canonical period, e.g. "FY2024"
    scope: Optional[str] = None
    original_value: str                          # preserved as-extracted, for provenance
    original_unit: Optional[str] = None


# ---------- Embeddings ----------

class EmbeddingStatus(str, Enum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"


class EmbeddingRecord(BaseModel):
    fact_id: int
    status: EmbeddingStatus
    vector: Optional[list[float]] = None
    model: Optional[str] = None
    dim: Optional[int] = None
    error: Optional[str] = None

    @field_validator("vector")
    @classmethod
    def _valid_vector(cls, v):
        if v is None:
            return v
        if len(v) == 0:
            raise ValueError("embedding vector must not be empty")
        if all(x == 0 for x in v):
            raise ValueError("embedding vector must not be all-zero")
        return v


# ---------- Relations ----------

class RelationType(str, Enum):
    CORROBORATES = "corroborates"
    CONTRADICTS = "contradicts"
    RECONCILABLE = "reconcilable"
    UNRELATED = "unrelated"


class ReconcileReason(str, Enum):
    DIFFERENT_PERIOD = "different_reporting_period"
    DIFFERENT_SCOPE = "different_scope"
    DIFFERENT_DEFINITION = "different_definition"
    DIFFERENT_GEOGRAPHY = "different_geography"
    DIFFERENT_UNIT = "different_unit"
    DIFFERENT_CURRENCY = "different_currency"
    UPDATED_INFORMATION = "updated_information"
    DIFFERENT_ESTIMATION_METHOD = "different_estimation_method"
    NONE = "none"


class Relation(BaseModel):
    id: Optional[int] = None
    fact_a_id: int
    fact_b_id: int
    relation_type: RelationType
    confidence: float = 0.5
    reason: ReconcileReason = ReconcileReason.NONE
    explanation: str = ""
    fact_a_evidence: str = ""
    fact_b_evidence: str = ""

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalize_confidence(cls, v):
        """Coerce whatever the LLM (or deterministic path) returns into a sensible
        0-1 indicative score rather than rejecting the whole relation. Handles common
        LLM slip-ups: percentages (0-100), out-of-range floats, strings, None."""
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 0.5
        if f != f:  # NaN
            return 0.5
        if 5.0 < f <= 100.0:
            # Clearly given as a percentage (e.g. 85 instead of 0.85)
            f = f / 100.0
        return max(0.0, min(1.0, f))

    @field_validator("relation_type", mode="before")
    @classmethod
    def _coerce_relation_type(cls, v):
        s = str(v).strip().lower()
        if "corroborat" in s:
            return RelationType.CORROBORATES
        if "contradict" in s:
            return RelationType.CONTRADICTS
        if "reconcil" in s:
            return RelationType.RECONCILABLE
        return RelationType.UNRELATED

    @field_validator("reason", mode="before")
    @classmethod
    def _coerce_reason(cls, v):
        if not v:
            return ReconcileReason.NONE
        s = str(v).strip().lower().replace("-", "_").replace(" ", "_")
        if "period" in s:
            return ReconcileReason.DIFFERENT_PERIOD
        if "scope" in s:
            return ReconcileReason.DIFFERENT_SCOPE
        if "def" in s:
            return ReconcileReason.DIFFERENT_DEFINITION
        if "geo" in s:
            return ReconcileReason.DIFFERENT_GEOGRAPHY
        if "unit" in s:
            return ReconcileReason.DIFFERENT_UNIT
        if "curr" in s:
            return ReconcileReason.DIFFERENT_CURRENCY
        if "update" in s or "restate" in s:
            return ReconcileReason.UPDATED_INFORMATION
        if "estimat" in s or "method" in s:
            return ReconcileReason.DIFFERENT_ESTIMATION_METHOD
        try:
            return ReconcileReason(s)
        except ValueError:
            return ReconcileReason.NONE
