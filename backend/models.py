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

    @field_validator("confidence")
    @classmethod
    def _valid_confidence(cls, v):
        if v not in ("low", "medium", "high"):
            return "medium"
        return v


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
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    reason: ReconcileReason = ReconcileReason.NONE
    explanation: str = ""
    fact_a_evidence: str = ""
    fact_b_evidence: str = ""
