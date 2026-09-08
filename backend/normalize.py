"""
Normalize raw extracted facts into a canonical representation so that semantically
equivalent facts (e.g. "revenue grew 15.2%" vs "revenue increased by 15.2 percent")
become comparable on structured fields rather than only on embedding similarity.

The original extracted value/unit/quote is always preserved (see NormalizedFact) —
normalization never destroys source information.
"""
import re
import logging
from backend.models import Fact, NormalizedFact

logger = logging.getLogger("factloom.normalize")

# --- metric canonicalization -------------------------------------------------
# Maps common surface forms -> canonical snake_case metric name. Not exhaustive;
# unmatched metrics fall back to a slugified version of the original string, so
# nothing is dropped for lack of a mapping.
_METRIC_ALIASES = {
    "revenue": "revenue",
    "total revenue": "revenue",
    "revenue from operations": "revenue",
    "revenue growth": "revenue_growth",
    "net loss": "net_loss",
    "net profit": "net_profit",
    "profit after tax": "net_profit",
    "ebitda": "ebitda",
    "ebitda margin": "ebitda_margin",
    "gdp growth": "gdp_growth",
    "gdp growth rate": "gdp_growth",
    "real gdp growth": "gdp_growth",
    "inflation": "inflation_rate",
    "inflation rate": "inflation_rate",
    "cpi inflation": "inflation_rate",
    "repo rate": "repo_rate",
    "current account deficit": "current_account_deficit",
    "fiscal deficit": "fiscal_deficit",
    "foreign exchange reserves": "forex_reserves",
    "forex reserves": "forex_reserves",
}


def _slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def canonical_metric(metric: str) -> str:
    key = metric.strip().lower()
    return _METRIC_ALIASES.get(key, _slugify(metric))


# --- unit canonicalization ----------------------------------------------------
_UNIT_ALIASES = {
    "%": "percent", "percent": "percent", "percentage": "percent", "pct": "percent",
    "inr crore": "inr_crore", "rs. crore": "inr_crore", "rs crore": "inr_crore", "crore": "inr_crore",
    "inr lakh": "inr_lakh", "lakh": "inr_lakh",
    "inr million": "inr_million", "rs million": "inr_million",
    "inr billion": "inr_billion",
    "usd": "usd", "$": "usd", "us$": "usd",
    "usd million": "usd_million", "usd billion": "usd_billion",
    "bps": "basis_points", "basis points": "basis_points",
    "days": "days",
}


def canonical_unit(unit: str | None) -> str:
    if not unit:
        return "unitless"
    key = unit.strip().lower()
    return _UNIT_ALIASES.get(key, _slugify(unit))


# --- period canonicalization ---------------------------------------------------
_FY_RE = re.compile(r"fy\s*'?(\d{2,4})", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_QTR_RE = re.compile(r"\bq([1-4])\b", re.IGNORECASE)


def canonical_period(period: str | None) -> str | None:
    if not period:
        return None
    p = period.strip()
    qtr_match = _QTR_RE.search(p)
    fy_match = _FY_RE.search(p)
    if fy_match:
        yr = fy_match.group(1)
        yr = ("20" + yr) if len(yr) == 2 else yr
        prefix = f"Q{qtr_match.group(1)} " if qtr_match else ""
        return f"{prefix}FY{yr}"
    year_match = _YEAR_RE.search(p)
    if year_match:
        return year_match.group(0)
    return p  # leave as-is if nothing recognizable; don't destroy information


# --- value parsing ---------------------------------------------------------
_NUM_RE = re.compile(r"-?[\d,]+\.?\d*")


def parse_numeric_value(value: str) -> float | None:
    """Extract a float from a value string like '8,993.6' or '15.2%'. Returns None if non-numeric
    (e.g. a status like 'resigned') — such facts are simply not normalized numerically."""
    cleaned = value.replace(",", "")
    match = _NUM_RE.search(cleaned)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def normalize_fact(fact: Fact) -> NormalizedFact | None:
    """Produce a NormalizedFact for a validated Fact. Returns None if the fact's value
    isn't numeric (non-numeric facts, e.g. qualitative status changes, aren't normalized
    here and are still retrievable via embeddings on the raw fact)."""
    numeric_value = parse_numeric_value(fact.value)
    if numeric_value is None:
        logger.info("skipping numeric normalization for non-numeric fact id=%s value=%r", fact.id, fact.value)
        return None

    return NormalizedFact(
        fact_id=fact.id,
        entity=fact.entity.strip(),
        metric=canonical_metric(fact.metric),
        value=numeric_value,
        unit=canonical_unit(fact.unit),
        period=canonical_period(fact.period),
        scope=(fact.scope or "unspecified").strip().lower(),
        original_value=fact.value,
        original_unit=fact.unit,
    )
