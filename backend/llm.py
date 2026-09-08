"""
LLM orchestration: provider-agnostic entrypoint used by the rest of the app.

Architecture:
    Groq (PRIMARY) --(transient error)--> exponential backoff + retry (MAX_RETRIES)
                                                 |
                                         still failing?
                                                 |
                                                YES
                                                 |
                                        Gemini (FALLBACK) --> retry
                                                 |
                                         still failing?
                                                 |
                                               LLMUnavailableError

Provider order is controlled by LLM_PRIMARY_PROVIDER env var (default: groq).
Identical prompts are cached in SQLite so repeated runs don't burn API quota.

Embeddings are intentionally NOT handled here. They use a local Transformer model
(backend/embed.py) which requires no API key.
"""
import os
import time
import random
import logging
import hashlib
import threading
from dataclasses import dataclass, field

from backend.providers.base import TransientError, QuotaExhaustedError
from backend import cache

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger("factloom.llm")

MAX_RETRIES   = int(os.environ.get("LLM_MAX_RETRIES", "4"))
BASE_DELAY    = float(os.environ.get("LLM_BASE_DELAY_SECONDS", "1.0"))
MAX_DELAY     = float(os.environ.get("LLM_MAX_DELAY_SECONDS", "20.0"))
CACHE_ENABLED = os.environ.get("LLM_CACHE_ENABLED", "true").lower() == "true"
PRIMARY_PROVIDER = os.environ.get("LLM_PRIMARY_PROVIDER", "groq").lower()

# ---------- Custom Exceptions ----------

class LLMUnavailableError(Exception):
    """Raised when every configured provider has failed."""


# ---------- Metrics ----------

@dataclass
class _Metrics:
    groq_calls: int = 0
    groq_successes: int = 0
    groq_failures: int = 0
    gemini_calls: int = 0
    gemini_successes: int = 0
    gemini_failures: int = 0
    cache_hits: int = 0
    extraction_calls: int = 0
    relation_calls: int = 0
    deterministic_results: int = 0

    def to_dict(self) -> dict:
        return {
            "groq_calls": self.groq_calls,
            "groq_successes": self.groq_successes,
            "groq_failures": self.groq_failures,
            "gemini_calls": self.gemini_calls,
            "gemini_successes": self.gemini_successes,
            "gemini_failures": self.gemini_failures,
            "cache_hits": self.cache_hits,
            "extraction_calls": self.extraction_calls,
            "relation_calls": self.relation_calls,
            "deterministic_results": self.deterministic_results,
            "total_llm_calls": self.groq_calls + self.gemini_calls,
        }


_metrics = _Metrics()
_metrics_lock = threading.Lock()


def get_metrics() -> dict:
    """Return current session metrics as a dict."""
    with _metrics_lock:
        return _metrics.to_dict()


def reset_metrics():
    """Reset all metrics counters (e.g. between uploads)."""
    global _metrics
    with _metrics_lock:
        _metrics = _Metrics()


def increment_deterministic():
    """Called by compare.py when a pair is resolved without an LLM call."""
    with _metrics_lock:
        _metrics.deterministic_results += 1


def increment_extraction():
    with _metrics_lock:
        _metrics.extraction_calls += 1


def increment_relation():
    with _metrics_lock:
        _metrics.relation_calls += 1


# ---------- Provider registry ----------

def _build_providers():
    """Instantiate providers in the configured order.
    Default: Groq first (primary), Gemini second (fallback).
    Set LLM_PRIMARY_PROVIDER=gemini to swap if needed.
    """
    def _try_groq():
        if not os.environ.get("GROQ_API_KEY"):
            logger.warning("GROQ_API_KEY not set — Groq provider skipped")
            return None
        try:
            from backend.providers.groq import GroqProvider
            return GroqProvider()
        except Exception as e:
            logger.warning("Groq provider unavailable: %s", e)
            return None

    def _try_gemini():
        if not os.environ.get("GEMINI_API_KEY"):
            logger.warning("GEMINI_API_KEY not set — Gemini provider skipped")
            return None
        try:
            from backend.providers.gemini import GeminiProvider
            return GeminiProvider()
        except Exception as e:
            logger.warning("Gemini provider unavailable: %s", e)
            return None

    if PRIMARY_PROVIDER == "gemini":
        ordered = [_try_gemini(), _try_groq()]
    else:
        # Default: Groq primary, Gemini fallback
        ordered = [_try_groq(), _try_gemini()]

    return [p for p in ordered if p is not None]


_PROVIDERS = None


def _providers():
    global _PROVIDERS
    if _PROVIDERS is None:
        _PROVIDERS = _build_providers()
        names = [p.name for p in _PROVIDERS]
        logger.info("LLM provider chain: %s", " → ".join(names) if names else "(none)")
    return _PROVIDERS


# ---------- Retry logic ----------

def _backoff_sleep(attempt: int):
    delay = min(MAX_DELAY, BASE_DELAY * (2 ** attempt))
    delay = delay * (0.5 + random.random())  # jitter: 0.5x–1.5x
    logger.debug("backoff sleep %.2fs (attempt %d)", delay, attempt + 1)
    time.sleep(delay)


def _call_with_retry(provider, prompt: str):
    """Retry one provider on transient errors with exponential backoff."""
    last_err = None
    is_groq = provider.name == "groq"
    is_gemini = provider.name == "gemini"

    for attempt in range(MAX_RETRIES):
        with _metrics_lock:
            if is_groq:
                _metrics.groq_calls += 1
            elif is_gemini:
                _metrics.gemini_calls += 1

        try:
            result = provider.generate_json(prompt)
            with _metrics_lock:
                if is_groq:
                    _metrics.groq_successes += 1
                elif is_gemini:
                    _metrics.gemini_successes += 1
            if attempt > 0:
                logger.info("provider=%s succeeded on attempt=%d", provider.name, attempt + 1)
            return result

        except QuotaExhaustedError as e:
            with _metrics_lock:
                if is_groq:
                    _metrics.groq_failures += 1
                elif is_gemini:
                    _metrics.gemini_failures += 1
            logger.warning("provider=%s hard quota exhausted, not retrying: %s", provider.name, e)
            raise

        except TransientError as e:
            last_err = e
            with _metrics_lock:
                if is_groq:
                    _metrics.groq_failures += 1
                elif is_gemini:
                    _metrics.gemini_failures += 1
            logger.warning(
                "provider=%s attempt=%d/%d transient error: %s",
                provider.name, attempt + 1, MAX_RETRIES, e,
            )
            if attempt < MAX_RETRIES - 1:
                _backoff_sleep(attempt)

    raise last_err  # type: ignore[misc]


# ---------- Public API ----------

def generate_json(prompt: str, use_cache: bool = True, call_type: str = "generic") -> dict | list:
    """Call the LLM with automatic retry + provider fallback + response caching.

    call_type: 'extraction' | 'relation' | 'generic' — used for metrics only.
    """
    with _metrics_lock:
        if call_type == "extraction":
            _metrics.extraction_calls += 1
        elif call_type == "relation":
            _metrics.relation_calls += 1

    cache_key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if CACHE_ENABLED and use_cache:
        cached = cache.get(cache_key)
        if cached is not None:
            with _metrics_lock:
                _metrics.cache_hits += 1
            return cached

    providers = _providers()
    if not providers:
        raise LLMUnavailableError(
            "No LLM providers configured. Set GROQ_API_KEY (primary) and/or GEMINI_API_KEY (fallback)."
        )

    errors = []
    for i, provider in enumerate(providers):
        try:
            result = _call_with_retry(provider, prompt)
            if CACHE_ENABLED and use_cache:
                cache.set(cache_key, result)
            if i > 0:
                logger.info("succeeded on fallback provider=%s after primary failed", provider.name)
            return result
        except (TransientError, QuotaExhaustedError) as e:
            logger.error("provider=%s exhausted%s: %s",
                         provider.name,
                         ", falling back" if i < len(providers) - 1 else "",
                         e)
            errors.append(f"{provider.name}: {e}")
            continue

    raise LLMUnavailableError(f"All providers failed. Details: {' | '.join(errors)}")
