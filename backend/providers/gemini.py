import os
import json
from google import genai
from google.genai import types
from google.genai import errors as genai_errors

from backend.providers.base import LLMProvider, TransientError, QuotaExhaustedError

GEN_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")

# error codes that are worth retrying (transient)
_TRANSIENT_CODES = {429, 500, 502, 503, 504}


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self):
        self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def generate_json(self, prompt: str) -> dict | list:
        model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        try:
            resp = self._client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
        except genai_errors.APIError as e:
            code = getattr(e, "code", None)
            msg = str(e).lower()
            # Gemini reports hard daily quota exhaustion as 429 too, but usually
            # with "quota" / "exceeded your current quota" in the body while a
            # transient rate limit says "please retry" / has a retry-after.
            if code == 429 and ("exceeded" in msg or "daily" in msg) and "retry" not in msg:
                raise QuotaExhaustedError(str(e)) from e
            if code in _TRANSIENT_CODES:
                raise TransientError(str(e)) from e
            raise TransientError(str(e)) from e  # unknown API error: treat as transient, retries are cheap
        except Exception as e:
            raise TransientError(str(e)) from e

        text = (resp.text or "").strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            raise TransientError(f"invalid JSON from gemini: {e}") from e

        if isinstance(parsed, dict):
            for list_key in ("facts", "items", "data", "results"):
                if list_key in parsed and isinstance(parsed[list_key], list):
                    return parsed[list_key]
        return parsed
