import os
import json
from backend.providers.base import LLMProvider, TransientError, QuotaExhaustedError

GROQ_MODEL = os.environ.get("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
_TRANSIENT_CODES = {429, 500, 502, 503, 504}


class GroqProvider(LLMProvider):
    name = "groq"

    def __init__(self):
        from groq import Groq  # imported lazily so groq is only required if used
        self._client = Groq(api_key=os.environ["GROQ_API_KEY"])

    def generate_json(self, prompt: str) -> dict | list:
        try:
            resp = self._client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": "Respond with strict JSON only. No markdown, no commentary."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"} if "{{" not in prompt else None,
                temperature=0.1,
            )
        except Exception as e:
            msg = str(e).lower()
            status = getattr(e, "status_code", None)
            if status == 429 and ("daily" in msg or "quota" in msg) and "retry" not in msg:
                raise QuotaExhaustedError(str(e)) from e
            if status in _TRANSIENT_CODES or status is None:
                raise TransientError(str(e)) from e
            raise TransientError(str(e)) from e

        text = resp.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            raise TransientError(f"invalid JSON from groq: {e}") from e

        # Groq's json_object mode only allows a JSON *object*; our prompts sometimes
        # want a top-level array, so we ask for that wrapped as {"items": [...]} and
        # unwrap it here to keep the provider interface uniform.
        if isinstance(parsed, dict) and set(parsed.keys()) == {"items"}:
            return parsed["items"]
        return parsed
