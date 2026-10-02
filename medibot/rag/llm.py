"""One thin door to the LLM (Groq). Every generation step in the app goes through chat().

Keeping it in one place means: one model name (env-overridable), one retry policy, and one spot
to add tracing later (Assignment 3).
"""

from __future__ import annotations

import time

from groq import APIConnectionError, APIStatusError, AuthenticationError, Groq, RateLimitError

from medibot.config import GROQ_API_KEY, GROQ_MODEL

_client: Groq | None = None


class LLMError(RuntimeError):
    """Raised with a user-safe message; the real exception is chained for the logs."""


def client() -> Groq:
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise LLMError("GROQ_API_KEY is not set (see .env.example)")
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def chat(system: str, user: str, *, temperature: float = 0.2, max_tokens: int = 700, retries: int = 2) -> str:
    """Single-turn call: standing rules in `system`, per-request data in `user`. Returns the text."""
    for attempt in range(retries + 1):
        try:
            resp = client().chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning_effort="low",  # gpt-oss models: keep hidden reasoning short, we want the answer
            )
            return (resp.choices[0].message.content or "").strip()
        except RateLimitError as exc:  # free tier: 8k tokens/min — wait and retry once or twice
            if attempt == retries:
                raise LLMError("The language model is rate-limited right now; please retry in a minute.") from exc
            time.sleep(2 + 3 * attempt)
        except AuthenticationError as exc:
            raise LLMError("The language model rejected the API key.") from exc
        except APIConnectionError as exc:
            raise LLMError("Could not reach the language model service.") from exc
        except APIStatusError as exc:
            raise LLMError(f"Language model error ({exc.status_code}).") from exc
    raise LLMError("Language model call failed.")  # unreachable, keeps type checkers happy
