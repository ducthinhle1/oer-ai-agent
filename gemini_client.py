"""
Thin wrapper around the Google Gemini API.

Why a wrapper?
- Centralises the model name + retry logic in one place.
- Lets us swap models or providers later without touching app code.
"""
from __future__ import annotations

import json
import time
from typing import Optional

import google.generativeai as genai
from google.api_core import exceptions

# Default model. Use the lighter "flash" by default for cheaper / faster calls.
# Switch to "gemini-2.5-pro" for the rubric-evaluation step where quality matters.
DEFAULT_MODEL = "gemini-2.5-flash"
QUALITY_MODEL = "gemini-2.5-pro"


def initialize_gemini(api_key: str) -> None:
    """Configure the Gemini SDK with your API key. Call once at app start."""
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is missing. Copy .env.example -> .env and fill it in.")
    genai.configure(api_key=api_key)


def ask_gemini(
    prompt: str,
    *,
    model_name: str = DEFAULT_MODEL,
    system_instruction: Optional[str] = None,
    temperature: float = 0.2,
    max_retries: int = 3,
) -> str:
    """Send a text prompt to Gemini and return the text response.

    Retries on rate-limit errors with exponential backoff.
    """
    model = genai.GenerativeModel(
        model_name,
        system_instruction=system_instruction,
        generation_config={"temperature": temperature},
    )

    wait = 5
    for attempt in range(max_retries):
        try:
            resp = model.generate_content(prompt)
            return (resp.text or "").strip()
        except exceptions.ResourceExhausted:
            if attempt == max_retries - 1:
                return "Error: Gemini quota exceeded. Try again in a minute."
            time.sleep(wait)
            wait *= 2
        except Exception as e:  # noqa: BLE001 - surface anything else verbatim
            if "429" in str(e) and attempt < max_retries - 1:
                time.sleep(wait)
                wait *= 2
                continue
            return f"Error: {e}"
    return "Error: unknown failure"


def ask_gemini_json(prompt: str, **kwargs) -> dict:
    """Same as ask_gemini, but parses the response as JSON.

    The model is asked (via prompt convention) to return only JSON. We strip
    common code-fence wrappers before parsing so it survives ``` blocks.
    """
    raw = ask_gemini(prompt, **kwargs)
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        # remove ```json ... ``` or ``` ... ```
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"_parse_error": True, "raw": raw}
