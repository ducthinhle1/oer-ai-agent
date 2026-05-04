"""
Thin wrapper around google-genai (the current Gemini SDK) for the OER agent.

Two entry points:
  ask_gemini      — plain-text response (used for query refinement)
  ask_gemini_json — JSON-parsed response (used for rubric evaluation)

Model constants let callers pick the right power/cost trade-off without
knowing internal model IDs.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

from google import genai
from google.genai import types

DEFAULT_MODEL = "gemini-2.0-flash"   # fast, free tier: 1500 RPD / 15 RPM
QUALITY_MODEL = "gemini-2.0-flash"   # rubric evaluation — same model, 3× more daily quota than 2.5-flash

_client: genai.Client | None = None


def initialize_gemini(api_key: str) -> None:
    """Call once at app startup to configure the API key."""
    global _client
    _client = genai.Client(api_key=api_key)


def _get_client() -> genai.Client:
    if _client is None:
        raise RuntimeError("Call initialize_gemini(api_key) before using ask_gemini.")
    return _client


def _thinking_cfg(model_name: str) -> dict:
    """ThinkingConfig is only valid for Gemini 2.5+; older models reject it."""
    if "2.5" in model_name:
        return {"thinking_config": types.ThinkingConfig(thinking_budget=0)}
    return {}


def ask_gemini(
    prompt: str,
    *,
    system_instruction: str = "",
    model_name: str = DEFAULT_MODEL,
) -> str:
    """Send a plain-text prompt and return the response text."""
    try:
        cfg = types.GenerateContentConfig(
            system_instruction=system_instruction or None,
            **_thinking_cfg(model_name),
        )
        response = _get_client().models.generate_content(
            model=model_name,
            contents=prompt,
            config=cfg,
        )
        return response.text or ""
    except Exception as e:
        return f"Error: {e}"


def ask_gemini_json(
    prompt: str,
    *,
    system_instruction: str = "",
    model_name: str = QUALITY_MODEL,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Send a prompt expecting JSON back and return a parsed dict.

    Strips markdown code fences if the model wraps the JSON in them.
    On parse failure, returns {"_parse_error": True, "raw": <raw_text>}.
    """
    cfg = types.GenerateContentConfig(
        system_instruction=system_instruction or None,
        temperature=temperature,
        response_mime_type="application/json",
        **_thinking_cfg(model_name),
    )
    text = ""
    for attempt in range(3):
        try:
            response = _get_client().models.generate_content(
                model=model_name,
                contents=prompt,
                config=cfg,
            )
            text = (response.text or "").strip()
            break
        except Exception as e:
            err = str(e)
            is_quota = "429" in err or "quota" in err.lower()
            if is_quota and attempt < 2:
                time.sleep(30 * (attempt + 1))  # 30s, then 60s
                continue
            if is_quota:
                return {"_parse_error": True, "raw": "API quota exceeded — try again in a moment."}
            return {"_parse_error": True, "raw": err}

    # Safety net: strip ``` fences in case they appear despite response_mime_type
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"_parse_error": True, "raw": text}
