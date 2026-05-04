"""
Thin wrapper around google-genai (Gemini) with automatic Groq fallback.

When Gemini hits a quota limit (429), calls are transparently retried via
Groq (llama-3.3-70b-versatile) — no error shown to the user.

Initialization:
    initialize_gemini(api_key)          # required
    initialize_groq(api_key)            # optional — enables fallback
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

from google import genai
from google.genai import types

DEFAULT_MODEL = "gemini-2.0-flash"   # fast, free tier: 1500 RPD / 15 RPM
QUALITY_MODEL = "gemini-2.0-flash"   # rubric evaluation
GROQ_MODEL    = "llama-3.3-70b-versatile"  # fallback — 14,400 free RPD

_gemini_client: genai.Client | None = None
_groq_client = None


def initialize_gemini(api_key: str) -> None:
    global _gemini_client
    _gemini_client = genai.Client(api_key=api_key)


def initialize_groq(api_key: str) -> None:
    global _groq_client
    from groq import Groq
    _groq_client = Groq(api_key=api_key)


def groq_available() -> bool:
    return _groq_client is not None


def _get_gemini() -> genai.Client:
    if _gemini_client is None:
        raise RuntimeError("Call initialize_gemini(api_key) before using ask_gemini.")
    return _gemini_client


def _thinking_cfg(model_name: str) -> dict:
    """ThinkingConfig is only valid for Gemini 2.5+; older models reject it."""
    if "2.5" in model_name:
        return {"thinking_config": types.ThinkingConfig(thinking_budget=0)}
    return {}


# ---------------------------------------------------------------------------
# Groq fallback helpers
# ---------------------------------------------------------------------------

def _groq_text(prompt: str, system_instruction: str = "") -> str:
    """Plain-text response via Groq."""
    if _groq_client is None:
        return "Error: Groq not configured"
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})
    try:
        resp = _groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0,
            max_tokens=256,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        return f"Error: {e}"


def _groq_json(
    prompt: str,
    system_instruction: str = "",
    temperature: float = 0.1,
) -> dict[str, Any]:
    """JSON response via Groq."""
    if _groq_client is None:
        return {
            "_parse_error": True,
            "raw": "API quota reached and no Groq fallback configured. Add GROQ_API_KEY to your secrets.",
        }
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})
    try:
        resp = _groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        text = (resp.choices[0].message.content or "").strip()
        return json.loads(text)
    except Exception as e:
        return {"_parse_error": True, "raw": str(e)}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ask_gemini(
    prompt: str,
    *,
    system_instruction: str = "",
    model_name: str = DEFAULT_MODEL,
) -> str:
    """Plain-text prompt. Falls back to Groq on quota error."""
    try:
        cfg = types.GenerateContentConfig(
            system_instruction=system_instruction or None,
            **_thinking_cfg(model_name),
        )
        response = _get_gemini().models.generate_content(
            model=model_name,
            contents=prompt,
            config=cfg,
        )
        return response.text or ""
    except Exception as e:
        err = str(e)
        if "429" in err or "quota" in err.lower():
            return _groq_text(prompt, system_instruction)
        return f"Error: {e}"


def ask_gemini_json(
    prompt: str,
    *,
    system_instruction: str = "",
    model_name: str = QUALITY_MODEL,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """JSON prompt. Retries Gemini up to 3×, then falls back to Groq on quota."""
    cfg = types.GenerateContentConfig(
        system_instruction=system_instruction or None,
        temperature=temperature,
        response_mime_type="application/json",
        **_thinking_cfg(model_name),
    )
    text = ""
    for attempt in range(3):
        try:
            response = _get_gemini().models.generate_content(
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
                time.sleep(30 * (attempt + 1))  # 30 s, then 60 s
                continue
            if is_quota:
                return _groq_json(prompt, system_instruction, temperature)
            return {"_parse_error": True, "raw": err}

    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"_parse_error": True, "raw": text}
