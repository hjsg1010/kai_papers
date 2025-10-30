"""Utilities for interacting with the LLM backend."""
from __future__ import annotations

import math
from typing import Dict, List

import requests

from .config import LLM_API_KEY, LLM_BASE_URL, LLM_MAX_TOKENS, LLM_MODEL, logger


def _estimate_tokens(text: str) -> int:
    """Rough token estimator assuming ~4 characters per token."""
    return max(1, math.ceil(len(text) / 4))


def call_llm(messages: List[Dict], max_tokens: int = 2000) -> str:
    """Call the configured chat-completions endpoint."""
    headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": LLM_MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0.2}
    try:
        base_url = LLM_BASE_URL.rstrip("/") if LLM_BASE_URL else ""
        url = f"{base_url}/chat/completions"

        logger.info("Calling LLM: %s (model: %s, max_tokens: %s)", url, LLM_MODEL, max_tokens)

        response = requests.post(url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()

        if "choices" not in data or not data["choices"]:
            raise ValueError(f"Invalid response: {list(data.keys())}")

        message = data["choices"][0].get("message", {})
        finish_reason = data["choices"][0].get("finish_reason")

        if finish_reason == "length":
            logger.warning("⚠️  Response truncated due to max_tokens limit!")

        content = message.get("content") or message.get("reasoning_content") or message.get("text") or ""

        if not content.strip():
            logger.warning(
                "Empty LLM response. finish_reason: %s, message keys: %s",
                finish_reason,
                list(message.keys()),
            )
            if "reasoning_content" in message and not message.get("content"):
                logger.error("❌ Only reasoning_content available, no actual answer!")
                logger.error("reasoning_content: %s", message["reasoning_content"][:200])
                logger.error("This usually means max_tokens is too low for reasoning models")
            return ""

        if "reasoning_content" in message and "content" not in message:
            logger.info("⚠️  Using reasoning_content (reasoning model, but content missing)")

        return content

    except requests.exceptions.HTTPError as exc:
        logger.error("LLM API HTTP Error: %s", exc.response.status_code if exc.response else exc)
        if exc.response is not None:
            logger.error("Response: %s", exc.response.text[:500])
        raise
    except Exception as exc:
        logger.error("LLM call failed: %s: %s", type(exc).__name__, exc)
        raise


__all__ = ["LLM_MAX_TOKENS", "call_llm", "_estimate_tokens"]
