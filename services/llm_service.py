"""
LLM service for calling language models
"""
import logging
import math
import requests
from typing import List, Dict

from config.settings import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

logger = logging.getLogger(__name__)


def estimate_tokens(s: str) -> int:
    """
    Simple token estimation (1 token ≈ 4 chars)

    Args:
        s: Input string

    Returns:
        Estimated token count
    """
    return max(1, math.ceil(len(s) / 4))


def call_llm(messages: List[Dict], max_tokens: int = 2000) -> str:
    """
    Call LLM API (supports reasoning models)

    Args:
        messages: List of message dictionaries with 'role' and 'content'
        max_tokens: Maximum tokens in response

    Returns:
        LLM response text

    Raises:
        ValueError: If response format is invalid
        requests.exceptions.HTTPError: If API call fails
    """
    headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": LLM_MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0.2}

    try:
        base_url = LLM_BASE_URL.rstrip('/') if LLM_BASE_URL else ""
        url = f"{base_url}/chat/completions"

        logger.info(f"Calling LLM: {url} (model: {LLM_MODEL}, max_tokens: {max_tokens})")

        r = requests.post(url, headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        j = r.json()

        if "choices" not in j or not j["choices"]:
            raise ValueError(f"Invalid response: {list(j.keys())}")

        message = j["choices"][0].get("message", {})
        finish_reason = j["choices"][0].get("finish_reason")

        if finish_reason == "length":
            logger.warning(f"⚠️  Response truncated due to max_tokens limit!")

        content = (
            message.get("content") or
            message.get("reasoning_content") or
            message.get("text") or
            ""
        )

        if not content.strip():
            logger.warning(f"Empty LLM response. finish_reason: {finish_reason}, message keys: {message.keys()}")

            if "reasoning_content" in message and not message.get("content"):
                logger.error(f"❌ Only reasoning_content available, no actual answer!")
                logger.error(f"reasoning_content: {message['reasoning_content'][:200]}")
                logger.error(f"This usually means max_tokens is too low for reasoning models")

            return ""

        if "reasoning_content" in message and "content" not in message:
            logger.info("⚠️  Using reasoning_content (reasoning model, but content missing)")

        return content

    except requests.exceptions.HTTPError as e:
        logger.error(f"LLM API HTTP Error: {e.response.status_code}")
        logger.error(f"Response: {e.response.text[:500]}")
        raise
    except ValueError as e:
        logger.error(f"LLM response format error: {e}")
        raise
    except KeyError as e:
        logger.error(f"LLM response missing key: {e}")
        raise
    except Exception as e:
        logger.error(f"LLM call failed: {type(e).__name__}: {e}")
        raise
