"""Utilities for splitting markdown content into manageable chunks."""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

from .config import logger
from .llm import _estimate_tokens


def extract_all_headers(markdown_content: str) -> Dict[str, str]:
    """Split markdown into sections using ``#`` and ``##`` headers."""
    logger.info("Extracting main headers (# and ##) from markdown...")
    sections = []
    current_section = {"header": "preamble", "level": 0, "content": []}

    for line in markdown_content.splitlines():
        header_match = re.match(r"^(#{1,2})\s+(.+)$", line.strip())

        if header_match:
            if current_section["content"]:
                sections.append(current_section)

            level = len(header_match.group(1))
            header = header_match.group(2).strip()
            current_section = {"header": header, "level": level, "content": []}
        else:
            current_section["content"].append(line)

    if current_section["content"]:
        sections.append(current_section)

    result: Dict[str, str] = {}
    for i, sec in enumerate(sections):
        header_clean = re.sub(r"[^\w\s-]", "", sec["header"])[:30]
        key = f"section_{i:02d}_{header_clean}"
        content = "\n".join(sec["content"])

        if content.strip():
            result[key] = content
            logger.info("Section '%s': %d chars, ~%d toks", sec["header"], len(content), _estimate_tokens(content))

    return result


def chunk_by_tokens(text: str, chunk_size: int = 4000, overlap: int = 400) -> List[str]:
    """Split text evenly based on an approximate token budget."""
    logger.info("Chunking by tokens (chunk_size=%s, overlap=%s)...", chunk_size, overlap)

    total_chars = len(text)
    if total_chars <= chunk_size * 4:
        return [text]

    char_per_chunk = chunk_size * 4
    overlap_chars = overlap * 4

    chunks: List[str] = []
    start = 0

    while start < total_chars:
        end = start + char_per_chunk

        if end < total_chars:
            for i in range(end, max(start + char_per_chunk // 2, end - 500), -1):
                if i < len(text) and text[i] in ".!?\n":
                    end = i + 1
                    break

        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap_chars

        if start >= total_chars:
            break

    logger.info("Created %d chunks by tokens", len(chunks))
    return chunks


def smart_chunk_hybrid(markdown_content: str, min_headers: int = 10) -> Tuple[Dict[str, str], str]:
    """Choose between header-based and token-based chunking depending on structure."""
    main_headers = re.findall(r"^#\s+[^#].+$", markdown_content, re.MULTILINE)
    num_headers = len(main_headers)

    logger.info("Found %d main headers (# only) in markdown", num_headers)

    if num_headers >= min_headers:
        logger.info("Using HEADER-BASED chunking (%d headers >= %d)", num_headers, min_headers)
        chunks = extract_all_headers(markdown_content)
        return chunks, "header_based"

    logger.info("Using TOKEN-BASED chunking (%d headers < %d)", num_headers, min_headers)
    chunk_list = chunk_by_tokens(markdown_content, chunk_size=6000, overlap=600)
    chunks = {f"chunk_{i:02d}": chunk for i, chunk in enumerate(chunk_list)}
    return chunks, "token_based"


def smart_chunk_with_overlap(text: str, chunk_size: int = 4000, overlap: int = 400) -> List[str]:
    """Split *text* while keeping an overlap between adjacent segments."""
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            chunk_end = end
            for i in range(end, max(start + chunk_size // 2, end - 200), -1):
                if text[i] in ".!?\n":
                    chunk_end = i + 1
                    break
            chunks.append(text[start:chunk_end])
            start = chunk_end - overlap
        else:
            chunks.append(text[start:])
            break

    logger.info("Split into %d chunks with overlap=%s", len(chunks), overlap)
    return chunks


__all__ = [
    "chunk_by_tokens",
    "extract_all_headers",
    "smart_chunk_hybrid",
    "smart_chunk_with_overlap",
]
