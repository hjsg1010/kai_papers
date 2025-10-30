"""Utility helpers for interacting with S3."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import fnmatch
import os
import tempfile

from .config import logger, s3_client


def _iter_s3_objects(bucket: str, prefix: str) -> Iterable[Dict]:
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            yield obj


def _preview(text: str, limit: int) -> str:
    if not text or limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...\n(truncated)"


def get_paper_list_from_s3(bucket: str, prefix: str) -> Optional[List[str]]:
    """Return URLs listed in ``paper_list.txt`` if present."""
    paper_list_key = f"{prefix.rstrip('/')}/paper_list.txt"

    try:
        logger.info("Checking for paper list: s3://%s/%s", bucket, paper_list_key)
        response = s3_client.get_object(Bucket=bucket, Key=paper_list_key)
        content = response["Body"].read().decode("utf-8")

        urls: List[str] = []
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("http"):
                urls.append(line)
            else:
                logger.warning("Invalid URL format in paper_list.txt: %s", line)

        logger.info("Found %d URLs in paper_list.txt", len(urls))
        return urls if urls else None

    except s3_client.exceptions.NoSuchKey:
        logger.info("paper_list.txt not found: %s", paper_list_key)
        return None
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Error reading paper_list.txt: %s", exc)
        return None


def get_s3_papers(
    bucket: str,
    prefix: str,
    file_pattern: str = "*.pdf",
    process_subdirectories: bool = True,
    min_size_bytes: int = 1024,
    max_size_bytes: int = 1024 * 1024 * 100,
) -> List[Dict]:
    logger.info("Fetching papers from S3: s3://%s/%s", bucket, prefix)
    papers: List[Dict] = []
    for obj in _iter_s3_objects(bucket, prefix):
        key = obj["Key"]
        rel = key[len(prefix):].lstrip("/") if key.startswith(prefix) else key
        if not fnmatch.fnmatch(rel, file_pattern):
            continue
        if not process_subdirectories and "/" in rel:
            continue
        size = obj.get("Size", 0)
        if size < min_size_bytes or size > max_size_bytes:
            continue
        papers.append({
            "title": Path(key).stem.replace("_", " ").replace("-", " "),
            "s3_key": key,
            "s3_bucket": bucket,
            "last_modified": obj["LastModified"].isoformat(),
            "size_bytes": size,
            "source": "s3",
        })
    logger.info("Found %d papers in S3", len(papers))
    return papers


def download_pdf_from_s3(s3_key: str, s3_bucket: str) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        logger.info("Downloading PDF: s3://%s/%s", s3_bucket, s3_key)
        s3_client.download_fileobj(s3_bucket, s3_key, tmp)
        tmp.close()
        return tmp.name
    except Exception as exc:
        tmp.close()
        if os.path.exists(tmp.name):
            try:
                os.unlink(tmp.name)
            except Exception:  # pragma: no cover - cleanup best effort
                pass
        raise Exception(f"Failed to download PDF: {exc}") from exc


__all__ = [
    "_iter_s3_objects",
    "_preview",
    "download_pdf_from_s3",
    "get_paper_list_from_s3",
    "get_s3_papers",
]
