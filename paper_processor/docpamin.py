"""Docpamin parsing helpers including caching on S3."""
from __future__ import annotations

import hashlib
import io
import json
import re
import time
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import requests

from .config import (
    DOCPAMIN_API_KEY,
    DOCPAMIN_BASE_URL,
    DOCPAMIN_CRT_FILE,
    S3_BUCKET,
    S3_PAPERS_PREFIX,
    logger,
    s3_client,
)
from .images import (
    extract_figure_pairs_from_json,
    is_valid_caption,
    match_images_with_captions_from_json,
    match_images_with_figure_pairs,
    process_markdown_images,
    select_representative_images,
)


def get_docpamin_cache_key(source_identifier: str) -> str:
    """Return a deterministic cache key for an arXiv ID or generic file name."""
    arxiv_match = re.search(r"(\d{4}\.\d{5})", source_identifier)
    if arxiv_match:
        return arxiv_match.group(1)
    return hashlib.md5(source_identifier.encode()).hexdigest()[:16]


def save_docpamin_cache_to_s3(bucket: str, prefix: str, cache_key: str, markdown: str, metadata: Dict) -> bool:
    """Persist Docpamin output (markdown + metadata) back to S3 for reuse."""
    try:
        cache_prefix = f"{prefix.rstrip('/')}/cache"

        md_key = f"{cache_prefix}/{cache_key}.md"
        s3_client.put_object(
            Bucket=bucket,
            Key=md_key,
            Body=markdown.encode("utf-8"),
            ContentType="text/markdown",
        )
        logger.info("Saved markdown cache: s3://%s/%s", bucket, md_key)

        json_key = f"{cache_prefix}/{cache_key}.json"
        s3_client.put_object(
            Bucket=bucket,
            Key=json_key,
            Body=json.dumps(metadata, ensure_ascii=False).encode("utf-8"),
            ContentType="application/json",
        )
        logger.info("Saved metadata cache: s3://%s/%s", bucket, json_key)
        return True
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Failed to save Docpamin cache: %s", exc)
        return False


def load_docpamin_cache_from_s3(bucket: str, prefix: str, cache_key: str) -> Tuple[Optional[str], Optional[Dict]]:
    """Load previously cached Docpamin output from S3."""
    try:
        cache_prefix = f"{prefix.rstrip('/')}/cache"

        md_key = f"{cache_prefix}/{cache_key}.md"
        logger.info("Checking cache: s3://%s/%s", bucket, md_key)
        md_response = s3_client.get_object(Bucket=bucket, Key=md_key)
        markdown = md_response["Body"].read().decode("utf-8")

        json_key = f"{cache_prefix}/{cache_key}.json"
        json_response = s3_client.get_object(Bucket=bucket, Key=json_key)
        metadata = json.loads(json_response["Body"].read().decode("utf-8"))

        logger.info("✅ Loaded from cache: %s (md_len=%s)", cache_key, len(markdown))
        return markdown, metadata

    except s3_client.exceptions.NoSuchKey:
        logger.info("Cache not found: %s", cache_key)
        return None, None
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Failed to load cache: %s", exc)
        return None, None


def _session_with_auth() -> requests.Session:
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {DOCPAMIN_API_KEY}"})
    return session


def parse_pdf_with_docpamin(pdf_path: str) -> Tuple[str, Dict]:
    """Parse a local PDF using Docpamin and return markdown plus metadata."""
    logger.info("Parsing via Docpamin: %s", pdf_path)
    session = _session_with_auth()
    req_timeout = 30

    try:
        with open(pdf_path, "rb") as handle:
            files = {"file": handle}
            data = {
                "alarm_options": json.dumps({"enabled": False}),
                "workflow_options": json.dumps({"workflow": "dp-o1", "image_export_mode": "embedded"}),
            }
            response = session.post(
                f"{DOCPAMIN_BASE_URL}/tasks",
                files=files,
                data=data,
                verify=DOCPAMIN_CRT_FILE,
                timeout=req_timeout,
            )
        response.raise_for_status()
        task_id = response.json().get("task_id")
        if not task_id:
            raise Exception("Docpamin: no task_id returned")

        logger.info("Docpamin task: %s", task_id)
        max_wait, waited, backoff = 600, 0, 2
        while waited < max_wait:
            status_response = session.get(
                f"{DOCPAMIN_BASE_URL}/tasks/{task_id}",
                verify=DOCPAMIN_CRT_FILE,
                timeout=req_timeout,
            )
            status_response.raise_for_status()
            status = status_response.json().get("status")
            if status == "DONE":
                break
            if status in {"FAILED", "ERROR"}:
                raise Exception(f"Docpamin task failed: {status}")
            time.sleep(backoff)
            waited += backoff
            backoff = min(backoff * 1.5, 10)
        if waited >= max_wait:
            raise Exception("Docpamin timeout")

        export_opts = {"task_ids": [task_id], "output_types": ["markdown", "json"]}
        export_response = session.post(
            f"{DOCPAMIN_BASE_URL}/tasks/export",
            json=export_opts,
            verify=DOCPAMIN_CRT_FILE,
            timeout=req_timeout,
        )
        export_response.raise_for_status()

        markdown = ""
        metadata: Dict = {}
        with zipfile.ZipFile(io.BytesIO(export_response.content)) as zipped:
            for filename in zipped.namelist():
                with zipped.open(filename) as fh:
                    if filename.endswith(".md"):
                        text = fh.read().decode("utf-8", errors="ignore")
                        if len(text) > len(markdown):
                            markdown = text
                    elif filename.endswith(".json"):
                        try:
                            metadata = json.loads(fh.read().decode("utf-8", errors="ignore"))
                        except Exception:
                            pass
        if not markdown:
            raise Exception("Docpamin: no markdown in export")

        paper_title = extract_title_from_markdown(markdown)
        metadata["extracted_title"] = paper_title
        logger.info("Docpamin parsed OK (md_len=%s)", len(markdown))

        md_cleaned, extracted_images = process_markdown_images(
            markdown,
            remove_for_llm=True,
            keep_representative=1,
        )

        if extracted_images:
            representative = select_representative_images(
                extracted_images,
                max_count=1,
                paper_title=paper_title,
            )
            metadata["images_info"] = {
                "total_images": len(extracted_images),
                "representative_images": representative,
            }
            logger.info(
                "Image preprocessing: %s images, markdown size reduced from %s to %s chars",
                len(extracted_images),
                len(markdown),
                len(md_cleaned),
            )

        return md_cleaned, metadata
    except Exception as exc:
        logger.error("Docpamin error: %s", exc)
        raise


def parse_pdf_with_docpamin_url(pdf_url: str, arxiv_id: str = "") -> Tuple[str, Dict]:
    """Parse a PDF via URL, using cache where possible."""
    cache_key = get_docpamin_cache_key(arxiv_id or pdf_url)
    bucket = S3_BUCKET
    prefix = S3_PAPERS_PREFIX

    cached_md, cached_meta = load_docpamin_cache_from_s3(bucket, prefix, cache_key)

    if cached_md and cached_meta:
        logger.info("📦 Using cached Docpamin for %s", cache_key)

        paper_title = extract_title_from_markdown(cached_md)
        cached_meta["extracted_title"] = paper_title

        md_cleaned, extracted_images = process_markdown_images(
            cached_md,
            remove_for_llm=True,
        )

        if extracted_images and cached_meta:
            figure_pairs = extract_figure_pairs_from_json(cached_meta)

            if figure_pairs:
                extracted_images = match_images_with_figure_pairs(extracted_images, figure_pairs)

            images_with_caption = [
                img
                for img in extracted_images
                if img.get("caption") and is_valid_caption(img.get("caption"))
            ]

            logger.info(
                "Images with valid captions: %s/%s",
                len(images_with_caption),
                len(extracted_images),
            )

            if images_with_caption:
                representative = select_representative_images(
                    images_with_caption,
                    max_count=1,
                    paper_title=paper_title,
                )

                cached_meta["images_info"] = {
                    "total_images": len(extracted_images),
                    "images_with_caption": len(images_with_caption),
                    "representative_images": representative,
                }

        return md_cleaned, cached_meta

    logger.info("Parsing via Docpamin (URL): %s", pdf_url)
    session = _session_with_auth()
    req_timeout = 30

    try:
        data = {
            "file_url": pdf_url,
            "alarm_options": json.dumps({"enabled": False}),
            "workflow_options": json.dumps({"workflow": "dp-o1", "image_export_mode": "embedded"}),
        }

        response = session.post(
            f"{DOCPAMIN_BASE_URL}/tasks",
            data=data,
            verify=DOCPAMIN_CRT_FILE,
            timeout=req_timeout,
        )
        response.raise_for_status()
        task_id = response.json().get("task_id")
        if not task_id:
            raise Exception("Docpamin: no task_id returned")

        logger.info("Docpamin task: %s", task_id)

        max_wait, waited, backoff = 600, 0, 2
        while waited < max_wait:
            status_response = session.get(
                f"{DOCPAMIN_BASE_URL}/tasks/{task_id}",
                verify=DOCPAMIN_CRT_FILE,
                timeout=req_timeout,
            )
            status_response.raise_for_status()
            status = status_response.json().get("status")
            if status == "DONE":
                break
            if status in {"FAILED", "ERROR"}:
                raise Exception(f"Docpamin task failed: {status}")
            time.sleep(backoff)
            waited += backoff
            backoff = min(backoff * 1.5, 10)

        if waited >= max_wait:
            raise Exception("Docpamin timeout")

        export_opts = {"task_ids": [task_id], "output_types": ["markdown", "json"]}
        export_response = session.post(
            f"{DOCPAMIN_BASE_URL}/tasks/export",
            json=export_opts,
            verify=DOCPAMIN_CRT_FILE,
            timeout=req_timeout,
        )
        export_response.raise_for_status()

        markdown = ""
        metadata: Dict = {}
        with zipfile.ZipFile(io.BytesIO(export_response.content)) as zipped:
            for filename in zipped.namelist():
                with zipped.open(filename) as fh:
                    if filename.endswith(".md"):
                        text = fh.read().decode("utf-8", errors="ignore")
                        if len(text) > len(markdown):
                            markdown = text
                    elif filename.endswith(".json"):
                        try:
                            metadata = json.loads(fh.read().decode("utf-8", errors="ignore"))
                        except Exception:
                            pass

        if not markdown:
            raise Exception("No markdown in export")

        save_docpamin_cache_to_s3(bucket, prefix, cache_key, markdown, metadata)

        paper_title = extract_title_from_markdown(markdown)
        metadata["extracted_title"] = paper_title
        metadata["from_cache"] = False

        logger.info("Docpamin parsed (md_len=%s, title=%s)", len(markdown), paper_title)

        md_cleaned, extracted_images = process_markdown_images(
            markdown,
            remove_for_llm=True,
            keep_representative=1,
        )

        if extracted_images:
            extracted_images = match_images_with_captions_from_json(extracted_images, metadata)

            representative = select_representative_images(
                extracted_images,
                max_count=1,
                paper_title=paper_title,
            )

            metadata["images_info"] = {
                "total_images": len(extracted_images),
                "representative_images": representative,
            }

        return md_cleaned, metadata

    except Exception as exc:
        logger.error("Docpamin error: %s", exc)
        raise


def extract_title_from_url(url: str) -> str:
    """Extract a friendly title (arXiv ID or filename) from a URL."""
    match = re.search(r"(\d{4}\.\d{5})", url)
    if match:
        return match.group(1)

    from urllib.parse import urlparse

    path = urlparse(url).path
    return Path(path).stem or "unknown"


def extract_title_from_markdown(markdown: str) -> str:
    """Return the first ``##`` heading in Docpamin markdown as the title."""
    try:
        lines = markdown.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("##") and not line.startswith("###"):
                title = line.lstrip("#").strip()
                if title:
                    logger.info("Extracted title from markdown: %s", title)
                    return title

        logger.warning("No title found in markdown")
        return "Unknown"

    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Error extracting title: %s", exc)
        return "Unknown"


__all__ = [
    "extract_title_from_markdown",
    "extract_title_from_url",
    "get_docpamin_cache_key",
    "load_docpamin_cache_from_s3",
    "parse_pdf_with_docpamin",
    "parse_pdf_with_docpamin_url",
    "save_docpamin_cache_to_s3",
]
