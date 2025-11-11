"""
Docpamin Service for PDF Parsing

This module handles all interactions with the Docpamin API including:
- PDF parsing via file upload or URL
- Caching parsed results to S3
- Image extraction and processing
- Title extraction from markdown
"""

import logging
import json
import time
import requests
import zipfile
import io
import re
import hashlib
from typing import Dict, Tuple, Optional
from pathlib import Path
from urllib.parse import urlparse

# Import S3 client
from services.s3_service import s3_client

# Import image processing functions
from utils.image_processing import (
    process_markdown_images,
    select_representative_images,
    extract_figure_pairs_from_json,
    match_images_with_figure_pairs,
    match_images_with_captions_from_json,
    is_valid_caption
)

# Import configuration
from config.settings import (
    DOCPAMIN_API_KEY,
    DOCPAMIN_BASE_URL,
    DOCPAMIN_CRT_FILE,
    S3_BUCKET,
    S3_PAPERS_PREFIX
)

logger = logging.getLogger(__name__)


# ===== Cache Management =====

def get_docpamin_cache_key(source_identifier: str) -> str:
    """
    캐시 키 생성 (arXiv ID 또는 파일 해시)

    Args:
        source_identifier: arXiv ID (예: 2510.11701) 또는 S3 key

    Returns:
        캐시 키 (예: "2510.11701" 또는 MD5 해시)
    """
    # arXiv ID 추출
    arxiv_match = re.search(r'(\d{4}\.\d{5})', source_identifier)
    if arxiv_match:
        return arxiv_match.group(1)

    # 일반 파일명에서 해시 생성
    return hashlib.md5(source_identifier.encode()).hexdigest()[:16]


def save_docpamin_cache_to_s3(
    bucket: str,
    prefix: str,
    cache_key: str,
    markdown: str,
    metadata: Dict
) -> bool:
    """
    Docpamin 결과를 S3에 캐싱

    Args:
        bucket: S3 버킷
        prefix: S3 prefix (예: kai_papers/w44)
        cache_key: 캐시 키
        markdown: 파싱된 markdown
        metadata: JSON metadata

    Returns:
        성공 여부
    """
    try:
        cache_prefix = f"{prefix.rstrip('/')}/cache"

        # Markdown 저장
        md_key = f"{cache_prefix}/{cache_key}.md"
        s3_client.put_object(
            Bucket=bucket,
            Key=md_key,
            Body=markdown.encode('utf-8'),
            ContentType='text/markdown'
        )
        logger.info(f"Saved markdown cache: s3://{bucket}/{md_key}")

        # Metadata 저장
        json_key = f"{cache_prefix}/{cache_key}.json"
        s3_client.put_object(
            Bucket=bucket,
            Key=json_key,
            Body=json.dumps(metadata, ensure_ascii=False).encode('utf-8'),
            ContentType='application/json'
        )
        logger.info(f"Saved metadata cache: s3://{bucket}/{json_key}")

        return True

    except Exception as e:
        logger.error(f"Failed to save Docpamin cache: {e}")
        return False


def load_docpamin_cache_from_s3(
    bucket: str,
    prefix: str,
    cache_key: str
) -> Tuple[Optional[str], Optional[Dict]]:
    """
    S3에서 Docpamin 캐시 로드

    Args:
        bucket: S3 버킷
        prefix: S3 prefix (예: kai_papers/w44)
        cache_key: 캐시 키

    Returns:
        (markdown, metadata) 또는 (None, None)
    """
    try:
        cache_prefix = f"{prefix.rstrip('/')}/cache"

        # Markdown 로드
        md_key = f"{cache_prefix}/{cache_key}.md"
        logger.info(f"Checking cache: s3://{bucket}/{md_key}")

        md_response = s3_client.get_object(Bucket=bucket, Key=md_key)
        markdown = md_response['Body'].read().decode('utf-8')

        # Metadata 로드
        json_key = f"{cache_prefix}/{cache_key}.json"
        json_response = s3_client.get_object(Bucket=bucket, Key=json_key)
        metadata = json.loads(json_response['Body'].read().decode('utf-8'))

        logger.info(f"✅ Loaded from cache: {cache_key} (md_len={len(markdown)})")
        return markdown, metadata

    except s3_client.exceptions.NoSuchKey:
        logger.info(f"Cache not found: {cache_key}")
        return None, None
    except Exception as e:
        logger.error(f"Failed to load cache: {e}")
        return None, None


# ===== PDF Parsing =====

def parse_pdf_with_docpamin(pdf_path: str) -> Tuple[str, Dict]:
    """
    Docpamin API를 사용하여 PDF 파싱 (파일 업로드)

    Args:
        pdf_path: 로컬 PDF 파일 경로

    Returns:
        (cleaned_markdown, metadata): 이미지가 정리된 마크다운과 메타데이터

    Raises:
        Exception: 파싱 실패 시
    """
    logger.info(f"Parsing via Docpamin: {pdf_path}")
    headers = {"Authorization": f"Bearer {DOCPAMIN_API_KEY}"}
    session = requests.Session()
    session.headers.update(headers)
    REQ_TIMEOUT = 30

    try:
        with open(pdf_path, "rb") as f:
            files = {"file": f}
            data = {
                "alarm_options": json.dumps({"enabled": False}),
                "workflow_options": json.dumps({
                    "workflow": "dp-o1",
                    "image_export_mode": "embedded"
                }),
            }
            r = session.post(f"{DOCPAMIN_BASE_URL}/tasks", files=files, data=data,
                             verify=DOCPAMIN_CRT_FILE, timeout=REQ_TIMEOUT)
        r.raise_for_status()
        task_id = r.json().get("task_id")
        if not task_id:
            raise Exception("Docpamin: no task_id returned")

        logger.info(f"Docpamin task: {task_id}")
        max_wait, waited, backoff = 600, 0, 2
        while waited < max_wait:
            s = session.get(f"{DOCPAMIN_BASE_URL}/tasks/{task_id}",
                            verify=DOCPAMIN_CRT_FILE, timeout=REQ_TIMEOUT)
            s.raise_for_status()
            status = s.json().get("status")
            if status == "DONE":
                break
            if status in {"FAILED", "ERROR"}:
                raise Exception(f"Docpamin task failed: {status}")
            time.sleep(backoff)
            waited += backoff
            backoff = min(backoff * 1.5, 10)
        if waited >= max_wait:
            raise Exception("Docpamin timeout")

        opts = {"task_ids": [task_id], "output_types": ["markdown", "json"]}
        e = session.post(f"{DOCPAMIN_BASE_URL}/tasks/export", json=opts,
                         verify=DOCPAMIN_CRT_FILE, timeout=REQ_TIMEOUT)
        e.raise_for_status()

        md, meta = "", {}
        with zipfile.ZipFile(io.BytesIO(e.content)) as zf:
            for fn in zf.namelist():
                with zf.open(fn) as fh:
                    if fn.endswith(".md"):
                        s = fh.read().decode("utf-8", errors="ignore")
                        if len(s) > len(md):
                            md = s
                    elif fn.endswith(".json"):
                        try:
                            meta = json.loads(fh.read().decode("utf-8", errors="ignore"))
                        except Exception:
                            pass
        if not md:
            raise Exception("Docpamin: no markdown in export")

        paper_title = extract_title_from_markdown(md)
        meta['extracted_title'] = paper_title
        logger.info(f"Docpamin parsed OK (md_len={len(md)})")

        # 이미지 전처리: base64 제거, 대표 이미지 추출
        md_cleaned, extracted_images = process_markdown_images(
            md,
            remove_for_llm=True,  # LLM 입력용으로 base64 제거
            keep_representative=1
        )

        # 메타데이터에 이미지 정보 추가
        if extracted_images:
            representative = select_representative_images(
                extracted_images,
                max_count=1,
                paper_title=paper_title
            )
            meta['images_info'] = {
                'total_images': len(extracted_images),
                'representative_images': representative
            }
            logger.info(f"Image preprocessing: {len(extracted_images)} images, "
                       f"markdown size reduced from {len(md)} to {len(md_cleaned)} chars")

        return md_cleaned, meta
    except Exception as e:
        logger.error(f"Docpamin error: {e}")
        raise


def parse_pdf_with_docpamin_url(pdf_url: str, arxiv_id: str = "") -> Tuple[str, Dict]:
    """
    Docpamin API를 사용하여 PDF 파싱 (URL 기반, 캐싱 지원)

    Args:
        pdf_url: PDF 다운로드 URL
        arxiv_id: arXiv ID (캐싱용, 선택사항)

    Returns:
        (cleaned_markdown, metadata): 이미지가 정리된 마크다운과 메타데이터

    Raises:
        Exception: 파싱 실패 시

    Note:
        - S3 캐시를 먼저 확인하고, 없으면 Docpamin API 호출
        - JSON metadata에서 Figure-Caption 쌍을 추출하여 이미지에 매칭
        - Caption이 있는 이미지 중 대표 이미지 선택
    """
    cache_key = get_docpamin_cache_key(arxiv_id or pdf_url)
    bucket = S3_BUCKET
    prefix = S3_PAPERS_PREFIX

    cached_md, cached_meta = load_docpamin_cache_from_s3(bucket, prefix, cache_key)

    if cached_md and cached_meta:
        logger.info(f"📦 Using cached Docpamin for {cache_key}")

        paper_title = extract_title_from_markdown(cached_md)
        cached_meta['extracted_title'] = paper_title

        # ⭐ 이미지만 추출 (caption 없음)
        md_cleaned, extracted_images = process_markdown_images(
            cached_md,
            remove_for_llm=True
        )

        if extracted_images and cached_meta:
            # ⭐ JSON에서만 caption 매칭!
            figure_pairs = extract_figure_pairs_from_json(cached_meta)

            if figure_pairs:
                extracted_images = match_images_with_figure_pairs(
                    extracted_images,
                    figure_pairs
                )

            # ⭐ Caption 있는 이미지만 선택
            images_with_caption = [
                img for img in extracted_images
                if img.get('caption') and is_valid_caption(img.get('caption'))
            ]

            logger.info(f"Images with valid captions: {len(images_with_caption)}/{len(extracted_images)}")

            if images_with_caption:
                representative = select_representative_images(
                    images_with_caption,
                    max_count=1,
                    paper_title=paper_title
                )

                cached_meta['images_info'] = {
                    'total_images': len(extracted_images),
                    'images_with_caption': len(images_with_caption),
                    'representative_images': representative
                }

        return md_cleaned, cached_meta

    # 캐시 없음 → Docpamin 파싱
    logger.info(f"Parsing via Docpamin (URL): {pdf_url}")
    headers = {"Authorization": f"Bearer {DOCPAMIN_API_KEY}"}
    session = requests.Session()
    session.headers.update(headers)
    REQ_TIMEOUT = 30

    try:
        data = {
            "file_url": pdf_url,
            "alarm_options": json.dumps({"enabled": False}),
            "workflow_options": json.dumps({
                "workflow": "dp-o1",
                "image_export_mode": "embedded"
            }),
        }

        r = session.post(
            f"{DOCPAMIN_BASE_URL}/tasks",
            data=data,
            verify=DOCPAMIN_CRT_FILE,
            timeout=REQ_TIMEOUT
        )
        r.raise_for_status()
        task_id = r.json().get("task_id")
        if not task_id:
            raise Exception("Docpamin: no task_id returned")

        logger.info(f"Docpamin task: {task_id}")

        # 상태 폴링
        max_wait, waited, backoff = 600, 0, 2
        while waited < max_wait:
            s = session.get(
                f"{DOCPAMIN_BASE_URL}/tasks/{task_id}",
                verify=DOCPAMIN_CRT_FILE,
                timeout=REQ_TIMEOUT
            )
            s.raise_for_status()
            status = s.json().get("status")
            if status == "DONE":
                break
            if status in {"FAILED", "ERROR"}:
                raise Exception(f"Docpamin task failed: {status}")
            time.sleep(backoff)
            waited += backoff
            backoff = min(backoff * 1.5, 10)

        if waited >= max_wait:
            raise Exception("Docpamin timeout")

        # Export
        opts = {"task_ids": [task_id], "output_types": ["markdown", "json"]}
        e = session.post(
            f"{DOCPAMIN_BASE_URL}/tasks/export",
            json=opts,
            verify=DOCPAMIN_CRT_FILE,
            timeout=REQ_TIMEOUT
        )
        e.raise_for_status()

        md, meta = "", {}
        with zipfile.ZipFile(io.BytesIO(e.content)) as zf:
            for fn in zf.namelist():
                with zf.open(fn) as fh:
                    if fn.endswith(".md"):
                        s = fh.read().decode("utf-8", errors="ignore")
                        if len(s) > len(md):
                            md = s
                    elif fn.endswith(".json"):
                        try:
                            meta = json.loads(fh.read().decode("utf-8", errors="ignore"))
                        except Exception:
                            pass

        if not md:
            raise Exception("No markdown in export")

        # 캐시 저장
        save_docpamin_cache_to_s3(bucket, prefix, cache_key, md, meta)

        paper_title = extract_title_from_markdown(md)
        meta['extracted_title'] = paper_title
        meta['from_cache'] = False

        logger.info(f"Docpamin parsed (md_len={len(md)}, title={paper_title})")

        md_cleaned, extracted_images = process_markdown_images(
            md,
            remove_for_llm=True,
            keep_representative=1
        )

        if extracted_images:
            # ⭐ JSON metadata 사용
            extracted_images = match_images_with_captions_from_json(
                extracted_images,
                meta
            )

            representative = select_representative_images(
                extracted_images,
                max_count=1,
                paper_title=paper_title
            )

            meta['images_info'] = {
                'total_images': len(extracted_images),
                'representative_images': representative
            }

        return md_cleaned, meta

    except Exception as e:
        logger.error(f"Docpamin error: {e}")
        raise


# ===== Title Extraction =====

def extract_title_from_url(url: str) -> str:
    """
    URL에서 논문 제목 추출

    Args:
        url: arXiv URL (예: https://arxiv.org/pdf/2312.12391.pdf)

    Returns:
        제목 (arXiv ID 또는 파일명)
    """
    # arXiv ID 추출
    match = re.search(r'(\d{4}\.\d{5})', url)
    if match:
        return match.group(1)

    # 일반 URL에서 파일명 추출
    path = urlparse(url).path
    return Path(path).stem or "unknown"


def extract_title_from_markdown(markdown: str) -> str:
    """
    Docpamin markdown에서 논문 제목 추출

    Args:
        markdown: Docpamin이 반환한 markdown

    Returns:
        논문 제목 (첫 번째 ## 헤딩)
    """
    try:
        # 첫 번째 ## 헤딩 찾기
        lines = markdown.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('##') and not line.startswith('###'):
                # ## 제거하고 제목만 추출
                title = line.lstrip('#').strip()
                if title:
                    logger.info(f"Extracted title from markdown: {title}")
                    return title

        logger.warning("No title found in markdown")
        return "Unknown"

    except Exception as e:
        logger.error(f"Error extracting title: {e}")
        return "Unknown"
