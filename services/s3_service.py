"""
AWS S3 service for paper storage and retrieval
"""
import boto3
from botocore.config import Config
from typing import List, Optional, Dict, Iterable
from pathlib import Path
import tempfile
import logging
import fnmatch
import os

from config.settings import AWS_ACCESS_KEY, AWS_SECRET_KEY

logger = logging.getLogger(__name__)

# ===== boto3 client =====
boto_config = Config(
    retries={"max_attempts": 5, "mode": "standard"},
    connect_timeout=10,
    read_timeout=60,
)
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    config=boto_config,
)


def _iter_s3_objects(bucket: str, prefix: str) -> Iterable[Dict]:
    """Iterate through S3 objects with pagination"""
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            yield obj


def get_paper_list_from_s3(bucket: str, prefix: str) -> Optional[List[str]]:
    """
    S3에서 paper_list.txt 파일을 읽어 URL 리스트 반환

    Args:
        bucket: S3 버킷명
        prefix: S3 prefix (예: kai_papers/w43)

    Returns:
        URL 리스트 또는 None (파일 없음)
    """
    paper_list_key = f"{prefix.rstrip('/')}/paper_list.txt"

    try:
        logger.info(f"Checking for paper list: s3://{bucket}/{paper_list_key}")

        response = s3_client.get_object(Bucket=bucket, Key=paper_list_key)
        content = response['Body'].read().decode('utf-8')

        # URL 파싱 (빈 줄, 주석 제외)
        urls = []
        for line in content.splitlines():
            line = line.strip()
            # 빈 줄이나 # 주석 제외
            if not line or line.startswith('#'):
                continue
            # URL 형식 확인
            if line.startswith('http'):
                urls.append(line)
            else:
                logger.warning(f"Invalid URL format: {line}")

        logger.info(f"Found {len(urls)} URLs in paper_list.txt")
        return urls if urls else None

    except s3_client.exceptions.NoSuchKey:
        logger.info(f"paper_list.txt not found: {paper_list_key}")
        return None
    except Exception as e:
        logger.error(f"Error reading paper_list.txt: {e}")
        return None


def get_s3_papers(
    bucket: str,
    prefix: str,
    file_pattern: str = "*.pdf",
    process_subdirectories: bool = True,
    min_size_bytes: int = 1024,
    max_size_bytes: int = 1024 * 1024 * 100,
) -> List[Dict]:
    """
    Get list of PDF papers from S3

    Args:
        bucket: S3 bucket name
        prefix: S3 prefix path
        file_pattern: File pattern to match (default: *.pdf)
        process_subdirectories: Include subdirectories
        min_size_bytes: Minimum file size
        max_size_bytes: Maximum file size

    Returns:
        List of paper metadata dictionaries
    """
    logger.info(f"Fetching papers from S3: s3://{bucket}/{prefix}")
    papers = []
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
    logger.info(f"Found {len(papers)} papers in S3")
    return papers


def download_pdf_from_s3(s3_key: str, s3_bucket: str) -> str:
    """
    Download PDF from S3 to temporary file

    Args:
        s3_key: S3 object key
        s3_bucket: S3 bucket name

    Returns:
        Path to temporary file

    Raises:
        Exception: If download fails
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        logger.info(f"Downloading PDF: s3://{s3_bucket}/{s3_key}")
        s3_client.download_fileobj(s3_bucket, s3_key, tmp)
        tmp.close()
        return tmp.name
    except Exception as e:
        tmp.close()
        if os.path.exists(tmp.name):
            try:
                os.unlink(tmp.name)
            except Exception:
                pass
        raise Exception(f"Failed to download PDF: {e}")
