#!/usr/bin/env python3

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Tuple, Iterable
from datetime import datetime
from pathlib import Path
import boto3
from botocore.config import Config
import tempfile
import logging
import json
import time
import requests
import zipfile
import io
import re
import fnmatch
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import textwrap
import base64
import hashlib
from dotenv import load_dotenv, dotenv_values

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== Image Preprocessing Functions =====

def remove_base64_images(markdown: str, replacement: str = "[Image]") -> Tuple[str, int]:
    """
    Base64 이미지를 플레이스홀더로 대체
    
    Returns:
        (cleaned_markdown, num_removed)
    """
    pattern = r'!\[([^\]]*)\]\(data:image/[^;]+;base64,[A-Za-z0-9+/=]+\)'
    cleaned, count = re.subn(pattern, replacement, markdown)
    if count > 0:
        logger.info(f"Removed {count} base64 images from markdown")
    return cleaned, count

def extract_base64_images(markdown: str) -> List[Dict]:
    """Markdown에서 base64 이미지 추출"""
    pattern = r'!\[([^\]]*)\]\(data:image/([^;]+);base64,([A-Za-z0-9+/=]+)\)'
    images = []
    for match in re.finditer(pattern, markdown):
        base64_data = match.group(3)
        size_bytes = len(base64_data) * 3 // 4
        images.append({
            'full_match': match.group(0),
            'alt_text': match.group(1),
            'mime_type': match.group(2),
            'base64_data': base64_data,
            'size_kb': size_bytes / 1024,
            'position': match.start()
        })
    return images

def extract_figure_caption(markdown: str, figure_index: int) -> str:
    """
    Markdown에서 특정 Figure의 caption 추출
    
    Args:
        markdown: 전체 markdown
        figure_index: Figure 번호 (0-based)
    
    Returns:
        Figure caption 텍스트
    """
    try:
        # Figure 번호 (1-based로 검색)
        fig_num = figure_index + 1
        
        # 패턴들 시도
        patterns = [
            # "Figure 1: Caption text"
            rf'Figure\s+{fig_num}[:\.]?\s*([^\n]+?)(?:\n|$)',
            # "Fig. 1: Caption text"
            rf'Fig\.?\s+{fig_num}[:\.]?\s*([^\n]+?)(?:\n|$)',
            # "Figure 1. Caption text"
            rf'Figure\s+{fig_num}\.\s*([^\n]+?)(?:\n|$)',
            # "**Figure 1:** Caption text"
            rf'\*\*Figure\s+{fig_num}\*\*[:\.]?\s*([^\n]+?)(?:\n|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, markdown, re.IGNORECASE)
            if match:
                caption = match.group(1).strip()
                # 너무 짧거나 의미없는 caption 제외
                if len(caption) > 10 and not caption.startswith('!['):
                    # 최대 200자로 제한
                    caption = caption[:200] if len(caption) > 200 else caption
                    logger.info(f"Found caption for Figure {fig_num}: {caption[:50]}...")
                    return caption
        
        logger.debug(f"No caption found for Figure {fig_num}")
        return ""
        
    except Exception as e:
        logger.error(f"Error extracting caption for Figure {figure_index + 1}: {e}")
        return ""

def select_representative_image(images: List[Dict], min_kb: float = 10, max_kb: float = 200) -> Optional[Dict]:
    """대표 이미지 선정 (크기 + 위치 기준)"""
    if not images:
        return None
    candidates = [img for img in images if min_kb <= img['size_kb'] <= max_kb]
    if not candidates:
        candidates = sorted(images, key=lambda x: abs(x['size_kb'] - (min_kb + max_kb) / 2))[:3]
    return min(candidates, key=lambda x: x['position']) if candidates else None

app = FastAPI(
    title="AI Paper Newsletter Processor",
    description="Processes AI papers from S3, parses with Docpamin, summarizes via LLM. v2: Smart Hybrid Chunking (header-based or token-based)",
    version="2.0.0",
    docs_url="/docs",
    redoc_url=None,
    openapi_url="/openapi.json",
    root_path="/proxy/7070",
)

# ===== Configuration (env) =====
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET = os.getenv("S3_BUCKET_NAME")
S3_PAPERS_PREFIX = os.getenv("S3_PAPERS_PREFIX", "papers/")

DOCPAMIN_API_KEY = os.getenv("DOCPAMIN_API_KEY")
DOCPAMIN_BASE_URL = os.getenv("DOCPAMIN_BASE_URL", "https://docpamin.superaip.samsungds.net/api/v1")
DOCPAMIN_CRT_FILE = os.getenv("DOCPAMIN_CRT_FILE", "/etc/ssl/certs/ca-certificates.crt")

print(f"crt : {DOCPAMIN_CRT_FILE}")

LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "30000"))

CONFLUENCE_URL = os.getenv("CONFLUENCE_URL")
CONFLUENCE_EMAIL = os.getenv("CONFLUENCE_EMAIL")
CONFLUENCE_API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN")
CONFLUENCE_SPACE_KEY = os.getenv("CONFLUENCE_SPACE_KEY")

MAX_WORKERS = int(os.getenv("MAX_WORKERS", "8"))

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

# ===== Models =====
class S3PapersRequest(BaseModel):
    bucket: Optional[str] = None
    prefix: Optional[str] = None
    file_pattern: Optional[str] = "*.pdf"
    process_subdirectories: bool = True
    week_label: Optional[str] = None
    upload_confluence: Optional[bool] = False
    # 개선 옵션
    use_hierarchical: bool = True  # 계층적 요약 사용
    use_overlap: bool = True  # overlap chunking 사용

class BatchProcessRequest(BaseModel):
    bucket: Optional[str] = None
    prefix: Optional[str] = None
    confluence_page_title: Optional[str] = None
    tags: List[str] = ["AI", "Research"]

class PaperAnalysis(BaseModel):
    title: str
    authors: List[str]
    abstract: str
    summary: str
    key_contributions: List[str]
    methodology: str
    results: str
    relevance_score: int
    tags: List[str]
    source_file: str

class DebugParseS3Request(BaseModel):
    bucket: Optional[str] = None
    key: str
    include_markdown: bool = False
    markdown_max_chars: int = 5000

class DebugSummarizeMarkdownRequest(BaseModel):
    title: str = "Untitled Paper"
    markdown: str
    include_section_summaries: bool = True
    include_final_analysis: bool = True
    return_markdown_preview_chars: int = 0
    # 개선 옵션
    use_hierarchical: bool = True
    use_overlap: bool = True
    show_intermediate_steps: bool = False  # 중간 단계 출력

class DebugSummarizeSectionsRequest(BaseModel):
    title: str = "Untitled Paper"
    sections: Dict[str, str] = Field(default_factory=dict)
    only_sections: Optional[List[str]] = None
    # 개선 옵션
    use_overlap: bool = True

# ===== Utilities =====
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
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        logger.info(f"Downloading PDF: s3://{s3_bucket}/{s3_key}")
        s3_client.download_fileobj(s3_bucket, s3_key, tmp)
        tmp.close()
        return tmp.name
    except Exception as e:
        tmp.close()
        if os.path.exists(tmp.name):
            try: os.unlink(tmp.name)
            except Exception: pass
        raise Exception(f"Failed to download PDF: {e}")

# ===== Docpamin =====
def parse_pdf_with_docpamin(pdf_path: str) -> Tuple[str, Dict]:
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
    URL을 사용하여 Docpamin으로 PDF 파싱
    
    Args:
        pdf_url: PDF URL
        arxiv_id: arXiv ID (임시 제목용)
    
    Returns:
        (markdown, metadata)
    """
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
            raise Exception("Docpamin: no markdown in export")
        
        paper_title = extract_title_from_markdown(md)
        meta['extracted_title'] = paper_title
        
        logger.info(f"Docpamin parsed OK (md_len={len(md)}, title={paper_title})")
        
        # 이미지 전처리
        md_cleaned, extracted_images = process_markdown_images(
            md, 
            remove_for_llm=True,
            keep_representative=1
        )
        
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
            logger.info(f"Image preprocessing: {len(extracted_images)} total, selected Figure {representative[0]['index']+1}")
        
        return md_cleaned, meta
        
    except Exception as e:
        logger.error(f"Docpamin error: {e}")
        raise

def extract_title_from_url(url: str) -> str:
    """
    URL에서 논문 제목 추출
    
    Args:
        url: arXiv URL (예: https://arxiv.org/pdf/2312.12391.pdf)
    
    Returns:
        제목 (arXiv ID)
    """
    # arXiv ID 추출
    match = re.search(r'(\d{4}\.\d{5})', url)
    if match:
        return match.group(1)
    
    # 일반 URL에서 파일명 추출
    from urllib.parse import urlparse
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

# ===== Image Processing =====
def process_markdown_images(
    markdown: str, 
    remove_for_llm: bool = True,
    keep_representative: int = 1
) -> Tuple[str, List[Dict]]:
    """
    Markdown에서 이미지 처리 + Caption 추출
    """
    pattern = r'!\[(.*?)\]\(data:image/([^;]+);base64,([A-Za-z0-9+/=]+)\)'
    
    images = []
    
    def extract_image(match):
        alt_text = match.group(1)
        img_type = match.group(2)
        base64_data = match.group(3)
        
        full_img = match.group(0)
        img_size = len(base64_data)
                
        images.append({
            'index': len(images),
            'alt': alt_text.strip(),
            'type': img_type,
            'size': img_size,
            'size_kb': img_size * 3 / 4 / 1024,
            'base64_data': base64_data,
            'full': full_img
        })
        
        if remove_for_llm:
            if alt_text.strip():
                return f"\n[Figure {len(images)}: {alt_text}]\n"
            else:
                return f"\n[Figure {len(images)}]\n"
        else:
            return full_img
    
    # 이미지 추출
    processed_md = re.sub(pattern, extract_image, markdown)
    
    # Caption 추출 (원본 markdown 사용)
    if images:
        for img in images:
            caption = extract_figure_caption(markdown, img['index'])
            if caption:
                img['caption'] = caption
        
        total_img_size = sum(img['size'] for img in images)
        logger.info(f"Processed {len(images)} images. Total: {total_img_size:,} chars")
    
    return processed_md, images

def select_representative_image_with_llm(images: List[Dict], paper_title: str = "") -> Dict:
    """
    LLM을 사용하여 가장 대표적인 이미지 선택 (Caption 포함)
    """
    if not images:
        return None
    
    if len(images) == 1:
        return images[0]
    
    try:
        # 이미지 정보 포맷 (Caption 우선 사용)
        image_descriptions = []
        for img in images:
            desc = f"Figure {img['index'] + 1}"
            
            # Caption 우선 (markdown에서 추출)
            if img.get('caption'):
                desc += f": {img['caption']}"
            elif img.get('alt') and img['alt'] != 'Image':
                desc += f": {img['alt']}"
            
            desc += f" (Size: {img['size_kb']:.1f}KB)"
            image_descriptions.append(desc)
        
        # 프롬프트
        prompt = f"""You are analyzing a research paper titled: "{paper_title}"

Below is a list of figures from this paper. Select the ONE figure that best represents the main contribution or overview of the paper. 

Prioritize figures that show (Important): 
1. Overall architecture/framework diagrams
2. System overview illustrations 
3. Main workflow diagrams

Avoid selecting:
- Detailed experimental result graphs or tables
- Comparison tables
- Ablation study charts
- Small component diagrams

Consider:
- Representative figure's caption usually include keywords such as **"Architecture," "Overview,"** or **"main workflow."**
- Image size is the least opponent that you should consider.

Figures:
{chr(10).join(f"{i+1}. {desc}" for i, desc in enumerate(image_descriptions))}

Respond with ONLY the number (1-{len(images)}) of the best representative figure. No explanation needed."""

        # call_llm 사용
        messages = [{"role": "user", "content": prompt}]
        response = call_llm(messages, max_tokens=1000)
        
        # 숫자 추출
        response_text = response.strip()
        numbers = re.findall(r'\d+', response_text)
        
        if not numbers:
            raise ValueError(f"No number found in LLM response: {response_text}")
        
        selected_num = int(numbers[0])
        selected_idx = selected_num - 1
        
        if 0 <= selected_idx < len(images):
            logger.info(f"LLM selected Figure {selected_num} as representative image (from response: '{response_text}')")
            return images[selected_idx]
        else:
            logger.warning(f"LLM returned invalid index: {selected_num} (valid: 1-{len(images)}), using largest")
            return max(images, key=lambda x: x['size_kb'])
            
    except Exception as e:
        logger.error(f"LLM image selection failed: {e}, falling back to size-based")
        return max(images, key=lambda x: x['size_kb'])


def select_representative_images(images: List[Dict], max_count: int = 1, paper_title: str = "") -> List[Dict]:
    """
    논문의 대표 이미지 선택 (LLM 기반)
    
    Args:
        images: 이미지 리스트
        max_count: 최대 선택 개수 (현재는 1개만)
        paper_title: 논문 제목
    
    Returns:
        선택된 이미지 리스트
    """
    if not images:
        return []
    
    if len(images) <= max_count:
        return images[:max_count]
    
    # LLM으로 대표 이미지 선택
    selected = select_representative_image_with_llm(images, paper_title)
    return [selected] if selected else []


# ===== LLM utils =====
def _estimate_tokens(s: str) -> int:
    """간단한 토큰 추정 (1 token ≈ 4 chars)"""
    return max(1, math.ceil(len(s) / 4))

def call_llm(messages: List[Dict], max_tokens: int = 2000) -> str:
    """LLM API 호출 (reasoning model 지원)"""
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
        
    except Exception as e:
        logger.error(f"LLM call failed: {type(e).__name__}: {e}")
        raise
        
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

def extract_all_headers(markdown_content: str) -> Dict[str, str]:
    """
    섹션 이름에 관계없이 주요 헤더만 추출하여 섹션으로 분리
    # (Level 1) 과 ## (Level 2) 헤더만 사용 (### 제외)
    """
    logger.info("Extracting main headers (# and ##) from markdown...")
    sections = []
    current_section = {"header": "preamble", "level": 0, "content": []}
    
    for line in markdown_content.splitlines():
        # 헤더 감지 (# 또는 ## 만, ### 제외!)
        header_match = re.match(r'^(#{1,2})\s+(.+)$', line.strip())
        
        if header_match:
            # 이전 섹션 저장
            if current_section["content"]:
                sections.append(current_section)
            
            # 새 섹션 시작
            level = len(header_match.group(1))
            header = header_match.group(2).strip()
            current_section = {
                "header": header,
                "level": level,
                "content": []
            }
        else:
            current_section["content"].append(line)
    
    # 마지막 섹션 저장
    if current_section["content"]:
        sections.append(current_section)
    
    # Dict로 변환 (키: section_N_header)
    result = {}
    for i, sec in enumerate(sections):
        # 헤더를 키로 사용 (중복 방지를 위해 번호 추가)
        header_clean = re.sub(r'[^\w\s-]', '', sec['header'])[:30]
        key = f"section_{i:02d}_{header_clean}"
        content = "\n".join(sec["content"])
        
        if content.strip():  # 빈 섹션 제외
            result[key] = content
            logger.info(f"Section '{sec['header']}': {len(content)} chars, ~{_estimate_tokens(content)} toks")
    
    return result

def chunk_by_tokens(text: str, chunk_size: int = 4000, overlap: int = 400) -> List[str]:
    """
    토큰 기반으로 텍스트를 균등 분할
    - 섹션 구조가 없거나 불명확할 때 사용
    """
    logger.info(f"Chunking by tokens (chunk_size={chunk_size}, overlap={overlap})...")
    
    total_chars = len(text)
    if total_chars <= chunk_size * 4:  # 한 청크에 들어감
        return [text]
    
    # 대략적인 문자 수 계산 (1 token ≈ 4 chars)
    char_per_chunk = chunk_size * 4
    overlap_chars = overlap * 4
    
    chunks = []
    start = 0
    
    while start < total_chars:
        end = start + char_per_chunk
        
        # 마지막 청크가 아니면 문장 경계에서 자르기
        if end < total_chars:
            # 마침표, 줄바꿈 등에서 자연스럽게 자르기
            for i in range(end, max(start + char_per_chunk // 2, end - 500), -1):
                if i < len(text) and text[i] in '.!?\n':
                    end = i + 1
                    break
        
        chunk = text[start:end]
        chunks.append(chunk)
        
        # 다음 청크 시작 위치 (overlap 적용)
        start = end - overlap_chars
        
        if start >= total_chars:
            break
    
    logger.info(f"Created {len(chunks)} chunks by tokens")
    return chunks

def smart_chunk_hybrid(markdown_content: str, min_headers: int = 10) -> Tuple[Dict[str, str], str]:
    """
    하이브리드 청킹: 헤더 개수에 따라 방식 선택
    
    기본값을 8→10으로 상향 조정하여 청크 수 더욱 감소
    # (Level 1) 헤더만 카운트 (## 제외)
    
    Returns:
        (chunks_dict, method_used)
        method_used: "header_based" or "token_based"
    """
    # 메인 헤더만 카운트 (# 만, ## 제외)
    main_headers = re.findall(r'^#\s+[^#].+$', markdown_content, re.MULTILINE)
    num_headers = len(main_headers)
    
    logger.info(f"Found {num_headers} main headers (# only) in markdown")
    
    if num_headers >= min_headers:
        # 충분한 헤더 → 헤더 기반 섹션 분리 (하지만 거의 없을 것)
        logger.info(f"Using HEADER-BASED chunking ({num_headers} headers >= {min_headers})")
        chunks = extract_all_headers(markdown_content)
        return chunks, "header_based"
    else:
        # 헤더 부족 → 토큰 기반 균등 분할 (대부분 이쪽)
        logger.info(f"Using TOKEN-BASED chunking ({num_headers} headers < {min_headers})")
        chunk_list = chunk_by_tokens(markdown_content, chunk_size=6000, overlap=600)
        chunks = {f"chunk_{i:02d}": chunk for i, chunk in enumerate(chunk_list)}
        return chunks, "token_based"

# ===== 개선된 요약 함수들 =====

def smart_chunk_with_overlap(text: str, chunk_size: int = 4000, overlap: int = 400) -> List[str]:
    """
    텍스트를 overlap을 가지고 청크로 분할
    - chunk_size: 각 청크의 대략적인 크기 (문자 단위)
    - overlap: 청크 간 겹치는 부분 (문자 단위)
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        
        # 마지막 청크가 아니면 overlap 적용
        if end < len(text):
            # 문장 경계에서 자르기 시도
            chunk_end = end
            # 마침표, 줄바꿈 등을 찾아서 자연스러운 경계 찾기
            for i in range(end, max(start + chunk_size // 2, end - 200), -1):
                if text[i] in '.!?\n':
                    chunk_end = i + 1
                    break
            chunks.append(text[start:chunk_end])
            start = chunk_end - overlap  # overlap만큼 뒤로
        else:
            chunks.append(text[start:])
            break
    
    logger.info(f"Split into {len(chunks)} chunks with overlap={overlap}")
    return chunks

def summarize_chunk_with_overlap(
    chunk_key: str, 
    chunk_content: str, 
    paper_title: str,
    use_overlap: bool = True,
    prev_summary: str = ""
) -> str:
    """
    청크를 overlap을 가지고 요약 (섹션 이름 무관)
    - chunk_key: "section_01_abstract" 또는 "chunk_00" 형식
    - prev_summary: 이전 청크의 요약 (context로 사용)
    """
    if not chunk_content.strip():
        return ""
    
    # 청크 타입에 따라 프롬프트 조정
    if chunk_key.startswith("section_"):
        # 섹션 기반: 헤더 이름 추출
        header_part = chunk_key.split("_", 2)[-1] if "_" in chunk_key else "content"
        prompt = f"다음 '{header_part}' 섹션의 핵심 내용을 요약하세요."
    else:
        # 토큰 기반: 순서만 표시
        chunk_num = chunk_key.split("_")[-1] if "_" in chunk_key else "0"
        prompt = f"논문의 일부분 (Part {chunk_num})을 요약하세요."
    
    budget_tokens = min(LLM_MAX_TOKENS - 800, 6000)
    approx_tokens = _estimate_tokens(chunk_content)
    
    # 청크가 너무 크면 다시 분할
    if approx_tokens <= budget_tokens:
        # 한 번에 처리 가능
        context_prompt = ""
        if prev_summary and use_overlap:
            context_prompt = f"\n이전 내용 요약: {prev_summary}\n"
        
        msgs = [
            {"role": "system", "content": "You are an expert AI paper analyst. Keep technical terms in English."},
            {"role": "user", "content": f"[{paper_title}] {prompt}{context_prompt}\n\n내용:\n{chunk_content}"},
        ]
        return call_llm(msgs, max_tokens=min(3000, LLM_MAX_TOKENS - 500))
    
    # 너무 크면 sub-chunk로 분할
    chunk_size_chars = budget_tokens * 4
    overlap_chars = 400 if use_overlap else 0
    
    sub_chunks = smart_chunk_with_overlap(chunk_content, chunk_size_chars, overlap_chars)
    
    summaries = []
    sub_prev_summary = prev_summary
    
    for i, sub_chunk in enumerate(sub_chunks):
        context_prompt = ""
        if sub_prev_summary and use_overlap:
            context_prompt = f"\n이전 내용: {sub_prev_summary}\n"
        
        msgs = [
            {"role": "system", "content": "You are an expert AI paper analyst. Keep technical terms in English."},
            {"role": "user", "content": f"[{paper_title}] {prompt} (sub-part {i+1}/{len(sub_chunks)}){context_prompt}\n\n{sub_chunk}"},
        ]
        summary = call_llm(msgs, max_tokens=min(3000, LLM_MAX_TOKENS - 500))
        summaries.append(summary)
        sub_prev_summary = summary
    
    # 여러 sub-chunk 요약을 병합
    if len(summaries) == 1:
        return summaries[0]
    
    merge_msgs = [
        {"role": "system", "content": "You are an expert AI paper analyst."},
        {"role": "user", "content": f"다음은 [{paper_title}]의 '{chunk_key}' 부분을 여러 sub-part로 나눠 요약한 결과입니다. 이를 하나의 일관된 요약으로 병합하세요:\n\n" + "\n\n---\n\n".join(summaries)},
    ]
    return call_llm(merge_msgs, max_tokens=3000)

def create_hierarchical_summary_v2(chunk_summaries: Dict[str, str], paper_title: str) -> Dict[str, str]:
    """
    계층적 요약 생성 v2 (위치 기반)
    Level 1: 청크 요약 (이미 완료)
    Level 2: 위치별 그룹 요약 (beginning, middle, end)
    Level 3: 최종 통합 요약
    
    섹션 이름에 의존하지 않고 논문의 위치(앞/중간/뒤)로 그룹핑
    """
    logger.info("Creating hierarchical summary (position-based)...")
    
    num_chunks = len(chunk_summaries)
    if num_chunks == 0:
        return {}
    
    # Level 2: 위치 기반 그룹핑
    items = list(chunk_summaries.items())
    
    if num_chunks <= 2:
        # 청크가 너무 적으면 그대로 사용
        groups = {"full": items}
    elif num_chunks <= 5:
        # 청크가 적으면 2개 그룹
        mid_point = num_chunks // 2
        groups = {
            "beginning": items[:mid_point],
            "end": items[mid_point:]
        }
    else:
        # 청크가 많으면 3개 그룹
        third = num_chunks // 3
        groups = {
            "beginning": items[:third],
            "middle": items[third:third*2],
            "end": items[third*2:]
        }
    
    # 그룹별 프롬프트
    group_prompts = {
        "full": "논문의 전체 내용을 종합적으로 요약하세요.",
        "beginning": "논문의 도입부 (배경, 문제 정의, 목표, 관련 연구)를 종합적으로 요약하세요.",
        "middle": "논문의 핵심 부분 (방법론, 실험 설계, 결과, 성능)을 종합적으로 요약하세요.",
        "end": "논문의 결론 부분 (인사이트, 한계, 기여, 향후 과제)을 종합적으로 요약하세요."
    }
    
    intermediate_summaries = {}
    for group_name, group_items in groups.items():
        if not group_items:
            continue
        
        # 그룹 내 요약들을 결합
        group_texts = [f"### Part {i+1}\n{summary}" for i, (key, summary) in enumerate(group_items)]
        combined = "\n\n".join(group_texts)
        
        prompt = group_prompts.get(group_name, "다음 내용을 종합적으로 요약하세요.")
        
        msgs = [
            {"role": "system", "content": "You are an expert AI paper analyst."},
            {"role": "user", "content": f"[{paper_title}] {prompt}\n\n{combined}"},
        ]
        
        intermediate_summaries[group_name] = call_llm(msgs, max_tokens=3000)
        logger.info(f"Created intermediate summary for '{group_name}' ({len(group_items)} chunks)")
    
    return intermediate_summaries

def _json_extract(s: str) -> Optional[Dict]:
    """문자열에서 JSON 추출"""
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def analyze_paper_with_llm_improved(
    paper_info: Dict, 
    markdown_content: str, 
    json_metadata: Dict,
    use_hierarchical: bool = True,
    use_overlap: bool = True,
    return_intermediate: bool = False
) -> Tuple[PaperAnalysis, Optional[Dict]]:
    """
    개선된 논문 분석 함수 v2 (하이브리드 청킹)
    - 헤더 많으면 → 섹션 기반
    - 헤더 적으면 → 토큰 기반
    - use_hierarchical: 계층적 요약 사용
    - use_overlap: overlap chunking 사용
    - return_intermediate: 중간 단계 결과 반환
    """
    logger.info(f"Analyzing paper (hierarchical={use_hierarchical}, overlap={use_overlap}): {paper_info.get('title','Unknown')}")
    
    # Step 0: 이미지 처리 (LLM 토큰 절약)
    clean_markdown, extracted_images = process_markdown_images(
        markdown_content, 
        remove_for_llm=True,
        keep_representative=1
    )
    
    if extracted_images:
        logger.info(f"Removed {len(extracted_images)} images for LLM processing")
    
    # Step 1: 스마트 청킹 (하이브리드) - clean_markdown 사용
    chunks, chunking_method = smart_chunk_hybrid(clean_markdown, min_headers=10)
    logger.info(f"Chunking method: {chunking_method}, total chunks: {len(chunks)}")
    
    # Step 2: 각 청크 요약
    chunk_summaries: Dict[str, str] = {}
    prev_summary = ""
    
    for chunk_key, content in chunks.items():
        if content.strip() and len(content.strip()) > 100:  # 100자 이상만
            summary = summarize_chunk_with_overlap(
                chunk_key, 
                content, 
                paper_info.get("title", "Unknown"),
                use_overlap=use_overlap,
                prev_summary=prev_summary if use_overlap else ""
            )
            chunk_summaries[chunk_key] = summary
            prev_summary = summary
    
    logger.info(f"Created {len(chunk_summaries)} chunk summaries")
    
    # Step 3: 계층적 요약 (옵션)
    intermediate_summaries = {}
    if use_hierarchical and len(chunk_summaries) > 0:
        intermediate_summaries = create_hierarchical_summary_v2(
            chunk_summaries, 
            paper_info.get("title", "Unknown")
        )
        # 계층적 요약을 사용하여 최종 분석
        combined = "\n\n".join([f"## {k.title()}\n{v}" for k, v in intermediate_summaries.items() if v.strip()])
    else:
        # 기존 방식: 청크 요약을 직접 결합
        combined = "\n\n".join([f"## {k}\n{v}" for k, v in chunk_summaries.items() if v.strip()])
    
    # Step 4: 최종 종합 분석
    format_hint = {
        "title": paper_info.get("title", "Unknown"),
        "tldr": "",
        "key_contributions": [],
        "methodology": "",
        "results": "",
        "novelty": "",
        "limitations": [],
        "relevance_score": 7,
        "tags": [],
    }
    
    final_prompt = f"""논문 "{paper_info.get('title','Unknown')}"의 {'계층적 ' if use_hierarchical else ''}요약이 아래에 있습니다:

{combined}

아래 JSON 스키마에 맞게 결과만 JSON으로 출력하세요(설명문 금지):
{json.dumps(format_hint, ensure_ascii=False, indent=2)}

규칙:
- key_contributions: 3~6개 bullet 수준의 간결 문장
- relevance_score: 1~10 정수
- tags: 5~8개 짧은 표제어 (영문)
- 한글로 작성
- 전문 용어는 English 그대로 유지
"""
    
    msgs = [
        {"role": "system", "content": "You are an expert AI/ML researcher. Return ONLY valid JSON."},
        {"role": "user", "content": final_prompt},
    ]
    final_out = call_llm(msgs, max_tokens=min(3000, LLM_MAX_TOKENS))
    parsed = _json_extract(final_out) or {}
    
    # Abstract 추출 (첫 번째 청크에서)
    abstract_text = ""
    for key, content in chunks.items():
        if len(content) < 2000:  # abstract는 보통 짧음
            abstract_text = content[:800]
            break
    
    analysis = PaperAnalysis(
        title=paper_info.get("title", "Unknown"),
        authors=paper_info.get("authors", []),
        abstract=abstract_text if abstract_text else "",
        summary=final_out if isinstance(final_out, str) else json.dumps(final_out, ensure_ascii=False),
        key_contributions=parsed.get("key_contributions", []),
        methodology=parsed.get("methodology", ""),
        results=parsed.get("results", ""),
        relevance_score=int(parsed.get("relevance_score", 7)),
        tags=parsed.get("tags", []),
        source_file=paper_info.get("s3_key", ""),
    )
    
    # 중간 단계 결과
    intermediate_data = None
    if return_intermediate:
        # 대표 이미지 선택
        representative_imgs = select_representative_images(extracted_images, max_count=1)
        
        intermediate_data = {
            "chunking_method": chunking_method,
            "num_chunks": len(chunks),
            "chunks_detected": list(chunks.keys()),
            "chunk_summaries": chunk_summaries,
            "intermediate_summaries": intermediate_summaries if use_hierarchical else {},
            "images": {
                "total_count": len(extracted_images),
                "removed_size": sum(img['size'] for img in extracted_images),
                "representative": [
                    {
                        "index": img['index'],
                        "alt": img['alt'],
                        "type": img['type'],
                        "size": img['size'],
                        "markdown": img['full']
                    }
                    for img in representative_imgs
                ]
            }
        }
    
    return analysis, intermediate_data

# 기존 함수 호환성 유지
def analyze_paper_with_llm(paper_info: Dict, markdown_content: str, json_metadata: Dict) -> PaperAnalysis:
    """기존 함수 (개선 버전 호출)"""
    analysis, _ = analyze_paper_with_llm_improved(
        paper_info, markdown_content, json_metadata,
        use_hierarchical=True, use_overlap=True, return_intermediate=False
    )
    return analysis

# ===== Confluence =====
def _conf_get_page_by_title(title: str) -> Optional[Dict]:
    url = f"{CONFLUENCE_URL}/rest/api/content"
    params = {"title": title, "spaceKey": CONFLUENCE_SPACE_KEY, "expand": "version"}
    r = requests.get(url, params=params, auth=(CONFLUENCE_EMAIL, CONFLUENCE_API_TOKEN), timeout=30)
    r.raise_for_status()
    res = r.json().get("results", [])
    return res[0] if res else None

def _conf_escape(s: str) -> str:
    return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

def upload_to_confluence(analyses: List[PaperAnalysis], page_title: str):
    logger.info(f"Uploading to Confluence: {page_title}")
    body = [f"<h1>AI Paper Newsletter - {datetime.now().strftime('%Y-%m-%d')}</h1>",
            "<p>이번 주의 주목할 만한 AI 논문들을 소개합니다.</p>",
            '<ac:structured-macro ac:name="info"><ac:rich-text-body>',
            f"<p>총 {len(analyses)}편의 논문이 분석되었습니다.</p>",
            "</ac:rich-text-body></ac:structured-macro><hr/>"]
    for i, a in enumerate(analyses, 1):
        body.append(f"<h2>{i}. {_conf_escape(a.title)}</h2>")
        if a.authors:
            body.append(f"<p><strong>Authors:</strong> {_conf_escape(', '.join(a.authors[:8]))}</p>")
        if a.tags:
            body.append(f"<p><strong>Tags:</strong> {_conf_escape(', '.join(a.tags))}</p>")
        if a.abstract:
            body.append("<h3>Abstract</h3><p>" + _conf_escape(a.abstract) + "</p>")
        body.append("<h3>Analysis</h3>")
        body.append(a.summary)
        body.append(f"<p><em>Source:</em> s3://{a.source_file}</p>")
        body.append("<hr/>")
    content_html = "\n".join(body)

    create_url = f"{CONFLUENCE_URL}/rest/api/content"
    headers = {"Content-Type": "application/json"}
    try:
        existing = _conf_get_page_by_title(page_title)
        if existing:
            page_id = existing["id"]
            version = existing.get("version", {}).get("number", 1) + 1
            payload = {
                "id": page_id, "type": "page", "title": page_title,
                "space": {"key": CONFLUENCE_SPACE_KEY},
                "body": {"storage": {"value": content_html, "representation": "storage"}},
                "version": {"number": version},
            }
            r = requests.put(f"{create_url}/{page_id}", json=payload, headers=headers,
                             auth=(CONFLUENCE_EMAIL, CONFLUENCE_API_TOKEN), timeout=60)
            r.raise_for_status()
            result = r.json()
        else:
            payload = {
                "type": "page", "title": page_title, "space": {"key": CONFLUENCE_SPACE_KEY},
                "body": {"storage": {"value": content_html, "representation": "storage"}},
            }
            r = requests.post(create_url, json=payload, headers=headers,
                              auth=(CONFLUENCE_EMAIL, CONFLUENCE_API_TOKEN), timeout=60)
            r.raise_for_status()
            result = r.json()

        base = CONFLUENCE_URL.rstrip("/")
        webui = result.get("_links", {}).get("webui")
        tiny = result.get("_links", {}).get("tinyui")
        page_url = f"{base}{webui}" if webui else (f"{base}{tiny}" if tiny else f"{base}/pages/{result['id']}")
        logger.info(f"Confluence page: {page_url}")
        return {"success": True, "page_url": page_url, "page_id": result["id"]}
    except Exception as e:
        logger.exception("Confluence upload error")
        raise

# ===== Markdown builder =====
def derive_week_label(prefix: str) -> str:
    m = re.search(r"w(\d{1,2})", prefix or "", re.IGNORECASE)
    if m:
        return f"w{int(m.group(1))}"
    iso_year, iso_week, _ = datetime.utcnow().isocalendar()
    return f"w{iso_week}"

def build_markdown(
    analyses: List[PaperAnalysis], 
    papers_metadata: Optional[List[Dict]] = None,
    week_label: str = "", 
    prefix: str = ""
) -> Tuple[str, str]:
    """
    논문 분석 결과를 Markdown으로 변환
    """
    if not week_label:
        week_label = derive_week_label(prefix)
    
    header = f"""# AI Paper Newsletter – {week_label}
_Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}_

Source prefix: `{prefix}`

---

"""
    
    # 이미지 매핑 생성
    image_map = {}
    if papers_metadata:
        for meta in papers_metadata:
            if meta.get('images_info') and meta.get('images_info', {}).get('representative_images'):
                s3_key = meta.get('s3_key', '')
                image_map[s3_key] = meta['images_info']
    
    parts = [header]
    
    for i, a in enumerate(analyses, 1):
        tags = f"**Tags:** {', '.join(a.tags)}" if a.tags else ""
        authors = f"**Authors:** {', '.join(a.authors[:8])}" if a.authors else ""
        
        abstract_block = ""
        if a.abstract and a.abstract.strip():
            abstract_block = f"\n**Abstract**\n\n> {a.abstract.strip()}\n\n"
        
        # ⭐ Summary JSON 파싱 및 개조식 변환
        summary_formatted = format_summary_as_markdown(a.summary)
        
        sec = f"""## {i}. {a.title}

{authors}
{tags}

{summary_formatted}

{abstract_block}"""
        
        # 이미지 섹션 추가
        if a.source_file in image_map:
            img_info = image_map[a.source_file]
            rep_imgs = img_info.get('representative_images', [])
            
            if rep_imgs:
                rep_img = rep_imgs[0]
                paper_name = Path(a.source_file).stem
                img_filename = f"{week_label}_{paper_name}_fig{rep_img['index'] + 1}.{rep_img['type']}"
                
                sec += f"""### 📊 대표 이미지

**전체 이미지:** {img_info['total_images']}개  
**대표 이미지:** Figure {rep_img['index'] + 1} ({rep_img['size_kb']:.1f}KB)

![Figure {rep_img['index'] + 1}](images/{img_filename})

"""
        
        sec += f"""**Source:** `s3://{a.source_file}`

---

"""
        parts.append(sec)
    
    md_content = "".join(parts)
    md_filename = f"{week_label}.md"
    return md_filename, md_content


def format_summary_as_markdown(summary: str) -> str:
    """
    Summary JSON을 보기 좋은 Markdown 개조식으로 변환
    
    Args:
        summary: JSON 형태의 summary 문자열
    
    Returns:
        포맷팅된 Markdown 문자열
    """
    try:
        # JSON 추출 시도
        summary_clean = summary.strip().replace('~', '–')
        
        # JSON 파싱
        json_match = re.search(r'\{[\s\S]*\}', summary_clean)
        if not json_match:
            # JSON이 없으면 원본 반환
            return f"**Summary**\n\n{summary_clean}\n"
        
        data = json.loads(json_match.group(0))
        
        # Markdown 개조식으로 변환
        lines = ["**Summary**  \n\n"]
        
        # TL;DR
        if data.get('tldr'):
            lines.append(f"**📌 TL;DR**\n")
            lines.append(f"{data['tldr']}\n\n")
        
        # 핵심 기여
        if data.get('key_contributions'):
            lines.append(f"**🎯 핵심 기여**\n")
            for contrib in data['key_contributions']:
                lines.append(f"- {contrib}\n")
            lines.append("\n")
        
        # 방법론
        if data.get('methodology'):
            lines.append(f"**🔬 방법론**\n")
            lines.append(f"{data['methodology']}\n\n")
        
        # 결과
        if data.get('results'):
            lines.append(f"**📊 결과**\n")
            lines.append(f"{data['results']}\n\n")
        
        # 새로운 점
        if data.get('novelty'):
            lines.append(f"**💡 새로운 점**\n")
            lines.append(f"{data['novelty']}\n\n")
        
        # 한계점
        if data.get('limitations'):
            lines.append(f"**⚠️ 한계점**\n")
            for limitation in data['limitations']:
                lines.append(f"- {limitation}\n")
            lines.append("\n")
        
        # Relevance Score
        if data.get('relevance_score'):
            score = data['relevance_score']
            stars = '⭐' * score
            lines.append(f"**관련성 점수:** {stars} ({score}/10)\n\n")
        
        return "".join(lines)
        
    except Exception as e:
        # JSON 파싱 실패 시 원본 반환
        logger.warning(f"Failed to parse summary JSON: {e}")
        return f"**Summary**\n\n{summary.strip().replace('~', '–')}\n"

# ===== Main Routes =====
@app.get("/")
async def root():
    return {
        "service": "AI Paper Newsletter Processor",
        "version": "2.0.0",
        "status": "running",
        "improvements": [
            "smart_hybrid_chunking (header-based or token-based)",
            "overlap_chunking",
            "hierarchical_summarization (position-based)"
        ],
        "available_endpoints": ["/health", "/list-s3-papers", "/process-s3-papers", "/debug/*"],
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat(), "mode": "S3-only (v2: hybrid chunking)"}

@app.get("/list-s3-papers")
async def list_s3_papers(bucket: Optional[str] = None, prefix: Optional[str] = None):
    """
    S3에서 논문 목록 조회
    - paper_list.txt가 있으면 URL 리스트 반환
    - 없으면 PDF 파일 목록 반환
    """
    bucket = bucket or S3_BUCKET
    prefix = prefix or S3_PAPERS_PREFIX
    
    try:
        paper_urls = get_paper_list_from_s3(bucket, prefix)
        
        if paper_urls:
            logger.info(f"Found paper_list.txt with {len(paper_urls)} URLs")
            papers = [
                {
                    "title": extract_title_from_url(url),
                    "s3_key": f"{prefix}/{extract_title_from_url(url)}.pdf",  # 가상 경로
                    "url": url,
                    "source": "url_list",
                    "last_modified": None,
                    "size_bytes": 0
                }
                for url in paper_urls
            ]
            
            return {
                "bucket": bucket,
                "prefix": prefix,
                "papers_found": len(papers),
                "source": "url_list",
                "paper_list_found": True,
                "papers": [
                    {
                        "title": p["title"],
                        "s3_key": p["s3_key"],
                        "url": p.get("url"),
                        "source": p.get("source"),
                        "last_modified": p.get("last_modified"),
                        "size_bytes": p.get("size_bytes", 0)
                    }
                    for p in papers
                ],
            }
        else:
            logger.info("paper_list.txt not found, listing PDF files")
            papers = get_s3_papers(bucket, prefix)
            
            return {
                "bucket": bucket,
                "prefix": prefix,
                "papers_found": len(papers),
                "source": "pdf_files",
                "paper_list_found": False,
                "papers": [
                    {
                        "title": p["title"],
                        "s3_key": p["s3_key"],
                        "source": p.get("source", "s3"),
                        "last_modified": p["last_modified"],
                        "size_bytes": p.get("size_bytes", 0)
                    }
                    for p in papers
                ],
            }
            
    except Exception as e:
        logger.error(f"list_s3_papers error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-s3-papers")
async def process_s3_papers(request: S3PapersRequest):
    logger.info(f"Processing S3 papers (hierarchical={request.use_hierarchical}, overlap={request.use_overlap})")
    bucket = request.bucket or S3_BUCKET
    prefix = request.prefix or S3_PAPERS_PREFIX
    
    try:
        # 1단계: paper_list.txt 확인
        paper_urls = get_paper_list_from_s3(bucket, prefix)
        
        if paper_urls:
            # URL 기반 처리
            logger.info(f"Processing {len(paper_urls)} papers from paper_list.txt")
            papers = [
                {
                    "title": extract_title_from_url(url),
                    "url": url,
                    "source": "url",
                    "s3_key": f"{prefix}/{extract_title_from_url(url)}.pdf"  # 가상 경로
                }
                for url in paper_urls
            ]
        else:
            # 기존 PDF 파일 기반 처리
            logger.info("paper_list.txt not found, processing PDF files")
            papers = get_s3_papers(
                bucket=bucket,
                prefix=prefix,
                file_pattern=request.file_pattern or "*.pdf",
                process_subdirectories=request.process_subdirectories,
            )
        
        if not papers:
            return {
                "message": "No papers found",
                "papers_found": 0,
                "papers_processed": 0,
                "bucket": bucket,
                "prefix": prefix,
                "md_filename": None,
                "md_content": "",
            }

        def _process_one(p: Dict) -> Tuple[Optional[PaperAnalysis], Optional[Dict], Optional[str]]:
            """논문 1개 처리"""
            tmp = None
            try:
                # URL 기반 vs 파일 기반 처리
                if p.get("source") == "url":
                    # URL에서 직접 파싱
                    md, meta = parse_pdf_with_docpamin_url(p["url"], p["title"])
                    
                    actual_title = meta.get('extracted_title', p["title"])
                    info = {
                        "title": actual_title,
                        "authors": [], 
                        "abstract": "", 
                        "s3_key": p["s3_key"]
                    }
                else:
                    # S3에서 다운로드 후 파싱
                    tmp = download_pdf_from_s3(p["s3_key"], p["s3_bucket"])
                    md, meta = parse_pdf_with_docpamin(tmp)
                    
                    actual_title = extract_title_from_markdown(md) if md else p["title"]
                    meta['extracted_title'] = actual_title
                    
                    info = {
                        "title": actual_title,
                        "authors": [], 
                        "abstract": "", 
                        "s3_key": p["s3_key"]
                    }
                
                a, _ = analyze_paper_with_llm_improved(
                    info, md, meta,
                    use_hierarchical=request.use_hierarchical,
                    use_overlap=request.use_overlap,
                    return_intermediate=False
                )
                a.source_file = p["s3_key"]
                return a, meta, None
                
            except Exception as e:
                return None, None, f"{p.get('title', 'unknown')}: {e}"
            finally:
                if tmp and os.path.exists(tmp):
                    try: os.unlink(tmp)
                    except Exception: pass

        analyses: List[PaperAnalysis] = []
        papers_metadata: List[Dict] = []
        errors: List[str] = []
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = [ex.submit(_process_one, p) for p in papers]
            for fut in as_completed(futs):
                a, meta, err = fut.result()
                if a: 
                    analyses.append(a)
                    if meta and meta.get('images_info'):
                        papers_metadata.append({
                            's3_key': a.source_file,
                            'title': a.title,
                            'images_info': meta['images_info']
                        })
                if err: 
                    errors.append(err)

        week_label = request.week_label or derive_week_label(prefix)
        md_filename, md_content = build_markdown(analyses, papers_metadata, week_label, prefix)

        confluence_result = None
        if request.upload_confluence and analyses:
            page_title = f"AI Paper Newsletter - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            confluence_result = upload_to_confluence(analyses, page_title)

        return {
            "message": "Processed",
            "papers_found": len(papers),
            "papers_processed": len(analyses),
            "errors": errors,
            "bucket": bucket,
            "prefix": prefix,
            "week_label": week_label,
            "md_filename": md_filename,
            "md_content": md_content,
            "papers_metadata": papers_metadata,
            "source": "url_list" if paper_urls else "pdf_files",
            "confluence_url": (confluence_result or {}).get("page_url"),
            "improvements_used": {
                "hierarchical": request.use_hierarchical,
                "overlap": request.use_overlap
            }
        }
    except Exception as e:
        logger.exception("process_s3_papers error")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-process")
async def batch_process_papers(request: BatchProcessRequest):
    logger.info("Batch processing papers")
    bucket = request.bucket or S3_BUCKET
    prefix = request.prefix or S3_PAPERS_PREFIX
    try:
        papers = get_s3_papers(bucket, prefix)
        if not papers:
            return {"message": "No papers found in S3", "papers_processed": 0}

        analyses: List[PaperAnalysis] = []
        for p in papers:
            tmp = None
            try:
                tmp = download_pdf_from_s3(p["s3_key"], p["s3_bucket"])
                md, meta = parse_pdf_with_docpamin(tmp)
                info = {"title": p["title"], "authors": [], "abstract": "", "s3_key": p["s3_key"]}
                a = analyze_paper_with_llm(info, md, meta)
                a.source_file = p["s3_key"]
                a.tags = request.tags
                analyses.append(a)
            except Exception as e:
                logger.error(f"Error processing: {e}")
            finally:
                if tmp and os.path.exists(tmp):
                    try: os.unlink(tmp)
                    except Exception: pass

        page_title = request.confluence_page_title or f"AI Paper Review - {datetime.now().strftime('%Y-%m-%d')}"
        confluence_result = upload_to_confluence(analyses, page_title)
        return {
            "message": "Successfully batch processed papers",
            "papers_found": len(papers),
            "papers_processed": len(analyses),
            "confluence_url": confluence_result.get("page_url"),
        }
    except Exception as e:
        logger.exception("batch_process error")
        raise HTTPException(status_code=500, detail=str(e))

# ===== Debug Endpoints =====

@app.post("/debug/parse-file")
async def debug_parse_file(
    file: UploadFile = File(...),
    include_markdown: bool = Form(False),
    markdown_max_chars: int = Form(5000),
):
    """로컬 PDF 파일 업로드 → Docpamin 파싱 테스트"""
    suffix = Path(file.filename).suffix or ".pdf"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        content = await file.read()
        tmp.write(content)
        tmp.close()

        md, meta = parse_pdf_with_docpamin(tmp.name)
        resp = {
            "filename": file.filename,
            "md_len": len(md),
            "md_preview": _preview(md, markdown_max_chars),
            "metadata": meta,
            "images_info": meta.get("images_info"),
            # 🔄  Backwards compatibility: older scripts expect `json_metadata`
            #     while newer ones read `metadata`.
            "json_metadata": meta,
        }
        if include_markdown:
            resp["markdown"] = md
        return resp
    finally:
        try:
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)
        except Exception:
            pass

@app.post("/debug/parse-s3")
async def debug_parse_s3(req: DebugParseS3Request):
    """S3 특정 key → Docpamin 파싱 테스트"""
    bucket = req.bucket or S3_BUCKET
    if not bucket:
        raise HTTPException(status_code=400, detail="Bucket is required")
    tmp = None
    try:
        tmp = download_pdf_from_s3(s3_key=req.key, s3_bucket=bucket)
        md, meta = parse_pdf_with_docpamin(tmp)
        resp = {
            "bucket": bucket,
            "key": req.key,
            "md_len": len(md),
            "md_preview": _preview(md, req.markdown_max_chars),
            "metadata": meta,
            "images_info": meta.get("images_info"),
            # 🔄  Maintain both keys so workflows consuming either continue working.
            "json_metadata": meta,
        }
        if req.include_markdown:
            resp["markdown"] = md
        return resp
    finally:
        try:
            if tmp and os.path.exists(tmp):
                os.unlink(tmp)
        except Exception:
            pass

@app.post("/debug/summarize-markdown")
async def debug_summarize_markdown(req: DebugSummarizeMarkdownRequest):
    """
    Markdown 텍스트 → 청크/최종 요약 테스트 (개선 버전 v2)
    - use_hierarchical: 계층적 요약 사용
    - use_overlap: overlap chunking 사용
    - show_intermediate_steps: 중간 단계 출력
    """
    if not req.markdown or not req.markdown.strip():
        raise HTTPException(status_code=400, detail="markdown is empty")

    # 스마트 청킹 (하이브리드)
    chunks, chunking_method = smart_chunk_hybrid(req.markdown, min_headers=5)
    
    chunk_summaries = {}
    if req.include_section_summaries:
        prev_summary = ""
        for chunk_key, chunk_text in chunks.items():
            if not chunk_text.strip() or len(chunk_text.strip()) < 100:
                continue
            chunk_summaries[chunk_key] = summarize_chunk_with_overlap(
                chunk_key, chunk_text, req.title, 
                use_overlap=req.use_overlap,
                prev_summary=prev_summary if req.use_overlap else ""
            )
            prev_summary = chunk_summaries[chunk_key]

    final_analysis = None
    intermediate_data = None
    if req.include_final_analysis:
        paper_info = {"title": req.title, "authors": [], "abstract": "", "s3_key": ""}
        final, intermediate = analyze_paper_with_llm_improved(
            paper_info, req.markdown, {},
            use_hierarchical=req.use_hierarchical,
            use_overlap=req.use_overlap,
            return_intermediate=req.show_intermediate_steps
        )
        final_analysis = {
            "title": final.title,
            "authors": final.authors,
            "abstract": final.abstract,
            "summary": final.summary,
            "key_contributions": final.key_contributions,
            "methodology": final.methodology,
            "results": final.results,
            "relevance_score": final.relevance_score,
            "tags": final.tags,
        }
        intermediate_data = intermediate

    return {
        "title": req.title,
        "chunking_method": chunking_method,
        "chunks_detected": list(chunks.keys()),
        "chunk_summaries": chunk_summaries if req.include_section_summaries else {},
        "final_analysis": final_analysis,
        "intermediate_steps": intermediate_data if req.show_intermediate_steps else None,
        "markdown_preview": _preview(req.markdown, req.return_markdown_preview_chars),
        "improvements_used": {
            "hierarchical": req.use_hierarchical,
            "overlap": req.use_overlap
        }
    }

@app.post("/debug/summarize-sections")
async def debug_summarize_sections(req: DebugSummarizeSectionsRequest):
    """섹션/청크 dict → 요약 테스트 (개선 버전)"""
    if not req.sections:
        raise HTTPException(status_code=400, detail="sections is empty")

    targets = req.only_sections or list(req.sections.keys())
    out: Dict[str, str] = {}
    prev_summary = ""
    
    for sec in targets:
        text = req.sections.get(sec, "")
        if not text.strip():
            continue
        out[sec] = summarize_chunk_with_overlap(
            sec, text, req.title, 
            use_overlap=req.use_overlap,
            prev_summary=prev_summary if req.use_overlap else ""
        )
        prev_summary = out[sec]

    return {
        "title": req.title,
        "summarized_sections": list(out.keys()),
        "summaries": out,
        "improvements_used": {
            "overlap": req.use_overlap
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("paper_processor:app", host="0.0.0.0", port=7070, reload=False, workers=2)

    