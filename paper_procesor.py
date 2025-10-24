#!/usr/bin/env python3
"""
AI Paper Newsletter Processor (S3-only, Markdown + Git-ready)
- S3에서 PDF 수집
- Docpamin(API)로 파싱 → LLM으로 섹션/전체 요약
- Markdown 문서 생성 (wNN.md)
- (옵션) Confluence 업로드
- n8n에서 md를 받아 Git에 업로드 가능
"""

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Paper Newsletter Processor",
    description="Processes AI papers from S3, parses with Docpamin, summarizes via LLM.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None,
    openapi_url="/openapi.json",
    root_path="/proxy/7070",   # ★ 고정 프리픽스
)


# ===== Configuration (env) =====
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET = os.getenv("S3_BUCKET_NAME")
S3_PAPERS_PREFIX = os.getenv("S3_PAPERS_PREFIX", "papers/")

DOCPAMIN_API_KEY = os.getenv("DOCPAMIN_API_KEY")
DOCPAMIN_BASE_URL = os.getenv("DOCPAMIN_BASE_URL", "https://docpamin.superaip.samsungds.net/api/v1")
DOCPAMIN_CRT_FILE = os.getenv("DOCPAMIN_CRT_FILE", "/etc/ssl/certs/ca-certificates.crt")

LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))

CONFLUENCE_URL = os.getenv("CONFLUENCE_URL")
CONFLUENCE_EMAIL = os.getenv("CONFLUENCE_EMAIL")
CONFLUENCE_API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN")
CONFLUENCE_SPACE_KEY = os.getenv("CONFLUENCE_SPACE_KEY")

MAX_WORKERS = int(os.getenv("MAX_WORKERS", "2"))

# ===== boto3 client (retry/timeout tuned) =====
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
    # 신규 옵션
    week_label: Optional[str] = None
    upload_confluence: Optional[bool] = False

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
    source_file: str  # S3 key

# ==== [요청/응답 모델] ====
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
    return_markdown_preview_chars: int = 0  # 0이면 미포함

class DebugSummarizeSectionsRequest(BaseModel):
    title: str = "Untitled Paper"
    sections: Dict[str, str] = Field(
        default_factory=dict,
        description="{'abstract': '...', 'introduction': '...', ...}"
    )
    only_sections: Optional[List[str]] = None  # 특정 섹션만 요약하고 싶으면 지정

# ===== Utilities =====
def _iter_s3_objects(bucket: str, prefix: str) -> Iterable[Dict]:
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            yield obj

def _preview(text: str, limit: int) -> str: # 미리보기 잘라내기
    if not text or limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...\n(truncated)"

def get_s3_papers(
    bucket: str,
    prefix: str,
    file_pattern: str = "*.pdf",
    process_subdirectories: bool = True,
    min_size_bytes: int = 1024,
    max_size_bytes: int = 1024 * 1024 * 100,
) -> List[Dict]:
    logger.info(f"Fetching papers from S3: s3://{bucket}/{prefix} (pattern={file_pattern})")
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

    logger.info(f"Found {len(papers)} papers in S3 (after filtering)")
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
                    "workflow": "docling",
                    "worker_options": {
                        "docling_to_formats": ["md", "json"],
                        "docling_image_export_mode": "embedded",
                    },
                }),
            }
            r = session.post(f"{DOCPAMIN_BASE_URL}/tasks", files=files, data=data,
                             verify=DOCPAMIN_CRT_FILE, timeout=REQ_TIMEOUT)
        r.raise_for_status()
        task_id = r.json().get("task_id")
        if not task_id:
            raise Exception("Docpamin: no task_id returned")

        logger.info(f"Docpamin task: {task_id}")
        # Poll
        max_wait, waited, backoff = 600, 0, 2
        while waited < max_wait:
            s = session.get(f"{DOCPAMIN_BASE_URL}/tasks/{task_id}",
                            verify=DOCPAMIN_CRT_FILE, timeout=REQ_TIMEOUT)
            s.raise_for_status()
            status = s.json().get("status")
            logger.info(f"Docpamin status={status}")
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
        logger.info(f"Docpamin parsed OK (md_len={len(md)})")
        return md, meta
    except Exception as e:
        logger.error(f"Docpamin error: {e}")
        raise

# ===== LLM utils =====
def _estimate_tokens(s: str) -> int:
    return max(1, math.ceil(len(s) / 4))

def call_llm(messages: List[Dict], max_tokens: int = 2000) -> str:
    headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": LLM_MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0.2}
    try:
        r = requests.post(f"{LLM_BASE_URL}/chat/completions", headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        j = r.json()
        return j["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise

def extract_sections_from_markdown(markdown_content: str) -> Dict[str, str]:
    logger.info("Extracting sections from markdown...")
    sections = {k: "" for k in ["abstract","introduction","methods","results","discussion","conclusion","other"]}
    header_map = {
        "abstract": r"^#{1,3}\s*(abstract|요약)\b",
        "introduction": r"^#{1,3}\s*(introduction|서론|배경)\b",
        "methods": r"^#{1,3}\s*(methods?|methodology|approach|방법)\b",
        "results": r"^#{1,3}\s*(results?|experiments?|결과|실험)\b",
        "discussion": r"^#{1,3}\s*(discussion|analysis|분석|논의)\b",
        "conclusion": r"^#{1,3}\s*(conclusion|summary|결론|요약)\b",
    }
    current = "other"
    buf: List[str] = []
    for line in markdown_content.splitlines():
        matched = None
        for sec, pat in header_map.items():
            if re.search(pat, line.strip(), re.IGNORECASE):
                matched = sec; break
        if matched:
            if buf:
                sections[current] += "\n".join(buf) + "\n"
                buf = []
            current = matched
        buf.append(line)
    if buf:
        sections[current] += "\n".join(buf)
    for k, v in sections.items():
        if v.strip():
            logger.info(f"Section '{k}': {len(v)} chars, ~{_estimate_tokens(v)} toks")
    return sections

def summarize_section(section_name: str, section_content: str, paper_title: str) -> str:
    if not section_content.strip():
        return ""
    prompts = {
        "abstract": "Abstract 핵심을 간결하게 요약.",
        "introduction": "Introduction의 배경/문제/목표를 요약.",
        "methods": "Methods의 핵심 알고리즘/모델/데이터/학습설정 요약.",
        "results": "Results의 주요 실험설정/지표/비교결과 요약.",
        "discussion": "Discussion의 인사이트/의미/한계 요약.",
        "conclusion": "Conclusion의 결론/기여/향후과제 요약.",
        "other": "다음 내용을 핵심 위주로 요약.",
    }
    prompt = prompts.get(section_name, prompts["other"])
    budget_tokens = min(LLM_MAX_TOKENS - 800, 3000)
    approx = _estimate_tokens(section_content)
    chunks = [section_content]
    if approx > budget_tokens:
        n = math.ceil(approx / budget_tokens)
        step = math.ceil(len(section_content) / n)
        chunks = [section_content[i:i+step] for i in range(0, len(section_content), step)]

    summaries = []
    for ch in chunks:
        msgs = [
            {"role": "system", "content": "You are an expert AI paper analyst. Keep technical terms in English."},
            {"role": "user", "content": f"[{paper_title}] {prompt}\n\nContent:\n{ch}"},
        ]
        summaries.append(call_llm(msgs, max_tokens=min(1500, LLM_MAX_TOKENS - 500)))

    if len(summaries) == 1:
        return summaries[0]
    merge_msgs = [
        {"role": "system", "content": "You are an expert AI paper analyst."},
        {"role": "user", "content": "Merge the partial summaries into one concise section summary:\n\n" + "\n\n".join(summaries)},
    ]
    return call_llm(merge_msgs, max_tokens=1200)

def _json_extract(s: str) -> Optional[Dict]:
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def analyze_paper_with_llm(paper_info: Dict, markdown_content: str, json_metadata: Dict) -> PaperAnalysis:
    logger.info(f"Analyzing paper with LLM: {paper_info.get('title','Unknown')}")
    sections = extract_sections_from_markdown(markdown_content)
    section_summaries: Dict[str, str] = {}
    for sec, content in sections.items():
        if content.strip():
            section_summaries[sec] = summarize_section(sec, content, paper_info.get("title","Unknown"))

    combined = "\n\n".join([f"## {k.title()}\n{v}" for k, v in section_summaries.items() if v.strip()])
    format_hint = {
        "title": paper_info.get("title","Unknown"),
        "tldr": "",
        "key_contributions": [],
        "methodology": "",
        "results": "",
        "novelty": "",
        "limitations": [],
        "relevance_score": 7,
        "tags": [],
    }
    final_prompt = f"""논문 "{paper_info.get('title','Unknown')}"의 섹션별 요약이 아래에 있습니다:

{combined}

아래 JSON 스키마에 맞게 결과만 JSON으로 출력하세요(설명문 금지):
{json.dumps(format_hint, ensure_ascii=False, indent=2)}

규칙:
- key_contributions: 3~6개 bullet 수준의 간결 문장
- relevance_score: 1~10 정수
- tags: 5~8개 짧은 표제어 (영문)
- 전문 용어는 English 그대로 유지
"""
    msgs = [
        {"role": "system", "content": "You are an expert AI/ML researcher. Return ONLY valid JSON."},
        {"role": "user", "content": final_prompt},
    ]
    final_out = call_llm(msgs, max_tokens=min(2500, LLM_MAX_TOKENS))
    parsed = _json_extract(final_out) or {}

    return PaperAnalysis(
        title=paper_info.get("title","Unknown"),
        authors=paper_info.get("authors", []),
        abstract=(sections.get("abstract","")[:800] if sections.get("abstract") else ""),
        summary=final_out if isinstance(final_out, str) else json.dumps(final_out, ensure_ascii=False),
        key_contributions=parsed.get("key_contributions", []),
        methodology=parsed.get("methodology", section_summaries.get("methods","")),
        results=parsed.get("results", section_summaries.get("results","")),
        relevance_score=int(parsed.get("relevance_score", 7)),
        tags=parsed.get("tags", []),
        source_file=paper_info.get("s3_key",""),
    )

# ===== Confluence (optional) =====
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

def build_markdown(analyses: List[PaperAnalysis], week_label: str, prefix: str) -> Tuple[str, str]:
    if not week_label:
        week_label = derive_week_label(prefix)
    header = textwrap.dedent(f"""\
    # AI Paper Newsletter – {week_label}
    _Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}_

    Source prefix: `{prefix}`

    ---
    """)
    parts = [header]
    for i, a in enumerate(analyses, 1):
        tags = f"**Tags:** {', '.join(a.tags)}" if a.tags else ""
        authors = f"**Authors:** {', '.join(a.authors[:8])}" if a.authors else ""
        abstract_block = ""
        if a.abstract and a.abstract.strip():
            abstract_block = "\n**Abstract**\n\n> " + a.abstract.strip() + "\n"
        sec = textwrap.dedent(f"""\
        ## {i}. {a.title}
        {authors}
        {tags}

        **Summary**
        {a.summary.strip()}

        {abstract_block}**Source:** `s3://{a.source_file}`

        ---
        """)
        parts.append(sec)
    md_content = "\n".join(parts)
    md_filename = f"{week_label}.md"
    return md_filename, md_content

# ===== Routes =====
@app.get("/")
async def root():
    return {
        "service": "AI Paper Newsletter Processor",
        "status": "running",
        "available_endpoints": ["/health", "/list-s3-papers", "/process-s3-papers"],
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat(), "mode": "S3-only (internal)"}

@app.get("/list-s3-papers")
async def list_s3_papers(bucket: Optional[str] = None, prefix: Optional[str] = None):
    bucket = bucket or S3_BUCKET
    prefix = prefix or S3_PAPERS_PREFIX
    try:
        papers = get_s3_papers(bucket, prefix)
        return {
            "bucket": bucket,
            "prefix": prefix,
            "papers_found": len(papers),
            "papers": [
                {"title": p["title"], "s3_key": p["s3_key"], "last_modified": p["last_modified"], "size_bytes": p.get("size_bytes", 0)}
                for p in papers
            ],
        }
    except Exception as e:
        logger.error(f"list_s3_papers error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-s3-papers")
async def process_s3_papers(request: S3PapersRequest):
    logger.info("Processing S3 papers")
    bucket = request.bucket or S3_BUCKET
    prefix = request.prefix or S3_PAPERS_PREFIX
    try:
        papers = get_s3_papers(
            bucket=bucket,
            prefix=prefix,
            file_pattern=request.file_pattern or "*.pdf",
            process_subdirectories=request.process_subdirectories,
        )
        if not papers:
            return {
                "message": "No papers found in S3",
                "papers_found": 0,
                "papers_processed": 0,
                "bucket": bucket,
                "prefix": prefix,
                "md_filename": None,
                "md_content": "",
            }

        def _process_one(p: Dict) -> Tuple[Optional[PaperAnalysis], Optional[str]]:
            tmp = None
            try:
                tmp = download_pdf_from_s3(p["s3_key"], p["s3_bucket"])
                md, meta = parse_pdf_with_docpamin(tmp)
                info = {"title": p["title"], "authors": [], "abstract": "", "s3_key": p["s3_key"]}
                a = analyze_paper_with_llm(info, md, meta)
                a.source_file = p["s3_key"]
                return a, None
            except Exception as e:
                return None, f"{p['s3_key']}: {e}"
            finally:
                if tmp and os.path.exists(tmp):
                    try: os.unlink(tmp)
                    except Exception: pass

        analyses: List[PaperAnalysis] = []
        errors: List[str] = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = [ex.submit(_process_one, p) for p in papers]
            for fut in as_completed(futs):
                a, err = fut.result()
                if a: analyses.append(a)
                if err: errors.append(err)

        week_label = request.week_label or derive_week_label(prefix)
        md_filename, md_content = build_markdown(analyses, week_label, prefix)

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
            "confluence_url": (confluence_result or {}).get("page_url"),
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

# ==== [1] 로컬 파일 업로드 → 파싱 테스트 ====
@app.post("/debug/parse-file")
async def debug_parse_file(
    file: UploadFile = File(...),
    include_markdown: bool = Form(False),
    markdown_max_chars: int = Form(5000),
):
    """
    로컬에서 PDF만 업로드해서 Docpamin 파싱을 테스트할 수 있는 디버그 엔드포인트.
    multipart/form-data로 파일 업로드.
    """
    # 임시 저장
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

# ==== [2] S3 단일 객체 key → 파싱 테스트 ====
@app.post("/debug/parse-s3")
async def debug_parse_s3(req: DebugParseS3Request):
    """
    S3의 특정 key만 Docpamin 파싱 테스트.
    """
    bucket = req.bucket or S3_BUCKET
    if not bucket:
        raise HTTPException(status_code=400, detail="Bucket is required (env S3_BUCKET_NAME or request.bucket)")
    # 다운로드 후 파싱
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

# ==== [3] Markdown 텍스트 → 섹션/최종 요약 테스트 ====
@app.post("/debug/summarize-markdown")
async def debug_summarize_markdown(req: DebugSummarizeMarkdownRequest):
    """
    임의의 markdown 문자열을 입력받아:
      - 섹션 분해 (abstract/introduction/methods/results/...)
      - 섹션별 요약 (옵션)
      - 최종 종합 분석(LLM) (옵션)
    을 수행.
    """
    if not req.markdown or not req.markdown.strip():
        raise HTTPException(status_code=400, detail="markdown is empty")

    sections = extract_sections_from_markdown(req.markdown)

    section_summaries = {}
    if req.include_section_summaries:
        for sec_name, sec_text in sections.items():
            if not sec_text.strip():
                continue
            section_summaries[sec_name] = summarize_section(sec_name, sec_text, req.title)

    final_analysis = None
    if req.include_final_analysis:
        # analyze_paper_with_llm는 내부에서 섹션 요약을 다시 수행하지만,
        # 이미 섹션 요약이 있다면 더 빠르게 하려면 별도 경로를 만들어도 됨.
        paper_info = {"title": req.title, "authors": [], "abstract": "", "s3_key": ""}
        final = analyze_paper_with_llm(paper_info, req.markdown, {})
        # Pydantic 모델 → dict
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

    return {
        "title": req.title,
        "sections_detected": [k for k, v in sections.items() if v.strip()],
        "section_summaries": section_summaries if req.include_section_summaries else {},
        "final_analysis": final_analysis,
        "markdown_preview": _preview(req.markdown, req.return_markdown_preview_chars),
    }

# ==== [4] 섹션 dict → 섹션 요약만 테스트 ====
@app.post("/debug/summarize-sections")
async def debug_summarize_sections(req: DebugSummarizeSectionsRequest):
    """
    섹션 dict(예: {'abstract': '...', 'results': '...'})을 넣어서
    섹션 요약만 빠르게 테스트.
    """
    if not req.sections:
        raise HTTPException(status_code=400, detail="sections is empty")

    targets = req.only_sections or list(req.sections.keys())
    out: Dict[str, str] = {}
    for sec in targets:
        text = req.sections.get(sec, "")
        if not text.strip():
            continue
        out[sec] = summarize_section(sec, text, req.title)

    return {
        "title": req.title,
        "summarized_sections": list(out.keys()),
        "summaries": out,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("paper_processor:app", host="0.0.0.0", port=7070, reload=False, workers=2)
