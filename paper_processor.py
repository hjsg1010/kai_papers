#!/usr/bin/env python3

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from typing import List, Optional, Dict, Tuple
from datetime import datetime
from pathlib import Path
import tempfile
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import from newly created modules
from config.settings import *
from models import *
from services.s3_service import *
from services.docpamin_service import *
from services.llm_service import *
from services.confluence_service import *
from utils.image_processing import *
from utils.text_processing import *
from utils.markdown_utils import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="AI Paper Newsletter Processor",
    description="Processes AI papers from S3, parses with Docpamin, summarizes via LLM. v2: Smart Hybrid Chunking (header-based or token-based)",
    version="2.0.0",
    docs_url="/docs",
    redoc_url=None,
    openapi_url="/openapi.json",
    root_path="/proxy/7070",
)


# ===== Helper Function =====
def _preview(text: str, limit: int) -> str:
    if not text or limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...\n(truncated)"


# ===== API Endpoints =====

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
    uvicorn.run(app, host="0.0.0.0", port=7070, log_level="info")
