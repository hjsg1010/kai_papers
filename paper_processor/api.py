"""FastAPI application exposing the paper processing workflow."""
from __future__ import annotations

import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .chunking import smart_chunk_hybrid
from .config import MAX_WORKERS, S3_BUCKET, S3_PAPERS_PREFIX, logger
from .docpamin import (
    extract_title_from_markdown,
    extract_title_from_url,
    parse_pdf_with_docpamin,
    parse_pdf_with_docpamin_url,
)
from .markdown import build_markdown, derive_week_label
from .models import (
    BatchProcessRequest,
    DebugParseS3Request,
    DebugSummarizeMarkdownRequest,
    DebugSummarizeSectionsRequest,
    PaperAnalysis,
    S3PapersRequest,
)
from .s3_utils import _preview, download_pdf_from_s3, get_paper_list_from_s3, get_s3_papers
from .summary import analyze_paper_with_llm, analyze_paper_with_llm_improved, summarize_chunk_with_overlap
from .confluence import upload_to_confluence


app = FastAPI(
    title="AI Paper Newsletter Processor",
    description="Processes AI papers from S3, parses with Docpamin, summarizes via LLM. v2: Smart Hybrid Chunking (header-based or token-based)",
    version="2.0.0",
    docs_url="/docs",
    redoc_url=None,
    openapi_url="/openapi.json",
    root_path="/proxy/7070",
)


@app.get("/")
async def root():
    return {
        "service": "AI Paper Newsletter Processor",
        "version": "2.0.0",
        "status": "running",
        "improvements": [
            "smart_hybrid_chunking (header-based or token-based)",
            "overlap_chunking",
            "hierarchical_summarization (position-based)",
        ],
        "available_endpoints": ["/health", "/list-s3-papers", "/process-s3-papers", "/debug/*"],
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat(), "mode": "S3-only (v2: hybrid chunking)"}


@app.get("/list-s3-papers")
async def list_s3_papers(bucket: Optional[str] = None, prefix: Optional[str] = None):
    bucket = bucket or S3_BUCKET
    prefix = prefix or S3_PAPERS_PREFIX

    try:
        paper_urls = get_paper_list_from_s3(bucket, prefix)

        if paper_urls:
            logger.info("Found paper_list.txt with %d URLs", len(paper_urls))
            papers = [
                {
                    "title": extract_title_from_url(url),
                    "s3_key": f"{prefix}/{extract_title_from_url(url)}.pdf",
                    "url": url,
                    "source": "url_list",
                    "last_modified": None,
                    "size_bytes": 0,
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
                        "title": paper["title"],
                        "s3_key": paper["s3_key"],
                        "url": paper.get("url"),
                        "source": paper.get("source"),
                        "last_modified": paper.get("last_modified"),
                        "size_bytes": paper.get("size_bytes", 0),
                    }
                    for paper in papers
                ],
            }

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
                    "title": paper["title"],
                    "s3_key": paper["s3_key"],
                    "source": paper.get("source", "s3"),
                    "last_modified": paper["last_modified"],
                    "size_bytes": paper.get("size_bytes", 0),
                }
                for paper in papers
            ],
        }

    except Exception as exc:
        logger.error("list_s3_papers error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/process-s3-papers")
async def process_s3_papers(request: S3PapersRequest):
    logger.info(
        "Processing S3 papers (hierarchical=%s, overlap=%s)", request.use_hierarchical, request.use_overlap
    )
    bucket = request.bucket or S3_BUCKET
    prefix = request.prefix or S3_PAPERS_PREFIX

    try:
        paper_urls = get_paper_list_from_s3(bucket, prefix)

        if paper_urls:
            logger.info("Processing %d papers from paper_list.txt", len(paper_urls))
            papers = [
                {
                    "title": extract_title_from_url(url),
                    "url": url,
                    "source": "url",
                    "s3_key": f"{prefix}/{extract_title_from_url(url)}.pdf",
                }
                for url in paper_urls
            ]
        else:
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

        def _process_one(paper: Dict) -> Tuple[Optional[PaperAnalysis], Optional[Dict], Optional[str]]:
            tmp_file = None
            try:
                if paper.get("source") == "url":
                    markdown, metadata = parse_pdf_with_docpamin_url(paper["url"], paper["title"])
                    actual_title = metadata.get("extracted_title", paper["title"])
                    info = {"title": actual_title, "authors": [], "abstract": "", "s3_key": paper["s3_key"]}
                else:
                    tmp_file = download_pdf_from_s3(paper["s3_key"], paper["s3_bucket"])
                    markdown, metadata = parse_pdf_with_docpamin(tmp_file)
                    actual_title = extract_title_from_markdown(markdown) if markdown else paper["title"]
                    metadata["extracted_title"] = actual_title
                    info = {"title": actual_title, "authors": [], "abstract": "", "s3_key": paper["s3_key"]}

                analysis, _ = analyze_paper_with_llm_improved(
                    info,
                    markdown,
                    metadata,
                    use_hierarchical=request.use_hierarchical,
                    use_overlap=request.use_overlap,
                    return_intermediate=False,
                )
                analysis.source_file = paper["s3_key"]
                return analysis, metadata, None

            except Exception as exc:  # pragma: no cover - defensive logging
                return None, None, f"{paper.get('title', 'unknown')}: {exc}"
            finally:
                if tmp_file and os.path.exists(tmp_file):
                    try:
                        os.unlink(tmp_file)
                    except Exception:
                        pass

        analyses: List[PaperAnalysis] = []
        papers_metadata: List[Dict] = []
        errors: List[str] = []

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(_process_one, paper) for paper in papers]
            for future in as_completed(futures):
                analysis, metadata, error = future.result()
                if analysis:
                    analyses.append(analysis)
                    if metadata and metadata.get("images_info"):
                        papers_metadata.append(
                            {
                                "s3_key": analysis.source_file,
                                "title": analysis.title,
                                "images_info": metadata["images_info"],
                            }
                        )
                if error:
                    errors.append(error)

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
            "improvements_used": {"hierarchical": request.use_hierarchical, "overlap": request.use_overlap},
        }
    except Exception as exc:
        logger.exception("process_s3_papers error")
        raise HTTPException(status_code=500, detail=str(exc))


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
        for paper in papers:
            tmp_file = None
            try:
                tmp_file = download_pdf_from_s3(paper["s3_key"], paper["s3_bucket"])
                markdown, metadata = parse_pdf_with_docpamin(tmp_file)
                info = {"title": paper["title"], "authors": [], "abstract": "", "s3_key": paper["s3_key"]}
                analysis = analyze_paper_with_llm(info, markdown, metadata)
                analysis.source_file = paper["s3_key"]
                analysis.tags = request.tags
                analyses.append(analysis)
            except Exception as exc:
                logger.error("Error processing: %s", exc)
            finally:
                if tmp_file and os.path.exists(tmp_file):
                    try:
                        os.unlink(tmp_file)
                    except Exception:
                        pass

        page_title = request.confluence_page_title or f"AI Paper Review - {datetime.now().strftime('%Y-%m-%d')}"
        confluence_result = upload_to_confluence(analyses, page_title)
        return {
            "message": "Successfully batch processed papers",
            "papers_found": len(papers),
            "papers_processed": len(analyses),
            "confluence_url": confluence_result.get("page_url"),
        }
    except Exception as exc:
        logger.exception("batch_process error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/debug/parse-file")
async def debug_parse_file(
    file: UploadFile = File(...),
    include_markdown: bool = Form(False),
    markdown_max_chars: int = Form(5000),
):
    suffix = Path(file.filename).suffix or ".pdf"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        content = await file.read()
        tmp.write(content)
        tmp.close()

        markdown, metadata = parse_pdf_with_docpamin(tmp.name)
        response = {
            "filename": file.filename,
            "md_len": len(markdown),
            "md_preview": _preview(markdown, markdown_max_chars),
            "metadata": metadata,
            "images_info": metadata.get("images_info"),
            "json_metadata": metadata,
        }
        if include_markdown:
            response["markdown"] = markdown
        return response
    finally:
        try:
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)
        except Exception:
            pass


@app.post("/debug/parse-s3")
async def debug_parse_s3(req: DebugParseS3Request):
    bucket = req.bucket or S3_BUCKET
    if not bucket:
        raise HTTPException(status_code=400, detail="Bucket is required")
    tmp = None
    try:
        tmp = download_pdf_from_s3(s3_key=req.key, s3_bucket=bucket)
        markdown, metadata = parse_pdf_with_docpamin(tmp)
        response = {
            "bucket": bucket,
            "key": req.key,
            "md_len": len(markdown),
            "md_preview": _preview(markdown, req.markdown_max_chars),
            "metadata": metadata,
            "images_info": metadata.get("images_info"),
            "json_metadata": metadata,
        }
        if req.include_markdown:
            response["markdown"] = markdown
        return response
    finally:
        try:
            if tmp and os.path.exists(tmp):
                os.unlink(tmp)
        except Exception:
            pass


@app.post("/debug/summarize-markdown")
async def debug_summarize_markdown(req: DebugSummarizeMarkdownRequest):
    if not req.markdown or not req.markdown.strip():
        raise HTTPException(status_code=400, detail="markdown is empty")

    chunks, chunking_method = smart_chunk_hybrid(req.markdown, min_headers=5)

    chunk_summaries: Dict[str, str] = {}
    if req.include_section_summaries:
        prev_summary = ""
        for chunk_key, chunk_text in chunks.items():
            if not chunk_text.strip() or len(chunk_text.strip()) < 100:
                continue
            chunk_summaries[chunk_key] = summarize_chunk_with_overlap(
                chunk_key,
                chunk_text,
                req.title,
                use_overlap=req.use_overlap,
                prev_summary=prev_summary if req.use_overlap else "",
            )
            prev_summary = chunk_summaries[chunk_key]

    final_analysis = None
    intermediate_data = None
    if req.include_final_analysis:
        paper_info = {"title": req.title, "authors": [], "abstract": "", "s3_key": ""}
        final, intermediate = analyze_paper_with_llm_improved(
            paper_info,
            req.markdown,
            {},
            use_hierarchical=req.use_hierarchical,
            use_overlap=req.use_overlap,
            return_intermediate=req.show_intermediate_steps,
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
        "improvements_used": {"hierarchical": req.use_hierarchical, "overlap": req.use_overlap},
    }


@app.post("/debug/summarize-sections")
async def debug_summarize_sections(req: DebugSummarizeSectionsRequest):
    if not req.sections:
        raise HTTPException(status_code=400, detail="sections is empty")

    targets = req.only_sections or list(req.sections.keys())
    output: Dict[str, str] = {}
    prev_summary = ""

    for section in targets:
        text = req.sections.get(section, "")
        if not text.strip():
            continue
        output[section] = summarize_chunk_with_overlap(
            section,
            text,
            req.title,
            use_overlap=req.use_overlap,
            prev_summary=prev_summary if req.use_overlap else "",
        )
        prev_summary = output[section]

    return {
        "title": req.title,
        "summarized_sections": list(output.keys()),
        "summaries": output,
        "improvements_used": {"overlap": req.use_overlap},
    }


__all__ = ["app"]
