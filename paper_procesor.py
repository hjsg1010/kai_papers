"""
AI Paper Newsletter Processor
FastAPI service for processing papers from S3
Uses Docpamin API for parsing and OpenAI-compatible LLM for analysis
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict, Tuple
import boto3
from datetime import datetime
import os
import tempfile
from pathlib import Path
import logging
import json
import time
import requests
import zipfile
import io
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Paper Newsletter Processor - Internal")

# Configuration
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET_NAME")
S3_PAPERS_PREFIX = os.getenv("S3_PAPERS_PREFIX", "papers/")

# Docpamin API Configuration
DOCPAMIN_API_KEY = os.getenv("DOCPAMIN_API_KEY")
DOCPAMIN_BASE_URL = os.getenv("DOCPAMIN_BASE_URL", "https://docpamin.superaip.samsungds.net/api/v1")
DOCPAMIN_CRT_FILE = os.getenv("DOCPAMIN_CRT_FILE", "/etc/ssl/certs/ca-certificates.crt")

# OpenAI Compatible LLM Configuration
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))

# Confluence Configuration
CONFLUENCE_URL = os.getenv("CONFLUENCE_URL")
CONFLUENCE_EMAIL = os.getenv("CONFLUENCE_EMAIL")
CONFLUENCE_API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN")
CONFLUENCE_SPACE_KEY = os.getenv("CONFLUENCE_SPACE_KEY")

# Initialize clients
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)


class S3PapersRequest(BaseModel):
    bucket: Optional[str] = None
    prefix: Optional[str] = None
    # Optional filters
    file_pattern: Optional[str] = "*.pdf"
    process_subdirectories: bool = True


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


def get_s3_papers(bucket: str, prefix: str, file_pattern: str = "*.pdf", 
                  process_subdirectories: bool = True) -> List[Dict]:
    """Get PDF files from S3 bucket"""
    logger.info(f"Fetching papers from S3: s3://{bucket}/{prefix}")
    
    papers = []
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        
        # List all objects in the prefix
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if 'Contents' not in page:
                continue
                
            for obj in page['Contents']:
                key = obj['Key']
                
                # Skip if not a PDF
                if not key.lower().endswith('.pdf'):
                    continue
                
                # Skip subdirectories if not processing them
                if not process_subdirectories:
                    # Check if file is directly in the prefix directory
                    relative_path = key[len(prefix):].lstrip('/')
                    if '/' in relative_path:
                        continue
                
                # Extract metadata from S3 object
                file_name = Path(key).stem
                
                papers.append({
                    "title": file_name.replace('_', ' ').replace('-', ' '),
                    "s3_key": key,
                    "s3_bucket": bucket,
                    "last_modified": obj['LastModified'].isoformat(),
                    "size_bytes": obj['Size'],
                    "source": "s3"
                })
        
        logger.info(f"Found {len(papers)} papers in S3")
    except Exception as e:
        logger.error(f"Error fetching from S3: {str(e)}")
        raise
    
    return papers


def download_pdf_from_s3(s3_key: str, s3_bucket: str) -> str:
    """Download PDF from S3 to temp file"""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    
    try:
        logger.info(f"Downloading PDF from S3: {s3_bucket}/{s3_key}")
        s3_client.download_fileobj(s3_bucket, s3_key, temp_file)
        temp_file.close()
        return temp_file.name
    except Exception as e:
        temp_file.close()
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        raise Exception(f"Failed to download PDF from S3: {str(e)}")


def parse_pdf_with_docpamin(pdf_path: str) -> Tuple[str, Dict]:
    """
    Parse PDF using Docpamin API
    Returns: (markdown_content, json_content)
    """
    logger.info(f"Parsing PDF with Docpamin API: {pdf_path}")
    
    headers = {"Authorization": f"bearer {DOCPAMIN_API_KEY}"}
    
    try:
        # Step 1: Create Task
        logger.info("Step 1: Creating Docpamin task...")
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
            
            resp = requests.post(
                f"{DOCPAMIN_BASE_URL}/tasks",
                headers=headers,
                files=files,
                data=data,
                verify=DOCPAMIN_CRT_FILE
            )
            resp.raise_for_status()
            
            task_id = resp.json().get("task_id")
            if not task_id:
                raise Exception("Failed to create task: no task_id returned")
            
            logger.info(f"Task created: {task_id}")
        
        # Step 2: Poll Task Status
        logger.info("Step 2: Waiting for task completion...")
        max_wait = 300  # 5 minutes
        wait_time = 0
        
        while wait_time < max_wait:
            resp = requests.get(
                f"{DOCPAMIN_BASE_URL}/tasks/{task_id}",
                headers=headers,
                verify=DOCPAMIN_CRT_FILE
            )
            resp.raise_for_status()
            
            status = resp.json().get("status")
            logger.info(f"Task status: {status}")
            
            if status == "DONE":
                logger.info("Task completed successfully")
                break
            elif status in ["FAILED", "ERROR"]:
                raise Exception(f"Task failed with status: {status}")
            
            time.sleep(3)
            wait_time += 3
        
        if wait_time >= max_wait:
            raise Exception("Task timeout: exceeded 5 minutes")
        
        # Step 3: Download Results
        logger.info("Step 3: Downloading results...")
        options = {
            "task_ids": [task_id],
            "output_types": ["markdown", "json"]
        }
        
        resp = requests.post(
            f"{DOCPAMIN_BASE_URL}/tasks/export",
            headers=headers,
            json=options,
            verify=DOCPAMIN_CRT_FILE
        )
        resp.raise_for_status()
        
        # Extract content from ZIP
        logger.info("Extracting content from ZIP...")
        markdown_content = ""
        json_content = {}
        
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            for filename in zf.namelist():
                if filename.endswith('.md'):
                    with zf.open(filename) as f:
                        markdown_content = f.read().decode('utf-8')
                        logger.info(f"Extracted markdown: {len(markdown_content)} characters")
                elif filename.endswith('.json'):
                    with zf.open(filename) as f:
                        json_content = json.loads(f.read().decode('utf-8'))
                        logger.info(f"Extracted JSON metadata")
        
        if not markdown_content:
            raise Exception("No markdown content found in export")
        
        logger.info(f"Successfully parsed PDF: {len(markdown_content)} characters")
        return markdown_content, json_content
        
    except Exception as e:
        logger.error(f"Error parsing PDF with Docpamin: {str(e)}")
        raise


def extract_sections_from_markdown(markdown_content: str) -> Dict[str, str]:
    """
    Extract sections from markdown content
    Identifies common paper sections: Abstract, Introduction, Methods, Results, Discussion, Conclusion
    """
    logger.info("Extracting sections from markdown...")
    
    sections = {
        "abstract": "",
        "introduction": "",
        "methods": "",
        "results": "",
        "discussion": "",
        "conclusion": "",
        "other": ""
    }
    
    # Common section headers (case-insensitive)
    section_patterns = {
        "abstract": r"##?\s*(?:abstract|요약)",
        "introduction": r"##?\s*(?:introduction|서론|배경)",
        "methods": r"##?\s*(?:methods?|methodology|approach|방법)",
        "results": r"##?\s*(?:results?|experiments?|결과|실험)",
        "discussion": r"##?\s*(?:discussion|analysis|분석|논의)",
        "conclusion": r"##?\s*(?:conclusion|summary|결론|요약)"
    }
    
    # Split by headers
    lines = markdown_content.split('\n')
    current_section = "other"
    current_content = []
    
    for line in lines:
        # Check if this line is a section header
        matched_section = None
        for section_name, pattern in section_patterns.items():
            if re.match(pattern, line.strip(), re.IGNORECASE):
                # Save previous section
                if current_content:
                    sections[current_section] += '\n'.join(current_content) + '\n'
                
                # Start new section
                matched_section = section_name
                current_content = [line]
                break
        
        if matched_section:
            current_section = matched_section
        else:
            current_content.append(line)
    
    # Save last section
    if current_content:
        sections[current_section] += '\n'.join(current_content)
    
    # Log section sizes
    for section, content in sections.items():
        if content.strip():
            logger.info(f"Section '{section}': {len(content)} characters")
    
    return sections


def call_llm(messages: List[Dict], max_tokens: int = 2000) -> str:
    """
    Call OpenAI-compatible LLM API
    """
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.3
    }
    
    try:
        response = requests.post(
            f"{LLM_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        
        result = response.json()
        return result['choices'][0]['message']['content']
        
    except Exception as e:
        logger.error(f"LLM API call failed: {str(e)}")
        raise


def summarize_section(section_name: str, section_content: str, paper_title: str) -> str:
    """
    Summarize a single section of the paper
    """
    if not section_content.strip():
        return ""
    
    logger.info(f"Summarizing section: {section_name} ({len(section_content)} chars)")
    
    # Adjust prompt based on section
    section_prompts = {
        "abstract": "이 논문의 Abstract를 핵심 내용 위주로 간결하게 요약해주세요.",
        "introduction": "Introduction 섹션의 주요 내용(배경, 문제 정의, 연구 목적)을 요약해주세요.",
        "methods": "Methods 섹션의 핵심 방법론과 접근법을 요약해주세요.",
        "results": "Results 섹션의 주요 실험 결과와 성능 지표를 요약해주세요.",
        "discussion": "Discussion 섹션의 핵심 인사이트와 분석을 요약해주세요.",
        "conclusion": "Conclusion 섹션의 주요 결론과 기여점을 요약해주세요.",
        "other": "다음 내용의 핵심을 요약해주세요."
    }
    
    prompt = section_prompts.get(section_name, section_prompts["other"])
    
    # Chunk if too long (assume ~4 chars per token, leave room for prompt)
    max_section_chars = (LLM_MAX_TOKENS - 500) * 4
    
    if len(section_content) > max_section_chars:
        logger.info(f"Section too long, chunking: {len(section_content)} chars")
        # Split into chunks
        chunks = []
        for i in range(0, len(section_content), max_section_chars):
            chunks.append(section_content[i:i+max_section_chars])
        
        # Summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            messages = [
                {"role": "system", "content": "당신은 학술 논문 분석 전문가입니다. 핵심 정보를 누락 없이 간결하게 요약합니다."},
                {"role": "user", "content": f"{prompt}\n\n내용:\n{chunk}"}
            ]
            chunk_summary = call_llm(messages, max_tokens=1000)
            chunk_summaries.append(chunk_summary)
        
        # Merge chunk summaries
        if len(chunk_summaries) > 1:
            merged_content = "\n\n".join(chunk_summaries)
            messages = [
                {"role": "system", "content": "당신은 학술 논문 분석 전문가입니다."},
                {"role": "user", "content": f"다음은 '{section_name}' 섹션의 부분 요약들입니다. 이를 하나의 통합된 요약으로 만들어주세요:\n\n{merged_content}"}
            ]
            return call_llm(messages, max_tokens=1500)
        else:
            return chunk_summaries[0]
    else:
        messages = [
            {"role": "system", "content": "당신은 학술 논문 분석 전문가입니다. 핵심 정보를 누락 없이 간결하게 요약합니다."},
            {"role": "user", "content": f"{prompt}\n\n내용:\n{section_content}"}
        ]
        return call_llm(messages, max_tokens=1500)


def analyze_paper_with_llm(paper_info: Dict, markdown_content: str, json_metadata: Dict) -> PaperAnalysis:
    """
    Analyze paper content using OpenAI-compatible LLM with hierarchical summarization
    """
    logger.info(f"Analyzing paper with LLM: {paper_info.get('title', 'Unknown')}")
    
    try:
        # Step 1: Extract sections
        sections = extract_sections_from_markdown(markdown_content)
        
        # Step 2: Summarize each section
        section_summaries = {}
        for section_name, section_content in sections.items():
            if section_content.strip():
                summary = summarize_section(
                    section_name, 
                    section_content, 
                    paper_info.get('title', 'Unknown')
                )
                if summary:
                    section_summaries[section_name] = summary
        
        logger.info(f"Created {len(section_summaries)} section summaries")
        
        # Step 3: Create comprehensive analysis by combining section summaries
        combined_summaries = "\n\n".join([
            f"**{name.upper()}**:\n{summary}" 
            for name, summary in section_summaries.items()
        ])
        
        # Step 4: Generate final comprehensive analysis
        logger.info("Generating final comprehensive analysis...")
        
        final_prompt = f"""다음은 논문 "{paper_info.get('title', 'Unknown')}"의 섹션별 요약입니다.

{combined_summaries}

위 정보를 바탕으로 다음 형식으로 종합 분석을 작성해주세요:

1. **핵심 요약** (3-4문장): 논문의 전체 내용을 가장 간결하게 요약
2. **주요 기여점** (3-5개): 이 논문의 핵심 기여를 bullet point로
3. **방법론**: 사용된 주요 방법과 접근법
4. **주요 결과**: 핵심 실험 결과와 성능
5. **의의 및 한계**: 연구의 의의와 한계점
6. **관련성 점수** (1-10): AI/ML 분야에서의 중요도와 영향력
7. **키워드** (3-5개): 이 논문을 대표하는 키워드

한국어로 작성하되, 전문 용어는 영어를 병기해주세요."""

        messages = [
            {"role": "system", "content": "당신은 AI/ML 분야의 선임 연구원입니다. 논문을 깊이 있게 분석하고 실무적 관점에서 평가합니다."},
            {"role": "user", "content": final_prompt}
        ]
        
        final_analysis = call_llm(messages, max_tokens=2500)
        
        # Parse the response (simple parsing, can be enhanced)
        analysis = PaperAnalysis(
            title=paper_info.get('title', 'Unknown'),
            authors=paper_info.get('authors', []),
            abstract=sections.get('abstract', '')[:500] if sections.get('abstract') else '',
            summary=final_analysis,
            key_contributions=[],  # Could parse from response
            methodology=section_summaries.get('methods', ''),
            results=section_summaries.get('results', ''),
            relevance_score=8,  # Could parse from response
            tags=[],  # Could parse from response
            source_file=paper_info.get('s3_key', '')
        )
        
        logger.info("Successfully analyzed paper with LLM")
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing with LLM: {str(e)}")
        raise


def upload_to_confluence(analyses: List[PaperAnalysis], page_title: str):
    """Upload paper analyses to Confluence"""
    logger.info(f"Uploading to Confluence: {page_title}")
    
    # Build Confluence page content
    content = f"""<h1>AI Paper Newsletter - {datetime.now().strftime('%Y-%m-%d')}</h1>
<p>이번 주의 주목할 만한 AI 논문들을 소개합니다.</p>

<ac:structured-macro ac:name="info">
<ac:rich-text-body>
<p>총 {len(analyses)}편의 논문이 분석되었습니다.</p>
</ac:rich-text-body>
</ac:structured-macro>

<hr/>
"""
    
    for i, analysis in enumerate(analyses, 1):
        content += f"""
<h2>{i}. {analysis.title}</h2>
<p><strong>저자:</strong> {', '.join(analysis.authors[:5])}</p>

<h3>초록</h3>
<p>{analysis.abstract[:500]}...</p>

<h3>분석</h3>
{analysis.summary}

<hr/>
"""
    
    # Confluence API
    url = f"{CONFLUENCE_URL}/rest/api/content"
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "type": "page",
        "title": page_title,
        "space": {"key": CONFLUENCE_SPACE_KEY},
        "body": {
            "storage": {
                "value": content,
                "representation": "storage"
            }
        }
    }
    
    try:
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            auth=(CONFLUENCE_EMAIL, CONFLUENCE_API_TOKEN)
        )
        response.raise_for_status()
        
        result = response.json()
        page_url = f"{CONFLUENCE_URL}{result['_links']['webui']}"
        logger.info(f"Successfully uploaded to Confluence: {page_url}")
        
        return {"success": True, "page_url": page_url, "page_id": result['id']}
        
    except Exception as e:
        logger.error(f"Error uploading to Confluence: {str(e)}")
        raise


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "mode": "S3-only (internal)"
    }


@app.post("/process-s3-papers")
async def process_s3_papers(request: S3PapersRequest):
    """Process papers from S3 bucket - Main endpoint"""
    logger.info("Processing S3 papers")
    
    bucket = request.bucket or S3_BUCKET
    prefix = request.prefix or S3_PAPERS_PREFIX
    
    try:
        # 1. Get papers from S3
        papers = get_s3_papers(
            bucket=bucket, 
            prefix=prefix,
            file_pattern=request.file_pattern,
            process_subdirectories=request.process_subdirectories
        )
        
        if not papers:
            return {
                "message": "No papers found in S3",
                "papers_processed": 0,
                "bucket": bucket,
                "prefix": prefix
            }
        
        # 2. Process each paper
        analyses = []
        errors = []
        
        for paper in papers:
            try:
                logger.info(f"Processing: {paper['title']}")
                
                # Download from S3
                pdf_path = download_pdf_from_s3(
                    s3_key=paper['s3_key'],
                    s3_bucket=paper['s3_bucket']
                )
                
                # Parse with Docpamin API
                markdown_content, json_metadata = parse_pdf_with_docpamin(pdf_path)
                
                # Analyze with LLM (hierarchical summarization)
                paper_info = {
                    "title": paper['title'],
                    "authors": [],
                    "abstract": "",
                    "s3_key": paper['s3_key']
                }
                analysis = analyze_paper_with_llm(paper_info, markdown_content, json_metadata)
                analysis.source_file = paper['s3_key']
                analyses.append(analysis)
                
                # Cleanup
                os.unlink(pdf_path)
                
                logger.info(f"Successfully processed: {paper['title']}")
                
            except Exception as e:
                error_msg = f"Error processing {paper['title']}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
                continue
        
        # 3. Upload to Confluence if we have analyses
        confluence_result = None
        if analyses:
            page_title = f"AI Paper Newsletter - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            confluence_result = upload_to_confluence(analyses, page_title)
        
        return {
            "message": "Successfully processed S3 papers",
            "papers_found": len(papers),
            "papers_processed": len(analyses),
            "errors": errors,
            "bucket": bucket,
            "prefix": prefix,
            "confluence_url": confluence_result.get("page_url") if confluence_result else None
        }
        
    except Exception as e:
        logger.error(f"Error in process_s3_papers: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-process")
async def batch_process_papers(request: BatchProcessRequest):
    """Batch process papers with custom settings"""
    logger.info("Batch processing papers")
    
    bucket = request.bucket or S3_BUCKET
    prefix = request.prefix or S3_PAPERS_PREFIX
    
    try:
        # Get and process papers
        papers = get_s3_papers(bucket, prefix)
        
        if not papers:
            return {
                "message": "No papers found in S3",
                "papers_processed": 0
            }
        
        analyses = []
        for paper in papers:
            try:
                pdf_path = download_pdf_from_s3(
                    s3_key=paper['s3_key'],
                    s3_bucket=paper['s3_bucket']
                )
                
                markdown_content, json_metadata = parse_pdf_with_docpamin(pdf_path)
                
                paper_info = {
                    "title": paper['title'],
                    "authors": [],
                    "abstract": "",
                    "s3_key": paper['s3_key']
                }
                analysis = analyze_paper_with_llm(paper_info, markdown_content, json_metadata)
                analysis.source_file = paper['s3_key']
                analysis.tags = request.tags
                analyses.append(analysis)
                
                os.unlink(pdf_path)
                
            except Exception as e:
                logger.error(f"Error processing paper: {str(e)}")
                continue
        
        # Upload to Confluence
        page_title = request.confluence_page_title or f"AI Paper Review - {datetime.now().strftime('%Y-%m-%d')}"
        confluence_result = upload_to_confluence(analyses, page_title)
        
        return {
            "message": "Successfully batch processed papers",
            "papers_found": len(papers),
            "papers_processed": len(analyses),
            "confluence_url": confluence_result.get("page_url")
        }
        
    except Exception as e:
        logger.error(f"Error in batch_process: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/list-s3-papers")
async def list_s3_papers(bucket: Optional[str] = None, prefix: Optional[str] = None):
    """List available papers in S3 without processing"""
    bucket = bucket or S3_BUCKET
    prefix = prefix or S3_PAPERS_PREFIX
    
    try:
        papers = get_s3_papers(bucket, prefix)
        
        return {
            "bucket": bucket,
            "prefix": prefix,
            "papers_found": len(papers),
            "papers": [
                {
                    "title": p['title'],
                    "s3_key": p['s3_key'],
                    "last_modified": p['last_modified'],
                    "size_bytes": p.get('size_bytes', 0)
                }
                for p in papers
            ]
        }
    except Exception as e:
        logger.error(f"Error listing S3 papers: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)