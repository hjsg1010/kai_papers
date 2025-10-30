"""Pydantic models shared across the service."""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


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


__all__ = [
    "S3PapersRequest",
    "BatchProcessRequest",
    "PaperAnalysis",
    "DebugParseS3Request",
    "DebugSummarizeMarkdownRequest",
    "DebugSummarizeSectionsRequest",
]
