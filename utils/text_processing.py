"""
Text Processing and Analysis Utilities for Paper Processor

This module contains all text processing and LLM analysis functions including:
- Header extraction and text chunking
- Token-based and header-based chunking strategies
- Hierarchical summarization
- LLM-based paper analysis
"""

import logging
import re
import json
import math
from typing import List, Dict, Tuple, Optional

# Import LLM service
from services.llm_service import call_llm, estimate_tokens

# Import image processing utilities
from utils.image_processing import process_markdown_images, select_representative_images

# Import configuration
from config.settings import LLM_MAX_TOKENS

# Import models
from models import PaperAnalysis

logger = logging.getLogger(__name__)


# ===== Header Extraction =====

def extract_all_headers(markdown_content: str) -> Dict[str, str]:
    """
    섹션 이름에 관계없이 주요 헤더만 추출하여 섹션으로 분리
    # (Level 1) 과 ## (Level 2) 헤더만 사용 (### 제외)

    Args:
        markdown_content: 마크다운 형식의 텍스트

    Returns:
        Dict[str, str]: 섹션 키와 내용의 매핑
            - 키 형식: "section_00_header_name"
            - 값: 해당 섹션의 내용
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
            logger.info(f"Section '{sec['header']}': {len(content)} chars, ~{estimate_tokens(content)} toks")

    return result


# ===== Token-Based Chunking =====

def chunk_by_tokens(text: str, chunk_size: int = 4000, overlap: int = 400) -> List[str]:
    """
    토큰 기반으로 텍스트를 균등 분할
    - 섹션 구조가 없거나 불명확할 때 사용

    Args:
        text: 분할할 텍스트
        chunk_size: 각 청크의 대략적인 토큰 크기
        overlap: 청크 간 겹치는 토큰 수

    Returns:
        List[str]: 분할된 텍스트 청크 리스트
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


# ===== Hybrid Chunking Strategy =====

def smart_chunk_hybrid(markdown_content: str, min_headers: int = 10) -> Tuple[Dict[str, str], str]:
    """
    하이브리드 청킹: 헤더 개수에 따라 방식 선택

    기본값을 8→10으로 상향 조정하여 청크 수 더욱 감소
    # (Level 1) 헤더만 카운트 (## 제외)

    Args:
        markdown_content: 마크다운 형식의 텍스트
        min_headers: 헤더 기반 청킹을 사용하기 위한 최소 헤더 개수

    Returns:
        Tuple[Dict[str, str], str]: (청크 딕셔너리, 사용된 방법)
            - chunks_dict: 청크 키와 내용의 매핑
            - method_used: "header_based" or "token_based"
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


# ===== Smart Chunking with Overlap =====

def smart_chunk_with_overlap(text: str, chunk_size: int = 4000, overlap: int = 400) -> List[str]:
    """
    텍스트를 overlap을 가지고 청크로 분할

    Args:
        text: 분할할 텍스트
        chunk_size: 각 청크의 대략적인 크기 (문자 단위)
        overlap: 청크 간 겹치는 부분 (문자 단위)

    Returns:
        List[str]: 분할된 텍스트 청크 리스트
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


# ===== Chunk Summarization =====

def summarize_chunk_with_overlap(
    chunk_key: str,
    chunk_content: str,
    paper_title: str,
    use_overlap: bool = True,
    prev_summary: str = ""
) -> str:
    """
    청크를 overlap을 가지고 요약 (섹션 이름 무관)

    Args:
        chunk_key: "section_01_abstract" 또는 "chunk_00" 형식
        chunk_content: 요약할 청크 내용
        paper_title: 논문 제목
        use_overlap: overlap chunking 사용 여부
        prev_summary: 이전 청크의 요약 (context로 사용)

    Returns:
        str: 요약된 텍스트
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
    approx_tokens = estimate_tokens(chunk_content)

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


# ===== Hierarchical Summarization =====

def create_hierarchical_summary_v2(chunk_summaries: Dict[str, str], paper_title: str) -> Dict[str, str]:
    """
    계층적 요약 생성 v2 (위치 기반)

    Level 1: 청크 요약 (이미 완료)
    Level 2: 위치별 그룹 요약 (beginning, middle, end)
    Level 3: 최종 통합 요약

    섹션 이름에 의존하지 않고 논문의 위치(앞/중간/뒤)로 그룹핑

    Args:
        chunk_summaries: 청크별 요약 딕셔너리
        paper_title: 논문 제목

    Returns:
        Dict[str, str]: 위치별 중간 요약 딕셔너리
            - 키: "beginning", "middle", "end", "full"
            - 값: 해당 부분의 요약
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


# ===== JSON Extraction Utility =====

def _json_extract(s: str) -> Optional[Dict]:
    """
    문자열에서 JSON 추출

    Args:
        s: JSON이 포함된 문자열

    Returns:
        Optional[Dict]: 추출된 JSON 딕셔너리 또는 None
    """
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


# ===== Paper Analysis with LLM =====

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

    Args:
        paper_info: 논문 기본 정보 (title, authors, s3_key 등)
        markdown_content: 논문의 마크다운 내용
        json_metadata: Docpamin API에서 반환된 JSON 메타데이터
        use_hierarchical: 계층적 요약 사용 여부
        use_overlap: overlap chunking 사용 여부
        return_intermediate: 중간 단계 결과 반환 여부

    Returns:
        Tuple[PaperAnalysis, Optional[Dict]]: (분석 결과, 중간 단계 데이터)
            - PaperAnalysis: 최종 분석 결과
            - Optional[Dict]: 중간 단계 데이터 (return_intermediate=True인 경우)
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

    final_prompt = f"""논문 "{paper_info.get('title','Unknown')}"의 섹션별 요약이 아래에 있습니다:

{combined}

아래 JSON 스키마에 맞게 결과만 JSON으로 한글로 출력하세요(설명문 금지):
{json.dumps(format_hint, ensure_ascii=False, indent=2)}

규칙:
- key_contributions: 3~6개 bullet 수준의 간결 문장
- relevance_score: **LLM 에이전트 연구/개발에 대한 관련성** (1~10 정수)
  * 1-3: 관련 없음 (전혀 다른 분야의 연구)
  * 4-5: 간접 관련 (기초 기술이나 배경 지식)
  * 6-7: 보통 관련 (참고할 만한 방법론이나 아이디어)
  * 8-9: 높은 관련성 (직접 적용 가능한 기술이나 방법)
  * 10: 필수 참고 (LLM 에이전트의 핵심 기술)

    특히 다음 주제는 높은 점수:
    - Agentic reasoning, tool use, planning
    - Reinforcement learning for LLM agents
    - Agent architectures, frameworks

- tags: 5~8개 짧은 표제어 (영문)
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


def analyze_paper_with_llm(paper_info: Dict, markdown_content: str, json_metadata: Dict) -> PaperAnalysis:
    """
    기존 함수 (개선 버전 호출)

    호환성을 위한 래퍼 함수

    Args:
        paper_info: 논문 기본 정보
        markdown_content: 논문의 마크다운 내용
        json_metadata: Docpamin API에서 반환된 JSON 메타데이터

    Returns:
        PaperAnalysis: 분석 결과
    """
    analysis, _ = analyze_paper_with_llm_improved(
        paper_info, markdown_content, json_metadata,
        use_hierarchical=True, use_overlap=True, return_intermediate=False
    )
    return analysis
