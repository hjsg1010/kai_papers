"""Higher level summarisation helpers built on top of the chunking utilities."""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

from .chunking import smart_chunk_hybrid, smart_chunk_with_overlap
from .config import logger
from .images import process_markdown_images, select_representative_images
from .llm import LLM_MAX_TOKENS, _estimate_tokens, call_llm
from .models import PaperAnalysis


def summarize_chunk_with_overlap(
    chunk_key: str,
    chunk_content: str,
    paper_title: str,
    use_overlap: bool = True,
    prev_summary: str = "",
) -> str:
    """Summarise a single chunk, optionally passing in the previous summary."""
    if not chunk_content.strip():
        return ""

    if chunk_key.startswith("section_"):
        header_part = chunk_key.split("_", 2)[-1] if "_" in chunk_key else "content"
        prompt = f"다음 '{header_part}' 섹션의 핵심 내용을 요약하세요."
    else:
        chunk_num = chunk_key.split("_")[-1] if "_" in chunk_key else "0"
        prompt = f"논문의 일부분 (Part {chunk_num})을 요약하세요."

    budget_tokens = min(LLM_MAX_TOKENS - 800, 6000)
    approx_tokens = _estimate_tokens(chunk_content)

    if approx_tokens <= budget_tokens:
        context_prompt = ""
        if prev_summary and use_overlap:
            context_prompt = f"\n이전 내용 요약: {prev_summary}\n"

        messages = [
            {"role": "system", "content": "You are an expert AI paper analyst. Keep technical terms in English."},
            {"role": "user", "content": f"[{paper_title}] {prompt}{context_prompt}\n\n내용:\n{chunk_content}"},
        ]
        return call_llm(messages, max_tokens=min(3000, LLM_MAX_TOKENS - 500))

    chunk_size_chars = budget_tokens * 4
    overlap_chars = 400 if use_overlap else 0

    sub_chunks = smart_chunk_with_overlap(chunk_content, chunk_size_chars, overlap_chars)

    summaries: List[str] = []
    sub_prev_summary = prev_summary

    for i, sub_chunk in enumerate(sub_chunks):
        context_prompt = ""
        if sub_prev_summary and use_overlap:
            context_prompt = f"\n이전 내용: {sub_prev_summary}\n"

        messages = [
            {"role": "system", "content": "You are an expert AI paper analyst. Keep technical terms in English."},
            {
                "role": "user",
                "content": f"[{paper_title}] {prompt} (sub-part {i+1}/{len(sub_chunks)}){context_prompt}\n\n{sub_chunk}",
            },
        ]
        summary = call_llm(messages, max_tokens=min(3000, LLM_MAX_TOKENS - 500))
        summaries.append(summary)
        sub_prev_summary = summary

    if len(summaries) == 1:
        return summaries[0]

    merge_messages = [
        {"role": "system", "content": "You are an expert AI paper analyst."},
        {
            "role": "user",
            "content": f"다음은 [{paper_title}]의 '{chunk_key}' 부분을 여러 sub-part로 나눠 요약한 결과입니다. 이를 하나의 일관된 요약으로 병합하세요:\n\n"
            + "\n\n---\n\n".join(summaries),
        },
    ]
    return call_llm(merge_messages, max_tokens=3000)


def create_hierarchical_summary_v2(chunk_summaries: Dict[str, str], paper_title: str) -> Dict[str, str]:
    """Create a hierarchical summary grouped by the position of the chunks."""
    logger.info("Creating hierarchical summary (position-based)...")

    num_chunks = len(chunk_summaries)
    if num_chunks == 0:
        return {}

    items = list(chunk_summaries.items())

    if num_chunks <= 2:
        groups = {"full": items}
    elif num_chunks <= 5:
        mid_point = num_chunks // 2
        groups = {"beginning": items[:mid_point], "end": items[mid_point:]}
    else:
        third = num_chunks // 3
        groups = {
            "beginning": items[:third],
            "middle": items[third : third * 2],
            "end": items[third * 2 :],
        }

    group_prompts = {
        "full": "논문의 전체 내용을 종합적으로 요약하세요.",
        "beginning": "논문의 도입부 (배경, 문제 정의, 목표, 관련 연구)를 종합적으로 요약하세요.",
        "middle": "논문의 핵심 부분 (방법론, 실험 설계, 결과, 성능)을 종합적으로 요약하세요.",
        "end": "논문의 결론 부분 (인사이트, 한계, 기여, 향후 과제)을 종합적으로 요약하세요.",
    }

    intermediate_summaries: Dict[str, str] = {}
    for group_name, group_items in groups.items():
        if not group_items:
            continue

        group_texts = [f"### Part {i+1}\n{summary}" for i, (key, summary) in enumerate(group_items)]
        combined = "\n\n".join(group_texts)

        prompt = group_prompts.get(group_name, "다음 내용을 종합적으로 요약하세요.")

        messages = [
            {"role": "system", "content": "You are an expert AI paper analyst."},
            {"role": "user", "content": f"[{paper_title}] {prompt}\n\n{combined}"},
        ]

        intermediate_summaries[group_name] = call_llm(messages, max_tokens=3000)
        logger.info("Created intermediate summary for '%s' (%d chunks)", group_name, len(group_items))

    return intermediate_summaries


def _json_extract(text: str) -> Optional[Dict]:
    """Extract a JSON document from *text* if present."""
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def analyze_paper_with_llm_improved(
    paper_info: Dict,
    markdown_content: str,
    json_metadata: Dict,
    use_hierarchical: bool = True,
    use_overlap: bool = True,
    return_intermediate: bool = False,
) -> Tuple[PaperAnalysis, Optional[Dict]]:
    """Analyse a paper by chunking and summarising Docpamin markdown output."""
    logger.info(
        "Analyzing paper (hierarchical=%s, overlap=%s): %s",
        use_hierarchical,
        use_overlap,
        paper_info.get("title", "Unknown"),
    )

    clean_markdown, extracted_images = process_markdown_images(
        markdown_content,
        remove_for_llm=True,
        keep_representative=1,
    )

    if extracted_images:
        logger.info("Removed %d images for LLM processing", len(extracted_images))

    chunks, chunking_method = smart_chunk_hybrid(clean_markdown, min_headers=10)
    logger.info("Chunking method: %s, total chunks: %d", chunking_method, len(chunks))

    chunk_summaries: Dict[str, str] = {}
    prev_summary = ""

    for chunk_key, content in chunks.items():
        if content.strip() and len(content.strip()) > 100:
            summary = summarize_chunk_with_overlap(
                chunk_key,
                content,
                paper_info.get("title", "Unknown"),
                use_overlap=use_overlap,
                prev_summary=prev_summary if use_overlap else "",
            )
            chunk_summaries[chunk_key] = summary
            prev_summary = summary

    logger.info("Created %d chunk summaries", len(chunk_summaries))

    intermediate_summaries: Dict[str, str] = {}
    if use_hierarchical and len(chunk_summaries) > 0:
        intermediate_summaries = create_hierarchical_summary_v2(chunk_summaries, paper_info.get("title", "Unknown"))

    summary_prompt = """다음은 AI 연구 논문에 대한 Docpamin markdown입니다. 아래 지침을 따라 JSON으로 분석 결과를 반환하세요.

- title, authors, abstract 는 입력 정보 기반으로 채웁니다.
- key_contributions, methodology, results, novelty, limitations 를 요약하세요.
- relevance_score 는 1~10 점수 (10이 가장 높음)로 평가하세요.
  * 1-3: 낮은 관련성 (과거 연구, 영향도 낮음)
  * 4-6: 중간 관련성 (아이디어는 있으나 직접 활용은 어려움)
  * 7: 참고할 만함 (흥미로운 접근이나 개선점 존재)
  * 8-9: 높은 관련성 (직접 적용 가능한 기술이나 방법)
  * 10: 필수 참고 (LLM 에이전트의 핵심 기술)

    특히 다음 주제는 높은 점수:
    - Agentic reasoning, tool use, planning
    - Reinforcement learning for LLM agents
    - Agent architectures, frameworks

- tags: 5~8개 짧은 표제어 (영문)
- 전문 용어는 English 그대로 유지
"""

    chunks_for_prompt = []
    for i, (key, summary) in enumerate(chunk_summaries.items(), 1):
        chunks_for_prompt.append(f"### Chunk {i} - {key}\n{summary}")
    combined_chunks = "\n\n".join(chunks_for_prompt) if chunks_for_prompt else clean_markdown[:4000]

    if intermediate_summaries:
        combined_chunks += "\n\n" + "\n\n".join(
            f"## {name.title()} Summary\n{summary}" for name, summary in intermediate_summaries.items()
        )

    final_prompt = f"""{summary_prompt}\n\n### Paper Metadata\n{json.dumps(paper_info, ensure_ascii=False)}\n\n### Combined Summaries\n{combined_chunks}\n"""

    messages = [
        {"role": "system", "content": "You are an expert AI/ML researcher. Return ONLY valid JSON."},
        {"role": "user", "content": final_prompt},
    ]
    final_out = call_llm(messages, max_tokens=min(3000, LLM_MAX_TOKENS))
    parsed = _json_extract(final_out) or {}

    abstract_text = ""
    for key, content in chunks.items():
        if len(content) < 2000:
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

    intermediate_data: Optional[Dict] = None
    if return_intermediate:
        representative_imgs = select_representative_images(extracted_images, max_count=1)

        intermediate_data = {
            "chunking_method": chunking_method,
            "num_chunks": len(chunks),
            "chunks_detected": list(chunks.keys()),
            "chunk_summaries": chunk_summaries,
            "intermediate_summaries": intermediate_summaries if use_hierarchical else {},
            "images": {
                "total_count": len(extracted_images),
                "removed_size": sum(img["size"] for img in extracted_images),
                "representative": [
                    {
                        "index": img["index"],
                        "alt": img["alt"],
                        "type": img["type"],
                        "size": img["size"],
                        "markdown": img["full"],
                    }
                    for img in representative_imgs
                ],
            },
        }

    return analysis, intermediate_data


def analyze_paper_with_llm(paper_info: Dict, markdown_content: str, json_metadata: Dict) -> PaperAnalysis:
    """Backward compatible helper returning only the final analysis."""
    analysis, _ = analyze_paper_with_llm_improved(
        paper_info,
        markdown_content,
        json_metadata,
        use_hierarchical=True,
        use_overlap=True,
        return_intermediate=False,
    )
    return analysis


__all__ = [
    "analyze_paper_with_llm",
    "analyze_paper_with_llm_improved",
    "create_hierarchical_summary_v2",
    "summarize_chunk_with_overlap",
]
