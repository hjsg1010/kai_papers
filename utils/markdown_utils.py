"""
Markdown formatting utilities for paper newsletters
"""
import logging
import re
import json
import base64
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Tuple

from models import PaperAnalysis

logger = logging.getLogger(__name__)


def derive_week_label(prefix: str) -> str:
    """
    prefix에서 주차 레이블을 추출하거나 현재 주차를 반환

    Args:
        prefix: S3 prefix 또는 주차 정보를 포함한 문자열

    Returns:
        주차 레이블 (예: "w42")
    """
    m = re.search(r"w(\d{1,2})", prefix or "", re.IGNORECASE)
    if m:
        return f"w{int(m.group(1))}"
    iso_year, iso_week, _ = datetime.utcnow().isocalendar()
    return f"w{iso_week}"


def save_images_to_files(
    papers_metadata: Optional[List[Dict]],
    week_label: str,
    output_dir: str = "images"
) -> Dict[str, str]:
    """
    대표 이미지들을 파일로 저장

    Args:
        papers_metadata: 논문 메타데이터 리스트 (이미지 정보 포함)
        week_label: 주차 레이블 (예: "w42")
        output_dir: 이미지 저장 디렉토리

    Returns:
        Dict[s3_key, saved_filename]: 저장된 이미지 파일명 매핑
    """
    if not papers_metadata:
        return {}

    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    saved_images = {}

    for meta in papers_metadata:
        images_info = meta.get('images_info', {})
        rep_imgs = images_info.get('representative_images', [])

        if not rep_imgs:
            continue

        rep_img = rep_imgs[0]
        s3_key = meta.get('s3_key', '')

        # base64 데이터가 있는지 확인
        base64_data = rep_img.get('base64_data')
        if not base64_data:
            logger.warning(f"No base64 data for image in {s3_key}")
            continue

        # 파일명 생성
        paper_name = Path(s3_key).stem
        img_type = rep_img.get('type', 'png')
        img_filename = f"{week_label}_{paper_name}_fig{rep_img['index'] + 1}.{img_type}"
        img_path = os.path.join(output_dir, img_filename)

        try:
            # base64 디코딩 및 파일 저장
            img_bytes = base64.b64decode(base64_data)
            with open(img_path, 'wb') as f:
                f.write(img_bytes)

            saved_images[s3_key] = img_filename
            logger.info(f"Saved image: {img_path} ({len(img_bytes)} bytes)")

        except Exception as e:
            logger.error(f"Failed to save image for {s3_key}: {e}")

    return saved_images


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
            lines.append(f"**📌 TL;DR**\n\n")
            lines.append(f"{data['tldr']}\n\n")

        # 핵심 기여
        if data.get('key_contributions'):
            lines.append(f"**🎯 핵심 기여**\n\n")
            lines.append("".join([f"- {contrib}\n" for contrib in data['key_contributions']]))
            lines.append("\n")

        # 방법론
        if data.get('methodology'):
            lines.append(f"**🔬 방법론**\n\n")
            lines.append(f"{data['methodology']}\n\n")

        # 결과
        if data.get('results'):
            lines.append(f"**📊 결과**\n\n")
            lines.append(f"{data['results']}\n\n")

        # 새로운 점
        if data.get('novelty'):
            lines.append(f"**💡 새로운 점**\n\n")
            lines.append(f"{data['novelty']}\n\n")

        # 한계점
        if data.get('limitations'):
            lines.append(f"**⚠️ 한계점**\n\n")
            lines.append("".join([f"- {limitation}\n" for limitation in data['limitations']]))
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


def build_markdown(
    analyses: List[PaperAnalysis],
    papers_metadata: Optional[List[Dict]] = None,
    week_label: str = "",
    prefix: str = "",
    save_images: bool = True
) -> Tuple[str, str]:
    """
    논문 분석 결과를 Markdown으로 변환

    Args:
        analyses: PaperAnalysis 객체 리스트
        papers_metadata: 논문 메타데이터 리스트 (이미지 정보 포함)
        week_label: 주차 레이블 (예: "w42")
        prefix: S3 prefix
        save_images: 이미지를 파일로 저장할지 여부

    Returns:
        Tuple[str, str]: (파일명, Markdown 콘텐츠)
    """
    if not week_label:
        week_label = derive_week_label(prefix)

    # 이미지를 파일로 저장 (GitHub 표시용)
    if save_images and papers_metadata:
        saved_images = save_images_to_files(papers_metadata, week_label)
        logger.info(f"Saved {len(saved_images)} images to files")

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

        # Summary JSON 파싱 및 개조식 변환
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
