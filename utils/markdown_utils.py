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
from io import BytesIO

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL not available - image resizing disabled")

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


def resize_image_base64(base64_data: str, max_width: int = 600, quality: int = 85) -> Tuple[str, int]:
    """
    base64 이미지를 리사이징하고 압축

    Args:
        base64_data: 원본 base64 이미지 데이터
        max_width: 최대 너비 (픽셀)
        quality: JPEG 품질 (1-100)

    Returns:
        (resized_base64, size_bytes): 리사이징된 base64와 바이트 크기
    """
    if not PIL_AVAILABLE:
        # PIL이 없으면 원본 반환
        return base64_data, len(base64_data) * 3 // 4

    try:
        # base64 디코딩
        img_bytes = base64.b64decode(base64_data)
        img = Image.open(BytesIO(img_bytes))

        # 이미 작으면 그대로 반환
        if img.width <= max_width:
            return base64_data, len(img_bytes)

        # 비율 유지하면서 리사이징
        ratio = max_width / img.width
        new_height = int(img.height * ratio)
        img_resized = img.resize((max_width, new_height), Image.Resampling.LANCZOS)

        # RGB로 변환 (RGBA인 경우)
        if img_resized.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', img_resized.size, (255, 255, 255))
            if img_resized.mode == 'P':
                img_resized = img_resized.convert('RGBA')
            background.paste(img_resized, mask=img_resized.split()[-1] if img_resized.mode == 'RGBA' else None)
            img_resized = background

        # JPEG로 압축
        output = BytesIO()
        img_resized.save(output, format='JPEG', quality=quality, optimize=True)
        resized_bytes = output.getvalue()

        # base64 인코딩
        resized_base64 = base64.b64encode(resized_bytes).decode('utf-8')

        logger.info(f"Image resized: {len(img_bytes)} → {len(resized_bytes)} bytes "
                   f"({img.width}x{img.height} → {max_width}x{new_height})")

        return resized_base64, len(resized_bytes)

    except Exception as e:
        logger.error(f"Failed to resize image: {e}")
        return base64_data, len(base64_data) * 3 // 4


def save_images_to_files(
    papers_metadata: Optional[List[Dict]],
    week_label: str,
    output_dir: str = "images",
    create_thumbnails: bool = True
) -> Dict[str, str]:
    """
    대표 이미지들을 파일로 저장

    Args:
        papers_metadata: 논문 메타데이터 리스트 (이미지 정보 포함)
        week_label: 주차 레이블 (예: "w42")
        output_dir: 이미지 저장 디렉토리
        create_thumbnails: 썸네일 버전도 생성할지 여부 (메일용)

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
            # 원본 이미지 저장
            img_bytes = base64.b64decode(base64_data)
            with open(img_path, 'wb') as f:
                f.write(img_bytes)

            saved_images[s3_key] = img_filename
            logger.info(f"Saved image: {img_path} ({len(img_bytes)} bytes)")

            # 썸네일 버전 생성 (메일용)
            if create_thumbnails and PIL_AVAILABLE:
                thumb_filename = f"{week_label}_{paper_name}_fig{rep_img['index'] + 1}_thumb.jpg"
                thumb_path = os.path.join(output_dir, thumb_filename)

                resized_base64, resized_size = resize_image_base64(base64_data, max_width=600, quality=85)
                thumb_bytes = base64.b64decode(resized_base64)

                with open(thumb_path, 'wb') as f:
                    f.write(thumb_bytes)

                logger.info(f"Saved thumbnail: {thumb_path} ({resized_size} bytes)")

        except Exception as e:
            logger.error(f"Failed to save image for {s3_key}: {e}")

    return saved_images


def _optimize_images_for_email(
    papers_metadata: List[Dict],
    max_size_kb: int = 950
) -> List[Dict]:
    """
    이메일 크기 제한을 위해 이미지들을 최적화

    Args:
        papers_metadata: 논문 메타데이터 리스트 (원본은 보존)
        max_size_kb: 최대 허용 크기 (KB)

    Returns:
        최적화된 papers_metadata 복사본
    """
    if not PIL_AVAILABLE:
        logger.warning("PIL not available - skipping image optimization")
        return papers_metadata

    # 원본 보존을 위해 deep copy
    import copy
    optimized_metadata = copy.deepcopy(papers_metadata)

    # 여러 품질 레벨로 시도
    quality_levels = [85, 75, 65, 55, 45]
    max_widths = [600, 500, 400, 350, 300]

    for quality_idx, (quality, max_width) in enumerate(zip(quality_levels, max_widths)):
        total_image_size = 0

        # 모든 이미지 리사이징
        for meta in optimized_metadata:
            images_info = meta.get('images_info', {})
            rep_imgs = images_info.get('representative_images', [])

            if not rep_imgs:
                continue

            for rep_img in rep_imgs:
                base64_data = rep_img.get('base64_data')
                if not base64_data:
                    continue

                # 리사이징
                resized_base64, resized_size = resize_image_base64(
                    base64_data,
                    max_width=max_width,
                    quality=quality
                )

                # 메타데이터 업데이트
                rep_img['base64_data'] = resized_base64
                rep_img['size_kb'] = resized_size / 1024
                total_image_size += resized_size

        # 텍스트 크기 추정 (이미지 제외한 마크다운)
        estimated_text_size = 50 * 1024  # 약 50KB로 추정 (헤더, 요약 등)
        total_size_kb = (total_image_size + estimated_text_size) / 1024

        logger.info(f"Optimization attempt {quality_idx + 1}: "
                   f"quality={quality}, max_width={max_width}, "
                   f"total_size={total_size_kb:.1f}KB")

        if total_size_kb <= max_size_kb:
            logger.info(f"✅ Image optimization successful: {total_size_kb:.1f}KB / {max_size_kb}KB")
            return optimized_metadata

        # 아직 크면 다음 품질 레벨 시도
        if quality_idx < len(quality_levels) - 1:
            logger.warning(f"⚠️  Still too large ({total_size_kb:.1f}KB), trying lower quality...")
            # Deep copy 다시 수행하여 원본에서 재시작
            optimized_metadata = copy.deepcopy(papers_metadata)

    # 최저 품질로도 안되면 경고하고 반환
    logger.warning(f"⚠️  Could not reduce size below {max_size_kb}KB even with lowest quality")
    return optimized_metadata


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
    save_images: bool = True,
    include_images: bool = True,
    optimize_for_email: bool = False,
    max_email_size_kb: int = 950
) -> Tuple[str, str]:
    """
    논문 분석 결과를 Markdown으로 변환

    Args:
        analyses: PaperAnalysis 객체 리스트
        papers_metadata: 논문 메타데이터 리스트 (이미지 정보 포함)
        week_label: 주차 레이블 (예: "w42")
        prefix: S3 prefix
        save_images: 이미지를 파일로 저장할지 여부
        include_images: 이미지 섹션을 포함할지 여부 (메일용: False, GitHub용: True)
        optimize_for_email: 이메일 크기 제한을 위해 이미지 리사이징 여부
        max_email_size_kb: 최대 이메일 크기 (KB)

    Returns:
        Tuple[str, str]: (파일명, Markdown 콘텐츠)
    """
    if not week_label:
        week_label = derive_week_label(prefix)

    # 이메일 최적화: 이미지 리사이징
    if optimize_for_email and papers_metadata and PIL_AVAILABLE:
        papers_metadata = _optimize_images_for_email(papers_metadata, max_email_size_kb)

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

        # 이미지 섹션 추가 (옵션)
        if include_images and a.source_file in image_map:
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
