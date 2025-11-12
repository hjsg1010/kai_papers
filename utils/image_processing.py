"""
Image Processing Utilities for Paper Processor

This module contains all image-related processing functions including:
- Base64 image extraction and removal
- Figure-caption pair extraction from JSON metadata
- Image matching with captions
- Representative image selection (with LLM support)
"""

import re
import logging
from typing import List, Dict, Optional, Tuple

# Import LLM service
from services.llm_service import call_llm

# Import configuration
from config.settings import LLM_MODEL

logger = logging.getLogger(__name__)


# ===== Base64 Image Processing =====

def remove_base64_images(markdown: str, replacement: str = "[Image]") -> Tuple[str, int]:
    """
    Base64 이미지를 플레이스홀더로 대체

    Args:
        markdown: 마크다운 텍스트
        replacement: 대체할 플레이스홀더 텍스트

    Returns:
        (cleaned_markdown, num_removed): 정리된 마크다운과 제거된 이미지 수
    """
    pattern = r'!\[([^\]]*)\]\(data:image/[^;]+;base64,[A-Za-z0-9+/=]+\)'
    cleaned, count = re.subn(pattern, replacement, markdown)
    if count > 0:
        logger.info(f"Removed {count} base64 images from markdown")
    return cleaned, count


def extract_base64_images(markdown: str) -> List[Dict]:
    """
    Markdown에서 base64 이미지 추출

    Args:
        markdown: 마크다운 텍스트

    Returns:
        이미지 정보 리스트 (alt_text, mime_type, base64_data, size_kb, position 포함)
    """
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


# ===== Figure-Caption Pair Extraction =====

def extract_figure_pairs_from_json(json_metadata: Dict) -> List[Dict]:
    """
    Docpamin JSON에서 PICTURE-CAPTION 쌍만 추출

    Args:
        json_metadata: Docpamin API에서 반환된 JSON 메타데이터

    Returns:
        [{'figure_number': 1, 'caption': '...', 'base64_preview': '...',
          'page_no': 1, 'picture_id': '...', 'caption_id': '...'}, ...]
    """
    figure_pairs = []

    try:
        pages = json_metadata.get('pages', [])

        for page in pages:
            layout = page.get('layout', [])

            # ⭐ PICTURE 다음에 CAPTION이 오는지 확인
            for i, block in enumerate(layout):
                if block.get('type') != 'PICTURE':
                    continue

                # 다음 블록 확인
                if i + 1 >= len(layout):
                    continue

                next_block = layout[i + 1]

                # ⭐ 다음 블록이 CAPTION인지 확인
                if next_block.get('type') != 'CAPTION':
                    logger.debug(f"PICTURE at id={block.get('id')} has no CAPTION (next: {next_block.get('type')})")
                    continue

                # Caption 추출
                caption_content = next_block.get('content', '').strip()

                if not caption_content:
                    continue

                # Figure 번호 추출
                fig_match = re.search(
                    r'Figure[~\s]+(\d+)[:\.]?\s*(.+?)$',
                    caption_content,
                    re.IGNORECASE
                )

                if not fig_match:
                    continue

                fig_num = int(fig_match.group(1))
                caption_text = fig_match.group(2).strip()

                # 유효성 검사
                if not is_valid_caption(caption_text):
                    logger.debug(f"Invalid caption for Figure {fig_num}")
                    continue

                # Base64 미리보기 (매칭용)
                picture_content = block.get('content', '')
                base64_match = re.search(r'base64,([A-Za-z0-9+/=]{50,100})', picture_content)
                base64_preview = base64_match.group(1) if base64_match else ''

                figure_pairs.append({
                    'figure_number': fig_num,
                    'caption': caption_text,
                    'base64_preview': base64_preview,
                    'page_no': page.get('page_no'),
                    'picture_id': block.get('id'),
                    'caption_id': next_block.get('id')
                })

                logger.info(f"📷 Figure {fig_num}: {caption_text[:60]}...")

        logger.info(f"Found {len(figure_pairs)} valid PICTURE-CAPTION pairs")

    except Exception as e:
        logger.error(f"Failed to extract figure pairs: {e}")

    return figure_pairs


# ===== Image-Caption Matching =====

def match_images_with_figure_pairs(
    images: List[Dict],
    figure_pairs: List[Dict]
) -> List[Dict]:
    """
    이미지와 Figure 쌍 매칭 (base64 기반)

    Args:
        images: 추출된 이미지 리스트
        figure_pairs: extract_figure_pairs_from_json에서 추출된 Figure-Caption 쌍

    Returns:
        Caption이 추가된 이미지 리스트
    """
    if not figure_pairs:
        logger.warning("No figure pairs to match")
        return images

    matched_count = 0

    for img in images:
        img_base64 = img.get('base64_data', '')

        if not img_base64 or len(img_base64) < 100:
            continue

        # ⭐ Base64 앞부분으로 매칭
        img_preview = img_base64[:100]

        for pair in figure_pairs:
            pair_preview = pair.get('base64_preview', '')

            # Base64가 매칭되면
            if pair_preview and pair_preview in img_preview:
                img['caption'] = pair['caption']
                img['figure_number'] = pair['figure_number']

                matched_count += 1
                logger.info(f"✅ Image {img['index']} → Figure {pair['figure_number']}: "
                           f"{pair['caption'][:60]}...")
                break

    logger.info(f"Matched {matched_count}/{len(images)} images with captions")

    return images


def match_images_with_captions_from_json(
    images: List[Dict],
    json_metadata: Dict
) -> List[Dict]:
    """
    JSON 메타데이터로부터 이미지와 Caption 매칭

    이 함수는 extract_figure_pairs_from_json과 match_images_with_figure_pairs를 결합한 편의 함수입니다.

    Args:
        images: 추출된 이미지 리스트
        json_metadata: Docpamin API에서 반환된 JSON 메타데이터

    Returns:
        Caption이 추가된 이미지 리스트
    """
    # JSON에서 Figure-Caption 쌍 추출
    figure_pairs = extract_figure_pairs_from_json(json_metadata)

    # 이미지와 매칭
    matched_images = match_images_with_figure_pairs(images, figure_pairs)

    return matched_images


# ===== Image Processing =====

def process_markdown_images(
    markdown: str,
    remove_for_llm: bool = True,
    keep_representative: int = 1
) -> Tuple[str, List[Dict]]:
    """
    Markdown에서 이미지만 추출 (Caption 매칭 없음!)

    Args:
        markdown: 마크다운 텍스트
        remove_for_llm: LLM 처리를 위해 이미지를 플레이스홀더로 대체할지 여부
        keep_representative: 유지할 대표 이미지 수 (현재는 사용되지 않음)

    Returns:
        (processed_markdown, images): 처리된 마크다운과 추출된 이미지 리스트

    Note:
        Caption 매칭은 여기서 하지 않습니다.
        parse_pdf_with_docpamin_url에서 JSON 기반으로 매칭합니다.
    """
    pattern = r'!\[(.*?)\]\(data:image/([^;]+);base64,([A-Za-z0-9+/=]+)\)'

    images = []

    def extract_image(match):
        alt_text = match.group(1)
        img_type = match.group(2)
        base64_data = match.group(3)

        images.append({
            'index': len(images),
            'alt': alt_text.strip(),
            'type': img_type,
            'size': len(base64_data),
            'size_kb': len(base64_data) * 3 / 4 / 1024,
            'base64_data': base64_data,
            'full': match.group(0)
        })

        if remove_for_llm:
            return f"\n[Image {len(images)}]\n"
        else:
            return match.group(0)

    processed_md = re.sub(pattern, extract_image, markdown)

    if images:
        logger.info(f"Extracted {len(images)} images from markdown")

    return processed_md, images


# ===== Caption Validation =====

def is_valid_caption(caption: str) -> bool:
    """
    Caption 유효성 검사 (base64, 해시값 등 제거)

    Args:
        caption: 검증할 캡션 텍스트

    Returns:
        유효한 캡션이면 True, 아니면 False
    """
    if not caption or len(caption) < 10:
        return False

    # ❌ Base64 패턴 거부
    base64_pattern = r'^[A-Za-z0-9+/=]{50,}$'
    if re.match(base64_pattern, caption):
        logger.debug(f"Rejected caption (base64): {caption[:50]}...")
        return False

    # ❌ 너무 긴 단어 하나로만 구성 (해시값)
    words = caption.split()
    if len(words) == 1 and len(words[0]) > 40:
        logger.debug(f"Rejected caption (hash): {caption[:50]}...")
        return False

    # ❌ 의미있는 영어 단어가 거의 없는 경우
    english_words = [w for w in words if re.match(r'^[a-zA-Z]+$', w) and len(w) > 2]
    if len(english_words) < 2:
        logger.debug(f"Rejected caption (no words): {caption[:50]}...")
        return False

    # ❌ 이미지 마크다운 거부
    if caption.startswith('![') or caption.startswith(']('):
        return False

    return True


# ===== Representative Image Selection =====

def select_representative_image(
    images: List[Dict],
    min_kb: float = 10,
    max_kb: float = 200
) -> Optional[Dict]:
    """
    대표 이미지 선정 (크기 + 위치 기준)

    Args:
        images: 이미지 리스트
        min_kb: 최소 이미지 크기 (KB)
        max_kb: 최대 이미지 크기 (KB)

    Returns:
        선택된 대표 이미지 또는 None
    """
    if not images:
        return None

    # 크기 조건에 맞는 후보 선택
    candidates = [img for img in images if min_kb <= img['size_kb'] <= max_kb]

    if not candidates:
        # 조건에 맞는 이미지가 없으면 크기가 중간값에 가까운 상위 3개 선택
        candidates = sorted(images, key=lambda x: abs(x['size_kb'] - (min_kb + max_kb) / 2))[:3]

    # 위치가 가장 앞에 있는 이미지 반환
    return min(candidates, key=lambda x: x['position']) if candidates else None


def select_representative_image_with_llm(
    images: List[Dict],
    paper_title: str = ""
) -> Optional[Dict]:
    """
    LLM을 사용하여 가장 대표적인 이미지 선택
    (사전 필터링 없이 LLM 프롬프트만 사용)

    Args:
        images: 이미지 리스트
        paper_title: 논문 제목 (프롬프트에 사용)

    Returns:
        LLM이 선택한 대표 이미지 또는 None
    """
    if not images:
        return None

    if len(images) == 1:
        return images[0]

    try:
        logger.info("=" * 60)
        logger.info("🎯 select_representative_image_with_llm")
        logger.info(f"Total images: {len(images)}")

        # ✅ Caption 유효성 검사만 수행
        images_with_valid_caption = []
        for img in images:
            caption = img.get('caption', '')

            if is_valid_caption(caption):
                images_with_valid_caption.append(img)
                logger.debug(f"  ✅ Image {img['index']}: {caption[:50]}...")
            else:
                logger.info(f"  ❌ Skipped image {img['index']}: Invalid caption")

        logger.info(f"Valid captions: {len(images_with_valid_caption)}/{len(images)}")
        logger.info("=" * 60)

        if not images_with_valid_caption:
            logger.warning("No valid captions, using first image")
            return images[0]

        if len(images_with_valid_caption) == 1:
            logger.info("Only one valid caption, auto-selected")
            return images_with_valid_caption[0]

        # ✅ 선택지 생성
        image_descriptions = []
        for choice_num, img in enumerate(images_with_valid_caption, 1):
            fig_num = img.get('figure_number', img['index'] + 1)
            caption = img.get('caption', '')

            desc = f"{choice_num}. (Figure {fig_num}): {caption} (Size: {img['size_kb']:.1f}KB)"
            image_descriptions.append(desc)

        # ✅ 강화된 프롬프트
        prompt = f"""You are selecting the BEST figure for a research paper: "{paper_title}"

**TASK:** Choose the figure showing the paper's MAIN ARCHITECTURE or SYSTEM DESIGN.

**STRICT ELIMINATION RULES (Apply FIRST):**
❌ REJECT if caption contains ANY of these keywords:
   - "Result", "Results", "Performance", "Accuracy", "Score"
   - "Comparison", "Compare", "Versus", "vs", "vs."
   - "Experiment", "Evaluation", "Benchmark", "Leaderboard"
   - "Ablation", "Analysis" (unless paired with "Architecture")
   - "Table", "Chart", "Graph" (unless about architecture)

**SELECTION PRIORITIES (After elimination):**
1. ✅ Keywords: "Architecture", "Framework", "System Design", "Workflow", "Pipeline", "Overview of method"
2. ✅ Descriptive captions explaining HOW the system works
3. ✅ Earlier figures (1-3) when tied

**IMPORTANT CLARIFICATIONS:**
- "Overall results" → ❌ REJECT (has "results")
- "Overall architecture" → ✅ GOOD (has "architecture")
- "Performance comparison" → ❌ REJECT (has both!)
- "System overview" → ✅ GOOD

**Figures:**
{chr(10).join(image_descriptions)}

**OUTPUT:** Respond with ONLY one number (1-{len(images_with_valid_caption)}). No explanation."""

        messages = [{"role": "user", "content": prompt}]

        # ✅ max_tokens 증가 (reasoning model 대응)
        response = call_llm(messages, max_tokens=500)

        response_text = response.strip()
        logger.info(f"LLM response: '{response_text}'")

        # 숫자 추출
        numbers = re.findall(r'\b(\d+)\b', response_text)

        if not numbers:
            logger.warning("No number in response, using first valid")
            return images_with_valid_caption[0]

        choice_num = int(numbers[0])
        choice_idx = choice_num - 1

        logger.info(f"LLM chose: choice={choice_num}, idx={choice_idx}")

        if 0 <= choice_idx < len(images_with_valid_caption):
            selected = images_with_valid_caption[choice_idx]

            logger.info("=" * 60)
            logger.info(f"✅ SELECTED:")
            logger.info(f"   Index: {selected['index']}")
            logger.info(f"   Figure: {selected.get('figure_number', 'N/A')}")
            logger.info(f"   Caption: {selected.get('caption', '')[:80]}...")
            logger.info(f"   Size: {selected['size_kb']:.1f}KB")
            logger.info("=" * 60)

            return selected
        else:
            logger.warning(f"Choice {choice_num} out of range, using first")
            return images_with_valid_caption[0]

    except Exception as e:
        logger.error(f"Selection failed: {e}")
        logger.exception("Full traceback:")
        return images[0] if images else None


def select_representative_images(
    images: List[Dict],
    max_count: int = 1,
    paper_title: str = ""
) -> List[Dict]:
    """
    논문의 대표 이미지 선택 (Caption 있는 것만 고려)

    Args:
        images: 이미지 리스트
        max_count: 선택할 최대 이미지 개수
        paper_title: 논문 제목 (LLM 선택 시 사용)

    Returns:
        선택된 대표 이미지 리스트
    """
    if not images:
        return []

    if len(images) <= max_count:
        return images[:max_count]

    # LLM으로 대표 이미지 선택 (내부에서 caption 필터링)
    selected = select_representative_image_with_llm(images, paper_title)
    return [selected] if selected else []
