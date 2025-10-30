"""Image parsing and selection utilities."""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from .config import logger
from .llm import call_llm


def remove_base64_images(markdown: str, replacement: str = "[Image]") -> Tuple[str, int]:
    """Replace embedded base64 images with a placeholder."""
    pattern = r"!\[[^\]]*\]\(data:image/[^;]+;base64,[A-Za-z0-9+/=]+\)"
    cleaned, count = re.subn(pattern, replacement, markdown)
    if count > 0:
        logger.info("Removed %d base64 images from markdown", count)
    return cleaned, count


def extract_base64_images(markdown: str) -> List[Dict]:
    """Return metadata for embedded base64 images found in *markdown*."""
    pattern = r"!\[([^\]]*)\]\(data:image/([^;]+);base64,([A-Za-z0-9+/=]+)\)"
    images: List[Dict] = []
    for match in re.finditer(pattern, markdown):
        base64_data = match.group(3)
        size_bytes = len(base64_data) * 3 // 4
        images.append(
            {
                "full_match": match.group(0),
                "alt_text": match.group(1),
                "mime_type": match.group(2),
                "base64_data": base64_data,
                "size_kb": size_bytes / 1024,
                "position": match.start(),
            }
        )
    return images


def extract_figure_pairs_from_json(json_metadata: Dict) -> List[Dict]:
    """Extract figure/caption pairs from Docpamin JSON output."""
    figure_pairs: List[Dict] = []

    try:
        pages = json_metadata.get("pages", [])

        for page in pages:
            layout = page.get("layout", [])

            for i, block in enumerate(layout):
                if block.get("type") != "PICTURE":
                    continue

                if i + 1 >= len(layout):
                    continue

                next_block = layout[i + 1]
                if next_block.get("type") != "CAPTION":
                    logger.debug(
                        "PICTURE at id=%s has no CAPTION (next: %s)",
                        block.get("id"),
                        next_block.get("type"),
                    )
                    continue

                caption_content = next_block.get("content", "").strip()
                if not caption_content:
                    continue

                fig_match = re.search(r"Figure[~\s]+(\d+)[:\.]?\s*(.+?)$", caption_content, re.IGNORECASE)
                if not fig_match:
                    continue

                fig_num = int(fig_match.group(1))
                caption_text = fig_match.group(2).strip()
                if not is_valid_caption(caption_text):
                    logger.debug("Invalid caption for Figure %s", fig_num)
                    continue

                picture_content = block.get("content", "")
                base64_match = re.search(r"base64,([A-Za-z0-9+/=]{50,100})", picture_content)
                base64_preview = base64_match.group(1) if base64_match else ""

                figure_pairs.append(
                    {
                        "figure_number": fig_num,
                        "caption": caption_text,
                        "base64_preview": base64_preview,
                        "page_no": page.get("page_no"),
                        "picture_id": block.get("id"),
                        "caption_id": next_block.get("id"),
                    }
                )

                logger.info("📷 Figure %s: %s...", fig_num, caption_text[:60])

        logger.info("Found %d valid PICTURE-CAPTION pairs", len(figure_pairs))

    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Failed to extract figure pairs: %s", exc)

    return figure_pairs


def match_images_with_figure_pairs(images: List[Dict], figure_pairs: List[Dict]) -> List[Dict]:
    """Augment images with caption data using base64 previews for matching."""
    if not figure_pairs:
        logger.warning("No figure pairs to match")
        return images

    matched_count = 0

    for img in images:
        img_base64 = img.get("base64_data", "")
        if not img_base64 or len(img_base64) < 100:
            continue

        img_preview = img_base64[:100]
        for pair in figure_pairs:
            pair_preview = pair.get("base64_preview", "")
            if pair_preview and pair_preview in img_preview:
                img["caption"] = pair["caption"]
                img["figure_number"] = pair["figure_number"]
                matched_count += 1
                logger.info(
                    "✅ Image %s → Figure %s: %s...",
                    img["index"],
                    pair["figure_number"],
                    pair["caption"][:60],
                )
                break

    logger.info("Matched %d/%d images with captions", matched_count, len(images))
    return images


def match_images_with_captions_from_json(images: List[Dict], json_metadata: Dict) -> List[Dict]:
    """Convenience helper that extracts figure pairs and matches them to images."""
    figure_pairs = extract_figure_pairs_from_json(json_metadata)
    if not figure_pairs:
        return images
    return match_images_with_figure_pairs(images, figure_pairs)


def select_representative_image(images: List[Dict], min_kb: float = 10, max_kb: float = 200) -> Optional[Dict]:
    """Pick a representative image based on file size and position."""
    if not images:
        return None
    candidates = [img for img in images if min_kb <= img["size_kb"] <= max_kb]
    if not candidates:
        candidates = sorted(images, key=lambda x: abs(x["size_kb"] - (min_kb + max_kb) / 2))[:3]
    return min(candidates, key=lambda x: x["position"]) if candidates else None


def process_markdown_images(markdown: str, remove_for_llm: bool = True, keep_representative: int = 1) -> Tuple[str, List[Dict]]:
    """Extract images embedded in markdown (without caption matching)."""
    pattern = r"!\[(.*?)\]\(data:image/([^;]+);base64,([A-Za-z0-9+/=]+)\)"

    images: List[Dict] = []

    def extract_image(match: re.Match) -> str:
        alt_text = match.group(1)
        img_type = match.group(2)
        base64_data = match.group(3)

        images.append(
            {
                "index": len(images),
                "alt": alt_text.strip(),
                "type": img_type,
                "size": len(base64_data),
                "size_kb": len(base64_data) * 3 / 4 / 1024,
                "base64_data": base64_data,
                "full": match.group(0),
            }
        )

        if remove_for_llm:
            return f"\n[Image {len(images)}]\n"
        return match.group(0)

    processed_md = re.sub(pattern, extract_image, markdown)

    if images:
        logger.info("Extracted %d images from markdown", len(images))

    return processed_md, images


def is_valid_caption(caption: str) -> bool:
    """Validate caption text to filter out artefacts such as hashes or base64."""
    if not caption or len(caption) < 10:
        return False

    base64_pattern = r"^[A-Za-z0-9+/=]{50,}$"
    if re.match(base64_pattern, caption):
        logger.debug("Rejected caption (base64): %s...", caption[:50])
        return False

    words = caption.split()
    if len(words) == 1 and len(words[0]) > 40:
        logger.debug("Rejected caption (hash): %s...", caption[:50])
        return False

    english_words = [w for w in words if re.match(r"^[a-zA-Z]+$", w) and len(w) > 2]
    if len(english_words) < 2:
        logger.debug("Rejected caption (no words): %s...", caption[:50])
        return False

    if caption.startswith("![") or caption.startswith("]("):
        return False

    return True


def select_representative_image_with_llm(images: List[Dict], paper_title: str = "") -> Optional[Dict]:
    """Use the LLM to pick the single best representative image."""
    if not images:
        return None
    if len(images) == 1:
        return images[0]

    try:
        logger.info("=" * 60)
        logger.info("🎯 select_representative_image_with_llm")
        logger.info("Total images: %d", len(images))

        images_with_valid_caption = []
        for img in images:
            caption = img.get("caption", "")
            if is_valid_caption(caption):
                images_with_valid_caption.append(img)
                logger.debug("  ✅ Image %s: %s...", img["index"], caption[:50])
            else:
                logger.info("  ❌ Skipped image %s: Invalid caption", img["index"])

        logger.info("Valid captions: %d/%d", len(images_with_valid_caption), len(images))
        logger.info("=" * 60)

        if not images_with_valid_caption:
            logger.warning("No valid captions, using first image")
            return images[0]
        if len(images_with_valid_caption) == 1:
            logger.info("Only one valid caption, auto-selected")
            return images_with_valid_caption[0]

        image_descriptions = []
        for choice_num, img in enumerate(images_with_valid_caption, 1):
            fig_num = img.get("figure_number", img["index"] + 1)
            caption = img.get("caption", "")
            desc = f"{choice_num}. (Figure {fig_num}): {caption} (Size: {img['size_kb']:.1f}KB)"
            image_descriptions.append(desc)

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
        response = call_llm(messages, max_tokens=500)

        response_text = response.strip()
        logger.info("LLM response: '%s'", response_text)

        numbers = re.findall(r"\b(\d+)\b", response_text)
        if not numbers:
            logger.warning("No number in response, using first valid")
            return images_with_valid_caption[0]

        choice_num = int(numbers[0])
        choice_idx = choice_num - 1
        logger.info("LLM chose: choice=%s, idx=%s", choice_num, choice_idx)

        if 0 <= choice_idx < len(images_with_valid_caption):
            selected = images_with_valid_caption[choice_idx]
            logger.info("=" * 60)
            logger.info("✅ SELECTED:")
            logger.info("   Index: %s", selected["index"])
            logger.info("   Figure: %s", selected.get("figure_number", "N/A"))
            logger.info("   Caption: %s...", selected.get("caption", "")[:80])
            logger.info("   Size: %.1fKB", selected["size_kb"])
            logger.info("=" * 60)
            return selected

        logger.warning("Choice %s out of range, using first", choice_num)
        return images_with_valid_caption[0]

    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Selection failed: %s", exc)
        logger.exception("Full traceback:")
        return images[0] if images else None


def select_representative_images(images: List[Dict], max_count: int = 1, paper_title: str = "") -> List[Dict]:
    """Return up to *max_count* representative images, using the LLM if needed."""
    if not images:
        return []
    if len(images) <= max_count:
        return images[:max_count]
    selected = select_representative_image_with_llm(images, paper_title)
    return [selected] if selected else []


__all__ = [
    "extract_base64_images",
    "extract_figure_pairs_from_json",
    "is_valid_caption",
    "match_images_with_captions_from_json",
    "match_images_with_figure_pairs",
    "process_markdown_images",
    "remove_base64_images",
    "select_representative_image",
    "select_representative_image_with_llm",
    "select_representative_images",
]
