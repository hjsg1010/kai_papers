"""Markdown rendering helpers for weekly reports."""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import logger
from .models import PaperAnalysis


def derive_week_label(prefix: str) -> str:
    match = re.search(r"w(\d{1,2})", prefix or "", re.IGNORECASE)
    if match:
        return f"w{int(match.group(1))}"
    _, iso_week, _ = datetime.utcnow().isocalendar()
    return f"w{iso_week}"


def build_markdown(
    analyses: List[PaperAnalysis],
    papers_metadata: Optional[List[Dict]] = None,
    week_label: str = "",
    prefix: str = "",
) -> Tuple[str, str]:
    if not week_label:
        week_label = derive_week_label(prefix)

    header = f"""# AI Paper Newsletter – {week_label}
_Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}_

Source prefix: `{prefix}`

---

"""

    image_map: Dict[str, Dict] = {}
    if papers_metadata:
        for meta in papers_metadata:
            if meta.get("images_info") and meta.get("images_info", {}).get("representative_images"):
                s3_key = meta.get("s3_key", "")
                image_map[s3_key] = meta["images_info"]

    parts = [header]

    for i, analysis in enumerate(analyses, 1):
        tags = f"**Tags:** {', '.join(analysis.tags)}" if analysis.tags else ""
        authors = f"**Authors:** {', '.join(analysis.authors[:8])}" if analysis.authors else ""

        abstract_block = ""
        if analysis.abstract and analysis.abstract.strip():
            abstract_block = f"\n**Abstract**\n\n> {analysis.abstract.strip()}\n\n"

        summary_formatted = format_summary_as_markdown(analysis.summary)

        section = f"""## {i}. {analysis.title}

{authors}
{tags}

{summary_formatted}

{abstract_block}"""

        if analysis.source_file in image_map:
            img_info = image_map[analysis.source_file]
            rep_imgs = img_info.get("representative_images", [])

            if rep_imgs:
                rep_img = rep_imgs[0]
                paper_name = Path(analysis.source_file).stem
                img_filename = f"{week_label}_{paper_name}_fig{rep_img['index'] + 1}.{rep_img['type']}"

                section += f"""### 📊 대표 이미지

**전체 이미지:** {img_info['total_images']}개
**대표 이미지:** Figure {rep_img['index'] + 1} ({rep_img['size_kb']:.1f}KB)

![Figure {rep_img['index'] + 1}](images/{img_filename})

"""

        section += f"""**Source:** `s3://{analysis.source_file}`

---

"""
        parts.append(section)

    md_content = "".join(parts)
    md_filename = f"{week_label}.md"
    return md_filename, md_content


def format_summary_as_markdown(summary: str) -> str:
    try:
        summary_clean = summary.strip().replace("~", "–")

        json_match = re.search(r"\{[\s\S]*\}", summary_clean)
        if not json_match:
            return f"**Summary**\n\n{summary_clean}\n"

        data = json.loads(json_match.group(0))

        lines = ["**Summary**  \n\n"]

        if data.get("tldr"):
            lines.append("**📌 TL;DR**\n")
            lines.append(f"{data['tldr']}\n\n")

        if data.get("key_contributions"):
            lines.append("**🎯 핵심 기여**\n")
            for contrib in data["key_contributions"]:
                lines.append(f"- {contrib}\n")
            lines.append("\n")

        if data.get("methodology"):
            lines.append("**🔬 방법론**\n")
            lines.append(f"{data['methodology']}\n\n")

        if data.get("results"):
            lines.append("**📊 결과**\n")
            lines.append(f"{data['results']}\n\n")

        if data.get("novelty"):
            lines.append("**💡 새로운 점**\n")
            lines.append(f"{data['novelty']}\n\n")

        if data.get("limitations"):
            lines.append("**⚠️ 한계점**\n")
            for limitation in data["limitations"]:
                lines.append(f"- {limitation}\n")
            lines.append("\n")

        if data.get("relevance_score"):
            score = data["relevance_score"]
            stars = "" * score
            lines.append(f"**관련성 점수:** {stars} ({score}/10)\n\n")

        return "".join(lines)

    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Failed to parse summary JSON: %s", exc)
        return f"**Summary**\n\n{summary.strip().replace('~', '–')}\n"


__all__ = ["build_markdown", "derive_week_label", "format_summary_as_markdown"]
