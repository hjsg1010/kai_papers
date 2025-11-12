"""
Confluence and Markdown service for uploading paper analyses
"""
import logging
import re
import json
import requests
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Tuple

from config.settings import (
    CONFLUENCE_URL,
    CONFLUENCE_EMAIL,
    CONFLUENCE_API_TOKEN,
    CONFLUENCE_SPACE_KEY
)
from models import PaperAnalysis

logger = logging.getLogger(__name__)


# ===== Confluence Functions =====

def _conf_get_page_by_title(title: str) -> Optional[Dict]:
    """
    Confluence 페이지를 제목으로 조회

    Args:
        title: 페이지 제목

    Returns:
        페이지 정보 딕셔너리 또는 None (페이지가 없는 경우)
    """
    url = f"{CONFLUENCE_URL}/rest/api/content"
    params = {"title": title, "spaceKey": CONFLUENCE_SPACE_KEY, "expand": "version"}
    r = requests.get(url, params=params, auth=(CONFLUENCE_EMAIL, CONFLUENCE_API_TOKEN), timeout=30)
    r.raise_for_status()
    res = r.json().get("results", [])
    return res[0] if res else None


def _conf_escape(s: str) -> str:
    """
    HTML 특수문자 이스케이프

    Args:
        s: 이스케이프할 문자열

    Returns:
        이스케이프된 문자열
    """
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def upload_to_confluence(analyses: List[PaperAnalysis], page_title: str):
    """
    논문 분석 결과를 Confluence에 업로드

    Args:
        analyses: PaperAnalysis 객체 리스트
        page_title: Confluence 페이지 제목

    Returns:
        업로드 결과 딕셔너리 (success, page_url, page_id)

    Raises:
        Exception: Confluence API 호출 실패 시
    """
    logger.info(f"Uploading to Confluence: {page_title}")
    body = [f"<h1>AI Paper Newsletter - {datetime.now().strftime('%Y-%m-%d')}</h1>",
            "<p>이번 주의 주목할 만한 AI 논문들을 소개합니다.</p>",
            '<ac:structured-macro ac:name="info"><ac:rich-text-body>',
            f"<p>총 {len(analyses)}편의 논문이 분석되었습니다.</p>",
            "</ac:rich-text-body></ac:structured-macro><hr/>"]
    for i, a in enumerate(analyses, 1):
        body.append(f"<h2>{i}. {_conf_escape(a.title)}</h2>")
        if a.authors:
            body.append(f"<p><strong>Authors:</strong> {_conf_escape(', '.join(a.authors[:8]))}</p>")
        if a.tags:
            body.append(f"<p><strong>Tags:</strong> {_conf_escape(', '.join(a.tags))}</p>")
        if a.abstract:
            body.append("<h3>Abstract</h3><p>" + _conf_escape(a.abstract) + "</p>")
        body.append("<h3>Analysis</h3>")
        body.append(a.summary)
        body.append(f"<p><em>Source:</em> s3://{a.source_file}</p>")
        body.append("<hr/>")
    content_html = "\n".join(body)

    create_url = f"{CONFLUENCE_URL}/rest/api/content"
    headers = {"Content-Type": "application/json"}
    try:
        existing = _conf_get_page_by_title(page_title)
        if existing:
            page_id = existing["id"]
            version = existing.get("version", {}).get("number", 1) + 1
            payload = {
                "id": page_id, "type": "page", "title": page_title,
                "space": {"key": CONFLUENCE_SPACE_KEY},
                "body": {"storage": {"value": content_html, "representation": "storage"}},
                "version": {"number": version},
            }
            r = requests.put(f"{create_url}/{page_id}", json=payload, headers=headers,
                             auth=(CONFLUENCE_EMAIL, CONFLUENCE_API_TOKEN), timeout=60)
            r.raise_for_status()
            result = r.json()
        else:
            payload = {
                "type": "page", "title": page_title, "space": {"key": CONFLUENCE_SPACE_KEY},
                "body": {"storage": {"value": content_html, "representation": "storage"}},
            }
            r = requests.post(create_url, json=payload, headers=headers,
                              auth=(CONFLUENCE_EMAIL, CONFLUENCE_API_TOKEN), timeout=60)
            r.raise_for_status()
            result = r.json()

        base = CONFLUENCE_URL.rstrip("/")
        webui = result.get("_links", {}).get("webui")
        tiny = result.get("_links", {}).get("tinyui")
        page_url = f"{base}{webui}" if webui else (f"{base}{tiny}" if tiny else f"{base}/pages/{result['id']}")
        logger.info(f"Confluence page: {page_url}")
        return {"success": True, "page_url": page_url, "page_id": result["id"]}
    except Exception as e:
        logger.exception("Confluence upload error")
        raise


# ===== Markdown Builder Functions =====

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


def build_markdown(
    analyses: List[PaperAnalysis],
    papers_metadata: Optional[List[Dict]] = None,
    week_label: str = "",
    prefix: str = ""
) -> Tuple[str, str]:
    """
    논문 분석 결과를 Markdown으로 변환

    Args:
        analyses: PaperAnalysis 객체 리스트
        papers_metadata: 논문 메타데이터 리스트 (이미지 정보 포함)
        week_label: 주차 레이블 (예: "w42")
        prefix: S3 prefix

    Returns:
        Tuple[str, str]: (파일명, Markdown 콘텐츠)
    """
    if not week_label:
        week_label = derive_week_label(prefix)

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
            lines.append(f"**📌 TL;DR**\n")
            lines.append(f"{data['tldr']}\n\n")

        # 핵심 기여
        if data.get('key_contributions'):
            lines.append(f"**🎯 핵심 기여**\n")
            for contrib in data['key_contributions']:
                lines.append(f"- {contrib}\n")
            lines.append("\n")

        # 방법론
        if data.get('methodology'):
            lines.append(f"**🔬 방법론**\n")
            lines.append(f"{data['methodology']}\n\n")

        # 결과
        if data.get('results'):
            lines.append(f"**📊 결과**\n")
            lines.append(f"{data['results']}\n\n")

        # 새로운 점
        if data.get('novelty'):
            lines.append(f"**💡 새로운 점**\n")
            lines.append(f"{data['novelty']}\n\n")

        # 한계점
        if data.get('limitations'):
            lines.append(f"**⚠️ 한계점**\n")
            for limitation in data['limitations']:
                lines.append(f"- {limitation}\n")
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
