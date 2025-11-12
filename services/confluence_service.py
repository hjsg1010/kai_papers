"""
Confluence service for uploading paper analyses
"""
import logging
import requests
from datetime import datetime
from typing import List, Optional, Dict

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
    Confluence에서 제목으로 페이지 검색

    Args:
        title: 페이지 제목

    Returns:
        페이지 정보 딕셔너리 또는 None
    """
    url = f"{CONFLUENCE_URL}/rest/api/content"
    params = {"title": title, "spaceKey": CONFLUENCE_SPACE_KEY, "expand": "version"}
    r = requests.get(url, params=params, auth=(CONFLUENCE_EMAIL, CONFLUENCE_API_TOKEN), timeout=30)
    r.raise_for_status()
    res = r.json().get("results", [])
    return res[0] if res else None


def _conf_escape(s: str) -> str:
    """
    HTML 특수 문자 이스케이프

    Args:
        s: 이스케이프할 문자열

    Returns:
        이스케이프된 문자열
    """
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def upload_to_confluence(analyses: List[PaperAnalysis], page_title: str) -> Dict:
    """
    논문 분석 결과를 Confluence에 업로드

    Args:
        analyses: PaperAnalysis 객체 리스트
        page_title: Confluence 페이지 제목

    Returns:
        업로드 결과 딕셔너리 (success, page_url, page_id)

    Raises:
        Exception: 업로드 실패 시
    """
    logger.info(f"Uploading to Confluence: {page_title}")

    body = [
        f"<h1>AI Paper Newsletter - {datetime.now().strftime('%Y-%m-%d')}</h1>",
        "<p>이번 주의 주목할 만한 AI 논문들을 소개합니다.</p>",
        '<ac:structured-macro ac:name="info"><ac:rich-text-body>',
        f"<p>총 {len(analyses)}편의 논문이 분석되었습니다.</p>",
        "</ac:rich-text-body></ac:structured-macro><hr/>"
    ]

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
                "id": page_id,
                "type": "page",
                "title": page_title,
                "space": {"key": CONFLUENCE_SPACE_KEY},
                "body": {"storage": {"value": content_html, "representation": "storage"}},
                "version": {"number": version},
            }
            r = requests.put(
                f"{create_url}/{page_id}",
                json=payload,
                headers=headers,
                auth=(CONFLUENCE_EMAIL, CONFLUENCE_API_TOKEN),
                timeout=60
            )
            r.raise_for_status()
            result = r.json()
        else:
            payload = {
                "type": "page",
                "title": page_title,
                "space": {"key": CONFLUENCE_SPACE_KEY},
                "body": {"storage": {"value": content_html, "representation": "storage"}},
            }
            r = requests.post(
                create_url,
                json=payload,
                headers=headers,
                auth=(CONFLUENCE_EMAIL, CONFLUENCE_API_TOKEN),
                timeout=60
            )
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
