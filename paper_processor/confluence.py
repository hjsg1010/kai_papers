"""Confluence publishing helpers."""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

import requests

from .config import (
    CONFLUENCE_API_TOKEN,
    CONFLUENCE_EMAIL,
    CONFLUENCE_SPACE_KEY,
    CONFLUENCE_URL,
    logger,
)
from .models import PaperAnalysis


def _conf_get_page_by_title(title: str) -> Optional[Dict]:
    url = f"{CONFLUENCE_URL}/rest/api/content"
    params = {"title": title, "spaceKey": CONFLUENCE_SPACE_KEY, "expand": "version"}
    response = requests.get(url, params=params, auth=(CONFLUENCE_EMAIL, CONFLUENCE_API_TOKEN), timeout=30)
    response.raise_for_status()
    results = response.json().get("results", [])
    return results[0] if results else None


def _conf_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def upload_to_confluence(analyses: List[PaperAnalysis], page_title: str):
    logger.info("Uploading to Confluence: %s", page_title)
    body = [
        f"<h1>AI Paper Newsletter - {datetime.now().strftime('%Y-%m-%d')}</h1>",
        "<p>이번 주의 주목할 만한 AI 논문들을 소개합니다.</p>",
        '<ac:structured-macro ac:name="info"><ac:rich-text-body>',
        f"<p>총 {len(analyses)}편의 논문이 분석되었습니다.</p>",
        "</ac:rich-text-body></ac:structured-macro><hr/>",
    ]
    for i, analysis in enumerate(analyses, 1):
        body.append(f"<h2>{i}. {_conf_escape(analysis.title)}</h2>")
        if analysis.authors:
            body.append(f"<p><strong>Authors:</strong> {_conf_escape(', '.join(analysis.authors[:8]))}</p>")
        if analysis.tags:
            body.append(f"<p><strong>Tags:</strong> {_conf_escape(', '.join(analysis.tags))}</p>")
        if analysis.abstract:
            body.append("<h3>Abstract</h3><p>" + _conf_escape(analysis.abstract) + "</p>")
        body.append("<h3>Analysis</h3>")
        body.append(analysis.summary)
        body.append(f"<p><em>Source:</em> s3://{analysis.source_file}</p>")
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
            response = requests.put(
                f"{create_url}/{page_id}",
                json=payload,
                headers=headers,
                auth=(CONFLUENCE_EMAIL, CONFLUENCE_API_TOKEN),
                timeout=60,
            )
            response.raise_for_status()
            result = response.json()
        else:
            payload = {
                "type": "page",
                "title": page_title,
                "space": {"key": CONFLUENCE_SPACE_KEY},
                "body": {"storage": {"value": content_html, "representation": "storage"}},
            }
            response = requests.post(
                create_url,
                json=payload,
                headers=headers,
                auth=(CONFLUENCE_EMAIL, CONFLUENCE_API_TOKEN),
                timeout=60,
            )
            response.raise_for_status()
            result = response.json()

        base = CONFLUENCE_URL.rstrip("/")
        webui = result.get("_links", {}).get("webui")
        tiny = result.get("_links", {}).get("tinyui")
        page_url = f"{base}{webui}" if webui else (f"{base}{tiny}" if tiny else f"{base}/pages/{result['id']}")
        logger.info("Confluence page: %s", page_url)
        return {"success": True, "page_url": page_url, "page_id": result["id"]}
    except Exception as exc:
        logger.exception("Confluence upload error")
        raise


__all__ = ["upload_to_confluence"]
