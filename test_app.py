# tests/test_app.py
import os
import json
import pytest
from fastapi.testclient import TestClient
import paper_processor as pp

@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    monkeypatch.setenv("S3_BUCKET_NAME", "dummy-bucket")
    monkeypatch.setenv("S3_PAPERS_PREFIX", "papers/")
    monkeypatch.setenv("DOCPAMIN_API_KEY", "x")
    monkeypatch.setenv("LLM_API_KEY", "x")
    monkeypatch.setenv("LLM_BASE_URL", "http://llm.local")
    monkeypatch.setenv("CONFLUENCE_URL", "http://conf.local")
    monkeypatch.setenv("CONFLUENCE_EMAIL", "u@c.com")
    monkeypatch.setenv("CONFLUENCE_API_TOKEN", "tok")
    yield

@pytest.fixture
def client():
    return TestClient(pp.app)

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "healthy"

def test_list_s3_papers(client, monkeypatch):
    # stub S3 listing
    def fake_iter(bucket, prefix, **kw):
        return [{"title":"test","s3_key":"papers/a.pdf","last_modified":"2025-10-23","size_bytes":123}]
    monkeypatch.setattr(pp, "get_s3_papers", lambda bucket, prefix, **kw: [
        {"title":"test","s3_key":"papers/a.pdf","s3_bucket":bucket, "last_modified":"2025-10-23","size_bytes":123}
    ])
    r = client.get("/list-s3-papers?bucket=b&prefix=papers/")
    assert r.status_code == 200
    j = r.json()
    assert j["papers_found"] == 1
    assert j["papers"][0]["s3_key"] == "papers/a.pdf"

def test_process_flow(client, monkeypatch):
    # 1) mock S3
    monkeypatch.setattr(pp, "get_s3_papers", lambda **kw: [
        {"title":"p1", "s3_key":"papers/p1.pdf", "s3_bucket":"b", "last_modified":"...", "size_bytes":123}
    ])
    monkeypatch.setattr(pp, "download_pdf_from_s3", lambda s3_key, s3_bucket: "/tmp/fake.pdf")
    # 2) mock Docpamin
    monkeypatch.setattr(pp, "parse_pdf_with_docpamin", lambda path: ("# Abstract\ntext", {}))
    # 3) mock LLM summarize → PaperAnalysis
    def fake_analyze(info, md, meta):
        return pp.PaperAnalysis(
            title=info["title"], authors=[], abstract="abs", summary="summary",
            key_contributions=["A","B"], methodology="meth", results="res",
            relevance_score=8, tags=["AI","LLM"], source_file=info["s3_key"]
        )
    monkeypatch.setattr(pp, "analyze_paper_with_llm", fake_analyze)
    # 4) mock Confluence
    monkeypatch.setattr(pp, "upload_to_confluence", lambda analyses, title: {"success":True,"page_url":"http://conf/p/1","page_id":"1"})

    r = client.post("/process-s3-papers", json={"bucket":"b","prefix":"papers/"})
    assert r.status_code == 200
    j = r.json()
    assert j["papers_processed"] == 1
    assert "confluence_url" in j
