#!/usr/bin/env python3
"""
Improved test script for AI Paper Newsletter API (S3-only)
- CLI args (base-url, timeout, no-input, process)
- Robust HTTP session with retries
- JSON schema validation for key endpoints
- Optional JUnit XML output for CI
"""

import os
import sys
import json
import time
import argparse
import requests
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# ---------- CLI & ENV ----------
load_dotenv()

def parse_args():
    p = argparse.ArgumentParser(description="AI Paper Newsletter API test")
    p.add_argument("--base-url", default=os.getenv("TEST_BASE_URL", "http://localhost:7070"))
    p.add_argument("--timeout", type=int, default=int(os.getenv("TEST_TIMEOUT", "300")))
    p.add_argument("--no-input", action="store_true", help="Do not ask interactive input")
    p.add_argument("--process", action="store_true", help="Run processing test even without prompt")
    p.add_argument("--no-color", action="store_true", help="Disable ANSI colors")
    p.add_argument("--junit", default=None, help="Write JUnit XML result to this path")
    return p.parse_args()

ARGS = parse_args()
BASE_URL = ARGS.base_url
TEST_TIMEOUT = ARGS.timeout

# ---------- Colors ----------
def use_color():
    if ARGS.no_color:
        return False
    return sys.stdout.isatty()

USE_COLOR = use_color()

class Colors:
    GREEN = '\033[92m' if USE_COLOR else ''
    YELLOW = '\033[93m' if USE_COLOR else ''
    RED = '\033[91m' if USE_COLOR else ''
    BLUE = '\033[94m' if USE_COLOR else ''
    END = '\033[0m' if USE_COLOR else ''

def print_success(msg: str): print(f"{Colors.GREEN}✅ {msg}{Colors.END}")
def print_error(msg: str): print(f"{Colors.RED}❌ {msg}{Colors.END}")
def print_info(msg: str): print(f"{Colors.BLUE}ℹ️  {msg}{Colors.END}")
def print_warning(msg: str): print(f"{Colors.YELLOW}⚠️  {msg}{Colors.END}")

# ---------- Robust Session ----------
from requests.adapters import HTTPAdapter, Retry
session = requests.Session()
retries = Retry(
    total=3, connect=3, read=3,
    backoff_factor=0.5,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=frozenset(["GET", "POST"])
)
session.mount("http://", HTTPAdapter(max_retries=retries))
session.mount("https://", HTTPAdapter(max_retries=retries))

def get_json(url: str, **kw) -> Optional[Dict[str, Any]]:
    r = session.get(url, timeout=kw.pop("timeout", 10))
    r.raise_for_status()
    return r.json()

def post_json(url: str, payload: Dict[str, Any], **kw) -> Optional[Dict[str, Any]]:
    r = session.post(url, json=payload, timeout=kw.pop("timeout", TEST_TIMEOUT))
    r.raise_for_status()
    return r.json()

# ---------- Light JSON Schema Checks ----------
def assert_has_keys(d: Dict[str, Any], keys, context=""):
    for k in keys:
        if k not in d:
            raise AssertionError(f"[{context}] missing key: {k}")

def test_health_check() -> bool:
    print_info("Testing /health ...")
    try:
        data = get_json(f"{BASE_URL}/health", timeout=5)
        assert_has_keys(data, ["status", "timestamp", "mode"], "health")
        print_success("Health check passed")
        print(f"   Status: {data.get('status')}, Mode: {data.get('mode')}")
        return True
    except Exception as e:
        print_error(f"Health check failed: {e}")
        return False

def test_api_documentation() -> bool:
    print_info("Checking /docs ...")
    try:
        r = session.get(f"{BASE_URL}/docs", timeout=5)
        if r.status_code == 200 and ("Swagger UI" in r.text or "FastAPI" in r.text):
            print_success("API documentation is accessible")
            return True
        print_error(f"Docs not accessible (status={r.status_code})")
        return False
    except Exception as e:
        print_error(f"Docs check failed: {e}")
        return False

def test_confluence_connection() -> Optional[bool]:
    print_info("Testing Confluence connection ...")
    url = os.getenv("CONFLUENCE_URL")
    email = os.getenv("CONFLUENCE_EMAIL")
    token = os.getenv("CONFLUENCE_API_TOKEN")
    if not all([url, email, token]):
        print_warning("Confluence credentials not configured, skipping")
        return None
    try:
        r = session.get(f"{url}/rest/api/space", auth=(email, token), timeout=10)
        if r.status_code == 200:
            print_success("Confluence connection OK")
            space_key = os.getenv("CONFLUENCE_SPACE_KEY")
            if space_key:
                try:
                    res = r.json().get("results", [])
                    exists = any(s.get("key") == space_key for s in res)
                    if exists:
                        print_success(f"Space '{space_key}' exists")
                    else:
                        print_warning(f"Space '{space_key}' not found in first page (may be paginated)")
                except Exception:
                    pass
            return True
        print_error(f"Confluence failed: {r.status_code}")
        return False
    except Exception as e:
        print_error(f"Confluence check error: {e}")
        return False

def test_s3_list() -> Optional[bool]:
    print_info("Testing /list-s3-papers ...")
    bucket = os.getenv("S3_BUCKET_NAME")
    prefix = os.getenv("S3_PAPERS_PREFIX", "papers/")
    if not bucket:
        print_warning("S3_BUCKET_NAME not set; skipping list")
        return None
    try:
        url = f"{BASE_URL}/list-s3-papers?bucket={bucket}&prefix={prefix}"
        data = get_json(url, timeout=30)
        assert_has_keys(data, ["bucket", "prefix", "papers_found", "papers"], "list-s3-papers")
        print_success("S3 listing OK")
        print(f"   Found: {data['papers_found']}")
        if data["papers_found"] == 0:
            print_warning("No PDFs in S3; upload one to proceed")
        else:
            for p in data["papers"][:3]:
                print(f"   - {p.get('title')} ({p.get('s3_key')})")
        return True
    except Exception as e:
        print_error(f"S3 list failed: {e}")
        return False

def test_s3_processing(force=False) -> Optional[bool]:
    print_info("Testing /process-s3-papers ...")
    bucket = os.getenv("S3_BUCKET_NAME")
    prefix = os.getenv("S3_PAPERS_PREFIX", "papers/")
    if not bucket:
        print_warning("S3_BUCKET_NAME not set; skipping processing")
        return None

    # Optionally confirm
    if not (ARGS.no_input or force or ARGS.process):
        try:
            choice = input("Run processing? (y/n): ").strip().lower()
            if choice != "y":
                print_info("Skipping processing")
                return None
        except EOFError:
            print_info("Skipping (no TTY)")
            return None

    # First, check there is at least one paper
    try:
        listing = get_json(f"{BASE_URL}/list-s3-papers?bucket={bucket}&prefix={prefix}", timeout=30)
        if listing.get("papers_found", 0) == 0:
            print_warning("No papers; skipping processing")
            return None
    except Exception as e:
        print_warning(f"List failed before processing: {e}")

    try:
        payload = {
            "bucket": bucket,
            "prefix": prefix,
            "file_pattern": "*.pdf",
            "process_subdirectories": True
        }
        data = post_json(f"{BASE_URL}/process-s3-papers", payload, timeout=TEST_TIMEOUT)
        # expected keys
        assert_has_keys(data, ["message", "papers_found", "papers_processed", "bucket", "prefix"], "process-s3-papers")
        print_success("Processing completed")
        print(f"   Found={data['papers_found']}  Processed={data['papers_processed']}")
        if data.get("errors"):
            print_warning(f"Errors: {len(data['errors'])}")
            for e in data["errors"][:3]:
                print("   -", e)
        if data.get("confluence_url"):
            print_info(f"Confluence: {data['confluence_url']}")
        return True
    except requests.Timeout:
        print_warning("Processing timed out; check service logs")
        return False
    except Exception as e:
        print_error(f"Processing failed: {e}")
        return False

# ---------- JUnit XML (optional) ----------
def write_junit(results: Dict[str, Optional[bool]], path: str):
    """Minimal JUnit XML for CI dashboards."""
    import xml.etree.ElementTree as ET
    testsuite = ET.Element("testsuite", name="paper-processor-tests", tests=str(len(results)))
    for name, res in results.items():
        case = ET.SubElement(testsuite, "testcase", name=name)
        if res is False:
            failure = ET.SubElement(case, "failure", message="failed")
            failure.text = f"{name} failed"
        elif res is None:
            skipped = ET.SubElement(case, "skipped")
            skipped.text = "skipped"
    tree = ET.ElementTree(testsuite)
    with open(path, "wb") as f:
        tree.write(f, encoding="utf-8", xml_declaration=True)
    print_info(f"JUnit XML written: {path}")

def main():
    print("="*60)
    print("🧪 AI Paper Newsletter - API Test Suite (S3-only)")
    print("="*60)
    # Wait until healthy
    print_info("Waiting for service ...")
    ready = False
    for i in range(15):
        try:
            r = session.get(f"{BASE_URL}/health", timeout=2)
            if r.status_code == 200:
                ready = True; break
        except:
            pass
        time.sleep(2)
    if not ready:
        print_error("Service not responding. Is it running?")
        print_info(f"Try: uvicorn paper_processor:app --host 0.0.0.0 --port 7070")
        sys.exit(1)

    results: Dict[str, Optional[bool]] = {}
    print("\n== Test 1: Health ==")
    results["health"] = test_health_check()
    print("\n== Test 2: Docs ==")
    results["docs"] = test_api_documentation()
    print("\n== Test 3: Confluence ==")
    results["confluence"] = test_confluence_connection()
    print("\n== Test 4: S3 List ==")
    results["s3_list"] = test_s3_list()
    print("\n== Test 5: Process ==")
    results["s3_process"] = test_s3_processing(force=ARGS.process)

    # Summary
    print("\n" + "="*60)
    print("📊 Summary")
    print("="*60)
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)
    for k, v in results.items():
        status = "✅ PASSED" if v is True else "❌ FAILED" if v is False else "⏭️  SKIPPED"
        print(f"{status}: {k}")
    print(f"\nTotal={len(results)} Passed={passed} Failed={failed} Skipped={skipped}")

    if ARGS.junit:
        write_junit(results, ARGS.junit)

    sys.exit(0 if failed == 0 else 2)

if __name__ == "__main__":
    main()
