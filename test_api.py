#!/usr/bin/env python3
"""
Test script for AI Paper Newsletter API (S3-only version)
Tests all endpoints and validates responses
"""

import requests
import json
import time
from typing import Dict, Any
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configuration
BASE_URL = "http://localhost:8000"
TEST_TIMEOUT = 300  # 5 minutes


class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    END = '\033[0m'


def print_success(msg: str):
    print(f"{Colors.GREEN}✅ {msg}{Colors.END}")


def print_error(msg: str):
    print(f"{Colors.RED}❌ {msg}{Colors.END}")


def print_info(msg: str):
    print(f"{Colors.BLUE}ℹ️  {msg}{Colors.END}")


def print_warning(msg: str):
    print(f"{Colors.YELLOW}⚠️  {msg}{Colors.END}")


def test_health_check() -> bool:
    """Test the health check endpoint"""
    print_info("Testing health check endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"Health check passed")
            print(f"   Status: {data.get('status')}")
            print(f"   Mode: {data.get('mode')}")
            return True
        else:
            print_error(f"Health check failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Health check failed: {str(e)}")
        return False


def test_api_documentation():
    """Check if API documentation is accessible"""
    print_info("Checking API documentation...")
    
    try:
        response = requests.get(f"{BASE_URL}/docs", timeout=10)
        
        if response.status_code == 200:
            print_success("API documentation is accessible")
            print(f"   URL: {BASE_URL}/docs")
            return True
        else:
            print_error("API documentation not accessible")
            return False
            
    except Exception as e:
        print_error(f"Failed to check API docs: {str(e)}")
        return False


def test_s3_list() -> bool:
    """Test S3 paper listing"""
    print_info("Testing S3 paper listing...")
    
    bucket = os.getenv("S3_BUCKET_NAME")
    prefix = os.getenv("S3_PAPERS_PREFIX", "papers/")
    
    if not bucket:
        print_warning("S3_BUCKET_NAME not configured, skipping test")
        return None
    
    try:
        url = f"{BASE_URL}/list-s3-papers?bucket={bucket}&prefix={prefix}"
        print_info(f"Checking S3: s3://{bucket}/{prefix}")
        
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print_success("S3 listing successful!")
            print(f"   Bucket: {data.get('bucket')}")
            print(f"   Prefix: {data.get('prefix')}")
            print(f"   Papers found: {data.get('papers_found', 0)}")
            
            if data.get('papers_found', 0) == 0:
                print_warning("No papers found in S3")
                print_info("Upload a PDF to test:")
                print(f"   aws s3 cp test.pdf s3://{bucket}/{prefix}")
            else:
                print_info("Papers in S3:")
                for paper in data.get('papers', [])[:3]:
                    print(f"   - {paper.get('title')}")
            
            return True
        else:
            print_error(f"S3 listing failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print_error(f"S3 listing test failed: {str(e)}")
        return False


def test_s3_processing() -> bool:
    """Test S3 paper processing (if papers exist)"""
    print_info("Testing S3 paper processing...")
    
    bucket = os.getenv("S3_BUCKET_NAME")
    prefix = os.getenv("S3_PAPERS_PREFIX", "papers/")
    
    if not bucket:
        print_warning("S3_BUCKET_NAME not configured, skipping test")
        return None
    
    # First check if papers exist
    try:
        list_response = requests.get(
            f"{BASE_URL}/list-s3-papers?bucket={bucket}&prefix={prefix}",
            timeout=30
        )
        
        if list_response.status_code == 200:
            data = list_response.json()
            if data.get('papers_found', 0) == 0:
                print_warning("No papers in S3 to process. Skipping processing test.")
                print_info("Upload papers first:")
                print(f"   aws s3 cp papers/ s3://{bucket}/{prefix} --recursive")
                return None
        
        print_info(f"Found {data.get('papers_found', 0)} papers. Starting processing...")
        print_warning("This may take several minutes...")
        
        # Process papers
        payload = {
            "bucket": bucket,
            "prefix": prefix,
            "file_pattern": "*.pdf",
            "process_subdirectories": True
        }
        
        response = requests.post(
            f"{BASE_URL}/process-s3-papers",
            json=payload,
            timeout=TEST_TIMEOUT
        )
        
        if response.status_code == 200:
            result = response.json()
            print_success("S3 processing completed successfully!")
            print(f"   Papers found: {result.get('papers_found', 0)}")
            print(f"   Papers processed: {result.get('papers_processed', 0)}")
            
            if result.get('errors'):
                print_warning(f"   Errors: {len(result['errors'])}")
                for error in result['errors'][:3]:
                    print(f"     - {error}")
            
            if result.get('confluence_url'):
                print(f"   Confluence: {result['confluence_url']}")
            
            return True
        else:
            print_error(f"S3 processing failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.Timeout:
        print_warning("S3 processing timed out (this may be normal for first run)")
        print_info("Check the service logs for progress")
        return False
    except Exception as e:
        print_error(f"S3 processing test failed: {str(e)}")
        return False


def test_confluence_connection() -> bool:
    """Test Confluence connection (without creating a page)"""
    print_info("Testing Confluence connection...")
    
    confluence_url = os.getenv("CONFLUENCE_URL")
    confluence_email = os.getenv("CONFLUENCE_EMAIL")
    confluence_token = os.getenv("CONFLUENCE_API_TOKEN")
    
    if not all([confluence_url, confluence_email, confluence_token]):
        print_warning("Confluence credentials not configured, skipping test")
        return None
    
    try:
        # Test by getting space info
        url = f"{confluence_url}/rest/api/space"
        response = requests.get(
            url,
            auth=(confluence_email, confluence_token),
            timeout=10
        )
        
        if response.status_code == 200:
            print_success("Confluence connection successful!")
            spaces = response.json().get('results', [])
            print(f"   Found {len(spaces)} spaces")
            
            space_key = os.getenv("CONFLUENCE_SPACE_KEY")
            if space_key:
                space_found = any(s['key'] == space_key for s in spaces)
                if space_found:
                    print_success(f"   Target space '{space_key}' exists")
                else:
                    print_warning(f"   Target space '{space_key}' not found")
            
            return True
        else:
            print_error(f"Confluence connection failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Confluence connection test failed: {str(e)}")
        return False


def main():
    print("=" * 60)
    print("🧪 AI Paper Newsletter - API Test Suite (S3-only)")
    print("=" * 60)
    print()
    
    # Wait for service to be ready
    print_info("Waiting for service to be ready...")
    for i in range(10):
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=2)
            if response.status_code == 200:
                print_success("Service is ready!")
                break
        except:
            pass
        time.sleep(2)
        if i == 9:
            print_error("Service is not responding. Is it running?")
            print_info(f"Start the service: python paper_processor.py")
            return
    
    print()
    
    # Run tests
    results = {}
    
    # Test 1: Health Check
    print("\n" + "=" * 60)
    print("Test 1: Health Check")
    print("=" * 60)
    results['health'] = test_health_check()
    
    # Test 2: API Documentation
    print("\n" + "=" * 60)
    print("Test 2: API Documentation")
    print("=" * 60)
    results['docs'] = test_api_documentation()
    
    # Test 3: Confluence Connection
    print("\n" + "=" * 60)
    print("Test 3: Confluence Connection")
    print("=" * 60)
    results['confluence'] = test_confluence_connection()
    
    # Test 4: S3 Listing
    print("\n" + "=" * 60)
    print("Test 4: S3 Paper Listing")
    print("=" * 60)
    results['s3_list'] = test_s3_list()
    
    # Test 5: S3 Processing (optional, takes time)
    print("\n" + "=" * 60)
    print("Test 5: S3 Paper Processing (Optional)")
    print("=" * 60)
    
    if results.get('s3_list'):
        user_input = input("Do you want to test paper processing? (y/n): ").lower()
        if user_input == 'y':
            results['s3_process'] = test_s3_processing()
        else:
            print_info("Skipping processing test")
            results['s3_process'] = None
    else:
        print_info("Skipping processing test (S3 list failed)")
        results['s3_process'] = None
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result is True else "❌ FAILED" if result is False else "⏭️  SKIPPED"
        print(f"{status}: {test_name}")
    
    print()
    print(f"Total: {len(results)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print()
    
    if failed > 0:
        print_warning("Some tests failed. Common issues:")
        print("  - Service not running: python paper_processor.py")
        print("  - Missing .env credentials")
        print("  - No papers in S3 bucket")
        print("  - Confluence access issues")
    else:
        print_success("All tests passed! 🎉")
        print()
        print_info("Next steps:")
        print("  1. Upload papers to S3:")
        print(f"     aws s3 cp papers/ s3://{os.getenv('S3_BUCKET_NAME')}/papers/ --recursive")
        print("  2. Import n8n workflow from n8n_workflow.json")
        print("  3. Configure n8n credentials (AWS S3, Slack)")
        print("  4. Run the workflow manually")
        print()
        print_info("Quick test command:")
        print(f"  curl http://localhost:8000/list-s3-papers")


if __name__ == "__main__":
    main()