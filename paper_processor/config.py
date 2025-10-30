"""Configuration and shared resources for the paper processor service."""
from __future__ import annotations

import logging
import os

import boto3
from botocore.config import Config as BotoConfig
from dotenv import load_dotenv

# Load environment variables as early as possible so other modules can rely on them
load_dotenv(override=True)

# Configure a module-level logger that other modules can reuse
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("paper_processor")

# --- AWS / S3 -----------------------------------------------------------------
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET = os.getenv("S3_BUCKET_NAME")
S3_PAPERS_PREFIX = os.getenv("S3_PAPERS_PREFIX", "papers/")

# --- Docpamin ------------------------------------------------------------------
DOCPAMIN_API_KEY = os.getenv("DOCPAMIN_API_KEY")
DOCPAMIN_BASE_URL = os.getenv("DOCPAMIN_BASE_URL", "https://docpamin.superaip.samsungds.net/api/v1")
DOCPAMIN_CRT_FILE = os.getenv("DOCPAMIN_CRT_FILE", "/etc/ssl/certs/ca-certificates.crt")

# --- LLM -----------------------------------------------------------------------
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "30000"))

# --- Confluence ----------------------------------------------------------------
CONFLUENCE_URL = os.getenv("CONFLUENCE_URL")
CONFLUENCE_EMAIL = os.getenv("CONFLUENCE_EMAIL")
CONFLUENCE_API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN")
CONFLUENCE_SPACE_KEY = os.getenv("CONFLUENCE_SPACE_KEY")

# --- Runtime behaviour ---------------------------------------------------------
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "8"))

boto_config = BotoConfig(
    retries={"max_attempts": 5, "mode": "standard"},
    connect_timeout=10,
    read_timeout=60,
)

s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    config=boto_config,
)

__all__ = [
    "AWS_ACCESS_KEY",
    "AWS_SECRET_KEY",
    "S3_BUCKET",
    "S3_PAPERS_PREFIX",
    "DOCPAMIN_API_KEY",
    "DOCPAMIN_BASE_URL",
    "DOCPAMIN_CRT_FILE",
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "LLM_MAX_TOKENS",
    "CONFLUENCE_URL",
    "CONFLUENCE_EMAIL",
    "CONFLUENCE_API_TOKEN",
    "CONFLUENCE_SPACE_KEY",
    "MAX_WORKERS",
    "logger",
    "s3_client",
]
