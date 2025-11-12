"""
Configuration settings for the Paper Processor
"""
import os
from dotenv import load_dotenv

load_dotenv(override=True)

# ===== AWS Configuration =====
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET = os.getenv("S3_BUCKET_NAME")
S3_PAPERS_PREFIX = os.getenv("S3_PAPERS_PREFIX", "papers/")

# ===== Docpamin Configuration =====
DOCPAMIN_API_KEY = os.getenv("DOCPAMIN_API_KEY")
DOCPAMIN_BASE_URL = os.getenv("DOCPAMIN_BASE_URL", "https://docpamin.superaip.samsungds.net/api/v1")
DOCPAMIN_CRT_FILE = os.getenv("DOCPAMIN_CRT_FILE", "/etc/ssl/certs/ca-certificates.crt")

# ===== LLM Configuration =====
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "30000"))

# ===== Confluence Configuration =====
CONFLUENCE_URL = os.getenv("CONFLUENCE_URL")
CONFLUENCE_EMAIL = os.getenv("CONFLUENCE_EMAIL")
CONFLUENCE_API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN")
CONFLUENCE_SPACE_KEY = os.getenv("CONFLUENCE_SPACE_KEY")

# ===== Processing Configuration =====
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "8"))
