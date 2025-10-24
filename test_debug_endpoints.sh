#!/usr/bin/env bash
# ============================================================
# Test script for FastAPI Debug Endpoints (Docpamin / LLM)
# ------------------------------------------------------------
# This script tests:
#   1. /debug/parse-file
#   2. /debug/parse-s3
#   3. /debug/summarize-markdown
#   4. /debug/summarize-sections
# ============================================================

BASE_URL="http://localhost:7070"
S3_BUCKET="aip-llm"
S3_KEY="kai_papers/w43/2503.09516v5.pdf"
TEST_PDF="sample.pdf"

echo "=============================================="
echo "🧪 Testing AI Paper Debug Endpoints"
echo "Base URL: $BASE_URL"
echo "=============================================="
echo

# # 1️⃣  Docpamin parsing (local file)
# if [ -f "$TEST_PDF" ]; then
#   echo "📄 [1] Testing /debug/parse-file with local file: $TEST_PDF"
#   curl -sS -X POST "$BASE_URL/debug/parse-file" \
#     -F "file=@$TEST_PDF" \
#     -F "include_markdown=false" \
#     -F "markdown_max_chars=1000" | jq .
# else
#   echo "⚠️  [1] Skipped /debug/parse-file (no $TEST_PDF found)"
# fi

echo
echo "----------------------------------------------"
# 2️⃣  Docpamin parsing (single S3 file)
echo "☁️  [2] Testing /debug/parse-s3 for $S3_BUCKET/$S3_KEY"
curl -sS -X POST "$BASE_URL/debug/parse-s3" \
  -H "Content-Type: application/json" \
  -d "{
        \"bucket\": \"$S3_BUCKET\",
        \"key\": \"$S3_KEY\",
        \"include_markdown\": false,
        \"markdown_max_chars\": 1000
      }" | jq .
echo
echo "----------------------------------------------"

# # 3️⃣  Markdown summarization
# echo "🧠 [3] Testing /debug/summarize-markdown"
# MARKDOWN_EXAMPLE=$(cat <<'EOF'
# # Abstract
# This paper proposes a new agentic LLM framework for autonomous research.
# # Introduction
# Recent works such as DeepAgents and Chain-of-Agents show multi-agent reasoning can outperform static prompting.
# # Methods
# We use a reinforcement learning loop based on policy gradient and advantage estimation.
# # Results
# Experiments demonstrate higher efficiency on long-horizon reasoning benchmarks.
# # Conclusion
# We believe this approach opens new opportunities in reasoning-oriented model training.
# EOF
# )

# curl -sS -X POST "$BASE_URL/debug/summarize-markdown" \
#   -H "Content-Type: application/json" \
#   -d "{
#         \"title\": \"DeepAgents: Agentic Reasoning with LLMs\",
#         \"markdown\": \"$MARKDOWN_EXAMPLE\",
#         \"include_section_summaries\": true,
#         \"include_final_analysis\": true,
#         \"return_markdown_preview_chars\": 300
#       }" | jq .
# echo
# echo "----------------------------------------------"

# # 4️⃣  Section-level summarization
# echo "🧩 [4] Testing /debug/summarize-sections"
# curl -sS -X POST "$BASE_URL/debug/summarize-sections" \
#   -H "Content-Type: application/json" \
#   -d '{
#         "title": "Ablation Study of MoR",
#         "sections": {
#           "methods": "We combine multiple retrievers and weight them dynamically per query.",
#           "results": "The mixture retriever achieves state-of-the-art results on BEIR benchmark."
#         },
#         "only_sections": ["methods", "results"]
#       }' | jq .
# echo
# echo "✅ All debug endpoints tested."
