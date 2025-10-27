#!/usr/bin/env bash
# ============================================================
# v2.0 하이브리드 청킹 테스트 스크립트
# ============================================================

BASE_URL="http://localhost:7070"
MARKDOWN_FILE="paper_markdown.txt"
# MARKDOWN_FILE="small_test.txt"

echo "=============================================="
echo "🎯 v2.0 Hybrid Chunking Test"
echo "=============================================="

# Health check
echo "1️⃣  Health Check"
curl -sS "$BASE_URL/health" | jq .
echo
echo "----------------------------------------------"

# Version check
echo "2️⃣  Version Check"
curl -sS "$BASE_URL/" | jq '{version, improvements}'
echo
echo "----------------------------------------------"

# Markdown 로드
if [ ! -f "$MARKDOWN_FILE" ]; then
  echo "❌ File not found: $MARKDOWN_FILE"
  echo "💡 Specify file: $0 your_markdown.txt"
  exit 1
fi

if [ -n "$1" ] && [ -f "$1" ]; then
  MARKDOWN_FILE="$1"
fi

echo "3️⃣  Loading Markdown"
echo "File: $MARKDOWN_FILE"
MARKDOWN=$(cat "$MARKDOWN_FILE")
echo "Size: ${#MARKDOWN} chars"
echo
echo "----------------------------------------------"

# 청킹 방법 확인 (요약 없이 빠르게)
echo "4️⃣  Chunking Method Detection"
echo "⏱️  Testing..."

RESULT=$(curl -sS -X POST "$BASE_URL/debug/summarize-markdown" \
  -H "Content-Type: application/json" \
  -d "{
    \"markdown\": $(echo "$MARKDOWN" | jq -Rs .),
    \"include_section_summaries\": false,
    \"include_final_analysis\": false
  }")

echo "$RESULT" | jq '{
  chunking_method,
  num_chunks: (.chunks_detected | length),
  chunks: (.chunks_detected | .[0:5])
}'

METHOD=$(echo "$RESULT" | jq -r '.chunking_method')
NUM_CHUNKS=$(echo "$RESULT" | jq -r '.chunks_detected | length')

echo
if [ "$METHOD" == "header_based" ]; then
  echo "✅ HEADER-BASED chunking"
  echo "   → 논문에 충분한 헤더가 있습니다 (${NUM_CHUNKS}개)"
  echo "   → 각 섹션을 실제 헤더 이름으로 처리합니다"
elif [ "$METHOD" == "token_based" ]; then
  echo "✅ TOKEN-BASED chunking"
  echo "   → 논문에 헤더가 부족합니다 (${NUM_CHUNKS}개 청크로 균등 분할)"
  echo "   → 토큰 기반으로 균등하게 처리합니다"
else
  echo "❌ Unknown method: $METHOD"
fi

echo
echo "----------------------------------------------"

# 전체 청크 목록
echo "5️⃣  All Chunks"
echo "$RESULT" | jq -r '.chunks_detected[]' | nl

echo
echo "----------------------------------------------"

# 청크 요약 테스트 (옵션)
read -p "📝 Continue with chunk summarization? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
  echo "6️⃣  Chunk Summarization"
  echo "⏱️  This may take 1-2 minutes..."
  
  START_TIME=$(date +%s)
  
  SUMMARY_RESULT=$(curl -sS -X POST "$BASE_URL/debug/summarize-markdown" \
    -H "Content-Type: application/json" \
    -d "{
      \"markdown\": $(echo "$MARKDOWN" | jq -Rs .),
      \"include_section_summaries\": true,
      \"include_final_analysis\": false,
      \"use_hierarchical\": false,
      \"use_overlap\": true
    }" 2>&1)
  
  END_TIME=$(date +%s)
  DURATION=$((END_TIME - START_TIME))
  
  # 에러 체크
  if echo "$SUMMARY_RESULT" | jq -e . >/dev/null 2>&1; then
    echo
    echo "✅ Chunk summaries created"
    echo "⏱️  Duration: ${DURATION}s"
    echo
    
    # 각 청크 요약 미리보기
    echo "$SUMMARY_RESULT" | jq -r '.chunk_summaries | to_entries[] | "\n### \(.key)\n\(.value | .[0:200])...\n"'
  else
    echo
    echo "❌ Chunk summarization failed"
    echo "⏱️  Duration: ${DURATION}s"
    echo
    echo "Error response:"
    echo "$SUMMARY_RESULT" | head -n 20
    echo
    echo "💡 Check server logs: docker logs paper-processor --tail 50"
    echo
  fi
  
  echo
  echo "----------------------------------------------"
fi

# 계층적 요약 테스트 (옵션)
read -p "📊 Continue with hierarchical summarization? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
  echo "7️⃣  Hierarchical Summarization"
  echo "⏱️  This may take 3-5 minutes..."
  
  START_TIME=$(date +%s)
  
  HIER_RESULT=$(curl -sS -X POST "$BASE_URL/debug/summarize-markdown" \
    -H "Content-Type: application/json" \
    -d "{
      \"markdown\": $(echo "$MARKDOWN" | jq -Rs .),
      \"include_section_summaries\": true,
      \"include_final_analysis\": true,
      \"use_hierarchical\": true,
      \"use_overlap\": true,
      \"show_intermediate_steps\": true
    }" 2>&1)
  
  END_TIME=$(date +%s)
  DURATION=$((END_TIME - START_TIME))
  
  # 에러 체크
  if echo "$HIER_RESULT" | jq -e . >/dev/null 2>&1; then
    echo
    echo "✅ Hierarchical summary created"
    echo "⏱️  Duration: ${DURATION}s"
    echo
    
    # 중간 요약 확인
    echo "📊 Intermediate Summaries (Position-based):"
    echo "$HIER_RESULT" | jq '.intermediate_steps.intermediate_summaries | keys'
    
    # Beginning 요약
    echo
    echo "### Beginning (앞 1/3)"
    echo "$HIER_RESULT" | jq -r '.intermediate_steps.intermediate_summaries.beginning | .[0:300]'
    echo "..."
    
    # Middle 요약
    echo
    echo "### Middle (중간 1/3)"
    echo "$HIER_RESULT" | jq -r '.intermediate_steps.intermediate_summaries.middle | .[0:300]'
    echo "..."
    
    # End 요약
    echo
    echo "### End (뒤 1/3)"
    echo "$HIER_RESULT" | jq -r '.intermediate_steps.intermediate_summaries.end | .[0:300]'
    echo "..."
    
    # 최종 분석
    echo
    echo "📋 Final Analysis:"
    echo "$HIER_RESULT" | jq '.final_analysis | {
      title,
      relevance_score,
      tags,
      key_contributions: (.key_contributions | map(.[0:80]))
    }'
    
    # 전체 결과 저장
    echo "$HIER_RESULT" > v2_test_result.json
    echo
    echo "💾 Full result saved to: v2_test_result.json"
  else
    echo
    echo "❌ Hierarchical summarization failed"
    echo "⏱️  Duration: ${DURATION}s"
    echo
    echo "Error response:"
    echo "$HIER_RESULT" | head -n 20
    echo
    echo "💡 Check server logs: docker logs paper-processor --tail 50"
    echo
  fi
  
  echo
  echo "----------------------------------------------"
fi

echo
echo "✅ v2.0 Hybrid Chunking Test Complete!"
echo
echo "📊 Summary:"
echo "  - Chunking method: $METHOD"
echo "  - Number of chunks: $NUM_CHUNKS"
echo "  - Markdown size: ${#MARKDOWN} chars"
echo
echo "💡 Next steps:"
echo "  - Review chunk structure: cat v2_test_result.json | jq '.chunks_detected'"
echo "  - Review intermediate summaries: cat v2_test_result.json | jq '.intermediate_steps'"
echo "  - Compare with other papers to see different chunking strategies"