#!/usr/bin/env bash
# ============================================================
# 전체 워크플로우 테스트: PDF → 이미지 전처리 → 요약
# ============================================================

BASE_URL="http://localhost:7070"
S3_BUCKET="${1:-aip-llm}"
S3_KEY="${2:-kai_papers/w43/2503.09516v5.pdf}"

echo "=============================================="
echo "🎯 Full Workflow Test: PDF → Summary"
echo "=============================================="
echo
echo "S3 Path: s3://$S3_BUCKET/$S3_KEY"
echo

# Health check
echo "1️⃣  Health Check"
HEALTH=$(curl -sS "$BASE_URL/health")
echo "$HEALTH" | jq .

if ! echo "$HEALTH" | jq -e '.status == "healthy"' > /dev/null 2>&1; then
  echo "❌ Server not healthy!"
  exit 1
fi

echo
echo "----------------------------------------------"

# PDF 파싱 (이미지 자동 전처리 포함!)
echo
echo "2️⃣  PDF Parsing (with image preprocessing)"
echo "⏱️  This may take 30-60 seconds..."

START=$(date +%s)

cat > /tmp/parse_request.json << EOF
{
  "bucket": "$S3_BUCKET",
  "key": "$S3_KEY",
  "include_markdown": true
}
EOF

PARSE_RESULT=$(curl -sS -X POST "$BASE_URL/debug/parse-s3" \
  -H "Content-Type: application/json" \
  -d @/tmp/parse_request.json 2>&1)

END=$(date +%s)
PARSE_DURATION=$((END - START))

if echo "$PARSE_RESULT" | jq -e . >/dev/null 2>&1; then
  echo
  echo "✅ PDF parsed successfully"
  echo "⏱️  Duration: ${PARSE_DURATION}s"
  echo
  
  # Markdown 분석
  MD_SIZE=$(echo "$PARSE_RESULT" | jq -r '.markdown | length')
  HAS_BASE64=$(echo "$PARSE_RESULT" | jq -r '.markdown | contains("data:image")')
  HAS_FIGURES=$(echo "$PARSE_RESULT" | jq -r '.markdown | contains("[Figure")')
  
  echo "📄 Markdown Analysis:"
  echo "  Size: $(numfmt --to=iec $MD_SIZE) ($MD_SIZE chars)"
  
  if [ "$HAS_BASE64" == "true" ]; then
    echo "  ⚠️  Base64 images: YES (preprocessing failed!)"
  else
    echo "  ✅ Base64 images: NO (cleaned)"
  fi
  
  if [ "$HAS_FIGURES" == "true" ]; then
    FIGURE_COUNT=$(echo "$PARSE_RESULT" | jq -r '.markdown' | grep -o '\[Figure' | wc -l || echo "0")
    echo "  ✅ Figure placeholders: $FIGURE_COUNT"
  else
    echo "  ℹ️  Figure placeholders: NO"
  fi
  
  # 이미지 정보
  echo
  echo "🎨 Image Preprocessing:"
  
  IMAGES_INFO=$(echo "$PARSE_RESULT" | jq -r '.json_metadata.images_info')
  
  if [ "$IMAGES_INFO" != "null" ]; then
    TOTAL_IMAGES=$(echo "$IMAGES_INFO" | jq -r '.total_images // 0')
    echo "  Total images: $TOTAL_IMAGES"
    
    if [ "$TOTAL_IMAGES" -gt 0 ]; then
      echo
      echo "  🎯 Representative image:"
      echo "$IMAGES_INFO" | jq -r '.representative_images[0] | 
        "    Figure \(.index + 1): \(.size_kb)KB (\(.type))" +
        (if .alt != "" then "\n    Alt: \(.alt)" else "" end)'
    fi
  else
    echo "  ℹ️  No images_info metadata (no images or old version)"
  fi
  
  # Markdown 저장
  echo "$PARSE_RESULT" | jq -r '.markdown' > /tmp/parsed_markdown.txt
  echo
  echo "💾 Markdown saved to: /tmp/parsed_markdown.txt"
  
else
  echo
  echo "❌ PDF parsing failed"
  echo "⏱️  Duration: ${PARSE_DURATION}s"
  echo
  echo "Error response:"
  echo "$PARSE_RESULT" | head -n 30
  echo
  echo "💡 Check server logs: docker logs paper-processor --tail 100"
  exit 1
fi

echo
echo "----------------------------------------------"

# 청킹 테스트
echo
echo "3️⃣  Chunking Test"
echo "⏱️  Testing..."

cat > /tmp/chunk_request.json << EOF
{
  "markdown": $(cat /tmp/parsed_markdown.txt | jq -Rs .),
  "include_section_summaries": false,
  "include_final_analysis": false
}
EOF

CHUNK_RESULT=$(curl -sS -X POST "$BASE_URL/debug/summarize-markdown" \
  -H "Content-Type: application/json" \
  -d @/tmp/chunk_request.json)

if echo "$CHUNK_RESULT" | jq -e . >/dev/null 2>&1; then
  METHOD=$(echo "$CHUNK_RESULT" | jq -r '.chunking_method')
  NUM_CHUNKS=$(echo "$CHUNK_RESULT" | jq -r '.chunks_detected | length')
  
  echo
  echo "✅ Chunking succeeded"
  echo "  Method: $METHOD"
  echo "  Chunks: $NUM_CHUNKS"
  
  if [ "$NUM_CHUNKS" -le 10 ]; then
    echo "  ✅ Good chunk count!"
  elif [ "$NUM_CHUNKS" -le 20 ]; then
    echo "  ⚠️  Many chunks (may be slow)"
  else
    echo "  ⚠️  Too many chunks! ($NUM_CHUNKS)"
  fi
  
  echo
  echo "  First 5 chunks:"
  echo "$CHUNK_RESULT" | jq -r '.chunks_detected[0:5][]' | sed 's/^/    /'
else
  echo
  echo "❌ Chunking failed"
  echo "$CHUNK_RESULT" | head -n 20
fi

echo
echo "----------------------------------------------"

# 최종 분석 (옵션)
read -p "📋 Continue with final analysis? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
  echo "4️⃣  Final Analysis"
  echo "⏱️  This may take 2-5 minutes..."
  
  START=$(date +%s)
  
  cat > /tmp/final_request.json << EOF
{
  "markdown": $(cat /tmp/parsed_markdown.txt | jq -Rs .),
  "include_section_summaries": true,
  "include_final_analysis": true,
  "use_hierarchical": false,
  "use_overlap": true
}
EOF
  
  FINAL_RESULT=$(curl -sS -X POST "$BASE_URL/debug/summarize-markdown" \
    -H "Content-Type: application/json" \
    -d @/tmp/final_request.json 2>&1)
  
  END=$(date +%s)
  FINAL_DURATION=$((END - START))
  
  if echo "$FINAL_RESULT" | jq -e . >/dev/null 2>&1; then
    echo
    echo "✅ Final analysis succeeded"
    echo "⏱️  Duration: ${FINAL_DURATION}s"
    echo
    
    echo "📋 Analysis Results:"
    echo
    echo "$FINAL_RESULT" | jq '{
      title: .final_analysis.title,
      relevance_score: .final_analysis.relevance_score,
      tags: .final_analysis.tags,
      key_contributions: .final_analysis.key_contributions
    }'
    
    # 전체 결과 저장
    echo "$FINAL_RESULT" > full_workflow_result.json
    echo
    echo "💾 Full result saved to: full_workflow_result.json"
  else
    echo
    echo "❌ Final analysis failed"
    echo "⏱️  Duration: ${FINAL_DURATION}s"
    echo
    echo "Error:"
    echo "$FINAL_RESULT" | head -n 30
  fi
fi

echo
echo "=============================================="
echo "✅ Full Workflow Test Complete!"
echo "=============================================="

# Cleanup
rm -f /tmp/parse_request.json /tmp/chunk_request.json /tmp/final_request.json

echo
echo "📊 Summary:"
echo "  Parse duration: ${PARSE_DURATION}s"
if [[ $REPLY =~ ^[Yy]$ ]]; then
  echo "  Final analysis duration: ${FINAL_DURATION}s"
  echo "  Total time: $((PARSE_DURATION + FINAL_DURATION))s"
fi
echo
echo "💾 Outputs:"
echo "  - /tmp/parsed_markdown.txt (cleaned markdown)"
if [[ $REPLY =~ ^[Yy]$ ]]; then
  echo "  - full_workflow_result.json (complete analysis)"
fi
echo
echo "💡 Next steps:"
echo "  - Review markdown: less /tmp/parsed_markdown.txt"
echo "  - Check images: grep '\[Figure' /tmp/parsed_markdown.txt | head"
if [[ $REPLY =~ ^[Yy]$ ]]; then
  echo "  - View analysis: cat full_workflow_result.json | jq '.final_analysis'"
fi