# 🤖 AI Paper Newsletter - Docpamin + OpenAI LLM

회사 내부 Docpamin API와 OpenAI-compatible LLM을 사용하는 S3 기반 논문 분석 시스템입니다.

## 📋 주요 특징

- 📁 **S3 전용**: AWS S3에서 PDF 자동 수집
- 📄 **Docpamin API**: 내부 PDF 파싱 API 사용
- 🧠 **계층적 요약**: Section-Aware Hierarchical Summarization
- 🤖 **OpenAI LLM**: OpenAI-compatible API로 분석
- 📝 **Confluence**: 자동 페이지 생성
- 🔒 **내부 전용**: 외부 API 호출 없음

## 🎯 시스템 개요

### 워크플로우

```
1. S3 업로드 (PDF)
   ↓
2. n8n 수동 실행
   ↓
3. Docpamin API로 파싱
   - Task 생성
   - 상태 조회 (polling)
   - 결과 다운로드 (Markdown + JSON)
   ↓
4. LLM 계층적 분석
   - 섹션 추출 (Abstract, Introduction, Methods, etc.)
   - 각 섹션 개별 요약
   - 통합 최종 분석
   ↓
5. Confluence 업로드
   ↓
6. Slack 알림
```

## ✨ 핵심 기능

### 1. Section-Aware Hierarchical Summarization

긴 논문을 효과적으로 처리하는 3단계 방법:

```
Phase 1: 섹션 추출
├── Abstract
├── Introduction  
├── Methods
├── Results
├── Discussion
└── Conclusion

Phase 2: 섹션별 요약 (병렬)
├── 각 섹션을 독립적으로 요약
├── 섹션이 너무 길면 청크로 분할
└── 청크 요약 후 병합

Phase 3: 통합 분석
└── 모든 섹션 요약을 종합
    └── 최종 포괄적 분석 생성
```

**장점**:
- ✅ 정보 손실 최소화
- ✅ 논문 구조 활용으로 맥락 보존
- ✅ 토큰 제한 극복
- ✅ 병렬 처리 가능

### 2. Docpamin API 통합

3단계 API 호출:
1. **Task 생성**: PDF 업로드 및 파싱 요청
2. **Status 조회**: Polling으로 완료 대기
3. **결과 다운로드**: Markdown + JSON 형식

### 3. OpenAI Compatible LLM

표준 OpenAI API 형식 사용:
- `/chat/completions` 엔드포인트
- 모든 OpenAI-compatible 모델 지원

## 🚀 빠른 시작

### 1. 환경 설정

```bash
cp .env.example .env
nano .env
```

필수 설정:
```bash
# AWS S3
AWS_ACCESS_KEY_ID=...
S3_BUCKET_NAME=company-papers

# Docpamin API
DOCPAMIN_API_KEY=your_api_key
DOCPAMIN_BASE_URL=https://docpamin.superaip.samsungds.net/api/v1

# OpenAI Compatible LLM
LLM_API_KEY=your_llm_key
LLM_BASE_URL=https://your-llm-endpoint.com/v1
LLM_MODEL=gpt-4
LLM_MAX_TOKENS=4096

# Confluence
CONFLUENCE_URL=https://company.atlassian.net/wiki
CONFLUENCE_API_TOKEN=...
```

### 2. 실행

#### Docker (권장)

```bash
docker-compose up -d
```

#### 로컬

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m paper_processor
```

### 3. 사용

```bash
# 1. S3 업로드
aws s3 cp paper.pdf s3://company-papers/papers/

# 2. n8n 실행 (http://localhost:5678)

# 3. 결과 확인 (Confluence)
```

## 📚 API 엔드포인트

### GET /health
시스템 상태

```bash
curl http://localhost:8000/health
```

### GET /list-s3-papers
S3 논문 목록

```bash
curl "http://localhost:8000/list-s3-papers?bucket=company-papers"
```

### POST /process-s3-papers
논문 처리

```bash
curl -X POST http://localhost:8000/process-s3-papers \
  -H "Content-Type: application/json" \
  -d '{
    "bucket": "company-papers",
    "prefix": "papers/",
    "process_subdirectories": true
  }'
```

**API 문서**: http://localhost:8000/docs

## 🔧 고급 설정

### Docpamin 설정

```python
# 파싱 옵션 커스터마이징
workflow_options = {
    "workflow": "docling",
    "worker_options": {
        "docling_to_formats": ["md", "json"],
        "docling_image_export_mode": "embedded"
    }
}
```

### LLM 토큰 설정

```bash
# .env
LLM_MAX_TOKENS=8192  # 더 긴 문서용
```

### 섹션 패턴 커스터마이징

`paper_processor/summary.py`의 청크/섹션 감지 로직 수정:

```python
section_patterns = {
    "abstract": r"##?\s*(?:abstract|요약|초록)",
    "methods": r"##?\s*(?:methods?|methodology|방법론)",
    # ... 추가 패턴
}
```

## 🐛 문제 해결

### Docpamin API 오류

```bash
# API 키 확인
echo $DOCPAMIN_API_KEY

# 인증서 경로 확인
ls -l /etc/ssl/certs/ca-certificates.crt
```

### LLM API 오류

```bash
# 엔드포인트 테스트
curl -X POST $LLM_BASE_URL/chat/completions \
  -H "Authorization: Bearer $LLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4", "messages": [{"role": "user", "content": "test"}]}'
```

### 토큰 제한 오류

```bash
# .env에서 토큰 수 증가
LLM_MAX_TOKENS=8192
```

## 📊 처리 예시

### 일반 논문 (20페이지)

```
1. Docpamin 파싱: ~30초
2. 섹션 추출: <1초
3. LLM 분석:
   - Abstract: ~10초
   - Introduction: ~15초
   - Methods: ~20초
   - Results: ~15초
   - Discussion: ~15초
   - 통합 분석: ~20초
총 소요 시간: ~2.5분
```

### 긴 논문 (50페이지)

```
1. Docpamin 파싱: ~60초
2. 섹션 추출 및 청킹: ~2초
3. LLM 분석 (청크 처리):
   - 각 섹션 청크별 요약
   - 청크 병합
   - 최종 통합
총 소요 시간: ~5-7분
```

## 📁 파일 구조

```
ai-paper-newsletter/
├── paper_processor/       # Docpamin + LLM 통합 패키지
├── n8n_workflow.json      # 수동 실행 워크플로우
├── requirements.txt       # 간소화된 의존성
├── Dockerfile            # Docpamin 전용
├── .env.example          # 새 환경 변수
└── README.md             # 이 파일
```

## 🔒 보안

- ✅ 내부 Docpamin API만 사용
- ✅ 내부 LLM 엔드포인트
- ✅ S3 IAM 역할 기반 접근
- ✅ API 키 환경 변수 관리
- ✅ 인증서 검증 (CRT file)

## 💡 Best Practices

### 1. 배치 크기

```bash
# 한 번에 5-10개 논문 처리 권장
aws s3 sync ./batch/ s3://bucket/papers/ --exclude "*" --include "*.pdf" | head -10
```

### 2. 에러 핸들링

```python
# 개별 논문 실패 시 계속 진행
# errors 배열에 수집
```

### 3. 모니터링

```bash
# 실시간 로그
docker-compose logs -f paper-processor
```

## 📧 지원

- **빠른 시작**: QUICKSTART.md
- **변경사항**: CHANGELOG_S3_ONLY.md
- **배포**: DEPLOYMENT.md
- **테스트**: `python test_api.py`

---

**Docpamin API + OpenAI LLM + Hierarchical Summarization**