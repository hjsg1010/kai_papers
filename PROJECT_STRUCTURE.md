# 📁 프로젝트 구조

AI Paper Newsletter의 전체 파일 구조와 각 파일의 역할을 설명합니다.

## 디렉토리 구조

```
ai-paper-newsletter/
├── 📄 README.md                    # 프로젝트 메인 문서
├── 📄 API_RESEARCH.md             # 논문 API 조사 문서
├── 📄 DEPLOYMENT.md               # 배포 가이드
├── 📄 PROJECT_STRUCTURE.md        # 이 파일
│
├── 📁 paper_processor/            # FastAPI 메인 서비스 패키지
├── 🐍 arxiv_example.py            # arXiv API 사용 예제
├── 🐍 test_api.py                 # API 테스트 스크립트
│
├── ⚙️ .env.example                # 환경 변수 템플릿
├── ⚙️ .gitignore                  # Git 무시 파일
├── 📦 requirements.txt            # Python 의존성
├── 🐳 Dockerfile                  # Docker 이미지 빌드
├── 🐳 docker-compose.yml          # Docker Compose 설정
│
├── 🔧 setup.sh                    # 초기 설정 스크립트
└── 🔄 n8n_workflow.json           # n8n 워크플로우 정의
```

## 파일 설명

### 📘 문서 파일

#### README.md
- **역할**: 프로젝트의 메인 문서
- **내용**: 
  - 시스템 개요 및 아키텍처
  - 설치 방법 (Docker / 로컬)
  - 사용법 및 API 문서
  - 논문 소스 API 설명
  - 트러블슈팅 가이드

#### API_RESEARCH.md
- **역할**: 논문 수집 API 조사 결과
- **내용**:
  - arXiv API (Primary)
  - Semantic Scholar API
  - Papers with Code API
  - OpenAlex API
  - 각 API의 장단점 비교

#### DEPLOYMENT.md
- **역할**: 프로덕션 배포 가이드
- **내용**:
  - AWS EC2 배포 방법
  - Docker 프로덕션 설정
  - 보안 설정 (SSL, 방화벽)
  - 모니터링 설정 (CloudWatch, Grafana)
  - 백업 및 복구 절차

### 🐍 Python 파일

#### paper_processor/ 패키지
- **역할**: FastAPI 기반 메인 서비스 및 유틸리티 모듈 모음
- **구성**:
  - `api.py`: FastAPI 엔드포인트 정의
  - `docpamin.py`, `images.py`, `summary.py` 등: 기능별 유틸리티 분리
  - `__main__.py`: `python -m paper_processor` 로 앱 실행 지원
- **주요 기능**:
  - AWS S3에서 논문 가져오기 및 Docpamin 파싱
  - LLM 기반 요약/분석 및 계층적 리포트 생성
  - 대표 이미지 선별 및 Markdown/Confluence 업로드 지원
- **엔드포인트**:
  - `GET /health` - 헬스 체크
  - `GET /list-s3-papers` - S3 논문 목록 조회
  - `POST /process-s3-papers` - S3 또는 paper_list 기반 논문 처리
  - `POST /batch-process` - 일괄 처리 및 Confluence 업로드
  - `POST /debug/*` - 디버그용 파싱/요약 API

#### arxiv_example.py
- **역할**: arXiv API 사용 예제 스크립트
- **기능**:
  - 키워드 기반 논문 검색
  - 논문 정보 출력
  - PDF 다운로드
  - JSON 형식으로 결과 저장
- **실행**: `python arxiv_example.py`

#### test_api.py
- **역할**: API 엔드포인트 테스트
- **테스트 항목**:
  - 헬스 체크
  - API 문서 접근
  - Confluence 연결
  - arXiv 논문 처리
  - S3 논문 처리
- **실행**: `python test_api.py`

### ⚙️ 설정 파일

#### .env.example
- **역할**: 환경 변수 템플릿
- **포함 내용**:
  - AWS 자격증명 및 S3 설정
  - Anthropic API 키
  - Confluence 설정
  - Slack 설정 (선택)
- **사용법**: `.env`로 복사 후 실제 값 입력

#### requirements.txt
- **역할**: Python 의존성 패키지 목록
- **주요 패키지**:
  - `fastapi` - 웹 프레임워크
  - `boto3` - AWS SDK
  - `docling` - PDF 파싱
  - `anthropic` - Claude API
  - `atlassian-python-api` - Confluence API

#### .gitignore
- **역할**: Git에서 무시할 파일 지정
- **무시 대상**:
  - Python 캐시 및 빌드 파일
  - 환경 변수 파일 (.env)
  - 로그 파일
  - IDE 설정 파일

### 🐳 Docker 파일

#### Dockerfile
- **역할**: Docker 이미지 빌드 정의
- **포함 내용**:
  - Python 3.11 slim 기반
  - 시스템 의존성 (poppler-utils)
  - Python 패키지 설치
  - FastAPI 서비스 실행

#### docker-compose.yml
- **역할**: 멀티 컨테이너 Docker 애플리케이션 정의
- **서비스**:
  - `paper-processor` - Python FastAPI 서비스
  - `n8n` - 워크플로우 자동화
- **네트워크**: 두 서비스 간 통신 설정
- **볼륨**: 데이터 영속성 보장

### 🔧 스크립트 파일

#### setup.sh
- **역할**: 자동 설치 스크립트
- **기능**:
  - 환경 확인 (Python, Docker)
  - 설치 방법 선택 (Docker / 로컬)
  - 의존성 설치
  - S3 버킷 생성
  - 초기 테스트
- **실행**: `bash setup.sh`

### 🔄 워크플로우 파일

#### n8n_workflow.json
- **역할**: n8n 워크플로우 정의 (JSON)
- **트리거**:
  - Schedule Trigger - 매주 월요일 오전 9시
  - Manual Trigger - 수동 실행
- **노드**:
  - HTTP Request - API 호출
  - Condition - 성공/실패 분기
  - Slack - 알림 발송
  - S3 Upload - 파일 업로드
- **사용법**: n8n UI에서 Import

## 실행 순서

### 1. 초기 설정
```bash
# 1. 프로젝트 클론
git clone <repository-url>
cd ai-paper-newsletter

# 2. 환경 변수 설정
cp .env.example .env
nano .env  # 실제 값 입력

# 3. 자동 설치 (권장)
bash setup.sh

# 또는 수동 설치
pip install -r requirements.txt
```

### 2. 서비스 시작

#### Docker 사용
```bash
docker-compose up -d
docker-compose logs -f
```

#### 로컬 실행
```bash
# Terminal 1: Paper Processor
python -m paper_processor

# Terminal 2: n8n
n8n start
```

### 3. 테스트
```bash
# API 테스트
python test_api.py

# arXiv 예제 실행
python arxiv_example.py
```

### 4. n8n 워크플로우 설정
1. http://localhost:5678 접속
2. Workflows → Import from File
3. `n8n_workflow.json` 선택
4. 크레덴셜 설정 (AWS S3, Slack)

## 개발 가이드

### 새로운 기능 추가

1. **새로운 논문 소스 추가**
   - `paper_processor.py`에 새 함수 추가
   - 예: `def search_semantic_scholar(...)`
   - API 엔드포인트 추가
   - n8n 워크플로우에 노드 추가

2. **분석 방식 변경**
   - `analyze_paper_with_llm()` 함수 수정
   - 프롬프트 커스터마이징
   - 새로운 분석 카테고리 추가

3. **알림 채널 추가**
   - n8n 워크플로우에 새 노드 추가
   - 예: Email, Microsoft Teams, Discord

### 디버깅

```bash
# 로그 확인
docker-compose logs -f paper-processor

# 특정 함수 테스트
python -c "from paper_processor import search_arxiv_papers; print(search_arxiv_papers(['LLM'], 5, 7, ['cs.AI']))"

# API 직접 호출
curl -X POST http://localhost:8000/health
```

## 프로덕션 체크리스트

배포 전 확인 사항:

- [ ] `.env` 파일에 모든 필수 값 입력
- [ ] AWS S3 버킷 생성 및 권한 설정
- [ ] Confluence API 토큰 발급
- [ ] Claude API 키 발급
- [ ] Docker 이미지 빌드 성공
- [ ] 헬스 체크 통과
- [ ] 테스트 실행 성공
- [ ] n8n 워크플로우 import 완료
- [ ] SSL 인증서 설정 (프로덕션)
- [ ] 백업 스크립트 설정

## 도움이 필요하신가요?

- **README.md** - 전체 시스템 개요 및 사용법
- **API_RESEARCH.md** - 논문 API 상세 정보
- **DEPLOYMENT.md** - 프로덕션 배포 가이드
- **test_api.py** - API 테스트 및 검증
- **arxiv_example.py** - arXiv API 사용 예제

각 파일에 상세한 설명과 예제가 포함되어 있습니다!