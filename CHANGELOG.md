# CHANGELOG

이 프로젝트의 주요 변경 이력을 기록합니다.

---

## [V5] 2026-03-08 — LangGraph StateGraph 전환

### 변경 사항
- `rag/chain.py` 전체 재작성: LangChain LCEL `create_retrieval_chain` → LangGraph `StateGraph`
- 노드 4개: `retrieve`, `generate`, `format_sources`, `no_evidence`
- 조건부 엣지 2개:
  - `check_relevance` — 관련 문서 없음 → `no_evidence`로 라우팅 (환각 원천 차단)
  - `check_answer` — LLM 응답 이상 감지 → `no_evidence`로 라우팅
- 외부 인터페이스 동일 유지: `query()`, `query_sync()`, `is_ready` 변경 없음
- `requirements.txt`, `pyproject.toml`에 `langgraph>=0.2.0` 추가
- 테스트 37개 통과 확인

### 전환 이유
LCEL은 선형 파이프라인에 적합하지만 조건 분기가 if문으로 숨겨짐.  
LangGraph는 워크플로우를 그래프 구조로 명시적 표현 → 디버깅·추적 용이.

---

## [Refactor] 2026-03 — 프로덕션 수준 리팩토링

### 구조 개선
- `project practice/` 중첩 폴더 제거 → 루트로 플랫 구조 이동
- `api/app.py` 1,206줄 → 207줄로 축소, `api/routers/` 5개 라우터로 분리
  - `routers/predict.py` — YOLOv8 추론
  - `routers/ask.py` — RAG Q&A
  - `routers/analyze.py` — 통합 분석
  - `routers/vlm.py` — VLM 해석
  - `routers/monitoring.py` — 헬스체크 + Prometheus

### 신규 파일
- `pyproject.toml` — 표준 Python 패키징 + ruff + pytest 설정
- `Makefile` — `serve/test/lint/docker-build/ingest/export-onnx` 단축 명령
- `.env.example` — 환경변수 템플릿
- `config.yaml` — 학습/추론 하이퍼파라미터 중앙 관리
- `core/config.py` — 설정 로딩 모듈
- `scripts/export_onnx.py` — ONNX 모델 변환 스크립트
- `.github/workflows/ci.yml` — GitHub Actions CI (pytest + ruff)
- `.pre-commit-config.yaml` — 커밋 전 자동 린팅
- `MODEL_CARD.md` — 모델 한계 + 윤리 고려사항
- `docs/DENTEX_ANALYSIS.md` — DENTEX 성능 분석 + 개선 로드맵
- `CHANGELOG.md` — 이 파일

### 기술 개선
- CORS 미들웨어 추가
- `async_jobs` TTL 자동 정리 (메모리 누수 방지)
- structlog JSON 구조화 로깅
- `.gitignore` 강화 (`.sisyphus/`, `compare_csc/`, 내부용 docs 제외)

---

## [V4] Async + Monitoring

### 추가 기능
- `POST /vlm-analyze/async` — 비동기 VLM 작업 제출
- `GET /jobs/{job_id}` — 작업 상태 조회
- `GET /metrics` — Prometheus 메트릭 엔드포인트
- structlog 기반 JSON 구조화 로깅
- SQLite 실험 추적 DB (`db/experiment_db.py`)

---

## [V3] 검출 → RAG 자동 연동

### 추가 기능
- `POST /analyze` — 이미지 업로드 시 YOLOv8 검출 결과를 RAG 쿼리로 자동 변환
- 검출 클래스명 기반 자동 질의 생성 (`"polyp detected"` → RAG 검색)
- VLM(LLaVA) 정성 해석 통합

---

## [V2] RAG 파이프라인 + no-evidence 가드

### 추가 기능
- `POST /ask` — RAG 기반 의료 지식 Q&A
- ChromaDB + BGE-M3 벡터스토어 구축
- LangChain LCEL 파이프라인 (RetrievalQA deprecated → create_retrieval_chain)
- **no-evidence 가드**: relevance score 0.2 미만 청크 필터링 → 환각 차단
- 출처 파일명 + 페이지 번호 동시 반환
- PubMed 자동 수집 스크립트 (`rag/auto_ingest.py`) — 34개 논문 인덱싱

---

## [V1] 초기 YOLOv8 추론 API

### 추가 기능
- `POST /predict` — YOLOv8n-seg 인스턴스 세그멘테이션 (polyp/dental 모델 선택)
- `GET /health` — 서버 상태 + 로드된 모델 목록
- 멀티모델 지원: `best.pt` (Kvasir-SEG), `best_dentex.pt` (DENTEX)
- Docker + docker-compose 환경 구성

### 학습 결과
- Kvasir-SEG: mAP@50(Mask) = **0.942** (50 epochs, Colab T4)
- DENTEX: mAP@50(Mask) = **0.344** (100 epochs, 4-class, 700장)
