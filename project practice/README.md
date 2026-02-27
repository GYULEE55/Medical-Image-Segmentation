# Medical AI Assistant

의료 이미지에서 병변을 검출하고, 관련 의료 지식을 함께 제공하는 FastAPI 서비스입니다.

## 핵심 기능
- `POST /predict`: YOLOv8 기반 병변 검출/세그멘테이션 (`polyp`, `dental`)
- `POST /ask`: RAG 기반 의료 Q&A (근거 문서 포함)
- `POST /analyze`: 검출 + 의료 지식 통합 응답
- `POST /vlm-analyze`: VLM + 검출 + RAG 통합 (선택 기능)
- `GET /health`: 모델/RAG/VLM 로딩 상태 확인

## 성능 요약
- Kvasir-SEG (polyp): Mask mAP@50 `0.942`
- DENTEX (dental): Mask mAP@50 `0.344`

## 프로젝트 구조
```text
project practice/
├── api/app.py
├── rag/chain.py
├── rag/ingest.py
├── preprocessing/
├── training/
├── tests/test_api.py
├── Dockerfile
└── docker-compose.yml
```

## macOS 실행 가이드

### 1) 사전 준비
- Python 3.10+
- Git
- Docker Desktop (선택)
- OpenAI API Key

### 2) 프로젝트 진입
```bash
cd "project practice"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3) 환경변수 설정
`.env` 파일 생성:
```env
OPENAI_API_KEY=sk-...
MODEL_PATH=./best.pt
DENTAL_MODEL_PATH=./best_dentex.pt
```

### 4) RAG 인덱스 생성 (최초 1회)
```bash
python rag/ingest.py
```

### 5) 서버 실행
```bash
uvicorn api.app:app --reload --port 8000
```

### 6) 브라우저 확인
- Swagger: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

정상 기준:
- `/docs` 페이지 로딩
- `/health`에 `status: ok`
- `v1_yolo.available_models`에 최소 `polyp` 표시

## API 빠른 테스트

### Predict
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@bus.jpg" \
  -F "conf=0.25" \
  -F "model_type=polyp"
```

### Ask
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"폴립 제거 후 주의사항은?"}'
```

### Analyze
```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@bus.jpg" \
  -F "conf=0.25"
```

## Docker 실행
```bash
docker compose up --build
```

## 주의사항
- `best_dentex.pt`가 없으면 dental 모델은 health에 표시되지 않을 수 있습니다.
- `rag/vectorstore`가 없으면 `/ask`, `/analyze`에서 RAG 기능이 비활성화됩니다.
- VLM은 `ollama`와 모델 설치가 필요합니다. (없으면 `/vlm-analyze` 비활성)
