# Medical AI Assistant — Vision + LLM + VLM 통합 서비스

> 의료 영상 병변 검출(YOLOv8) + 의료 지식 RAG + VLM(LLaVA) End-to-End 파이프라인

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![YOLOv8](https://img.shields.io/badge/YOLOv8-ultralytics-red)
![Docker](https://img.shields.io/badge/Docker-CPU--only-blue)
![pytest](https://img.shields.io/badge/pytest-30개-brightgreen)

내시경/X-ray 이미지에서 병변을 검출하고, 검출 결과를 기반으로 **논문·가이드라인 기반 의료 지식을 자동 제공**하는 서비스입니다.  
위장 폴립(Kvasir-SEG)과 치과 질환(DENTEX) 두 도메인을 동일 아키텍처로 지원하며, 데이터 전처리부터 학습·서빙·배포까지 전체 MLOps 파이프라인을 구축했습니다.

---

## 🏗️ 아키텍처 진화 (V1 → V4)

| 버전 | 엔드포인트 | 기능 |
|------|-----------|------|
| **V1** | `POST /predict` | YOLOv8 병변 검출 + 세그멘테이션 (polyp / dental 선택) |
| **V2** | `POST /ask` | RAG 의료 지식 Q&A (ChromaDB + GPT-4o-mini + 출처 명시) |
| **V3** | `POST /analyze` | 검출 결과 → 자동 RAG 질의 생성 → 의료 지식 통합 응답 |
| **V4** | `POST /vlm-analyze` | VLM(LLaVA) 직접 해석 + YOLOv8 검출 + RAG 근거 보강 |

**V3 핵심 아이디어**: 검출된 클래스를 자동으로 RAG 질의로 변환 → 의사가 별도 검색 없이 관련 가이드라인 자동 수신  
**V4 확장**: VLM이 학습하지 않은 소견도 기술 가능 → 검출 모델의 한계를 보완

---

## 📊 성능 결과

### Kvasir-SEG (위장 폴립) — YOLOv8n-seg, 50 epochs, Colab T4

| Metric | Box Detection | Mask Segmentation |
|--------|:---:|:---:|
| Precision | 0.920 | 0.930 |
| Recall | 0.887 | **0.897** |
| mAP@50 | 0.939 | **0.942** |
| mAP@50-95 | 0.777 | 0.786 |

> Recall 89.7% = 폴립의 89.7%를 놓치지 않고 검출. 의료 AI에서 Recall은 가장 중요한 지표 (놓치면 조기 암 발견 실패).

### DENTEX (치과 X-ray, 4클래스) — YOLOv8n-seg, 100 epochs, Best: Epoch 83

| Metric | Box Detection | Mask Segmentation |
|--------|:---:|:---:|
| Precision | 0.485 | 0.485 |
| Recall | 0.334 | 0.334 |
| mAP@50 | 0.377 | 0.344 |
| mAP@50-95 | 0.242 | 0.225 |

### 성능 차이 원인

| 비교 항목 | Kvasir-SEG | DENTEX |
|-----------|:---:|:---:|
| 클래스 수 | 1개 (polyp) | 4개 (치과 질환) |
| 클래스당 학습 데이터 | 800장 | ~175장 |
| 영상 특성 | 컬러 내시경 (병변 뚜렷) | 흑백 X-ray (경계 불명확) |
| 병변 크기 | 비교적 큼 | 작고 다양 |

**개선 방향**: 데이터 증강 강화(contrast, 회전) / YOLOv8n → YOLOv8s 스케일업 / 해상도 640 → 1280 / Focal Loss로 클래스 불균형 대응

---

## 🔧 시스템 아키텍처

```
[의료 이미지 원본 데이터]
        │
        ▼
preprocessing/          # 바이너리 마스크 / COCO JSON → YOLO polygon 변환
        │
        ▼
training/               # Google Colab T4 GPU 학습
        │
        ▼
best.pt / best_dentex.pt (학습된 가중치, 각 6.5MB)
        │
        ▼
┌─────────────────────────────────────────────────┐
│  api/app.py — FastAPI 서버 (V4)                 │
│                                                 │
│  POST /predict      → YOLOv8 검출 (V1)         │
│  POST /ask          → RAG 의료 Q&A (V2)        │
│  POST /analyze      → Detection + RAG (V3)     │
│  POST /vlm-analyze  → VLM + Detection + RAG (V4)│
│                                                 │
│  rag/chain.py   → LangChain LCEL 파이프라인     │
│  rag/ingest.py  → PDF → ChromaDB 인덱싱        │
└─────────────────────────────────────────────────┘
        │
        ▼
Dockerfile + docker-compose.yml   # CPU-only 컨테이너 배포
```

---

## 📁 프로젝트 구조

```
project practice/
├── api/
│   └── app.py                    # FastAPI V4 서버 (4개 엔드포인트)
├── rag/
│   ├── chain.py                  # LangChain LCEL + ChromaDB + GPT-4o-mini
│   ├── ingest.py                 # PDF → 청킹 → BGE-M3 임베딩 → ChromaDB
│   ├── docs/                     # 의료 논문/가이드라인 PDF
│   └── vectorstore/              # ChromaDB 벡터스토어
├── preprocessing/
│   ├── prepare_dataset.py        # Kvasir-SEG 바이너리 마스크 → YOLO polygon
│   └── prepare_dataset_dentex.py # DENTEX COCO JSON → YOLO polygon (Colab용)
├── training/
│   ├── train_colab.py            # Kvasir-SEG Colab 학습 스크립트
│   ├── train_colab_dentex.py     # DENTEX Colab 학습 스크립트
│   └── results.csv               # 50 epoch 학습 로그
├── db/
│   ├── experiment_db.py          # SQLite 실험 결과 CRUD
│   └── experiments.db            # 실험 DB
├── tests/
│   ├── test_api.py               # pytest 30개 (V1~V4 전체)
│   └── test_yolo.py              # YOLOv8 추론 테스트
├── Dockerfile                    # CPU-only PyTorch + RAG
├── docker-compose.yml            # .env 연동 + 볼륨 마운트
├── requirements.txt              # Python 3.10+ 호환 의존성
└── .gitignore
```

---

## 🚀 Quick Start

### Local

```bash
pip install -r requirements.txt

# RAG 벡터스토어 구축 (최초 1회)
python rag/ingest.py

# 서버 실행
uvicorn api.app:app --reload --port 8000
```

`.env` 파일 생성:
```
OPENAI_API_KEY=sk-...
MODEL_PATH=./best.pt
DENTAL_MODEL_PATH=./best_dentex.pt
```

### Docker

```bash
docker compose up --build
```

서버 실행 후 → http://localhost:8000/docs (Swagger UI)

---

## 📡 API

### `GET /health` — 서버 상태 확인

```bash
curl http://localhost:8000/health
```

---

### `POST /predict` — V1: 병변 세그멘테이션

```bash
# 위장 폴립 검출 (기본값)
curl -X POST http://localhost:8000/predict \
  -F "file=@endoscopy.jpg" -F "conf=0.25"

# 치과 X-ray 검출
curl -X POST http://localhost:8000/predict \
  -F "file=@dental_xray.jpg" -F "conf=0.25" -F "model_type=dental"
```

---

### `POST /ask` — V2: 의료 지식 Q&A (RAG)

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "폴립 제거 후 주의사항은?"}'
```

---

### `POST /analyze` — V3: Detection + RAG 통합

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@endoscopy.jpg" -F "conf=0.25"
```

**Response:**
```json
{
  "model_type": "polyp",
  "detections": [
    {"class": "polyp", "confidence": 0.92, "bbox": [...], "polygon": [...]}
  ],
  "count": 1,
  "medical_knowledge": {
    "auto_question": "이 환자의 영상에서 위장 폴립이(가) 1건 검출되었습니다...",
    "answer": "폴립은 위장관 점막에서 돌출된 비정상적 조직으로...",
    "sources": [{"source_file": "polyp-guidelines-2024.pdf", "page": "3"}]
  }
}
```

---

### `POST /vlm-analyze` — V4: VLM + Detection + RAG 통합

```bash
curl -X POST http://localhost:8000/vlm-analyze \
  -F "file=@endoscopy.jpg" -F "conf=0.25" -F "model_type=polyp"
```

**Response:**
```json
{
  "model_type": "polyp",
  "vlm_analysis": {
    "interpretation": "내시경 영상에서 표면이 융기된 병변이 관찰됩니다...",
    "model": "llava",
    "duration_ms": 21437
  },
  "detections": [{"class": "polyp", "confidence": 0.92, "bbox": [...]}],
  "medical_evidence": {
    "query_source": "vlm",
    "answer": "해당 소견은 선종성 폴립 가능성이 있으며...",
    "sources": [{"source_file": "guideline.pdf", "page": "3"}]
  }
}
```

> V4는 ollama + LLaVA 로컬 실행 필요: `ollama pull llava`

---

## 🛠️ Tech Stack

| 분류 | 기술 | 선택 이유 |
|------|------|----------|
| **Model** | YOLOv8n-seg | 실시간 추론, CPU 배포 가능, 6.5MB 경량 |
| **Training** | Google Colab T4 | 무료 GPU (50 epochs ~48분) |
| **Serving** | FastAPI + Uvicorn | async 지원, 자동 Swagger 문서 |
| **RAG** | LangChain 0.3 LCEL | 최신 패턴, 체인 조립 유연 |
| **VLM** | ollama + LLaVA | 이미지 직접 해석, 로컬 실행 |
| **Embedding** | BAAI/bge-m3 | 한국어+영어 동시 지원 |
| **VectorDB** | ChromaDB | 경량, 로컬 실행, 별도 서버 불필요 |
| **LLM** | GPT-4o-mini | 저렴($0.15/1M tokens) + 충분한 품질 |
| **Database** | SQLite | 외부 의존성 없이 실험 결과 추적 |
| **Infra** | Docker (CPU-only) | GPU 없는 환경에서도 배포 가능 |
| **Test** | pytest 30개 | CI 환경 호환 (멀티모델 포함) |

---

## 📦 Dataset

### Kvasir-SEG
**[Kvasir-SEG](https://datasets.simula.no/kvasir-seg/)** — 위장 내시경 폴립 세그멘테이션 (SimulaMet)
- 1,000장 + 바이너리 마스크 / Train 800 · Val 200 (seed=42)
- 전처리: `contour 추출 → 폴리곤 좌표 → YOLO 정규화`

### DENTEX
**[DENTEX](https://huggingface.co/datasets/ibrahimhamamci/DENTEX)** — 치과 파노라마 X-ray 질환 검출
- 훈련 705장 + 검증 51장 (COCO format)
- 4클래스: Impacted(매복치) · Caries(충치) · Periapical Lesion(치근단병변) · Deep Caries(깊은충치)
- 전처리: `COCO JSON → polygon 정규화 → YOLO seg 포맷`

---

## 🔍 트러블슈팅

### 1. numpy 소스 빌드 실패 (Python 3.13 + Windows)

**증상**: `pip install` 시 numpy 빌드 에러
```
ERROR: Unknown compiler(s): [['icl'], ['cl'], ['cc'], ['gcc'], ...]
```
**원인**: `langchain==0.3.0`이 `numpy<2.0.0` 요구 → Python 3.13용 wheel 없음 → 소스 빌드 시도 → Windows C 컴파일러 없어서 실패  
**해결**: `langchain==0.3.27` + `ultralytics==8.4.14` (numpy 2.x 지원 버전)으로 업그레이드. 코드 변경 없음 (동일 0.3.x API).

---

### 2. OpenCV 한글 경로 문제

**증상**: `cv2.imread()`가 한글 경로에서 `None` 반환  
**원인**: OpenCV가 Windows 비ASCII 경로를 처리 못함  
**해결**:
```python
# ❌ 실패
img = cv2.imread("C:/경로/이미지.jpg")

# ✅ 성공
buf = np.fromfile("C:/경로/이미지.jpg", dtype=np.uint8)
img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
```

---

### 3. load_dotenv 환경변수 미적용 (uvicorn --reload)

**증상**: 서버 첫 시작 시 RAG 초기화 실패, 리로드하면 정상  
**원인**: `load_dotenv()`는 기존 환경변수를 덮어쓰지 않음. uvicorn `--reload` 프로세스 분기 시 환경변수 상태 불안정.  
**해결**: `override=True` 추가 + 각 모듈에서 독립적으로 `.env` 로드 (방어적 프로그래밍)
```python
load_dotenv(Path(__file__).parent.parent / ".env", override=True)
```

---

### 4. LangChain deprecated API

**해결 (최신 LCEL 패턴)**:
```python
# ❌ deprecated
from langchain.chains import RetrievalQA
chain = RetrievalQA.from_chain_type(...)

# ✅ 현재
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
qa_chain = create_stuff_documents_chain(llm, prompt)
chain = create_retrieval_chain(retriever, qa_chain)
```
```python
# ❌ deprecated
@app.on_event("startup")

# ✅ 현재
from contextlib import asynccontextmanager
@asynccontextmanager
async def lifespan(app: FastAPI): ...
```
