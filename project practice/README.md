# Medical AI Assistant — 현장 문제 해결형 의료영상 AI PoC

> 목표: "탐지만 되는 AI"를 넘어, **근거까지 확인 가능한 의료영상 보조 도구** 만들기

의료 현장에서는 다음 3가지가 동시에 필요합니다.
- 놓침을 줄이는 **정량 검출**
- 결과를 이해할 수 있는 **정성 해석**
- 신뢰할 수 있는 **문헌 근거**

이 프로젝트는 이를 위해 **YOLO(검출) + VLM(해석) + RAG(근거)**를 한 흐름으로 통합했습니다.
또한 근거가 부족하면 추측하지 않고 "근거 없음"을 명확히 반환하도록 설계했습니다.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![YOLOv8](https://img.shields.io/badge/YOLOv8-ultralytics-red)
![Docker](https://img.shields.io/badge/Docker-CPU--only-blue)
![pytest](https://img.shields.io/badge/pytest-29_passed%2C_1_skipped-brightgreen)

내시경/X-ray 이미지에서 병변을 검출하고, 검출 결과를 기반으로 **논문·가이드라인 기반 의료 지식을 자동 제공**합니다. 위장 폴립(Kvasir-SEG)과 치과 질환(DENTEX) 두 도메인을 동일 아키텍처로 지원합니다.

---

## 🩺 현장 문제에서 출발한 이유

이 프로젝트는 "모델 성능 데모"가 아니라, 실제 임상에서 자주 생기는 아래 문제를 줄이기 위한 PoC입니다.

- **문제 1: 놓침(미검출) 리스크**
  - 대장내시경에서도 폴립/선종은 크기에 따라 놓칠 수 있습니다. 특히 작은 병변 miss rate가 높게 보고됩니다.
  - 참고: Van Rijn et al., tandem colonoscopy 메타분석 (PubMed)
    - https://pubmed.ncbi.nlm.nih.gov/16716777/
- **문제 2: 판독 업무 과부하와 인력 부담**
  - 영상량은 증가하는데 전문 인력은 부족해, 판독 품질과 일관성 유지가 어려워집니다.
  - 참고: Neiman Health Policy Institute radiologist shortage 전망
    - https://www.neimanhpi.org/press-releases/new-study-projects-radiologist-shortage-through-2055/
- **문제 3: "왜 이런 결과인지" 설명/근거 부족**
  - 의료 AI는 단순 "탐지됨"만으로는 신뢰를 얻기 어렵고, 근거 문헌/설명 가능성이 중요합니다.
  - 참고: WHO AI for Health guidance (투명성/설명가능성/거버넌스 강조)
    - https://www.who.int/publications/i/item/9789240029200

이 문제를 줄이기 위해, 이 프로젝트는 **검출(정량) + 해석(정성) + 문헌근거(RAG)**를 한 화면/한 API 흐름으로 통합했습니다.

---

## 👤 누가 쓰는가 (사용자 관점)

- **1차 사용자: 의료 AI 검토자/개발자**
  - 모델 결과를 숫자만 보지 않고, 근거 문서와 함께 검증 가능
- **2차 사용자: 데모 평가자(면접관, 협업 파트너)**
  - "기술 나열"이 아니라 "현장 문제 -> 해결 설계 -> 한계"를 빠르게 이해 가능
- **실사용 가정 사용자: 임상의 보조 도구 운영팀**
  - 단독 진단 도구가 아니라, 판독 보조/우선순위 확인/설명 보강 목적

---

## 🎯 프로젝트 목표 (명확한 성공 기준)

- **Goal A: 놓치지 않기(민감도 중심)**
  - 폴립 검출 Recall을 핵심 지표로 관리
- **Goal B: 근거 없는 답변 줄이기**
  - RAG 근거가 없으면 "모른다"를 명확히 반환
- **Goal C: 사용자가 이해하기 쉬운 결과**
  - 결과를 "검출 요약 / VLM 해석 / RAG 근거"로 분리해 표시

상세한 문제-해결 매핑은 `CLINICAL_PROBLEM_MAP.md` 문서에 정리했습니다.

---

## 🗣️ 1분 설명 (면접/소개용)

"의료 영상 AI는 보통 두 가지 한계가 있습니다. 첫째, 작은 병변을 놓칠 수 있고, 둘째, 결과에 대한 근거 설명이 부족합니다.
이 프로젝트는 그 문제를 줄이기 위해 검출(YOLO), 해석(VLM), 문헌근거(RAG)를 하나의 흐름으로 연결했습니다.
그리고 문서 근거가 없을 때는 추측하지 않고 `근거 없음`을 명확히 반환하도록 안전장치를 넣었습니다.
즉, 정확도 데모가 아니라 '현장에서 신뢰할 수 있는 보조 도구'를 목표로 만든 PoC입니다."

---

## ✅ 사용자 친화 설계 원칙

- **쉽게 읽히는 결과 구조**: `검출 요약 / VLM 해석 / RAG 근거`로 분리 표시
- **오해 방지 우선**: RAG 근거가 없으면 길게 설명하지 않고 "근거 없음" 명확히 표시
- **실무 의사결정 관점**: "어떤 병변이, 얼마나 신뢰도로, 어떤 문헌으로 뒷받침되는지"를 한 번에 제공
- **빠른 검증 흐름**: `/demo`에서 업로드 1회로 전 과정 확인 가능

---

## 🏗️ 아키텍처 진화 (V1 → V4)

| 버전 | 엔드포인트 | 기능 |
|------|-----------|------|
| **V1** | `POST /predict` | YOLOv8 병변 검출 + 세그멘테이션 (polyp / dental 선택) |
| **V2** | `POST /ask` | RAG 의료 지식 Q&A (ChromaDB + GPT-4o-mini + 출처 명시) |
| **V3** | `POST /analyze` | 검출 결과 → 자동 RAG 질의 생성 → 의료 지식 통합 응답 |
| **V4** | `POST /vlm-analyze` | VLM(LLaVA) 직접 해석 + YOLOv8 검출 + RAG 근거 보강 |
| **Demo** | `GET /demo` | 이미지 업로드 + 실시간 진행 상태 + JSON 결과 시각화 |

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
│   ├── test_api.py               # pytest 30개 (최근 실행: 29 passed, 1 skipped)
│   └── test_yolo.py              # YOLOv8 추론 테스트
├── ui/
│   └── demo.html                 # 데모 UI (/demo)
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

# PubMed 초록 자동 수집 + 인덱싱
python rag/auto_ingest.py --topics "colonoscopy polyp guideline" "endoscopic mucosal resection" --max-results 20

# 서버 실행
python -m uvicorn api.app:app --reload --port 8000
```

`.env` 파일 생성:
```
OPENAI_API_KEY=sk-...
PUBMED_EMAIL=your_email@example.com
# 선택: NCBI 요청 한도 상향(초당 10건)용
# PUBMED_API_KEY=...
MODEL_PATH=./best.pt
DENTAL_MODEL_PATH=./best_dentex.pt
RAG_LLM_PROVIDER=ollama
RAG_OLLAMA_MODEL=llama3.1:8b
RAG_OLLAMA_BASE_URL=http://localhost:11434
RAG_USE_OPENAI_FALLBACK=false
VLM_TIMEOUT_SECONDS=180
VLM_MAX_EDGE=960
VLM_JPEG_QUALITY=75
VLM_NUM_PREDICT=512
```

`OPENAI_API_KEY`는 선택입니다. 로컬 모드(`RAG_LLM_PROVIDER=ollama`)에서는 OpenAI 과금 없이 RAG를 사용할 수 있습니다.

`PUBMED_EMAIL`은 `rag/auto_ingest.py` 실행 시 필수입니다. 자동 수집 문서는 `rag/docs/auto/`에 저장되고, 기존 `rag/ingest.py`로 함께 인덱싱됩니다.

### Docker

```bash
docker compose up --build
```

서버 실행 후:
- Swagger: http://localhost:8000/docs
- Demo UI: http://localhost:8000/demo

---

## 📡 API

### `GET /health` — 서버 상태 확인

```bash
curl http://localhost:8000/health
```

RAG 상태는 별도 엔드포인트에서 확인:

```bash
curl http://localhost:8000/ask/health
```

`rag_llm_provider`가 `ollama`면 로컬 RAG 생성 모델이 사용 중입니다.

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

치과 파노라마 X-ray는 `model_type=dental` 권장:

```bash
curl -X POST http://localhost:8000/vlm-analyze \
  -F "file=@dental_xray.png" -F "conf=0.25" -F "model_type=dental"
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

> 응답 지연(특히 `/vlm-analyze`)이 길면 `VLM_TIMEOUT_SECONDS`, `VLM_MAX_EDGE`, `VLM_JPEG_QUALITY`, `VLM_NUM_PREDICT`로 튜닝 가능

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
| **Test** | pytest 30개 | 최근 실행: 29 passed, 1 skipped (OpenAI quota 영향) |

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

---

### 5. VLM 추론 타임아웃 (대용량 이미지)

**증상**: `/vlm-analyze`에서 `VLM 추론 타임아웃` 응답

**원인**: 고해상도 이미지 base64 전송 + LLaVA 생성 토큰 수가 크면 응답 지연 증가

**해결**:
- VLM 입력 전 이미지 자동 리사이즈/압축 적용 (`VLM_MAX_EDGE=960`, `VLM_JPEG_QUALITY=75`)
- VLM 클라이언트 타임아웃 환경변수화 (`VLM_TIMEOUT_SECONDS=180`)
- 생성 토큰 수 조정 (`VLM_NUM_PREDICT=512`)

이 설정으로 대용량 입력에서 타임아웃 발생률을 낮추고 응답 안정성을 개선했습니다.
