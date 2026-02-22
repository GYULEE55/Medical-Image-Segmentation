# Medical AI Assistant — Vision + LLM 통합 서비스

> 의료 영상 병변 검출(YOLOv8) + 의료 지식 RAG(LangChain/GPT-4o-mini) 통합 파이프라인

의료 영상에서 병변을 검출하고, 검출 결과를 기반으로 **논문/가이드라인 기반 의료 지식을 자동 제공**하는 End-to-End 의료 AI 서비스입니다.

**멀티모델 지원**: 위장 내시경(폴립) + 치과 X-ray(4개 치과 질환) — `model_type` 파라미터로 선택

---

## 핵심 기능: Vision + LLM 통합 (`POST /analyze`)

```
의료 이미지 업로드 (내시경 / X-ray)
        │
   [YOLOv8-seg] 병변 검출 + 세그멘테이션  ← model_type: polyp / dental
        │
        ├─ 검출 결과: 위치, 신뢰도, 마스크 좌표
        │
        └─ 자동 RAG 질의 생성
               │
          [ChromaDB + GPT-4o-mini]
               │
               └─ 의료 지식: 임상적 의미, 추적 관찰 주기, 환자 안내사항 + 출처(논문/가이드라인)
```

**왜 이 구조인가**: 검출 모델만 있으면 "폴립 있음"에서 끝남. 실제 임상에서는 **"이 폴립이 무엇이고, 어떻게 관리해야 하는지"** 가 필요함. Vision이 발견하고, LLM이 설명하는 구조.

**왜 멀티모델인가**: 하나의 도메인(폴립)만으로는 범용성이 부족. **동일 아키텍처를 다른 의료 도메인(치과)에 적용**하여 확장 가능성을 증명.

---

## Results

### Model 1: Kvasir-SEG (위장 폴립)

| Metric | Box Detection | Mask Segmentation |
|--------|:---:|:---:|
| **Precision** | 0.920 | 0.930 |
| **Recall** | 0.887 | **0.897** |
| **mAP@50** | 0.939 | **0.942** |
| **mAP@50-95** | 0.777 | 0.786 |

- **Model**: YOLOv8n-seg (nano, 6.5MB)
- **Training**: 50 epochs, batch 16, Google Colab T4 GPU
- **Dataset**: Kvasir-SEG 1,000장 (Train 800 / Val 200)

### Model 2: DENTEX (치과 X-ray)

| Metric | Box Detection | Mask Segmentation |
|--------|:---:|:---:|
| **Precision** | 0.485 | 0.485 |
| **Recall** | 0.334 | 0.334 |
| **mAP@50** | 0.377 | 0.344 |
| **mAP@50-95** | 0.242 | 0.225 |

- **Model**: YOLOv8n-seg (nano, 6.5MB)
- **Training**: 100 epochs, cos_lr, patience=15, Google Colab T4 GPU (Best: Epoch 83)
- **Dataset**: DENTEX ~700장, 4개 클래스 (Impacted, Caries, Periapical Lesion, Deep Caries)

### Kvasir-SEG vs DENTEX — 왜 성능 차이가 나는가?

| 비교 항목 | Kvasir-SEG (mAP50: 0.94) | DENTEX (mAP50: 0.38) |
|-----------|:---:|:---:|
| 클래스 수 | 1개 (polyp) | 4개 (치과 질환) |
| 학습 데이터 | 800장 (전부 1클래스) | ~700장 (클래스당 ~175장) |
| 영상 특성 | 컬러 내시경 (병변이 뚜렷) | 흑백 X-ray (병변 경계 불명확) |
| 병변 크기 | 비교적 큼 | 작고 다양 |

**개선 방향** (면접 포인트):
1. **데이터 증강 강화**: 치과 X-ray에 특화된 augmentation (contrast 조절, 회전)
2. **모델 스케일업**: YOLOv8n → YOLOv8s/m (파라미터 증가)
3. **클래스 불균형 대응**: Focal Loss, 클래스별 가중치 조절
4. **해상도 증가**: imgsz 640 → 1280 (X-ray는 고해상도에서 미세 병변 검출 유리)

### 의료 AI 관점 메트릭 해석

| 메트릭 | 값 | 의료 AI에서의 의미 |
|--------|-----|-------------------|
| **Recall 0.897** | 89.7% | 폴립의 89.7%를 놓치지 않고 검출 — **의료에서 가장 중요한 지표** (놓치면 조기 암 발견 실패) |
| **Precision 0.930** | 93.0% | 검출한 것 중 93%가 실제 폴립 — 불필요한 조직검사 최소화 |
| **mAP@50 0.942** | 94.2% | IoU 50% 기준 전반적 성능 — 임상 활용 가능 수준 |

---

## 모델 선택 근거: 왜 YOLOv8인가?

| 고려 사항 | YOLOv8-seg | Mask R-CNN | SAM | U-Net |
|-----------|:---:|:---:|:---:|:---:|
| **실시간 추론** | O (one-stage) | X (two-stage) | X (heavy) | X (seg only) |
| **Detection + Segmentation 동시** | O | O | X (프롬프트 필요) | X (seg only) |
| **경량 모델 (CPU 배포)** | O (6.5MB nano) | X (~170MB) | X (~2.4GB) | 가능 |
| **학습 데이터 1000장으로 충분** | O (transfer learning) | 가능 | 불필요 (zero-shot) | 가능 |

**선택 이유**:
1. **실시간 추론**: 내시경은 실시간 영상 — 빠른 추론이 필수 (YOLOv8 nano: ~5ms/frame GPU)
2. **CPU 배포 가능**: 병원에 GPU 서버가 없을 수 있음 — 6.5MB 모델은 CPU에서도 동작
3. **Detection + Segmentation 동시**: 폴립 위치(bbox)와 모양(mask)을 한 번에 출력
4. **적은 데이터에도 강건**: ImageNet 사전학습 + COCO 사전학습으로 1000장 Transfer Learning 충분

### Data Augmentation 전략

YOLOv8은 학습 시 **자동으로 아래 augmentation을 적용**합니다:

| 기법 | 설명 | 의료 영상 효과 |
|------|------|---------------|
| **Mosaic** | 4장을 합쳐 1장 생성 | 다양한 배경/크기 조합 학습 |
| **HSV 변환** | 색상/채도/명도 랜덤 조절 | 내시경 조명 변화 대응 |
| **Flip (좌우)** | 좌우 반전 | 폴립 위치 일반화 |
| **Scale** | 0.5~1.5x 크기 변환 | 다양한 폴립 크기 대응 |

추가로 `patience=10` EarlyStopping으로 **50 epoch 중 과적합 발생 시 자동 중단**.
`results.csv`에서 val_loss가 안정적으로 수렴한 것을 확인 → 과적합 없음.

---

## Architecture

```
[의료 이미지 원본 데이터]
        │
        ▼
preprocessing/prepare_dataset.py     # 바이너리 마스크 → YOLO polygon 변환
        │
        ▼
training/train_colab.py              # Colab T4 GPU에서 50 epochs 학습
        │
        ▼
best.pt (학습된 가중치, 6.5MB)
        │
        ▼
┌───────────────────────────────────────────────┐
│  api/app.py — FastAPI 서버 (V3)               │
│                                               │
│  POST /predict  → YOLOv8 검출 (V1)           │
│  POST /ask      → RAG 의료 Q&A (V2)         │
│  POST /analyze  → Vision + LLM 통합 (V3)    │  ← 핵심
│                                               │
│  rag/chain.py   → LangChain LCEL 파이프라인   │
│  rag/ingest.py  → PDF → ChromaDB 인덱싱      │
└───────────────────────────────────────────────┘
        │
        ▼
Dockerfile + docker-compose.yml      # 컨테이너화 배포
```

---

## Project Structure

```
├── api/                        # API 서버
│   └── app.py                  #   V1(predict) + V2(ask) + V3(analyze)
│
├── rag/                        # RAG 파이프라인
│   ├── chain.py                #   LangChain LCEL + ChromaDB + GPT-4o-mini
│   ├── ingest.py               #   PDF → 청킹 → BGE-M3 임베딩 → ChromaDB
│   ├── docs/                   #   의료 논문/가이드라인 PDF (5개)
│   └── vectorstore/            #   ChromaDB 벡터스토어
│
├── preprocessing/              # 데이터 전처리
│   ├── prepare_dataset.py      #   Kvasir-SEG 바이너리 마스크 → YOLO polygon
│   └── prepare_dataset_dentex.py #  DENTEX COCO JSON → YOLO polygon (Colab용)
│
├── training/                   # 모델 학습
│   ├── train_colab.py          #   Kvasir-SEG Colab 학습 스크립트
│   ├── train_colab_dentex.py   #   DENTEX Colab 학습 스크립트
│   └── results.csv             #   50 epoch 학습 로그
│
├── db/                         # 실험 관리
│   ├── experiment_db.py        #   SQLite 실험 결과 CRUD
│   └── experiments.db          #   실험 DB (Kvasir-SEG 결과 저장됨)
│
├── tests/                      # 테스트
│   ├── test_api.py             #   pytest 25개 (V1+V2+V3+멀티모델)
│   └── test_yolo.py            #   YOLOv8 추론 테스트
│
├── Dockerfile                  # Docker (CPU-only PyTorch + RAG)
├── docker-compose.yml          # .env 연동 + 벡터스토어 볼륨 마운트
├── requirements.txt            # Python 3.13 호환 의존성
└── .gitignore                  # .env / 데이터 / 모델 제외
```

---

## Quick Start

### Local

```bash
pip install -r requirements.txt

# RAG 벡터스토어 구축 (최초 1회)
python rag/ingest.py

# 서버 실행
uvicorn api.app:app --reload --port 8000
```

### Docker

```bash
docker compose up --build
```

서버 실행 후 http://localhost:8000/docs 에서 Swagger UI로 테스트 가능.

---

## API

### `GET /health` — 서버 상태 확인

### `POST /predict` — V1: 병변 세그멘테이션 (멀티모델)

```bash
# 위장 폴립 검출 (기본값)
curl -X POST http://localhost:8000/predict \
  -F "file=@endoscopy.jpg" -F "conf=0.25"

# 치과 X-ray 검출
curl -X POST http://localhost:8000/predict \
  -F "file=@dental_xray.jpg" -F "conf=0.25" -F "model_type=dental"
```

### `POST /ask` — V2: 의료 지식 Q&A (RAG)

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "폴립 제거 후 주의사항은?"}'
```

### `POST /analyze` — V3: Vision + LLM 통합 분석 (멀티모델)

```bash
# 내시경 이미지 분석 (폴립)
curl -X POST http://localhost:8000/analyze \
  -F "file=@endoscopy.jpg" -F "conf=0.25"

# 치과 X-ray 분석
curl -X POST http://localhost:8000/analyze \
  -F "file=@dental_xray.jpg" -F "conf=0.25" -F "model_type=dental"
```

**Response (V3):**
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
    "sources": [
      {"source_file": "polyp-guidelines-2024.pdf", "page": "3"}
    ]
  }
}
```

---

## Dataset

### Kvasir-SEG (위장 폴립)

**[Kvasir-SEG](https://datasets.simula.no/kvasir-seg/)** — 위장 내시경 폴립 세그멘테이션 데이터셋 (SimulaMet)

- 이미지 1,000장 + 바이너리 마스크 (1 클래스: polyp)
- 전처리: `contour 추출 → 폴리곤 좌표 → YOLO 정규화 포맷`
- Train/Val: 800/200 (8:2 랜덤 분할, seed=42 재현 가능)

### DENTEX (치과 X-ray)

**[DENTEX](https://huggingface.co/datasets/ibrahimhamamci/DENTEX)** — 치과 파노라마 X-ray 질환 검출 데이터셋

- 훈련 705장 + 검증 51장 (COCO format, segmentation polygon 포함)
- 4개 클래스: Impacted(매복치), Caries(충치), Periapical Lesion(치근단병변), Deep Caries(깊은충치)
- 전처리: `COCO JSON → polygon 좌표 정규화 → YOLO seg 포맷`

---

## Tech Stack

| 분류 | 기술 | 선택 이유 |
|------|------|----------|
| **Model** | YOLOv8n-seg | 실시간 추론, CPU 배포 가능, 6.5MB 경량 |
| **Training** | Google Colab T4 | 무료 GPU, 50 epochs ~48분 |
| **Serving** | FastAPI + Uvicorn | async 지원, 자동 API 문서 (Swagger) |
| **RAG** | LangChain 0.3.27 LCEL | 최신 패턴, 체인 조립 유연 |
| **Embedding** | BAAI/bge-m3 | 한국어+영어 동시 지원, 의료 문서 최적화 |
| **VectorDB** | ChromaDB | 경량, 로컬 실행, 별도 서버 불필요 |
| **LLM** | GPT-4o-mini | 저렴(약 $0.15/1M tokens) + 충분한 품질 |
| **Database** | SQLite | 외부 의존성 없음, stdlib만으로 실험 추적 |
| **Infra** | Docker (CPU-only) | GPU 없는 병원 환경에서도 배포 가능 |
| **Test** | pytest (25개) | CI 환경에서도 통과하도록 설계 (멀티모델 포함) |

---

## 개발 노트 (트러블슈팅 기록)

> 나중에 포트폴리오/면접 정리용. 실제로 겪은 문제들.

### 1. numpy 소스 빌드 실패 (Python 3.13 + Windows)

**증상**: `pip install -r requirements.txt` 시 numpy 빌드 에러
```
ERROR: Unknown compiler(s): [['icl'], ['cl'], ['cc'], ['gcc'], ...]
```

**원인 분석**:
- `langchain==0.3.0`이 `numpy<2.0.0,>=1.26.0` 요구
- Python 3.13용 numpy 1.26.4 **wheel(미리 컴파일 파일)이 존재하지 않음**
- pip가 소스에서 직접 빌드 시도 → Windows에 C 컴파일러(MSVC/GCC) 없어서 실패
- `ultralytics==8.3.0`도 동일하게 `numpy<2.0.0` 요구

**해결**:
- `langchain==0.3.0` → `langchain==0.3.27` (최신 0.3.x, numpy 제한 제거)
- `ultralytics==8.3.0` → `ultralytics==8.4.14` (numpy 2.x 지원)
- 연관 패키지 전부 Python 3.13 호환 버전으로 업데이트
- **코드 변경 없음** — 동일 0.3.x 라인이라 API 호환

**교훈**: 라이브러리 버전 고정 시 Python 버전과 wheel 가용성 반드시 확인.
`pip install [패키지] --dry-run` 으로 사전 검증 가능.

### 2. OpenCV 한글 경로 문제

**증상**: 한글이 포함된 경로의 이미지를 `cv2.imread()`로 읽으면 `None` 반환

**원인**: OpenCV의 `imread()`가 Windows에서 한글(비ASCII) 경로를 처리 못함

**해결**:
```python
# 기존 (실패)
img = cv2.imread("C:/경로/이미지.jpg")

# 수정 (성공)
img_array = np.fromfile("C:/경로/이미지.jpg", dtype=np.uint8)
img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
```

### 3. load_dotenv 환경변수 미적용 (uvicorn --reload)

**증상**: `.env` 파일에 `OPENAI_API_KEY`가 있는데, 서버 첫 시작 시 RAG 초기화 실패
```
[V2] RAG 초기화 실패: The api_key client option must be set either by passing api_key
to the client or by setting the OPENAI_API_KEY environment variable
```
서버를 리로드하면 정상 작동 — 첫 시작에서만 실패하는 간헐적 버그.

**원인 분석**:
- `load_dotenv()`는 기본적으로 **이미 존재하는 환경변수를 덮어쓰지 않음**
- uvicorn `--reload` 모드에서 reloader 프로세스 → worker 프로세스로 분기 시, 환경변수 상태가 불안정할 수 있음
- `app.py`에서만 `load_dotenv`를 호출했기 때문에, `chain.py`가 독립적으로 임포트되면 API 키를 못 읽음

**해결**:
```python
# app.py — override=True 추가
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

# chain.py — 독립적으로도 .env 로드 (방어적 프로그래밍)
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=True)
```

**교훈**: `load_dotenv()`의 기본 동작은 기존 환경변수를 **보존**한다.
확실하게 `.env` 값을 사용하려면 `override=True` 필수.
멀티모듈 구조에서는 각 모듈이 자기 의존성을 스스로 해결하는 **방어적 프로그래밍** 패턴이 안전하다.

### 4. LangChain deprecated API 패턴

**증상**: 구버전 코드 그대로 쓰면 DeprecationWarning 또는 에러

**해결 (최신 패턴)**:
```python
# ❌ deprecated
from langchain.chains import RetrievalQA
chain = RetrievalQA.from_chain_type(...)

# ✅ 현재 패턴 (LCEL)
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
qa_chain = create_stuff_documents_chain(llm, prompt)
chain = create_retrieval_chain(retriever, qa_chain)

# ❌ deprecated
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# ✅ 현재 패턴
from langchain_huggingface import HuggingFaceEmbeddings

# ❌ deprecated (FastAPI)
@app.on_event("startup")

# ✅ 현재 패턴
from contextlib import asynccontextmanager
@asynccontextmanager
async def lifespan(app: FastAPI): ...
```
