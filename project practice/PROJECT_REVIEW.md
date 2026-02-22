# 프로젝트 리뷰 — Medical AI Assistant

> 이력서/포트폴리오 작성 및 면접 준비용 종합 문서
> 프로젝트의 사고 과정, 기술 선택 근거, 문제 해결 경험을 정리합니다.

---

## 1. 프로젝트 한줄 요약

**내시경 영상에서 폴립을 실시간 검출(YOLOv8)하고, 검출 결과를 기반으로 논문/가이드라인 기반 의료 지식을 자동 제공(RAG)하는 End-to-End 의료 AI 서비스**

---

## 2. 왜 이 프로젝트를 만들었나 (동기)

### 문제 인식
- 대장 내시경 검사 시 폴립(용종) 놓침률(miss rate)이 **6~27%**에 달함 (Van Rijn et al., 2006)
- 검출 모델만으로는 "폴립 있음"에서 끝남 — **"이 폴립이 무엇이고, 어떻게 관리해야 하는지"**는 제공 못 함
- 기존 의료 AI 서비스는 Detection OR LLM 중 하나만 수행

### 해결 방안
- **Vision(YOLOv8)이 발견하고, LLM(RAG)이 설명하는 구조**
- 검출 → 자동 RAG 질의 생성 → 논문 기반 답변 + 출처 제공
- 하나의 API 호출(`POST /analyze`)로 Detection + 의료 지식을 동시에 반환

### 💡 면접 포인트
> "단순히 모델 하나 학습한 게 아니라, **임상 현장의 실제 문제(놓침률)에서 출발**해서 **Vision + LLM 통합 파이프라인**까지 구축한 End-to-End 프로젝트입니다."

---

## 3. 아키텍처 진화 — V1 → V2 → V3

### V1: 폴립 세그멘테이션 (YOLOv8n-seg)
```
이미지 → YOLOv8n-seg → bbox + mask + confidence
```
- Kvasir-SEG 1,000장으로 학습 (Train 800 / Val 200)
- mAP@50: **0.942** (Mask), Recall: **0.897**

### V2: 의료 지식 RAG (LangChain + ChromaDB + GPT-4o-mini)
```
질문 → BGE-M3 임베딩 → ChromaDB 검색 → GPT-4o-mini 답변 + 출처
```
- 의료 논문/가이드라인 PDF 5개 인덱싱
- 한국어+영어 동시 지원 (BGE-M3)

### V3: Vision + LLM 통합 (`POST /analyze`) ← 핵심
```
이미지 → YOLOv8 검출 → 검출 클래스 기반 RAG 질의 자동 생성 → 의료 지식 + 출처 반환
```
- `/predict`와 `/ask`가 독립적이었던 문제를 **하나의 파이프라인으로 연결**
- `class_names_kr` 매핑으로 한국어 질의 자동 생성

### 💡 면접 포인트
> "V1에서 V3까지 **점진적으로 발전**시켰습니다. V1은 Detection만, V2에서 RAG를 추가하고, V3에서 두 모듈을 **자동 연동**하는 파이프라인을 만들었습니다. 이 과정에서 **독립적인 모듈을 어떻게 연결할 것인가**라는 시스템 설계 문제를 해결했습니다."

---

## 4. 기술 선택 근거 (왜 이 기술을 선택했는가)

### 4-1. 모델: 왜 YOLOv8인가?

| 고려 사항 | YOLOv8-seg | Mask R-CNN | SAM | U-Net |
|-----------|:---:|:---:|:---:|:---:|
| 실시간 추론 | ✅ (~5ms/frame) | ❌ (two-stage) | ❌ (heavy) | ❌ (seg only) |
| Detection + Segmentation 동시 | ✅ | ✅ | ❌ | ❌ |
| 경량 (CPU 배포) | ✅ (6.5MB) | ❌ (~170MB) | ❌ (~2.4GB) | 가능 |
| 1,000장 Transfer Learning | ✅ | 가능 | 불필요 | 가능 |

**선택 이유**:
1. 내시경은 실시간 영상 → **빠른 추론 필수** (YOLOv8 nano: ~5ms/frame GPU)
2. 병원에 GPU 서버가 없을 수 있음 → **CPU 배포 가능한 6.5MB 모델**
3. 폴립 위치(bbox) + 모양(mask)을 **한 번에 출력**
4. ImageNet+COCO 사전학습으로 **1,000장으로도 충분한 성능**

### 4-2. RAG: 왜 직접 RAG를 구축했나?

**GPT-4에 직접 질문하면 안 되나?**
- GPT-4는 2023년 이전 데이터 → **최신 가이드라인 반영 불가**
- 의료 분야에서 **출처 없는 답변은 신뢰할 수 없음**
- RAG는 **우리가 검증한 논문/가이드라인**에서만 답변 → 할루시네이션 최소화

| 구성 요소 | 선택 | 이유 |
|-----------|------|------|
| Embedding | BGE-M3 | 한국어+영어 동시 지원, 의료 문서 최적화 |
| VectorDB | ChromaDB | 경량, 로컬 실행, 별도 서버 불필요 |
| LLM | GPT-4o-mini | 저렴($0.15/1M tokens) + 충분한 품질 |
| Framework | LangChain LCEL | 최신 패턴, 체인 조립 유연 |

### 4-3. 서빙: 왜 FastAPI인가?

- **async 지원** → 이미지 업로드 + RAG 호출을 비동기로 처리
- **자동 API 문서(Swagger)** → `/docs`에서 바로 테스트 가능
- **Pydantic 타입 검증** → 입력 유효성 자동 검사
- **Docker 친화적** → uvicorn 단독 실행, 경량 컨테이너

### 4-4. 인프라: 왜 Docker인가?

- **재현성**: `docker compose up` 한 줄로 동일 환경 보장
- **배포 용이성**: 병원 서버에 Docker만 있으면 즉시 배포
- **CPU-only PyTorch**: GPU 없는 환경에서도 동작
- **환경 분리**: `.env`로 API 키 관리, 볼륨으로 벡터스토어 영속화

### 💡 면접 포인트
> "각 기술을 **왜 선택했는지** 명확한 근거가 있습니다. 예를 들어 YOLOv8은 **실시간 추론과 CPU 배포**라는 의료 현장의 제약 조건을 고려해서 선택했고, RAG는 **출처 기반 답변으로 의료 AI의 신뢰성 문제**를 해결하기 위해 직접 구축했습니다."

---

## 5. 성과 지표

### 5-1. Kvasir-SEG 학습 결과 (50 Epochs)

| Metric | Box Detection | Mask Segmentation |
|--------|:---:|:---:|
| **Precision** | 0.920 | 0.930 |
| **Recall** | 0.887 | **0.897** |
| **mAP@50** | 0.939 | **0.942** |
| **mAP@50-95** | 0.777 | 0.786 |

### 5-2. DENTEX 학습 결과 (100 Epochs, Best: Epoch 83)

| Metric | Box Detection | Mask Segmentation |
|--------|:---:|:---:|
| **Precision** | 0.485 | 0.485 |
| **Recall** | 0.334 | 0.334 |
| **mAP@50** | 0.377 | 0.344 |
| **mAP@50-95** | 0.242 | 0.225 |

### 5-3. 두 데이터셋 성능 차이 분석

| 비교 항목 | Kvasir-SEG (mAP50: 0.94) | DENTEX (mAP50: 0.38) |
|-----------|:---:|:---:|
| 클래스 수 | 1개 (polyp) | 4개 (치과 질환) |
| 클래스당 학습 데이터 | 800장 | ~175장 |
| 영상 특성 | 컬러 내시경 (병변이 뚜렷) | 흑백 X-ray (병변 경계 불명확) |
| 병변 크기 | 비교적 큼 | 작고 다양 |

**개선 방향**:
1. **데이터 증강 강화**: contrast/brightness 조절 (X-ray 특화)
2. **모델 스케일업**: YOLOv8n → YOLOv8s (파라미터 3배 증가)
3. **클래스 불균형 대응**: Focal Loss, 오버샘플링
4. **해상도 증가**: 640 → 1280 (미세 병변 검출)
5. **전이 학습**: 치과 X-ray 사전학습 모델 활용

### 💡 면접 포인트
> "두 데이터셋에서 같은 모델을 돌려보니 **성능 차이가 극명**했습니다. 분석 결과 **클래스 수, 클래스당 데이터 양, 영상 특성**이 성능에 큰 영향을 미친다는 것을 확인했고, 구체적인 개선 방향을 제시할 수 있습니다. 이런 비교 실험 경험이 실무에서 **데이터/모델 전략을 세울 때 도움**이 됩니다."

### 5-4. 의료 AI 관점 해석

- **Recall 0.897**: 폴립의 89.7%를 놓치지 않고 검출 — **의료에서 가장 중요한 지표**
  - 놓치면 → 조기 대장암 발견 실패 → 환자 생명 위험
  - Precision보다 Recall이 중요한 이유: **False Negative(놓침) > False Positive(오탐)**
- **Precision 0.930**: 검출한 것 중 93%가 실제 폴립 → 불필요한 조직검사 최소화
- **mAP@50 0.942**: IoU 50% 기준 전반적 성능 → 임상 보조 도구로 활용 가능 수준

### 5-3. 학습 곡선 분석

- Epoch 1~10: 빠른 수렴 (Transfer Learning 효과)
- Epoch 10~30: 안정적 개선 (mAP50 0.55 → 0.93)
- Epoch 30~50: 미세 조정 (과적합 없이 수렴)
- `patience=10` EarlyStopping 설정했으나 50 epoch까지 val_loss가 계속 감소 → 조기 중단 미발동

### 💡 면접 포인트
> "의료 AI에서는 **Recall이 Precision보다 중요**합니다. False Negative(폴립을 놓침)은 환자 생명과 직결되지만, False Positive(오탐)은 추가 검사로 해결 가능하기 때문입니다. 저의 모델은 Recall 89.7%를 달성했습니다."

---

## 6. 문제 해결 경험 (트러블슈팅)

### 6-1. Python 3.13 + numpy 소스 빌드 실패

**상황**: `pip install -r requirements.txt` 시 numpy 빌드 에러

**원인 분석**:
- `langchain==0.3.0`이 `numpy<2.0.0` 요구
- Python 3.13용 numpy 1.26.4 **wheel이 존재하지 않음**
- 소스 빌드 시도 → Windows에 C 컴파일러 없어서 실패

**해결**:
- `langchain==0.3.0` → `0.3.27` (numpy 제한 제거)
- `ultralytics==8.3.0` → `8.4.14` (numpy 2.x 지원)
- **코드 변경 없음** — 동일 마이너 버전이라 API 호환

**교훈**: 버전 고정 시 Python 버전과 wheel 가용성 반드시 확인. `pip install --dry-run`으로 사전 검증.

### 6-2. OpenCV 한글 경로 문제

**상황**: `cv2.imread("한글경로/이미지.jpg")` → `None` 반환

**원인**: OpenCV가 Windows에서 비ASCII 경로를 처리하지 못함

**해결**:
```python
# ❌ 실패
img = cv2.imread("C:/한글/이미지.jpg")

# ✅ 성공
buf = np.fromfile("C:/한글/이미지.jpg", dtype=np.uint8)
img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
```

### 6-3. load_dotenv 환경변수 미적용 (간헐적 버그)

**상황**: `.env`에 `OPENAI_API_KEY`가 있는데 서버 첫 시작 시만 RAG 초기화 실패

**원인 분석**:
- `load_dotenv()`는 기본적으로 **기존 환경변수를 덮어쓰지 않음**
- uvicorn `--reload`에서 프로세스 분기 시 환경변수 상태 불안정
- `app.py`에서만 `load_dotenv` 호출 → `chain.py`가 독립 임포트 시 키를 못 읽음

**해결**:
```python
# app.py + chain.py 모두에 추가 (방어적 프로그래밍)
load_dotenv(Path(__file__).parent.parent / ".env", override=True)
```

**교훈**: 멀티모듈 구조에서 각 모듈이 자기 의존성을 스스로 해결하는 **방어적 프로그래밍** 패턴이 안전.

### 6-4. LangChain deprecated API 마이그레이션

**상황**: LangChain 구버전 API 사용 시 DeprecationWarning

**해결**: 최신 LCEL 패턴으로 전면 전환
```python
# ❌ deprecated
from langchain.chains import RetrievalQA
chain = RetrievalQA.from_chain_type(...)

# ✅ LCEL 패턴
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
```

### 💡 면접 포인트
> "개발 과정에서 겪은 문제들을 단순히 해결만 한 게 아니라, **왜 발생했는지 원인을 분석하고 문서화**했습니다. 특히 load_dotenv 이슈는 **멀티모듈 구조에서의 환경변수 관리**라는 일반적인 문제로 확장해서 방어적 프로그래밍 패턴을 적용했습니다."

---

## 7. 엔지니어링 실천 사항

### 7-1. 테스트
- pytest **25개** (Health 4, Predict 8, Validation 4, RAG 3, Analyze 6)
- V1/V2/V3 각 엔드포인트 + 멀티모델 파라미터 테스트 포함
- 잘못된 model_type 입력 시 400 에러 반환 검증
- CI 환경(OpenAI 키 없음)에서도 skip 처리로 깨지지 않음

### 7-2. 실험 관리
- SQLite 기반 `ExperimentDB` — 외부 의존성 없이 stdlib만 사용
- 에포크별 지표 저장 → 학습 곡선 분석 가능
- MLflow는 오버킬 → 가벼운 자체 구현 선택

### 7-3. Docker 배포
- **CPU-only PyTorch** — GPU 없는 병원 환경 고려
- `.env` 파일로 API 키 분리 (보안)
- `curl` healthcheck로 컨테이너 상태 모니터링
- 볼륨 마운트로 벡터스토어 영속화

### 7-4. 코드 품질
- 모든 코드에 상세한 한글 주석
- Pydantic 모델로 입력/출력 타입 명시
- `asynccontextmanager` lifespan 패턴 (deprecated `on_event` 대체)
- 에러 핸들링: `HTTPException` + 적절한 상태 코드

### 💡 면접 포인트
> "모델 학습뿐만 아니라 **테스트, Docker 배포, 실험 관리, API 설계**까지 전체 ML 파이프라인을 경험했습니다. 특히 pytest 20개로 각 엔드포인트의 정상/에러 케이스를 커버하고, SQLite로 실험 결과를 체계적으로 관리합니다."

---

## 8. 데이터 파이프라인

### Kvasir-SEG 전처리 흐름

```
원본 바이너리 마스크 (.jpg, 흑백)
    │
    ▼ cv2.findContours()
외곽선(contour) 추출
    │
    ▼ cv2.approxPolyDP()
폴리곤 단순화 (점 수 감소, 노이즈 제거)
    │
    ▼ 좌표 정규화 (x/width, y/height)
YOLO seg 포맷 (.txt)
    │
    ▼ random split (seed=42)
Train 800장 / Val 200장
```

**핵심 처리**:
- `min_area=100`: 100px² 미만 contour 제거 (노이즈)
- `epsilon=0.005 * arcLength`: 폴리곤 단순화 (학습 효율)
- `seed=42`: 재현 가능한 분할

### DENTEX 전처리 (치과 X-ray)

```
COCO JSON (segmentation polygon)
    │
    ▼ 카테고리 매핑
4개 클래스: Impacted(매복치), Caries(충치),
            Periapical Lesion(치근단병변), Deep Caries(깊은충치)
    │
    ▼ polygon 좌표 정규화
YOLO seg 포맷 (.txt)
    │
    ▼ random split (seed=42)
Train / Val 분할
```

### 💡 면접 포인트
> "두 가지 서로 다른 데이터 포맷(바이너리 마스크, COCO JSON)을 동일한 YOLO seg 포맷으로 변환하는 파이프라인을 구축했습니다. **데이터 전처리가 모델 성능에 직접적인 영향**을 미치기 때문에, contour 필터링과 폴리곤 단순화 같은 세부 처리에 신경을 썼습니다."

---

## 9. 프로젝트 구조

```
project practice/
├── api/app.py                  # FastAPI V1+V2+V3 멀티모델 (372줄)
├── rag/
│   ├── chain.py                # LangChain LCEL + ChromaDB + GPT-4o-mini
│   ├── ingest.py               # PDF → 청킹 → BGE-M3 임베딩 → ChromaDB
│   ├── docs/                   # 의료 논문/가이드라인 PDF (5개)
│   └── vectorstore/            # ChromaDB 벡터스토어
├── preprocessing/
│   ├── prepare_dataset.py      # Kvasir-SEG 마스크 → YOLO seg 변환
│   └── prepare_dataset_dentex.py  # DENTEX COCO JSON → YOLO seg 변환
├── training/
│   ├── train_colab.py          # Colab 학습 (Kvasir-SEG)
│   ├── train_colab_dentex.py   # Colab 학습 (DENTEX)
│   └── results.csv             # 50 epoch 학습 로그
├── db/
│   ├── experiment_db.py        # SQLite 실험 결과 CRUD
│   └── experiments.db          # 실험 DB 파일
├── tests/
│   ├── test_api.py             # pytest 25개 (멀티모델 포함)
│   └── test_yolo.py            # YOLOv8 추론 테스트
├── Dockerfile                  # CPU-only PyTorch + RAG
├── docker-compose.yml          # .env + 볼륨 마운트
└── requirements.txt            # Python 3.13 호환
```

---

## 10. 멀티모델 확장 (완료)

### 10-1. DENTEX (치과 X-ray) — 구현 완료
- 4개 치과 질환 Object Detection + Segmentation
- 동일 아키텍처(YOLOv8 + RAG)로 **도메인 확장 증명**
- `/predict`, `/analyze`에 `model_type` 파라미터 추가 (polyp/dental 선택)
- Colab 전처리(`prepare_dataset_dentex.py`) + 학습(`train_colab_dentex.py`) 스크립트 작성 완료

### 💡 면접 포인트
> "동일한 아키텍처를 **새로운 의료 도메인(치과)에 확장**했습니다. `model_type` 파라미터로 polyp/dental 모델을 선택할 수 있어, **하나의 API 서버로 여러 의료 도메인을 서빙**하는 구조입니다. 새로운 의료 도메인 추가 시 모델 학습 + `MODEL_PATHS` 등록만으로 확장 가능합니다."

### 10-2. Medical VLM (향후)
- Vision-Language Model로 의료 영상 직접 해석
- 현재 Vision + LLM이 분리된 구조 → VLM으로 통합 가능성

---

## 11. 면접 예상 질문 & 답변

### Q1. "이 프로젝트에서 가장 어려웠던 점은?"

> "**V3 통합 아키텍처 설계**가 가장 어려웠습니다. V1의 Detection 결과를 V2의 RAG 입력으로 자연스럽게 연결하는 과정에서, 검출 클래스를 한국어 의료 질의로 자동 변환하는 매핑 로직이 필요했습니다. 단순 연결이 아니라 **각 모듈의 출력/입력 인터페이스를 맞추는 시스템 설계**가 핵심이었습니다."

### Q2. "왜 MLflow 대신 SQLite를 사용했나요?"

> "프로젝트 규모 대비 **MLflow는 오버킬**이었습니다. 단일 모델, 단일 데이터셋 실험이라 SQLite 3개 테이블(experiments, metrics, predictions)로 충분했고, **외부 의존성 없이 stdlib만으로 구현**할 수 있어 배포가 간단합니다. 팀 단위나 대규모 실험에서는 MLflow/W&B를 쓰겠지만, 현 시점에서는 적절한 선택이었습니다."

### Q3. "모델 성능을 더 올리려면?"

> "세 가지 방향이 있습니다:
> 1. **데이터 증강**: CutMix, MixUp 같은 고급 augmentation 추가
> 2. **모델 스케일업**: YOLOv8n → YOLOv8s/m (파라미터 증가, 정확도↑)
> 3. **앙상블**: 여러 모델의 예측을 결합해 Recall 개선
> 다만 의료 AI에서는 **Recall 개선이 최우선**이므로, confidence threshold를 낮춰 민감도를 높이는 것이 첫 번째 시도입니다."

### Q4. "RAG에서 할루시네이션은 어떻게 방지했나요?"

> "세 가지 장치를 두었습니다:
> 1. **검증된 문서만 인덱싱**: 논문/가이드라인 PDF만 벡터스토어에 저장
> 2. **출처 반환**: 모든 답변에 원본 문서명 + 페이지 번호 포함
> 3. **프롬프트 제약**: '주어진 문서에 없는 내용은 답변하지 마세요' 지시
> 완전한 방지는 불가능하지만, 출처를 통해 **검증 가능한 답변**을 제공합니다."

### Q5. "Docker 배포 시 고려한 점은?"

> "**병원 환경의 제약**을 고려했습니다:
> 1. GPU가 없을 수 있으므로 **CPU-only PyTorch** 사용
> 2. API 키를 `.env`로 분리해 **보안** 확보
> 3. `curl` healthcheck로 **컨테이너 상태 자동 모니터링**
> 4. 벡터스토어를 **볼륨 마운트**해서 컨테이너 재시작 시 데이터 유지"

### Q6. "이 프로젝트를 실제 병원에 도입한다면?"

> "추가로 필요한 것이 있습니다:
> 1. **DICOM 지원**: 의료 영상 표준 포맷 입출력
> 2. **PACS 연동**: 병원 영상 저장 시스템과 통합
> 3. **의료기기 인허가**: 식약처/FDA 승인 절차
> 4. **개인정보 보호**: 환자 영상 비식별화, 온프레미스 배포
> 5. **성능 모니터링**: 드리프트 감지, 재학습 파이프라인
> 현재 프로젝트는 **기술 검증(PoC) 단계**이며, 프로덕션에는 위 요소들이 추가되어야 합니다."

---

## 12. 기술 스택 요약

| 분류 | 기술 | 버전 |
|------|------|------|
| Model | YOLOv8n-seg (Ultralytics) | 8.4.14 |
| Training | Google Colab T4 GPU | - |
| Serving | FastAPI + Uvicorn | 0.115+ |
| RAG | LangChain LCEL | 0.3.27 |
| Embedding | BAAI/bge-m3 (HuggingFace) | - |
| VectorDB | ChromaDB | 0.5+ |
| LLM | GPT-4o-mini (OpenAI) | - |
| Database | SQLite3 (stdlib) | - |
| Container | Docker + docker-compose | - |
| Test | pytest | 8.0+ |
| Language | Python | 3.13 |

---

*마지막 업데이트: 2026-02-22*
