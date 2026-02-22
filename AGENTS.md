# AGENTS.md — 프로젝트 컨텍스트 (AI 어시스턴트용)

> 이 파일은 AI 어시스턴트가 대화 맥락을 유지하기 위한 컨텍스트 파일입니다.
> 새 세션에서도 이 파일을 읽으면 이전 대화를 이어갈 수 있습니다.

---

## 사용자 정보

- **이름**: 이승규
- **나이**: 만 27세 (28세)
- **학력**: 을지대 의료공학과 졸업 (GPA 3.42/4.5), 가톨릭관동대 의료IT 중퇴→편입
- **교육**: 아시아경제 디지털헬스케어 AI 솔루션 6개월 (2025.05~11)
- **수상**: 최우수상(1위/25명), 유통데이터 우수상(3위/250팀)
- **자격증**: ADsP, TOEIC 820, TOEIC Speaking IM3, 컴활2급
- **목표**: AI 엔지니어 취업 (의료 AI 특화), 1~2개월 내 지원 시작
- **로컬 GPU**: GTX 1650 (4GB) → 학습은 Colab에서 진행
- **디스크**: 30GB 남음 → 큰 데이터는 Google Drive (데스크톱 앱 설치 완료)

---

## 대화 규칙 (필수)

1. **"너 혼자 다하지말고 나한테도 알려주면서 같이해"** — 모든 코드/개념을 설명하면서 같이 진행
2. **"뭐든건 내가 면접이나 이력서를 쓸때 참고할수있게 생각해서 대답해줘"** — 모든 설명에 면접 포인트 포함
3. **"내말의 의도가 잘 이해가 안되면 말해주고 같이 해보"** — 확인 필요 시 질문
4. 한국어로 대화
5. 코드에는 상세한 한글 주석 포함

---

## 프로젝트 로드맵

### 프로젝트1 (거의 완료): 의료 Object Detection/Segmentation + RAG
- **데이터**: Kvasir-SEG (위장 폴립, 1000장 ✅), DENTEX (치과 X-ray, 다운로드 완료 ✅)
- **모델**: YOLOv8n-seg (멀티모델: polyp + dental)
- **추가**: SQL 실험 DB + Docker + FastAPI + RAG + 멀티모델 API
- **상태**: Kvasir-SEG 완료, DENTEX 전처리/학습 스크립트 작성 완료 → Colab 실행 대기

### 프로젝트2 (다음): Medical VLM
- 의공학 배경 + 최신 비전 트렌드
- 1~2달 예상

### 프로젝트3 (선택): 논문 구현 또는 Medical SAM

---

## 완료된 작업

### ✅ 데이터 준비 — Kvasir-SEG
- `prepare_dataset.py` — Kvasir-SEG 마스크 → YOLO seg 포맷 변환 (Train 800, Val 200)
- `datasets/kvasir-seg/` — images/labels train/val 분할 완료
- `datasets/kvasir-seg/kvasir-seg.yaml` — YOLO 데이터셋 설정

### ✅ 데이터 준비 — DENTEX
- `prepare_dataset_dentex.py` — COCO JSON → YOLO seg 포맷 변환 (Colab용)
- DENTEX 데이터 Google Drive에 다운로드 완료 (training_data.zip)
- 어노테이션: COCO format, 4클래스 (Impacted, Caries, Periapical Lesion, Deep Caries)

### ✅ 학습 스크립트
- `train_colab.py` — Colab용 Kvasir-SEG 학습 (50 epochs 완료)
- `train_colab_dentex.py` — Colab용 DENTEX 학습 (100 epochs, cos_lr, patience=15)

### ✅ Kvasir-SEG 학습 결과
- 50 epochs, batch 16, YOLOv8n-seg, Colab T4 GPU
- mAP50(Box): 0.939, mAP50(Mask): 0.942
- Precision: 0.920, Recall: 0.887

### ✅ SQL 실험 DB
- `db/experiment_db.py` — SQLite 기반 실험 결과 관리
- `db/experiments.db` — Kvasir-SEG 학습 결과 저장 완료 (50 epoch × 6 지표 = 300개 + 최종)

### ✅ FastAPI 추론 API (V1 + V2 + V3, 멀티모델)
- `api/app.py` (372줄):
  - `POST /predict` — 병변 세그멘테이션 (model_type: polyp/dental 선택)
  - `POST /ask` — 의료 지식 Q&A (RAG)
  - `POST /analyze` — Vision + LLM 통합 분석 (V3 핵심, model_type 지원)
  - `GET /health` — 서버 상태 (로드된 모델 목록)
- `rag/chain.py` — LangChain LCEL + ChromaDB + GPT-4o-mini
- `rag/ingest.py` — PDF → 청킹 → BGE-M3 임베딩 → ChromaDB 인덱싱

### ✅ 테스트
- `tests/test_api.py` — pytest 20개 (Health 4, Predict 5, Validation 4, RAG 3, Analyze 4)
- 19 passed, 1 skipped (OpenAI 토큰 소진으로 RAG 응답 검증만 skip)

### ✅ Docker 환경 구성
- Dockerfile: V1(YOLOv8) + V2(RAG) 지원, CPU-only PyTorch, curl healthcheck
- docker-compose.yml: .env 연동, RAG 벡터스토어/docs 볼륨 마운트
- .dockerignore: 불필요 파일 제외

### ✅ 문서화
- `README.md` — 전면 개편 (모델 선택 근거 비교표, 의료 메트릭 해석, V3 아키텍처, Tech Stack)
- `PROJECT_REVIEW.md` — 포트폴리오/이력서용 종합 문서 (면접 예상 Q&A 포함)
- `CAREER_PLAN.md` — 커리어 플랜 정리

### ✅ 트러블슈팅 기록 (README에 문서화)
1. numpy 소스 빌드 실패 (Python 3.13 호환)
2. OpenCV 한글 경로 문제
3. load_dotenv 환경변수 미적용 (uvicorn --reload)
4. LangChain deprecated API → LCEL 최신 패턴 적용

### ✅ 이력서/포폴 평가
- 4개 PDF 분석 완료
- 강점: V1→V3 개선 스토리, 수상 2개, 의공학 차별점
- 약점: CV 프로젝트 부족, 엔지니어링 경험 없음 → 이 프로젝트로 보완 중

---

## 진행 중 / 남은 작업

### ⏳ DENTEX Colab 실행
- `prepare_dataset_dentex.py` → Colab에서 실행 (Google Drive 경로 확인 필요)
- `train_colab_dentex.py` → Colab에서 학습 (100 epochs)
- 학습 완료 후 `best_dentex.pt`를 프로젝트에 추가

### ❌ 이력서 오탈자 수정
- naver,com → naver.com
- 마침표 2개 → 1개
- Appel → Apple
- 의료I학과 → 의료IT학과
- 포트폴리오ㅓ → 포트폴리오

### ❌ GitHub README 정리
- 프로젝트별 README 작성
- 아키텍처 다이어그램 추가
- 코드 정리

---

## 파일 구조

```
C:\Users\user\Desktop\Python.Algrithm\
├── AGENTS.md                           # ← 이 파일 (AI 컨텍스트)
├── project practice/
│   ├── api/
│   │   └── app.py                      # ✅ FastAPI V3 (predict+ask+analyze, 멀티모델)
│   ├── rag/
│   │   ├── chain.py                    # ✅ RAG 체인 (LCEL + ChromaDB)
│   │   ├── ingest.py                   # ✅ PDF 인덱싱 파이프라인
│   │   ├── docs/                       # 의료 PDF 문서 (5개)
│   │   └── vectorstore/                # ChromaDB 벡터스토어
│   ├── preprocessing/
│   │   ├── prepare_dataset.py          # ✅ Kvasir-SEG 변환 (마스크→YOLO seg)
│   │   └── prepare_dataset_dentex.py   # ✅ DENTEX 변환 (COCO JSON→YOLO seg, Colab용)
│   ├── training/
│   │   ├── train_colab.py              # ✅ Kvasir-SEG Colab 학습 (완료)
│   │   ├── train_colab_dentex.py       # ✅ DENTEX Colab 학습 (실행 대기)
│   │   └── results.csv                 # ✅ Kvasir-SEG 50 epoch 학습 로그
│   ├── tests/
│   │   ├── test_api.py                 # ✅ pytest 20개 (19P/1S)
│   │   └── test_yolo.py               # 기존 YOLOv8 테스트
│   ├── db/
│   │   ├── experiment_db.py            # ✅ SQL 실험 DB
│   │   └── experiments.db              # ✅ Kvasir-SEG 결과 저장됨
│   ├── Dockerfile                      # ✅ V1+V2 Docker (CPU-only)
│   ├── docker-compose.yml              # ✅ 볼륨 마운트 + .env
│   ├── .dockerignore                   # ✅ 배포 최적화
│   ├── requirements.txt                # ✅ Python 3.13 호환
│   ├── README.md                       # ✅ 전면 개편 (모델 비교, 메트릭, 아키텍처)
│   ├── PROJECT_REVIEW.md               # ✅ 포트폴리오/면접용 종합 문서
│   ├── CAREER_PLAN.md                  # ✅ 커리어 플랜 정리
│   ├── .gitignore                      # ✅ .env/데이터 제외
│   ├── .env                            # OPENAI_API_KEY (git 제외)
│   ├── best.pt                         # ✅ Kvasir-SEG 학습 모델 (polyp)
│   ├── best_dentex.pt                  # ⏳ DENTEX 학습 후 추가 예정 (dental)
│   ├── yolov8n.pt                      # 사전학습 모델
│   └── bus.jpg                         # 테스트 이미지
│
└── resume_lee/                         # 이력서/포폴 PDF
    ├── 이력서 - Google Docs.pdf
    ├── 포트폴리오ㅓ.pdf
    ├── kt이력서.pdf
    └── 경험정리.pdf
```

---

## Colab 관련 경로

### Kvasir-SEG (완료)
- Google Drive 데이터셋: `/content/drive/MyDrive/kvasir-seg/`
- Google Drive 결과: `/content/drive/MyDrive/yolo-results/kvasir-seg-v1/`
- YAML 설정: `/content/kvasir-seg.yaml`

### DENTEX (실행 대기)
- Google Drive 원본: `DENTEX/DENTEX/training_data.zip` (경로 확인 필요)
- Google Drive 결과: `/content/drive/MyDrive/yolo-results/dentex-v1/`
- 변환 출력: `/content/drive/MyDrive/datasets/dentex/`
- YAML 설정: `/content/dentex.yaml`

---

## 기술적 발견사항

1. **한글 경로 문제**: OpenCV `imread`가 한글 경로 처리 못함 → `np.fromfile` + `cv2.imdecode`로 우회
2. **Colab HuggingFace**: `huggingface-cli` PATH 문제 → Python API `snapshot_download` 사용
3. **DENTEX 어노테이션**: COCO format, `category_id_3`이 4-class 키, segmentation polygon 포함
4. **Kvasir-SEG**: 1000장 (images + masks), polyp segmentation, 로컬에 100MB
5. **load_dotenv**: 기본 동작은 기존 환경변수 보존. uvicorn --reload 시 `override=True` 필수
6. **방어적 프로그래밍**: 멀티모듈 구조에서 각 모듈이 자기 의존성을 스스로 해결 (chain.py에도 load_dotenv 추가)
7. **LSP 에러**: `dotenv`, `fastapi`, `langchain_*` 등 import 에러는 시스템 Python 패키지를 LSP가 못 찾는 것 — 런타임 정상
8. **DENTEX vs Kvasir-SEG**: 데이터 적고(~700장) 클래스 많으므로(4개) epochs↑(100), patience↑(15), cos_lr 추가
