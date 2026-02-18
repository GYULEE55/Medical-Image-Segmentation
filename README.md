# Medical Image Segmentation — Polyp Detection

> YOLOv8n-seg 기반 위장 내시경 폴립(Polyp) 세그멘테이션 프로젝트

Kvasir-SEG 데이터셋(1,000장)으로 학습한 경량 세그멘테이션 모델을 FastAPI 추론 서버로 서빙하고, Docker로 배포할 수 있도록 구성한 End-to-End 파이프라인입니다.

---

## Results

| Metric | Box Detection | Mask Segmentation |
|--------|:---:|:---:|
| **Precision** | 0.920 | 0.930 |
| **Recall** | 0.887 | 0.897 |
| **mAP@50** | 0.939 | **0.942** |
| **mAP@50-95** | 0.777 | 0.786 |

- **Model**: YOLOv8n-seg (nano, 6.5MB)
- **Training**: 50 epochs, batch 16, Google Colab T4 GPU
- **Dataset**: Kvasir-SEG 1,000장 (Train 800 / Val 200)

---

## Architecture

```
[Kvasir-SEG 원본 데이터]
        │
        ▼
preprocessing/prepare_dataset.py    # 바이너리 마스크 → YOLO polygon 변환
        │
        ▼
training/train_colab.py             # Colab T4 GPU에서 50 epochs 학습
        │
        ▼
best.pt (학습된 가중치)
        │
        ▼
api/app.py                          # FastAPI 추론 서버 (POST /predict)
        │
        ▼
Dockerfile + docker-compose.yml     # 컨테이너화 배포
        │
        ▼
db/experiment_db.py                 # SQLite 실험 결과 추적
```

---

## Project Structure

```
├── api/                        # 추론 API 서버
│   └── app.py                  #   FastAPI - 이미지 업로드 → 세그멘테이션 결과 반환
│
├── preprocessing/              # 데이터 전처리
│   └── prepare_dataset.py      #   Kvasir-SEG 마스크 → YOLO polygon 포맷 변환
│
├── training/                   # 모델 학습
│   ├── train_colab.py          #   Google Colab 학습 스크립트
│   └── results.csv             #   50 epoch 학습 로그 (loss, mAP 등)
│
├── db/                         # 실험 관리
│   └── experiment_db.py        #   SQLite 기반 실험 결과 CRUD
│
├── tests/                      # 테스트
│   └── test_yolo.py            #   YOLOv8 추론 테스트
│
├── Dockerfile                  # Docker 이미지 빌드 (CPU-only PyTorch)
├── docker-compose.yml          # 원커맨드 컨테이너 실행
├── requirements.txt            # Python 의존성
└── .gitignore                  # 데이터/모델/venv 제외
```

---

## Quick Start

### Local

```bash
pip install -r requirements.txt
uvicorn api.app:app --reload --port 8000
```

### Docker

```bash
docker compose up --build
```

서버 실행 후 http://localhost:8000/docs 에서 Swagger UI로 테스트 가능.

---

## API

### `GET /health`

서버 상태 및 모델 로드 확인.

### `POST /predict`

이미지 업로드 → 폴립 세그멘테이션 추론.

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@test_image.jpg" \
  -F "conf=0.25"
```

**Response:**

```json
{
  "filename": "test_image.jpg",
  "image_size": { "width": 640, "height": 480 },
  "detections": [
    {
      "class": "polyp",
      "confidence": 0.92,
      "bbox": [120.5, 80.3, 340.1, 260.7],
      "polygon_points": 24,
      "polygon": [[120.5, 80.3], [125.0, 78.1], "..."]
    }
  ],
  "count": 1
}
```

---

## Dataset

**[Kvasir-SEG](https://datasets.simula.no/kvasir-seg/)** — 위장 내시경 폴립 세그멘테이션 데이터셋 (SimulaMet)

- 이미지 1,000장 + 바이너리 마스크
- 전처리: `contour 추출 → 폴리곤 좌표 → YOLO 정규화 포맷`
- Train/Val: 800/200 (8:2 랜덤 분할)

---

## Tech Stack

| 분류 | 기술 |
|------|------|
| **Model** | YOLOv8n-seg (Ultralytics) |
| **Training** | Google Colab (T4 GPU) |
| **Serving** | FastAPI + Uvicorn |
| **Database** | SQLite |
| **Infra** | Docker (CPU-only PyTorch) |
| **Language** | Python 3.10 |
