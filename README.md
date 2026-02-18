# Medical Image Segmentation — Polyp Detection

YOLOv8n-seg 기반 위장 내시경 폴립 세그멘테이션 프로젝트.  
Kvasir-SEG 데이터셋으로 학습하고, FastAPI + Docker로 추론 서버 구성.

## Results

| Metric | Box Detection | Mask Segmentation |
|--------|:---:|:---:|
| Precision | 0.920 | 0.930 |
| Recall | 0.887 | 0.897 |
| mAP@50 | 0.939 | 0.942 |
| mAP@50-95 | 0.777 | 0.786 |

- Model: YOLOv8n-seg (nano, 6.5MB)
- Training: 50 epochs, batch 16, Colab T4 GPU
- Dataset: Kvasir-SEG 1000장 (Train 800 / Val 200)

## Architecture

```
Kvasir-SEG (masks)
       │
       ▼
prepare_dataset.py ──→ YOLO format labels
       │
       ▼
train_colab.py ──→ best.pt (trained weights)
       │
       ▼
FastAPI (api/app.py) ──→ POST /predict
       │
       ▼
Docker container (Dockerfile)
       │
       ▼
experiment_db.py ──→ SQLite (results tracking)
```

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

## API Endpoints

### `GET /health`

서버 상태 확인.

### `POST /predict`

이미지 업로드 → 폴립 세그멘테이션 추론.

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@test_image.jpg" \
  -F "conf=0.25"
```

Response:
```json
{
  "filename": "test_image.jpg",
  "image_size": {"width": 640, "height": 480},
  "detections": [
    {
      "class": "polyp",
      "confidence": 0.92,
      "bbox": [120.5, 80.3, 340.1, 260.7],
      "polygon_points": 24,
      "polygon": [[120.5, 80.3], [125.0, 78.1], ...]
    }
  ],
  "count": 1
}
```

## Project Structure

```
├── api/
│   └── app.py              # FastAPI 추론 서버
├── db/
│   └── experiment_db.py     # SQLite 실험 결과 관리
├── prepare_dataset.py       # Kvasir-SEG 마스크 → YOLO 포맷 변환
├── train_colab.py           # Colab 학습 스크립트
├── results.csv              # 50 epoch 학습 로그
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Dataset

**Kvasir-SEG** — 위장 내시경 폴립 세그멘테이션 데이터셋

- 1000장 (이미지 + 바이너리 마스크)
- 출처: [Kvasir-SEG (SimulaMet)](https://datasets.simula.no/kvasir-seg/)
- 변환: 바이너리 마스크 → contour 추출 → YOLO polygon 포맷

## Tech Stack

- **Model**: YOLOv8n-seg (Ultralytics)
- **Training**: Google Colab (T4 GPU)
- **API**: FastAPI + Uvicorn
- **DB**: SQLite (실험 추적)
- **Infra**: Docker
- **Language**: Python 3.10
