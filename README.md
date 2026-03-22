# Medical Image AI Assistant

YOLOv8으로 병변을 탐지하고, LLaVA가 그 의미를 설명하며, RAG가 논문 근거를 붙이는 보조 진단 파이프라인입니다.

---

## 왜 만들었나

탐지 모델은 "어디에 무엇이 있다"는 정보만 줍니다. 임상에서는 설명과 근거가 함께 필요합니다.

세 가지를 하나의 흐름으로 연결했습니다.

- **검출(YOLO)** — 병변 위치와 클래스 (정량)
- **해석(VLM)** — LLaVA가 병변의 형태와 소견을 언어로 설명 (정성)
- **근거(RAG)** — PubMed 논문 38개 + PDF 문서 기반 답변 + 출처 반환

근거가 없으면 LLM이 추측하는 대신 고정 문구를 반환합니다.

---

## 데모

<p align="center">
  <img src="./assets/demo_result.jpg" width="720"/>
</p>

---

## 개발 과정

<p align="center">
  <img src="./assets/dev_process1.jpg" width="720"/>
</p>

<p align="center">
  <img src="./assets/dev_process2.jpg" width="720"/>
</p>

---

## 성능

**Kvasir-SEG — 폴립 세그멘테이션 (단일 클래스)**

| Precision | Recall | mAP@50 (Mask) |
|-----------|--------|---------------|
| 0.920 | 0.887 | **0.942** |

**DENTEX — 치과 X-ray (4-class)**

| mAP@50 (Mask) |
|---------------|
| 0.344 |

DENTEX 성능이 낮은 이유는 클래스당 데이터 ~175장, 파노라마 X-ray 저대비, 4-class 시각적 유사성 문제입니다.
원인 분석 → [docs/DENTEX_ANALYSIS.md](docs/DENTEX_ANALYSIS.md)

---

## 구조

```
api/
  routers/     # predict / ask / analyze / vlm / monitoring
  services.py
rag/
  chain.py     # LangGraph StateGraph + no-evidence 가드
  ingest.py    # PDF → ChromaDB 인덱싱
  docs/        # PubMed 논문 38개 + PDF 4개
vlm/
  client.py    # LLaVA via Ollama REST API (비동기)
training/
  train.py
  train_colab.py
preprocessing/
  prepare_dataset.py
  prepare_dataset_dentex.py
tests/         # pytest 36개
```

---

## 실행

```bash
git clone https://github.com/GYULEE55/Medical-Image-Segmentation.git
cd Medical-Image-Segmentation
pip install -e .
cp .env.example .env
make ingest    # RAG 문서 인덱싱 (최초 1회)
make serve     # API 서버 실행
```

---

## 주요 엔드포인트

| 엔드포인트 | 설명 |
|-----------|------|
| `POST /predict` | YOLOv8 병변 탐지 |
| `POST /analyze` | 탐지 + RAG 자동 연동 |
| `POST /ask` | RAG 기반 의료 Q&A |
| `POST /vlm-analyze/async` | 비동기 VLM 이미지 해석 |
| `GET /jobs/{job_id}` | 비동기 작업 결과 조회 |

---

## 기술 스택

- YOLOv8n-seg, PyTorch, OpenCV
- LLaVA via Ollama (VLM)
- LangGraph, ChromaDB, BGE-M3 (RAG)
- FastAPI, uvicorn
- SQLite (실험 추적)

---

> 연구용 PoC입니다. 실제 임상 진단에 사용할 수 없습니다.
