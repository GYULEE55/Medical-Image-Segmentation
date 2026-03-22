# Medical Image AI Assistant

YOLOv8 기반 병변 탐지와 LLaVA VLM 해석, RAG 문헌 근거를 하나의 파이프라인으로 연결한 의료 영상 보조 진단 시스템입니다.

---

## 왜 만들었나

기존 의료 AI는 "탐지됨"만 알려줍니다. 임상에서는 탐지 결과와 함께 설명과 근거가 필요합니다.

- **검출(YOLO)** → 병변 위치와 클래스
- **해석(VLM)** → 해당 병변이 어떤 형태인지 정성 설명
- **근거(RAG)** → 관련 논문 기반 답변 + 출처 반환

근거가 없으면 LLM이 답변하는 대신 고정 문구를 반환합니다(no-evidence 가드).

---

## 성능

**Kvasir-SEG (폴립 세그멘테이션)**

| 지표 | 결과 |
|------|------|
| Precision | 0.920 |
| Recall | 0.887 |
| mAP@50 (Mask) | **0.942** |

**DENTEX (치과 X-ray, 4-class)**

| 지표 | 결과 |
|------|------|
| mAP@50 (Mask) | 0.344 |

> DENTEX 성능이 낮은 이유는 데이터 부족(클래스당 ~175장)과 파노라마 X-ray 저대비 문제입니다. 원인 분석 문서 → [docs/DENTEX_ANALYSIS.md](docs/DENTEX_ANALYSIS.md)

---

## 주요 기능

- YOLOv8n-seg 기반 위장내시경 / 치과 X-ray 병변 탐지
- LLaVA(Ollama) 기반 병변 정성 해석 (비동기 처리)
- LangGraph StateGraph 기반 RAG 파이프라인 (no-evidence 가드)
- PubMed 논문 38개 + PDF 문서 인덱싱 (ChromaDB)
- FastAPI 기반 REST API

---

## 빠른 시작

```bash
git clone https://github.com/GYULEE55/Medical-Image-Segmentation.git
cd Medical-Image-Segmentation
pip install -e .
cp .env.example .env
make ingest   # RAG 문서 인덱싱 (최초 1회)
make serve    # API 서버 실행
```

---

## API 엔드포인트

| 엔드포인트 | 설명 |
|-----------|------|
| `POST /predict` | YOLOv8 병변 탐지 |
| `POST /ask` | RAG 기반 의료 Q&A |
| `POST /analyze` | 탐지 + RAG 자동 연동 |
| `POST /vlm-analyze/async` | 비동기 VLM 이미지 해석 |
| `GET /jobs/{job_id}` | 비동기 작업 결과 조회 |

---

## 기술 스택

- **탐지**: YOLOv8n-seg (Ultralytics)
- **VLM**: LLaVA via Ollama
- **RAG**: LangGraph, ChromaDB, BGE-M3
- **API**: FastAPI, uvicorn
- **실험 추적**: SQLite

---

## 주의사항

연구용 PoC입니다. 실제 임상 진단에 사용할 수 없습니다.
