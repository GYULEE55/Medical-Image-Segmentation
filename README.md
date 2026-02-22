# Medical AI Assistant — Vision + LLM 통합 서비스

> 의료 영상 병변 검출(YOLOv8) + 의료 지식 RAG(LangChain/GPT-4o-mini) 통합 파이프라인

의료 영상에서 병변을 검출하고, 검출 결과를 기반으로 **논문/가이드라인 기반 의료 지식을 자동 제공**하는 End-to-End 의료 AI 서비스입니다.

**멀티모델 지원**: 위장 내시경(폴립) + 치과 X-ray(4개 치과 질환) — `model_type` 파라미터로 선택

📁 **프로젝트 코드**: [`project practice/`](./project%20practice/) 폴더에서 확인하세요.

📄 **상세 문서**: [`project practice/README.md`](./project%20practice/README.md)

---

## Quick Overview

### 핵심 기능: Vision + LLM 통합 (`POST /analyze`)

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
               └─ 의료 지식: 임상적 의미, 추적 관찰 주기, 환자 안내사항 + 출처
```

### Results

| Model | Dataset | mAP@50 (Mask) | Precision | Recall |
|-------|---------|:---:|:---:|:---:|
| **Polyp** | Kvasir-SEG (1,000장) | **0.942** | 0.920 | 0.887 |
| **Dental** | DENTEX (~700장, 4클래스) | **0.344** | 0.485 | 0.334 |

### Tech Stack

| 분류 | 기술 | 선택 이유 |
|------|------|----------|
| **Model** | YOLOv8n-seg | 실시간 추론, CPU 배포 가능, 6.5MB 경량 |
| **Serving** | FastAPI + Uvicorn | async, 자동 Swagger 문서 |
| **RAG** | LangChain LCEL + ChromaDB | 최신 패턴, 한국어+영어 |
| **Embedding** | BAAI/bge-m3 | 다국어, 의료 문서 최적화 |
| **LLM** | GPT-4o-mini | 저렴 + 충분한 품질 |
| **Infra** | Docker (CPU-only) | GPU 없는 병원 환경 배포 |
| **Test** | pytest 25개 | CI 호환 설계 |
