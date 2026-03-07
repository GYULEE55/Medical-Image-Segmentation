# Resume Project Description — Medical Image Segmentation AI

## STAR Format (English)

### Situation
Medical professionals face increasing workload in image interpretation, with studies showing polyp miss rates in colonoscopy and the need for AI-assisted diagnostic tools. Existing AI solutions often provide detection results without interpretable evidence, limiting clinical trust and adoption.

### Task
Design and implement an end-to-end medical AI system that goes beyond simple detection — providing quantitative findings (YOLOv8), qualitative interpretation (VLM), and literature-backed evidence (RAG) in a single unified pipeline.

### Action
- **V1**: Built YOLOv8n-seg inference API with FastAPI, supporting multi-model inference (polyp + dental)
- **V2**: Integrated RAG pipeline (LangChain + ChromaDB + BGE-M3) for evidence-based medical Q&A
- **V3**: Connected detection results to RAG for automatic query generation from findings
- **V4**: Added VLM (LLaVA via Ollama) for direct image interpretation, async job processing, and Prometheus monitoring
- **Engineering**: Implemented structured logging (structlog), SQLite experiment tracking, Docker containerization, GitHub Actions CI, and 36-test pytest suite

### Result
- Kvasir-SEG polyp segmentation: **mAP@50 = 0.942** (50 epochs, Colab T4)
- DENTEX dental detection: mAP@50 = 0.344 (4-class, 700 images — documented challenges)
- Production-ready API: 9 endpoints, async processing, Prometheus metrics
- Test coverage: 36 tests passing, 0 failures
- Containerized: Docker + docker-compose, GitHub Actions CI/CD

---

## 이력서용 설명 (한국어)

### 프로젝트명
의료 영상 AI 어시스턴트 — YOLOv8 + RAG + VLM 통합 시스템

### 한 줄 요약
내시경/X-ray 영상에서 병변을 검출하고, VLM으로 해석하며, RAG로 문헌 근거를 제공하는 End-to-End 의료 AI PoC

### 주요 성과
- Kvasir-SEG 폴립 세그멘테이션: **mAP@50 = 0.942** (YOLOv8n-seg, 50 epochs)
- FastAPI 기반 9개 엔드포인트 (동기/비동기 처리, Prometheus 모니터링)
- LangChain + ChromaDB + BGE-M3 RAG 파이프라인 구축
- LLaVA VLM 통합 (Ollama REST API)
- pytest 36개 테스트, Docker 컨테이너화, GitHub Actions CI/CD

### 기술 스택
YOLOv8, FastAPI, LangChain, ChromaDB, LLaVA, Ollama, Docker, pytest, Prometheus, structlog, SQLite

---

## ATS Keywords (for resume optimization)

**ML/AI**: YOLOv8, Instance Segmentation, Object Detection, Transfer Learning, ONNX Export, Model Evaluation, mAP, Precision, Recall

**RAG/LLM**: LangChain, ChromaDB, BGE-M3, RAG, Vector Database, Embeddings, GPT-4o-mini, LLaVA, VLM, Ollama

**Engineering**: FastAPI, Docker, GitHub Actions, CI/CD, pytest, Prometheus, structlog, SQLite, REST API, Async Processing

**Data**: Kvasir-SEG, DENTEX, COCO Format, Data Preprocessing, Dataset Preparation, Medical Imaging

---

## Interview Q&A

**Q: 왜 YOLOv8을 선택했나요?**
A: 실시간 추론 속도, Detection + Segmentation 동시 처리, 소규모 데이터에서의 전이학습 효율성을 고려했습니다. 의료 현장의 워크플로우 통합을 위해 경량 배포(CPU)가 가능한 모델이 필요했고, YOLOv8n-seg가 이 조건을 가장 잘 충족했습니다.

**Q: DENTEX 성능이 낮은 이유는?**
A: 700장의 제한된 데이터로 4개 클래스를 학습하는 도전적인 설정이었습니다. 단일 클래스 Kvasir-SEG(1000장)와 비교하면 클래스당 데이터가 약 175장으로 부족하고, 파노라마 X-ray의 저대비 특성과 클래스 불균형이 복합적으로 작용했습니다. 이를 문서화하고 개선 로드맵을 제시했습니다.

**Q: RAG를 직접 구축한 이유는?**
A: 의료 도메인에서는 출처 없는 AI 응답이 신뢰성을 떨어뜨립니다. 최신 가이드라인을 반영하고 검증 가능한 설명을 제공하기 위해 ChromaDB + BGE-M3 + LangChain LCEL 스택으로 직접 구축했습니다.

**Q: 가장 어려웠던 기술적 도전은?**
A: 1206줄의 모놀리식 app.py를 라우터 기반 구조로 리팩토링하면서 36개 테스트를 모두 통과시키는 것이었습니다. 공유 상태(YOLO 모델 객체, async_jobs 딕셔너리)를 FastAPI의 app.state 패턴으로 안전하게 분리했습니다.

**Q: 프로덕션 배포 시 추가할 것은?**
A: FDA/CE 인증 프로세스, 독립 테스트셋 검증, 분산 학습, 모델 버전 관리(MLflow), 실시간 모니터링 대시보드, 그리고 의료진 피드백 루프를 통한 지속적 개선 파이프라인을 추가하겠습니다.
