# Clinical Problem Map (현장 문제-해결 매핑)

이 문서는 "무엇을 만들었는가"보다 "왜 만들었는가"를 설명하기 위한 문서입니다.
면접/포트폴리오에서 아래 표만 보여줘도 프로젝트 방향성이 명확해집니다.

---

## 1) 실제 현장 문제

| 현장 문제 | 왜 문제인가 | 현재 프로젝트에서 한 해결 | 남은 과제 |
|---|---|---|---|
| 작은 병변 놓침 가능성 | 조기 발견 실패로 이어질 수 있음 | YOLOv8 기반 병변 검출 + Recall 중심 평가 | 다기관 데이터로 일반화 검증 |
| 판독 업무 과부하 | 영상량 증가, 인력 부담 증가 | 이미지 1회 업로드로 검출+해석+근거 통합 응답 | PACS/EHR 연동으로 클릭 수 최소화 |
| "탐지됨"만으로는 신뢰 부족 | 임상에서는 설명과 출처가 중요 | RAG로 문헌 출처(파일/페이지) 동시 반환 | 문장 단위 citation 강제 |
| LLM 환각/과장 답변 위험 | 의료 영역에서 오답 리스크 큼 | no-evidence 가드: 근거 없으면 고정 문구 반환 | 임계치 튜닝/감사 로그 고도화 |
| 사용자 이해도 격차 | 개발자/임상의가 보는 포인트 다름 | 요약 카드 + 배지(RAG OK/WARN) 제공 | 초보자/전문가 모드 분리 |

---

## 2) 기술 선택 근거 (현장 제약 기반)

### YOLOv8n-seg
- 이유: 실시간성, 경량(배포 용이), detection+segmentation 동시 처리
- 현장 제약 대응: GPU 없는 환경에서도 상대적으로 운영 가능

### RAG (Chroma + BGE-M3 + LangChain)
- 이유: "모델 내부 지식"이 아니라 "문서 근거" 중심으로 답변
- 현장 제약 대응: 근거 없는 답변을 줄이고 검증 가능성 확보

### VLM (LLaVA via Ollama)
- 이유: 검출 클래스만으로 놓치는 정성적 소견 보완
- 현장 제약 대응: 정량 검출 + 정성 해석 결합

### FastAPI + Docker
- 이유: 배포/재현성/검증 자동화 용이
- 현장 제약 대응: 환경차로 인한 재현 실패 감소

---

## 3) 사용자 친화 설계 원칙

- 어려운 용어보다 "의사결정에 필요한 정보"를 먼저 보여준다.
- 답변을 한 덩어리로 주지 않고, `검출/해석/근거`로 분리한다.
- 근거가 없으면 그 사실을 먼저 명확히 말한다.
- 실패 시에도 "왜 실패했는지"를 사용자 문장으로 알려준다.

---

## 4) 이 프로젝트의 한 줄 문제해결력

"의료 영상 AI의 핵심 문제인 **놓침 위험, 설명 부족, 근거 부재**를 줄이기 위해,
검출-해석-근거를 하나의 흐름으로 연결하고 no-evidence 가드로 안전성을 높인 프로젝트"

---

## 5) 근거 레퍼런스 (README에 요약 반영 가능)

1. Van Rijn et al. Polyp miss rate meta-analysis (tandem colonoscopy)
   - https://pubmed.ncbi.nlm.nih.gov/16716777/
2. WHO. Ethics and governance of AI for health
   - https://www.who.int/publications/i/item/9789240029200
3. WHO. Guidance for LMM use in health (2024/2025)
   - https://www.who.int/publications/i/item/9789240084759
4. Neiman HPI. Radiologist shortage projection
   - https://www.neimanhpi.org/press-releases/new-study-projects-radiologist-shortage-through-2055/
5. ECRI. Top Health Technology Hazards (AI governance risk)
   - https://www.ecri.org/top-10-health-technology-hazards-2025

주의: 본 프로젝트는 임상 배포 제품이 아니라 PoC입니다. 실제 도입에는 인허가, 비식별화, 병원 시스템 연동이 필요합니다.
