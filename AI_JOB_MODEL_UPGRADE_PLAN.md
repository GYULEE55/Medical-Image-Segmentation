# AI Job Ready Model Upgrade Plan (Medical AI Project)

## Why this plan

현재 프로젝트는 "동작하는 데모"는 충분하지만, AI 엔지니어 채용 관점에서 더 강한 포인트가 필요합니다.
실무자는 보통 다음을 봅니다.

1. 모델 성능을 실제로 개선했는가
2. 개선 과정을 실험 기반으로 설명할 수 있는가
3. 배포/추론 최적화까지 해봤는가
4. 실패 케이스를 분석하고 다음 액션으로 연결했는가

이 문서는 위 4가지를 포트폴리오에서 바로 증명하기 위한 실행 계획입니다.

---

## Current baseline (as-is)

- Kvasir-SEG (polyp): mAP50 약 0.94, Recall 약 0.89
- DENTEX (dental): mAP50 약 0.34, Recall 약 0.24 (개선 여지 큼)
- API/Demo/RAG/VLM: 이미 구현 완료
- 목표: "UI 예쁜 데모"에서 "모델 개선 역량을 증명하는 프로젝트"로 전환

---

## Hiring manager가 혹하는 결과물 구조

아래 3개를 만들면 면접에서 강해집니다.

1. **Before vs After 실험표**
   - 예: YOLOv8n 기본 대비, augmentation + tuning + model size 변경 후 개선 수치
2. **실패 분석 리포트**
   - false negative/false positive 샘플, 원인, 대응 실험
3. **배포 최적화 리포트**
   - ONNX/CoreML 변환, latency/throughput 비교

---

## 4-Week execution plan (job-focused)

## Week 1 — Experiment discipline (필수)

### Task 1. 실험 트랙킹 표준화
- `db/experiment_db.py`에 모든 실험 결과를 누적
- 실험명 규칙: `dataset_modelsize_aug_opt_lr_imgsz_vX`
- 최소 기록 항목:
  - mAP50, mAP50-95, Precision, Recall
  - epoch, lr, batch, imgsz, optimizer
  - inference latency(ms)

### Task 2. 베이스라인 고정
- Kvasir, DENTEX 각각 baseline 재학습 1회
- seed 고정으로 재현성 확보

### Deliverable
- `reports/baseline_metrics.md`
- 면접 멘트: "재현 가능한 baseline부터 고정하고 실험을 시작했습니다."

---

## Week 2 — Model quality improvement (핵심)

### Task 3. Augmentation 실험
- Kvasir:
  - 기본 YOLO aug vs CLAHE 추가 vs 색/노이즈 강화
- DENTEX:
  - 기본 YOLO aug vs contrast/brightness 중심 강화

### Task 4. Model size 비교
- YOLOv8n-seg vs YOLOv8s-seg (필수)
- 가능하면 v8m까지 확장

### Task 5. Hyperparameter sweep
- lr0, patience, imgsz(640/768), conf threshold 영향 비교

### Deliverable
- `reports/model_quality_comparison.md`
- 그래프 2개:
  - mAP/Recall 비교
  - 속도-정확도 tradeoff
- 면접 멘트: "정확도만 보지 않고 latency까지 같이 최적화했습니다."

---

## Week 3 — Error analysis and medical trust

### Task 6. 실패 케이스 분석
- 클래스별 false negative 20개 수집
- false positive 20개 수집
- 원인 태깅:
  - low contrast
  - tiny lesion
  - boundary ambiguity
  - annotation noise

### Task 7. 임계값/후처리 전략
- conf/iou threshold 조합별 임상 안전성 비교
- 목표: recall 개선(특히 DENTEX)

### Deliverable
- `reports/error_analysis.md`
- 면접 멘트: "틀린 이유를 분해해서 다음 실험 가설로 연결했습니다."

---

## Week 4 — Deployment optimization (실무 어필)

### Task 8. Export + benchmark
- YOLO 모델을 ONNX로 export
- Mac 환경에서 CoreML export 시도
- 비교 지표:
  - model size
  - cold start
  - avg latency

### Task 9. Demo 연동
- 데모에 "모델 버전"과 "평균 추론시간" 표기
- 면접 시 "운영 관점" 설명 가능하게 구성

### Deliverable
- `reports/deployment_benchmark.md`
- 면접 멘트: "모델 성능뿐 아니라 운영 비용/속도까지 고려했습니다."

---

## Priority experiments (바로 시작 순서)

1. DENTEX recall 개선 (최우선)
   - 목표: Recall 0.24 -> 0.45+
2. YOLOv8n vs v8s 비교
3. Kvasir CLAHE 실험
4. ONNX export + latency 비교

---

## Resume/Interview bullet templates

아래 문장 구조로 작성하면 실무형으로 보입니다.

- "의료 영상 세그멘테이션 모델에서 baseline 대비 Recall을 X% 개선하여 미검출 위험을 낮춤"
- "YOLOv8n/s 모델 비교와 하이퍼파라미터 튜닝을 통해 정확도-속도 tradeoff 최적점 도출"
- "ONNX/CoreML 변환 및 추론 벤치마크를 수행해 배포 가능성을 검증"
- "RAG 근거 부족 시 답변 제한 로직을 적용해 의료 정보 환각 리스크를 제어"

---

## Acceptance criteria (완료 기준)

아래 5개가 충족되면 "AI 엔지니어 포트폴리오"로 충분히 강합니다.

1. 최소 20개 이상 실험 로그 정리
2. Before/After 수치가 명확한 개선 2개 이상
3. 실패 케이스 분석 문서 1개
4. ONNX/CoreML 벤치마크 결과 1개
5. README에 "문제 -> 해결 -> 성과" 구조 반영

---

## Immediate next action (today)

1. DENTEX baseline 재학습 1회
2. DENTEX augmentation 강화 실험 1회
3. 결과를 `db/experiment_db.py`와 `reports/baseline_metrics.md`에 기록

이 3개만 오늘 끝내도, 면접에서 "모델 개선을 실제로 실행한 사람"으로 보이기 시작합니다.
