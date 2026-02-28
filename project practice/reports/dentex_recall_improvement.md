# DENTEX Recall Improvement Report

## 1) 실무 문제 정의

현재 치과 X-ray 세그멘테이션 모델은 Precision 대비 Recall이 낮습니다.
즉, "맞춘 것"은 비교적 정확하지만 "놓치는 병변"이 많을 수 있습니다.

- baseline reference (기존 기록):
  - Box Recall: ~0.24
  - Box mAP50: ~0.34

실무에서 이 문제는 "미검출 위험"으로 연결됩니다.

---

## 2) 실험 가설

가설: 이미지 해상도와 학습 안정화(optimizer/lr/patience), augment 설정을 조정하면 작은 병변 검출이 개선되어 Recall이 오른다.

---

## 3) 실험 설계

### Exp-A: baseline 재현
- script: `training/train_colab_dentex_recall_boost.py` (셀 5)
- run name: `dentex-baseline-v2`
- 핵심 설정:
  - imgsz=640, optimizer=SGD, lr0=0.01, patience=15

### Exp-B: recall_boost
- script: `training/train_colab_dentex_recall_boost.py` (셀 6)
- run name: `dentex-recall-boost-v1`
- 핵심 설정:
  - imgsz=768
  - optimizer=AdamW, lr0=0.003, lrf=0.0005
  - patience=25
  - rotate/scale/translate/fliplr/mosaic/mixup 조정

---

## 4) 결과 기록 (채우는 칸)

| 실험명 | Box Recall | Box mAP50 | Mask Recall | Mask mAP50 | 비고 |
|---|---:|---:|---:|---:|---|
| dentex-baseline-v2 |  |  |  |  | baseline |
| dentex-recall-boost-v1 |  |  |  |  | recall 개선 실험 |

---

## 5) 결론 템플릿 (면접용)

- "baseline 대비 recall_boost 실험에서 **Recall을 X%p 개선**했습니다."
- "정확도만이 아니라 미검출 리스크(Recall)를 핵심 목표로 잡고 최적화했습니다."
- "실험 결과를 DB로 적재해 재현 가능한 비교 형태로 관리했습니다."

---

## 6) DB 적재 커맨드

baseline:
```bash
"/Users/iseung-gyu/bmad-project/i want to get a job!!/.venv/bin/python" training/log_experiment_results.py \
  --name dentex-baseline-v2 \
  --model yolov8n-seg \
  --dataset dentex \
  --results-csv training/results_dentex.csv \
  --epochs 100 --batch-size 16 --imgsz 640 --lr 0.01 --optimizer SGD \
  --notes "baseline reproduction"
```

recall_boost:
```bash
"/Users/iseung-gyu/bmad-project/i want to get a job!!/.venv/bin/python" training/log_experiment_results.py \
  --name dentex-recall-boost-v1 \
  --model yolov8n-seg \
  --dataset dentex \
  --results-csv /path/to/dentex-recall-boost-v1-results.csv \
  --epochs 140 --batch-size 12 --imgsz 768 --lr 0.003 --optimizer AdamW \
  --notes "recall-focused tuning"
```
