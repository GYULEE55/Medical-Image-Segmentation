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

## 4) 결과 기록

| 실험명 | Box Recall | Box mAP50 | Mask Recall | Mask mAP50 | 비고 |
|---|---:|---:|---:|---:|---|
| dentex-baseline-v2 | 0.12745 | 0.23932 | 0.12745 | 0.22950 | baseline |
| dentex-recall-boost-v1 | 0.18627 | 0.18147 | 0.17157 | 0.17038 | recall 개선 실험 |
| dentex-balance-recover-v1 | 0.22157 | 0.21004 | 0.20686 | 0.18354 | recall 유지 + mAP 회복 균형 |

요약:
- Box Recall: **+0.05882p** (약 +46.1% 상대 개선)
- Mask Recall: **+0.04412p** (약 +34.6% 상대 개선)
- 다만 mAP50은 하락 → 다음 라운드에서 precision/mAP 회복 튜닝 필요

balance_recover 반영 후 요약:
- Box Recall: baseline 대비 **+0.09412p** (0.12745 -> 0.22157, 약 +73.8% 상대 개선)
- Mask Recall: baseline 대비 **+0.07941p** (0.12745 -> 0.20686, 약 +62.3% 상대 개선)
- mAP50(B): recall_boost(0.18147) 대비 **회복**(0.21004), baseline(0.23932) 대비는 아직 낮음
- 실무 적용 관점 1차 결론: **미검출 감소가 우선이면 balance_recover가 현재 최선**

---

## 5) 결론 템플릿 (면접용)

- "baseline 대비 recall_boost 실험에서 **Box Recall을 0.127 -> 0.186(+0.059p)**로 개선했습니다."
- "정확도만이 아니라 미검출 리스크(Recall)를 핵심 목표로 잡고 최적화했습니다."
- "실험 결과를 DB로 적재해 재현 가능한 비교 형태로 관리했고, mAP 하락 trade-off도 다음 실험 과제로 명확히 정의했습니다."
- "3차 balance_recover 실험으로 Box Recall을 **0.221까지 추가 개선**하고, 하락했던 mAP를 일부 회복했습니다."

---

## 6) DB 적재 커맨드

baseline:
```bash
PYTHONPATH="." "/Users/iseung-gyu/bmad-project/i want to get a job!!/.venv/bin/python" training/log_experiment_results.py \
  --name dentex-baseline-v2 \
  --model yolov8n-seg \
  --dataset dentex \
  --results-csv ../compare_csc/baseresults.csv \
  --epochs 100 --batch-size 16 --imgsz 640 --lr 0.01 --optimizer SGD \
  --notes "baseline reproduction"
```

recall_boost:
```bash
PYTHONPATH="." "/Users/iseung-gyu/bmad-project/i want to get a job!!/.venv/bin/python" training/log_experiment_results.py \
  --name dentex-recall-boost-v1 \
  --model yolov8n-seg \
  --dataset dentex \
  --results-csv ../compare_csc/results.csv \
  --epochs 140 --batch-size 12 --imgsz 768 --lr 0.003 --optimizer AdamW \
  --notes "recall-focused tuning"
```
