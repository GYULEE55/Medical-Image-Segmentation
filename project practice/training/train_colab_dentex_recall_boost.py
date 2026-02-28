"""
[Colab용] DENTEX Recall 개선 실험 스크립트 (실무/면접 어필 버전)

목적:
  - 현재 DENTEX의 낮은 Recall(약 0.24) 문제를 개선하기 위한
    2개 실험을 같은 포맷으로 빠르게 반복 실행.

실험 구성:
  1) baseline 재현: 현재 설정을 그대로 다시 학습
  2) recall_boost: Recall 개선 중심 설정 (imgsz↑, patience↑, optimizer/lr 조정, augment 강화)

실무 포인트:
  - "좋아 보이는 데모"가 아니라 "왜 이 실험을 했는지"가 명확해야 함
  - 실험 이름과 설정을 구조화해 나중에 비교/보고서 작성이 쉽게 설계

사용법:
  - Google Colab에서 셀 단위로 실행
  - 마지막 셀에서 results.csv를 로컬 프로젝트로 가져와 DB에 기록
"""

"""
====================================
[셀 1] 환경 세팅
====================================
"""
# !nvidia-smi
# !pip install ultralytics

"""
====================================
[셀 2] Google Drive 마운트
====================================
"""
# from google.colab import drive
# drive.mount('/content/drive')

"""
====================================
[셀 3] 데이터 준비 확인
====================================
전제:
  - /content/dentex.yaml 생성 완료
  - train/val YOLO 포맷 변환 완료
"""
# import os
# assert os.path.exists('/content/dentex.yaml'), 'dentex.yaml 없음'
# print('OK: /content/dentex.yaml 확인')

"""
====================================
[셀 4] 공통 함수
====================================
"""
# from ultralytics import YOLO
#
# def run_experiment(exp_name: str, train_kwargs: dict):
#     model = YOLO('yolov8n-seg.pt')
#     print(f'\n===== RUN: {exp_name} =====')
#     print(train_kwargs)
#     result = model.train(**train_kwargs)
#     print(f'===== DONE: {exp_name} =====\n')
#     return result

"""
====================================
[셀 5] 실험 1 — baseline 재현
====================================
"""
# baseline_kwargs = dict(
#     data='/content/dentex.yaml',
#     epochs=100,
#     imgsz=640,
#     batch=16,
#     device=0,
#     optimizer='SGD',
#     lr0=0.01,
#     patience=15,
#     cos_lr=True,
#     conf=0.25,
#     save=True,
#     project='/content/drive/MyDrive/yolo-results',
#     name='dentex-baseline-v2',
# )
#
# run_experiment('dentex-baseline-v2', baseline_kwargs)

"""
====================================
[셀 6] 실험 2 — recall_boost
====================================
핵심 의도:
  - 작은 병변 미검출(False Negative) 줄이기

변경점:
  - imgsz: 640 -> 768 (작은 병변 가시성 개선)
  - optimizer: SGD -> AdamW (초기 수렴 안정화)
  - lr0: 0.01 -> 0.003 (과격한 업데이트 완화)
  - patience: 15 -> 25 (느린 수렴 허용)
  - augment 강화: rotate/scale/translate, fliplr, mosaic 조정
"""
# recall_boost_kwargs = dict(
#     data='/content/dentex.yaml',
#     epochs=140,
#     imgsz=768,
#     batch=12,          # imgsz 증가로 메모리 여유 확보
#     device=0,
#     optimizer='AdamW',
#     lr0=0.003,
#     lrf=0.0005,
#     weight_decay=0.0005,
#     patience=25,
#     cos_lr=True,
#     # ---- recall 개선용 augmentation ----
#     degrees=8.0,
#     translate=0.12,
#     scale=0.35,
#     shear=2.0,
#     perspective=0.0,
#     fliplr=0.5,
#     flipud=0.0,
#     mosaic=0.5,
#     mixup=0.1,
#     hsv_h=0.01,
#     hsv_s=0.2,
#     hsv_v=0.2,
#     conf=0.20,
#     save=True,
#     project='/content/drive/MyDrive/yolo-results',
#     name='dentex-recall-boost-v1',
# )
#
# run_experiment('dentex-recall-boost-v1', recall_boost_kwargs)

"""
====================================
[셀 7] 최종 검증 + 클래스별 출력
====================================
"""
# from ultralytics import YOLO
#
# def print_eval(run_name: str):
#     best_path = f'/content/drive/MyDrive/yolo-results/{run_name}/weights/best.pt'
#     model = YOLO(best_path)
#     m = model.val(data='/content/dentex.yaml', imgsz=768 if 'recall-boost' in run_name else 640, device=0)
#
#     print(f'\n===== {run_name} =====')
#     print(f'Box  mAP50:    {m.box.map50:.4f}')
#     print(f'Box  Recall:   {m.box.r:.4f}')
#     print(f'Mask mAP50:    {m.seg.map50:.4f}')
#     print(f'Mask Recall:   {m.seg.r:.4f}')
#
#     classes = ['Impacted', 'Caries', 'Periapical Lesion', 'Deep Caries']
#     print('--- Per-class Box AP50 ---')
#     for i, c in enumerate(classes):
#         ap50 = m.box.ap50[i] if i < len(m.box.ap50) else 0.0
#         print(f'{c:24s}: {ap50:.4f}')
#
# print_eval('dentex-baseline-v2')
# print_eval('dentex-recall-boost-v1')

"""
====================================
[셀 8] 결과 파일 복사 (DB 적재용)
====================================
"""
# import os
# import shutil
#
# for run_name in ['dentex-baseline-v2', 'dentex-recall-boost-v1']:
#     src = f'/content/drive/MyDrive/yolo-results/{run_name}/results.csv'
#     dst = f'/content/drive/MyDrive/yolo-results/{run_name}/{run_name}-results.csv'
#     if os.path.exists(src):
#         shutil.copy2(src, dst)
#         print(f'[OK] copied: {dst}')
#     else:
#         print(f'[WARN] missing: {src}')

"""
====================================
[셀 9] 면접용 한 줄 정리 템플릿
====================================
"""
# print('''
# [면접용 요약]
# - baseline 대비 recall_boost 실험에서 Recall이 개선되었는지 수치로 검증했습니다.
# - 단순 mAP가 아니라 Recall/클래스별 AP를 함께 비교해 미검출 리스크를 줄이는 방향으로 최적화했습니다.
# ''')
