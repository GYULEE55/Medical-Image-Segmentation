"""
[Colab용] DENTEX (치과 X-ray) YOLOv8 세그멘테이션 학습 스크립트
Google Colab에서 셀별로 복사해서 실행하세요.

DENTEX 데이터셋:
  - 치과 파노라마 X-ray 영상
  - 4개 클래스: Impacted(매복치), Caries(충치),
               Periapical Lesion(치근단병변), Deep Caries(깊은충치)

전제 조건:
  1. Google Drive에 training_data.zip이 있어야 함
  2. prepare_dataset_dentex.py로 YOLO 포맷 변환이 완료되어야 함
     (또는 아래 셀 3에서 직접 변환 실행)

====================================
[셀 1] 환경 세팅 — GPU 확인 & 라이브러리 설치
====================================
"""
# !nvidia-smi  # GPU 확인 (T4 or V100 나와야 함)
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
[셀 3] 데이터 전처리 (COCO → YOLO 변환)
====================================
prepare_dataset_dentex.py를 실행합니다.
이미 변환했으면 이 셀은 건너뛰세요.

⚠️ DRIVE_BASE 경로를 본인 Google Drive에 맞게 수정하세요!
"""
# # 방법 1: 스크립트 직접 실행 (Drive에 스크립트가 있을 때)
# %run "/content/drive/MyDrive/Python.Algrithm/project practice/preprocessing/prepare_dataset_dentex.py"
#
# # 방법 2: 직접 코드 실행 (스크립트 없을 때)
# import sys
# sys.path.append('/content/drive/MyDrive/Python.Algrithm/project practice/preprocessing')
# from prepare_dataset_dentex import main
# main()

"""
====================================
[셀 4] 데이터셋 경로 확인
====================================
변환된 YOLO 데이터셋의 구조를 확인합니다.

⚠️ DATASET_ROOT를 prepare_dataset_dentex.py의 OUTPUT_DIR에 맞추세요!
"""
# import os
#
# DATASET_ROOT = '/content/drive/MyDrive/datasets/dentex'
#
# # 폴더 구조 확인
# for split in ['train', 'val']:
#     img_dir = f'{DATASET_ROOT}/images/{split}'
#     lbl_dir = f'{DATASET_ROOT}/labels/{split}'
#     if os.path.exists(img_dir):
#         img_count = len(os.listdir(img_dir))
#         lbl_count = len(os.listdir(lbl_dir))
#         print(f'{split}: images={img_count}, labels={lbl_count}')
#     else:
#         print(f'{split}: 폴더 없음! 전처리(셀 3)를 먼저 실행하세요.')

"""
====================================
[셀 5] YAML 설정 파일 생성
====================================
YOLOv8이 데이터셋을 찾을 수 있도록 경로를 지정하는 설정 파일.

DENTEX는 4개 클래스:
  0: Impacted        — 매복치 (뼈/잇몸 속에 묻힌 치아)
  1: Caries           — 충치
  2: Periapical Lesion — 치근단병변 (치아 뿌리 끝 감염)
  3: Deep Caries      — 깊은충치 (신경 근처까지 진행)

Kvasir-SEG(1클래스)와 달리 4클래스 → 클래스 불균형 주의!
"""
# yaml_content = f"""
# path: {DATASET_ROOT}
# train: images/train
# val: images/val
#
# nc: 4
# names:
#   0: Impacted
#   1: Caries
#   2: Periapical Lesion
#   3: Deep Caries
# """
#
# with open('/content/dentex.yaml', 'w') as f:
#     f.write(yaml_content)
# print('YAML 설정 파일 생성 완료')

"""
====================================
[셀 6] 학습 실행
====================================
핵심 파라미터 설명 (Kvasir-SEG와 다른 점 ★ 표시):

★ 데이터 특성 차이:
  - Kvasir-SEG: 1000장, 1클래스 (polyp), 내시경 컬러 이미지
  - DENTEX: ~700장, 4클래스, X-ray 흑백 이미지
  → 데이터가 적고 클래스가 많으므로 더 신중한 학습 필요

- model: yolov8n-seg.pt (nano — 적은 데이터에 큰 모델은 과적합 위험)
- epochs: 100 (★ 50→100, 4클래스라 더 많은 학습 필요)
- imgsz: 640 (파노라마 X-ray는 원래 넓지만 640이 표준)
- batch: 16 (Colab T4 기준)
- patience: 15 (★ 10→15, 4클래스 수렴에 시간 필요)
- lr0: 0.01 (기본값, SGD)
- cos_lr: True (★ 추가 — Cosine LR로 후반 미세 조정)
"""
# from ultralytics import YOLO
#
# model = YOLO('yolov8n-seg.pt')  # 사전학습 모델 로드
#
# results = model.train(
#     data='/content/dentex.yaml',
#     epochs=100,                   # ★ 4클래스 → 더 많은 epoch
#     imgsz=640,
#     batch=16,
#     device=0,
#     patience=15,                  # ★ 15 epoch 연속 개선 없으면 중단
#     save=True,
#     cos_lr=True,                  # ★ Cosine LR 스케줄러 (후반 미세 조정)
#     project='/content/drive/MyDrive/yolo-results',
#     name='dentex-v1',
# )

"""
====================================
[셀 7] 학습 결과 확인
====================================
학습이 끝나면 자동으로 생성되는 결과물:
- confusion_matrix.png: 4x4 혼동 행렬 (클래스별 혼동 패턴 확인!)
- results.png: epoch별 loss/mAP 그래프
- weights/best.pt: 가장 성능 좋은 모델
- weights/last.pt: 마지막 epoch 모델

★ 4클래스라서 confusion matrix가 더 중요!
  → 어떤 클래스끼리 혼동하는지 확인 (예: Caries vs Deep Caries)
"""
# from IPython.display import Image, display
#
# result_dir = '/content/drive/MyDrive/yolo-results/dentex-v1'
# display(Image(filename=f'{result_dir}/results.png', width=800))
# display(Image(filename=f'{result_dir}/confusion_matrix.png', width=500))

"""
====================================
[셀 8] 검증 데이터로 평가 (클래스별 성능)
====================================
★ 4클래스이므로 전체 mAP뿐 아니라 클래스별 mAP도 확인!
  - Impacted(매복치): 크기가 커서 잘 검출될 가능성 높음
  - Deep Caries(깊은충치): Caries와 혼동 가능 → 주의
"""
# model_best = YOLO(f'{result_dir}/weights/best.pt')
#
# metrics = model_best.val(
#     data='/content/dentex.yaml',
#     imgsz=640,
#     device=0,
# )
#
# # 전체 성능
# print("=" * 50)
# print("전체 성능")
# print("=" * 50)
# print(f"mAP50 (Box):     {metrics.box.map50:.4f}")
# print(f"mAP50-95 (Box):  {metrics.box.map:.4f}")
# print(f"mAP50 (Mask):    {metrics.seg.map50:.4f}")
# print(f"mAP50-95 (Mask): {metrics.seg.map:.4f}")
#
# # ★ 클래스별 성능 (4클래스)
# class_names = ['Impacted', 'Caries', 'Periapical Lesion', 'Deep Caries']
# print("\n" + "=" * 50)
# print("클래스별 성능 (mAP50)")
# print("=" * 50)
# for i, name in enumerate(class_names):
#     box_ap = metrics.box.ap50[i] if i < len(metrics.box.ap50) else 0
#     seg_ap = metrics.seg.ap50[i] if i < len(metrics.seg.ap50) else 0
#     print(f"  {name:25s} | Box: {box_ap:.4f} | Mask: {seg_ap:.4f}")

"""
====================================
[셀 9] 샘플 이미지로 추론 테스트
====================================
"""
# import glob
# test_images = glob.glob(f'{DATASET_ROOT}/images/val/*.png')[:5]
# if not test_images:
#     test_images = glob.glob(f'{DATASET_ROOT}/images/val/*.jpg')[:5]
#
# results = model_best.predict(
#     source=test_images,
#     save=True,
#     conf=0.25,
#     device=0,
#     project='/content/drive/MyDrive/yolo-results',
#     name='dentex-v1-predict',
# )
#
# # 결과 이미지 표시
# import glob as g
# pred_images = sorted(g.glob('/content/drive/MyDrive/yolo-results/dentex-v1-predict/*'))
# for img_path in pred_images[:3]:
#     display(Image(filename=img_path, width=600))

"""
====================================
[셀 10] results.csv를 Google Drive에 복사
====================================
experiment_db.py로 DB에 저장하기 위해 results.csv를 Drive에 복사합니다.
"""
# import shutil
# src = f'{result_dir}/results.csv'
# dst = '/content/drive/MyDrive/yolo-results/dentex-v1/results.csv'
# if os.path.exists(src):
#     shutil.copy2(src, dst)
#     print(f'results.csv 복사 완료: {dst}')
# else:
#     print('results.csv를 찾을 수 없습니다')

"""
====================================
[셀 11] best.pt를 로컬 프로젝트에 사용하기 위해 복사
====================================
학습된 모델을 FastAPI 서버에서 사용하려면
best.pt를 프로젝트 폴더에 넣어야 합니다.

Colab에서 직접 다운로드하거나, Drive에서 복사하세요.
"""
# # Google Drive에 별도 저장
# shutil.copy2(
#     f'{result_dir}/weights/best.pt',
#     '/content/drive/MyDrive/yolo-results/dentex-best.pt'
# )
# print('best.pt 복사 완료 — 로컬에서 다운로드 후 project practice/best_dentex.pt로 저장하세요')
