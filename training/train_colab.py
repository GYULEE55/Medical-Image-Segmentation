"""
[Colab용] YOLOv8 세그멘테이션 학습 스크립트
Google Colab에서 셀별로 복사해서 실행하세요.

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
[셀 3] 데이터셋 경로 확인
====================================
Drive에 kvasir-seg 폴더를 업로드한 위치에 맞게 수정하세요.
"""
# import os
# DATASET_ROOT = '/content/drive/MyDrive/kvasir-seg'
#
# # 폴더 구조 확인
# for split in ['train', 'val']:
#     img_count = len(os.listdir(f'{DATASET_ROOT}/images/{split}'))
#     lbl_count = len(os.listdir(f'{DATASET_ROOT}/labels/{split}'))
#     print(f'{split}: images={img_count}, labels={lbl_count}')

"""
====================================
[셀 4] YAML 설정 파일 생성
====================================
YOLOv8이 데이터셋을 찾을 수 있도록 경로를 지정하는 설정 파일.
- path: 데이터셋 루트 폴더
- train/val: 이미지 폴더 경로 (path 기준 상대경로)
- names: 클래스 이름 (0번 = polyp)
"""
# yaml_content = f"""
# path: {DATASET_ROOT}
# train: images/train
# val: images/val
#
# names:
#   0: polyp
# """
#
# with open('/content/kvasir-seg.yaml', 'w') as f:
#     f.write(yaml_content)
# print('YAML 설정 파일 생성 완료')

"""
====================================
[셀 5] 학습 실행
====================================
핵심 파라미터 설명:
- model: yolov8n-seg.pt = nano 세그멘테이션 모델 (가장 가벼움)
- data: 위에서 만든 YAML 설정 파일
- epochs: 학습 반복 횟수 (50 = 빠른 테스트, 100 = 본격 학습)
- imgsz: 입력 이미지 크기 (640이 표준)
- batch: 한 번에 처리하는 이미지 수 (Colab T4는 16 가능)
- device: 0 = GPU 사용
- patience: N epoch 동안 성능 안 오르면 조기 중단
- save: 체크포인트 저장 여부
- project/name: 결과 저장 폴더
"""
# from ultralytics import YOLO
#
# model = YOLO('yolov8n-seg.pt')  # 사전학습 모델 로드
#
# results = model.train(
#     data='/content/kvasir-seg.yaml',
#     epochs=50,
#     imgsz=640,
#     batch=16,
#     device=0,
#     patience=10,           # 10 epoch 연속 개선 없으면 중단
#     save=True,
#     project='/content/drive/MyDrive/yolo-results',
#     name='kvasir-seg-v1',
# )

"""
====================================
[셀 6] 학습 결과 확인
====================================
학습이 끝나면 자동으로 생성되는 결과물:
- confusion_matrix.png: 혼동 행렬
- results.png: epoch별 loss/mAP 그래프
- weights/best.pt: 가장 성능 좋은 모델
- weights/last.pt: 마지막 epoch 모델
"""
# from IPython.display import Image, display
#
# result_dir = '/content/drive/MyDrive/yolo-results/kvasir-seg-v1'
# display(Image(filename=f'{result_dir}/results.png', width=800))
# display(Image(filename=f'{result_dir}/confusion_matrix.png', width=500))

"""
====================================
[셀 7] 검증 데이터로 평가
====================================
mAP (Mean Average Precision): 모델 성능 지표
- mAP50: IoU 50% 기준 정확도
- mAP50-95: IoU 50~95% 평균 (더 엄격)
"""
# model_best = YOLO(f'{result_dir}/weights/best.pt')
#
# metrics = model_best.val(
#     data='/content/kvasir-seg.yaml',
#     imgsz=640,
#     device=0,
# )
#
# print(f"mAP50:    {metrics.seg.map50:.4f}")
# print(f"mAP50-95: {metrics.seg.map:.4f}")

"""
====================================
[셀 8] 샘플 이미지로 추론 테스트
====================================
"""
# import glob
# test_images = glob.glob(f'{DATASET_ROOT}/images/val/*.jpg')[:5]
#
# results = model_best.predict(
#     source=test_images,
#     save=True,
#     conf=0.25,
#     device=0,
#     project='/content/drive/MyDrive/yolo-results',
#     name='kvasir-seg-v1-predict',
# )
#
# # 결과 이미지 표시
# import glob as g
# pred_images = sorted(g.glob('/content/drive/MyDrive/yolo-results/kvasir-seg-v1-predict/*.jpg'))
# for img_path in pred_images[:3]:
#     display(Image(filename=img_path, width=500))
