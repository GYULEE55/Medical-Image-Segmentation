"""
YOLOv8 첫 실행 테스트
- 사전학습된 yolov8n.pt 모델을 다운로드
- 샘플 이미지에서 객체 탐지 실행
- 결과 확인
"""
from ultralytics import YOLO

# 1. 모델 로드 (처음 실행 시 자동 다운로드됨, ~6MB)
#    'yolov8n.pt' = COCO 데이터셋(일상 사물 80종류)으로 학습된 nano 모델
model = YOLO('yolov8n.pt')

# 2. 추론 실행 — 버스 사진에서 사람, 차량 등을 찾아라
results = model.predict(
    source='https://ultralytics.com/images/bus.jpg',
    save=True,           # 결과 이미지를 파일로 저장
    conf=0.25,           # 신뢰도 25% 이상만 표시 (0.0~1.0)
    device=0,            # GPU 0번 사용 (CUDA)
)

# 3. 결과 해석
for result in results:
    print("\n===== 탐지 결과 =====")
    print(f"이미지 크기: {result.orig_shape}")
    print(f"탐지된 객체 수: {len(result.boxes)}")
    
    for box in result.boxes:
        # box.cls   = 클래스 번호 (0=person, 5=bus 등)
        # box.conf  = 신뢰도 (모델이 얼마나 확신하는지, 0.0~1.0)
        # box.xyxy  = 바운딩 박스 좌표 [x1, y1, x2, y2]
        cls_name = result.names[int(box.cls)]    # 번호 → 이름 변환
        confidence = float(box.conf)              # 텐서 → 숫자
        coords = box.xyxy[0].tolist()             # 텐서 → 리스트
        
        print(f"  {cls_name}: {confidence:.1%} 확신 | 위치: [{coords[0]:.0f}, {coords[1]:.0f}, {coords[2]:.0f}, {coords[3]:.0f}]")

print("\n결과 이미지가 runs/detect/predict/ 폴더에 저장되었습니다.")
