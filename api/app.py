"""
의료 이미지 세그멘테이션 추론 API

YOLOv8n-seg (Kvasir-SEG polyp) 모델로 추론하는 FastAPI 서버.
이미지 업로드 -> 폴립 검출 + 세그멘테이션 -> JSON 결과 반환.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from ultralytics import YOLO

# 모델 경로: 환경변수 우선, 없으면 프로젝트 루트의 best.pt
MODEL_PATH = os.getenv("MODEL_PATH", str(Path(__file__).parent.parent / "best.pt"))

app = FastAPI(
    title="Medical Segmentation API",
    description="위장 폴립 세그멘테이션 (YOLOv8n-seg, Kvasir-SEG 학습)",
    version="1.0.0",
)

# 서버 시작 시 모델 1회 로드 (매 요청마다 로드하면 느림)
model = None


@app.on_event("startup")
def load_model():
    global model
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"모델 파일 없음: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print(f"[API] 모델 로드 완료: {MODEL_PATH}")


@app.get("/health")
def health():
    """서버 상태 확인"""
    return {
        "status": "ok",
        "model": Path(MODEL_PATH).name,
        "model_loaded": model is not None,
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(..., description="추론할 이미지 파일"),
    conf: float = 0.25,
):
    """
    이미지 업로드 -> 폴립 세그멘테이션 추론

    - **file**: 이미지 파일 (jpg, png 등)
    - **conf**: confidence threshold (기본 0.25)

    반환: 검출된 폴립 목록 (bbox, 신뢰도, 폴리곤 좌표)
    """
    if model is None:
        raise HTTPException(503, "모델 로드 안됨")

    # 이미지 타입 검증
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(400, f"이미지 파일만 가능 (받은 타입: {file.content_type})")

    # 바이트 -> numpy 배열 -> OpenCV 이미지
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "이미지 디코딩 실패")

    # 추론
    results = model.predict(img, conf=conf, verbose=False)
    result = results[0]

    # 결과 파싱
    detections = []
    if result.boxes is not None:
        for i, box in enumerate(result.boxes):
            det = {
                "class": result.names[int(box.cls)],
                "confidence": round(float(box.conf), 4),
                "bbox": [round(v, 1) for v in box.xyxy[0].tolist()],
            }
            # 세그멘테이션 마스크 (폴리곤 좌표)
            if result.masks is not None and i < len(result.masks):
                polygon = result.masks[i].xy[0].tolist()
                det["polygon_points"] = len(polygon)
                # 전체 폴리곤 좌표 (클라이언트에서 시각화용)
                det["polygon"] = [[round(x, 1), round(y, 1)] for x, y in polygon]

            detections.append(det)

    return {
        "filename": file.filename,
        "image_size": {"width": img.shape[1], "height": img.shape[0]},
        "detections": detections,
        "count": len(detections),
    }
