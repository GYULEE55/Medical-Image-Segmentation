import time

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

import api.state as state  # pyright: ignore[reportMissingImports]
from api.services import (  # pyright: ignore[reportMissingImports]
    observe_inference,
    validate_upload_size,
)
from core.structured_logging import get_logger

router = APIRouter()
logger = get_logger("api.routers.predict")


@router.post("/predict")
async def predict(
    file: UploadFile = File(..., description="추론할 이미지 파일"),
    conf: float = 0.25,
    model_type: str = Form(
        "polyp", description="모델 선택: polyp(위장 폴립) 또는 dental(치과 X-ray)"
    ),
):
    if not state.models:
        raise HTTPException(503, "로드된 모델 없음")

    if model_type not in state.models:
        available = list(state.models.keys())
        raise HTTPException(400, f"'{model_type}' 모델이 없습니다. 사용 가능: {available}")

    model = state.models[model_type]

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(400, f"이미지 파일만 가능 (받은 타입: {file.content_type})")

    contents = await file.read()
    validate_upload_size(contents)
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "이미지 디코딩 실패")

    infer_started_at = time.perf_counter()
    results = model.predict(img, conf=conf, verbose=False)
    result = results[0]

    detections = []
    if result.boxes is not None:
        for i in range(len(result.boxes)):  # type: ignore[arg-type]
            box = result.boxes[i]
            det = {
                "class": result.names[int(box.cls)],
                "confidence": round(float(box.conf), 4),
                "bbox": [round(v, 1) for v in box.xyxy[0].tolist()],
            }
            if result.masks is not None and i < len(result.masks):
                polygon = result.masks[i].xy[0].tolist()
                det["polygon_points"] = len(polygon)
                det["polygon"] = [[round(x, 1), round(y, 1)] for x, y in polygon]
            detections.append(det)

    elapsed = observe_inference("predict", model_type, infer_started_at)
    logger.info(
        "predict_completed",
        model_type=model_type,
        detections=len(detections),
        duration_ms=round(elapsed * 1000, 2),
    )

    return {
        "filename": file.filename,
        "model_type": model_type,
        "image_size": {"width": img.shape[1], "height": img.shape[0]},
        "detections": detections,
        "count": len(detections),
    }
