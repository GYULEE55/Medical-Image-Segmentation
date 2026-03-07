import base64
import time

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

import api.state as state  # pyright: ignore[reportMissingImports]
from api.services import observe_inference, validate_upload_size  # pyright: ignore[reportMissingImports]
from core.structured_logging import get_logger

router = APIRouter()
logger = get_logger("api.routers.monitoring")


@router.get("/health")
def health():
    return {
        "status": "ok",
        "v1_yolo": {
            "available_models": list(state.models.keys()),
            "loaded_count": len(state.models),
        },
        "v2_rag": {
            "loaded": state.rag_chain is not None,
        },
        "v4_vlm": {
            "loaded": state.vlm_client is not None,
            "model": state.vlm_client.model if state.vlm_client else None,
            "host": state.vlm_client.host if state.vlm_client else None,
        },
    }


@router.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.post("/explain")
async def explain_with_heatmap(
    file: UploadFile = File(..., description="설명 가능한 분석 대상 이미지"),
    conf: float = 0.25,
    model_type: str = Form("polyp", description="YOLOv8 모델 선택: polyp / dental"),
):
    if not state.models:
        raise HTTPException(503, "로드된 모델 없음")

    if model_type not in state.models:
        available = list(state.models.keys())
        raise HTTPException(400, f"'{model_type}' 모델이 없습니다. 사용 가능: {available}")

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(400, f"이미지 파일만 가능 (받은 타입: {file.content_type})")

    contents = await file.read()
    validate_upload_size(contents)
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "이미지 디코딩 실패")

    model = state.models[model_type]
    infer_started_at = time.perf_counter()
    results = model.predict(img, conf=conf, verbose=False)
    result = results[0]

    h, w = img.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)
    detections = []

    if result.boxes is not None:
        for i, box in enumerate(result.boxes):  # type: ignore[arg-type]
            conf_score = float(box.conf)
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 > x1 and y2 > y1:
                heatmap[y1:y2, x1:x2] = np.maximum(heatmap[y1:y2, x1:x2], conf_score)

            if result.masks is not None and i < len(result.masks):
                polygon = np.array(result.masks[i].xy[0], dtype=np.int32)
                polygon[:, 0] = np.clip(polygon[:, 0], 0, w - 1)
                polygon[:, 1] = np.clip(polygon[:, 1], 0, h - 1)
                cv2.fillPoly(heatmap, [polygon], color=conf_score)

            detections.append(
                {
                    "class": result.names[int(box.cls)],
                    "confidence": round(conf_score, 4),
                    "bbox": [x1, y1, x2, y2],
                }
            )

    max_val = float(heatmap.max())
    if max_val > 0:
        heatmap = heatmap / max_val
    heatmap_u8 = np.clip(heatmap * 255.0, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, colored, 0.4, 0.0)

    ok, encoded = cv2.imencode(".jpg", overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise HTTPException(500, "설명용 오버레이 인코딩 실패")

    elapsed = observe_inference("explain", model_type, infer_started_at)
    logger.info(
        "explain_completed",
        model_type=model_type,
        detections=len(detections),
        duration_ms=round(elapsed * 1000, 2),
    )

    return {
        "filename": file.filename,
        "model_type": model_type,
        "method": "activation_heatmap",
        "count": len(detections),
        "detections": detections,
        "overlay_jpeg_base64": base64.b64encode(encoded.tobytes()).decode("utf-8"),
    }
