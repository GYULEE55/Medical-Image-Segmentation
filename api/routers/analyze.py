import time

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

import api.state as state  # pyright: ignore[reportMissingImports]
from api.constants import class_names_kr  # pyright: ignore[reportMissingImports]
from api.services import (  # pyright: ignore[reportMissingImports]
    observe_inference,
    validate_upload_size,
)
from core.structured_logging import get_logger

router = APIRouter()
logger = get_logger("api.routers.analyze")


@router.post("/analyze")
async def analyze_with_knowledge(
    file: UploadFile = File(..., description="분석할 의료 이미지"),
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
    detected_classes = set()

    if result.boxes is not None:
        for i, box in enumerate(result.boxes):  # type: ignore[arg-type]
            class_name = result.names[int(box.cls)]
            detected_classes.add(class_name)

            det = {
                "class": class_name,
                "confidence": round(float(box.conf), 4),
                "bbox": [round(v, 1) for v in box.xyxy[0].tolist()],
            }
            if result.masks is not None and i < len(result.masks):
                polygon = result.masks[i].xy[0].tolist()
                det["polygon_points"] = len(polygon)
                det["polygon"] = [[round(x, 1), round(y, 1)] for x, y in polygon]

            detections.append(det)

    medical_knowledge = None
    if detected_classes and state.rag_chain is not None:
        detected_kr = [class_names_kr.get(c) or c for c in detected_classes]
        class_text = ", ".join(detected_kr)

        auto_question = (
            f"이 환자의 영상에서 {class_text}이(가) {len(detections)}건 검출되었습니다. "
            f"{class_text}에 대한 임상적 의미, 권장 추적 관찰 주기, "
            f"그리고 환자에게 안내할 주의사항을 알려주세요."
        )

        try:
            rag_result = await state.rag_chain.query(auto_question)
            medical_knowledge = {
                "auto_question": auto_question,
                "answer": rag_result["answer"],
                "sources": [
                    {
                        "source_file": s["source_file"],
                        "page": s["page"],
                        "content_preview": s["content_preview"],
                    }
                    for s in rag_result["sources"]
                ],
                "num_sources": rag_result["num_sources"],
            }
        except Exception as e:
            medical_knowledge = {
                "auto_question": auto_question,
                "error": f"RAG 질의 실패: {str(e)}",
            }

    elapsed = observe_inference("analyze", model_type, infer_started_at)
    logger.info(
        "analyze_completed",
        model_type=model_type,
        detections=len(detections),
        rag_used=state.rag_chain is not None,
        duration_ms=round(elapsed * 1000, 2),
    )

    return {
        "filename": file.filename,
        "model_type": model_type,
        "image_size": {"width": img.shape[1], "height": img.shape[0]},
        "detections": detections,
        "count": len(detections),
        "medical_knowledge": medical_knowledge,
        "rag_available": state.rag_chain is not None,
    }
