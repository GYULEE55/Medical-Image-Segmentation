import asyncio
import io
import time
import uuid
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

import api.state as state  # pyright: ignore[reportMissingImports]
from api.constants import (  # pyright: ignore[reportMissingImports]
    NO_EVIDENCE_TEXT,
    VLM_JPEG_QUALITY,
    VLM_MAX_EDGE,
    class_names_kr,
)
from api.services import (  # pyright: ignore[reportMissingImports]
    fetch_pubmed_web_evidence,
    observe_inference,
    prepare_vlm_image_bytes,
    validate_upload_size,
)
from core.structured_logging import get_logger

router = APIRouter()
logger = get_logger("api.routers.vlm")


async def _process_vlm_job(
    job_id: str,
    filename: str,
    contents: bytes,
    conf: float,
    model_type: str,
    language: str,
) -> None:
    state.async_jobs[job_id]["status"] = "running"
    state.async_jobs[job_id]["started_at"] = time.time()

    upload = UploadFile(filename=filename, file=io.BytesIO(contents))

    try:
        result = await vlm_analyze_with_knowledge(
            file=upload,
            conf=conf,
            model_type=model_type,
            language=language,
        )
        state.async_jobs[job_id]["status"] = "completed"
        state.async_jobs[job_id]["result"] = result
    except Exception as e:
        state.async_jobs[job_id]["status"] = "failed"
        state.async_jobs[job_id]["error"] = str(e)
        logger.exception("async_vlm_job_failed", job_id=job_id, error=str(e))
    finally:
        state.async_jobs[job_id]["finished_at"] = time.time()
        await upload.close()


@router.post("/vlm-analyze/async", status_code=202)
async def vlm_analyze_async(
    file: UploadFile = File(..., description="비동기 분석할 의료 이미지"),
    conf: float = 0.25,
    model_type: str = Form("polyp", description="YOLOv8 모델 선택: polyp / dental"),
    language: str = Form("ko", description="VLM 분석 언어: ko(한국어) / en(영어)"),
):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(400, f"이미지 파일만 가능 (받은 타입: {file.content_type})")

    contents = await file.read()
    validate_upload_size(contents)
    job_id = str(uuid.uuid4())
    state.async_jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "created_at": time.time(),
        "result": None,
        "error": None,
        "filename": file.filename,
        "model_type": model_type,
    }

    asyncio.create_task(
        _process_vlm_job(
            job_id=job_id,
            filename=file.filename or "unknown",
            contents=contents,
            conf=conf,
            model_type=model_type,
            language=language,
        )
    )
    logger.info("async_vlm_job_created", job_id=job_id, model_type=model_type)
    return {
        "job_id": job_id,
        "status": "pending",
        "poll_endpoint": f"/jobs/{job_id}",
    }


@router.get("/jobs/{job_id}")
def get_job_status(job_id: str):
    job = state.async_jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"job_id를 찾을 수 없습니다: {job_id}")
    return job


@router.post("/vlm-analyze")
async def vlm_analyze_with_knowledge(
    file: UploadFile = File(..., description="분석할 의료 이미지"),
    conf: float = 0.25,
    model_type: str = Form("polyp", description="YOLOv8 모델 선택: polyp / dental"),
    language: str = Form("ko", description="VLM 분석 언어: ko(한국어) / en(영어)"),
):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(400, f"이미지 파일만 가능 (받은 타입: {file.content_type})")

    contents = await file.read()
    validate_upload_size(contents)
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "이미지 디코딩 실패")

    infer_started_at = time.perf_counter()

    response_data: dict[str, object] = {
        "filename": file.filename,
        "image_size": {"width": img.shape[1], "height": img.shape[0]},
    }

    vlm_analysis: dict[str, Any] | None = None
    if state.vlm_client is not None:
        try:
            detection_hints = None
            if state.models and model_type in state.models:
                quick_results = state.models[model_type].predict(img, conf=conf, verbose=False)
                quick_result = quick_results[0]
                if quick_result.boxes is not None and len(quick_result.boxes) > 0:
                    detection_hints = []
                    for box in quick_result.boxes:
                        detection_hints.append(
                            {
                                "class": quick_result.names[int(box.cls)],
                                "confidence": round(float(box.conf), 4),
                            }
                        )

            vlm_image_bytes, vlm_image_meta = prepare_vlm_image_bytes(img, contents)

            vlm_result = await state.vlm_client.analyze_with_context(
                image_bytes=vlm_image_bytes,
                detection_results=detection_hints,
                model_type=model_type,
            )
            vlm_analysis = {
                "interpretation": vlm_result["analysis"],
                "model": vlm_result["model"],
                "duration_ms": vlm_result["total_duration_ms"],
                "detection_context_provided": detection_hints is not None,
                "input_resize": {
                    "applied": vlm_image_meta["resized"],
                    "width": vlm_image_meta["width"],
                    "height": vlm_image_meta["height"],
                    "max_edge": VLM_MAX_EDGE,
                    "jpeg_quality": VLM_JPEG_QUALITY,
                },
            }
        except ConnectionError as e:
            vlm_analysis = {"error": f"ollama 서버 연결 실패: {str(e)}"}
        except Exception as e:
            vlm_analysis = {"error": f"VLM 분석 실패: {str(e)}"}
    else:
        vlm_analysis = {
            "error": "VLM 미초기화 — ollama serve 실행 후 서버 재시작 필요",
        }

    response_data["vlm_analysis"] = vlm_analysis

    detections = []
    detected_classes = set()

    if state.models and model_type in state.models:
        model = state.models[model_type]
        results = model.predict(img, conf=conf, verbose=False)
        result = results[0]

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

    response_data["model_type"] = model_type
    response_data["detections"] = detections
    response_data["count"] = len(detections)

    medical_evidence = None
    if state.rag_chain is not None:
        rag_query = None

        if isinstance(vlm_analysis, dict) and "interpretation" in vlm_analysis:
            vlm_text = vlm_analysis["interpretation"]
            detected_kr = [class_names_kr.get(c) or c for c in detected_classes]
            keyword_hint = ", ".join(detected_kr) if detected_kr else "검출 클래스 없음"
            rag_query = (
                "다음 의료 영상 분석 소견에 대한 임상 근거와 "
                "권장 조치를 문헌에서 찾아주세요. "
                f"핵심 키워드: {keyword_hint}\n\n{vlm_text[:500]}"
            )
        elif detected_classes:
            detected_kr = [class_names_kr.get(c) or c for c in detected_classes]
            class_text = ", ".join(detected_kr)
            rag_query = (
                f"{class_text}이(가) 검출되었습니다. "
                "임상적 의미와 권장 추적 관찰 주기를 알려주세요."
            )

        if rag_query:
            try:
                rag_result = await state.rag_chain.query(rag_query)
                has_evidence = rag_result.get("num_sources", 0) > 0
                answer_text = rag_result.get("answer", NO_EVIDENCE_TEXT)
                fallback_reason = None
                sources_payload = [
                    {
                        "source_file": s["source_file"],
                        "page": s["page"],
                        "content_preview": s["content_preview"],
                    }
                    for s in rag_result.get("sources", [])
                ]
                if not has_evidence:
                    web_evidence = await fetch_pubmed_web_evidence(
                        query=rag_query,
                        model_type=model_type,
                    )
                    if web_evidence:
                        answer_text = web_evidence.get("answer", NO_EVIDENCE_TEXT)
                        sources_payload = web_evidence.get("sources", [])
                        fallback_reason = web_evidence.get("reason", "web_pubmed_fallback")
                        has_evidence = web_evidence.get("num_sources", 0) > 0
                    else:
                        answer_text = NO_EVIDENCE_TEXT
                        fallback_reason = "no_evidence"

                medical_evidence = {
                    "query_used": rag_query[:200] + "..." if len(rag_query) > 200 else rag_query,
                    "query_source": "vlm"
                    if "interpretation" in (vlm_analysis or {})
                    else "detection",
                    "answer": answer_text,
                    "sources": sources_payload,
                    "num_sources": len(sources_payload),
                    "grounded": has_evidence,
                    "reason": fallback_reason,
                    "disclaimer": "인터넷(PubMed) 검색 기반 참고 정보입니다. 최종 의료 판단은 전문의 상담이 필요합니다."
                    if fallback_reason == "web_pubmed_fallback"
                    else None,
                }
            except Exception as e:
                error_text = str(e)
                if "insufficient_quota" in error_text:
                    medical_evidence = {
                        "error": "RAG 질의 실패: OpenAI 크레딧/요금제 한도를 초과했습니다 (insufficient_quota)."
                    }
                else:
                    medical_evidence = {"error": f"RAG 질의 실패: {error_text}"}

    response_data["medical_evidence"] = medical_evidence
    response_data["rag_available"] = state.rag_chain is not None
    response_data["vlm_available"] = state.vlm_client is not None

    elapsed = observe_inference("vlm_analyze", model_type, infer_started_at)
    logger.info(
        "vlm_analyze_completed",
        model_type=model_type,
        detections=len(detections),
        vlm_available=state.vlm_client is not None,
        rag_available=state.rag_chain is not None,
        duration_ms=round(elapsed * 1000, 2),
    )

    return response_data
