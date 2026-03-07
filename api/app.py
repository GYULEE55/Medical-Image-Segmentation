"""
Medical AI Assistant API — V4 (YOLOv8 + RAG + VLM + Multi-Model)
"""

import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import ultralytics

import api.state as state
from api.constants import MODEL_PATHS, SAMPLE_REGISTRY, VLM_TIMEOUT_SECONDS
from api.routers.analyze import router as analyze_router
from api.routers.ask import router as ask_router
from api.routers.monitoring import router as monitoring_router
from api.routers.predict import router as predict_router
from api.routers.vlm import router as vlm_router
from core.structured_logging import (
    bind_request_context,
    clear_request_context,
    configure_logging,
    get_logger,
)

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

configure_logging()
logger = get_logger("api.app")

SAMPLE_DIR = Path(__file__).parent.parent / "example_test"


@asynccontextmanager
async def lifespan(app: FastAPI):
    for model_name, model_path in MODEL_PATHS.items():
        if Path(model_path).exists():
            state.models[model_name] = getattr(ultralytics, "YOLO")(model_path)  # pyright: ignore[reportAttributeAccessIssue]
            logger.info(
                "model_loaded",
                stage="v1",
                model_name=model_name,
                model_path=model_path,
            )
        else:
            logger.warning(
                "model_missing",
                stage="v1",
                model_name=model_name,
                model_path=model_path,
            )

    if not state.models:
        logger.warning("no_models_loaded", stage="v1")

    rag_vectorstore = Path(__file__).parent.parent / "rag" / "vectorstore"
    if rag_vectorstore.exists():
        try:
            from rag.chain import MedicalRAGChain

            state.rag_chain = MedicalRAGChain()
            logger.info("rag_chain_initialized", stage="v2")
        except Exception as e:
            logger.exception("rag_chain_init_failed", stage="v2", error=str(e))
    else:
        logger.warning(
            "rag_vectorstore_missing",
            stage="v2",
            action="python rag/ingest.py",
        )

    try:
        from vlm.client import MedicalVLMClient

        state.vlm_client = MedicalVLMClient(timeout=VLM_TIMEOUT_SECONDS)
        if await state.vlm_client.is_available():
            logger.info(
                "vlm_initialized",
                stage="v4",
                model=state.vlm_client.model,
                timeout_seconds=VLM_TIMEOUT_SECONDS,
            )
        else:
            logger.warning(
                "vlm_unavailable",
                stage="v4",
                model=state.vlm_client.model,
                hint="ollama serve && ollama pull model",
            )
    except Exception as e:
        logger.exception("vlm_init_failed", stage="v4", error=str(e))
        state.vlm_client = None

    yield

    if state.vlm_client is not None:
        await state.vlm_client.close()


app = FastAPI(
    title="Medical AI Assistant API",
    description="""
## Medical AI Assistant — Vision + LLM + VLM 통합 서비스

### V1: 폴립 세그멘테이션
- `POST /predict` — 내시경 이미지 업로드 → 폴립 검출 + 마스크

### V2: 의료 지식 RAG
- `POST /ask` — 의료 질문 → 근거 문헌 기반 답변 + 출처

### V3: 통합 분석 (Detection + RAG)
- `POST /analyze` — 이미지 검출 → 검출 결과 기반 자동 RAG 질의 → 의료 지식 포함 응답

### V4: VLM 통합 분석 (VLM + Detection + RAG)
- `POST /vlm-analyze` — VLM이 이미지 직접 해석 → YOLOv8 검출 보완 → RAG 문헌 근거 보강
    """,
    version="4.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_context_middleware(request, call_next):
    request_id = bind_request_context(request)
    request_logger = get_logger("http.request")
    started_at = time.perf_counter()
    path = request.url.path
    method = request.method

    request_logger.info("request_started")
    try:
        response = await call_next(request)
    except Exception:
        elapsed_seconds = time.perf_counter() - started_at
        state.HTTP_REQUEST_TOTAL.labels(method=method, path=path, status_code="500").inc()
        state.HTTP_REQUEST_DURATION_SECONDS.labels(method=method, path=path).observe(
            elapsed_seconds
        )
        request_logger.exception("request_failed")
        raise
    else:
        elapsed_seconds = time.perf_counter() - started_at
        elapsed_ms = round(elapsed_seconds * 1000, 2)
        state.HTTP_REQUEST_TOTAL.labels(
            method=method, path=path, status_code=str(response.status_code)
        ).inc()
        state.HTTP_REQUEST_DURATION_SECONDS.labels(method=method, path=path).observe(
            elapsed_seconds
        )
        response.headers["X-Request-ID"] = request_id
        request_logger.info(
            "request_finished",
            status_code=response.status_code,
            duration_ms=elapsed_ms,
        )
        return response
    finally:
        clear_request_context()


@app.get("/demo", include_in_schema=False)
def demo_page():
    demo_path = Path(__file__).parent.parent / "ui" / "demo.html"
    if not demo_path.exists():
        raise HTTPException(404, "demo 페이지 파일을 찾을 수 없습니다")
    return FileResponse(demo_path)


@app.get("/demo/samples", include_in_schema=False)
def demo_sample_list():
    return [
        {"key": k, "label": v["label"], "model_type": v["model_type"]}
        for k, v in SAMPLE_REGISTRY.items()
        if (SAMPLE_DIR / v["file"]).exists()
    ]


@app.get("/demo/sample-image", include_in_schema=False)
def demo_sample_image(key: str = "colon_polyp"):
    entry = SAMPLE_REGISTRY.get(key)
    if not entry:
        raise HTTPException(404, f"알 수 없는 샘플 키: {key}")
    sample_path = SAMPLE_DIR / entry["file"]
    if not sample_path.exists():
        raise HTTPException(404, f"샘플 이미지를 찾을 수 없습니다: {entry['file']}")
    media = "image/png" if sample_path.suffix == ".png" else "image/jpeg"
    return FileResponse(sample_path, media_type=media)


app.include_router(monitoring_router)
app.include_router(predict_router)
app.include_router(ask_router)
app.include_router(analyze_router)
app.include_router(vlm_router)
