"""
Medical AI Assistant API — V4 (YOLOv8 + RAG + VLM + Multi-Model)

엔드포인트:
  GET  /health          — 서버 상태 확인 (모델 + RAG + VLM)
  POST /predict         — 병변 세그멘테이션 (YOLOv8) [V1] — model 파라미터로 polyp/dental 선택
  POST /ask             — 의료 지식 Q&A (RAG) [V2]
  POST /analyze         — 이미지 검출 + 의료 지식 통합 분석 [V3]
  POST /vlm-analyze     — VLM 직접 해석 + 검출 + RAG 근거 보강 [V4 핵심]
  GET  /ask/health      — RAG 상태 

V1: YOLOv8n-seg로 병변 검출 + 세그멘테이션 (polyp/dental 멀티모델)
V2: LangChain + ChromaDB + GPT-4o-mini로 의료 지식 Q&A (출처 포함)
V3: Vision(YOLOv8) + LLM(RAG) 통합 — 검출 결과를 기반으로 의료 지식 자동 제공
V4: VLM(LLaVA)이 이미지 직접 해석 → YOLOv8 검출 → RAG 문헌 근거 보강
"""

import os
import uuid
import asyncio
import base64
import io
import time
import re
import cv2
import numpy as np
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
import xml.etree.ElementTree as ET

# .env 파일에서 환경변수 로드 (OPENAI_API_KEY 등)
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel
from ultralytics import YOLO
import httpx
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from core.structured_logging import (
    bind_request_context,
    clear_request_context,
    configure_logging,
    get_logger,
)

configure_logging()
logger = get_logger("api.app")

# ── 경로 설정 ──────────────────────────────────────────────────
# 멀티모델 지원: polyp(위장 폴립) + dental(치과 X-ray)
MODEL_PATHS: Dict[str, str] = {
    "polyp": os.getenv("MODEL_PATH", str(Path(__file__).parent.parent / "best.pt")),
    "dental": os.getenv(
        "DENTAL_MODEL_PATH", str(Path(__file__).parent.parent / "best_dentex.pt")
    ),
}

VLM_MAX_EDGE = int(os.getenv("VLM_MAX_EDGE", "960"))
VLM_JPEG_QUALITY = int(os.getenv("VLM_JPEG_QUALITY", "75"))
VLM_TIMEOUT_SECONDS = float(os.getenv("VLM_TIMEOUT_SECONDS", "180"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", "10485760"))
WEB_EVIDENCE_TIMEOUT_SECONDS = float(os.getenv("WEB_EVIDENCE_TIMEOUT_SECONDS", "8"))
WEB_EVIDENCE_MAX_ARTICLES = int(os.getenv("WEB_EVIDENCE_MAX_ARTICLES", "3"))
PUBMED_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# ── 전역 상태 ──────────────────────────────────────────────────
models: Dict[str, YOLO] = {}  # 멀티모델 딕셔너리 {"polyp": YOLO, "dental": YOLO}
rag_chain = None  # RAG 체인 (V2)
vlm_client = None  # VLM 클라이언트 (V4)
NO_EVIDENCE_TEXT = "제공된 문서에서 해당 정보를 찾을 수 없습니다."
async_jobs: Dict[str, Dict[str, Any]] = {}

HTTP_REQUEST_TOTAL = Counter(
    "medical_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status_code"],
)
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "medical_http_request_duration_seconds",
    "HTTP request duration seconds",
    ["method", "path"],
)
MODEL_INFERENCE_TOTAL = Counter(
    "medical_model_inference_total",
    "Total model inference executions",
    ["endpoint", "model_type"],
)
MODEL_INFERENCE_DURATION_SECONDS = Histogram(
    "medical_model_inference_duration_seconds",
    "Model inference duration seconds",
    ["endpoint", "model_type"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 180.0),
)
PROCESS_MEMORY_BYTES = Gauge(
    "medical_process_memory_bytes",
    "Process resident memory bytes",
)


def _observe_inference(endpoint: str, model_type: str, started_at: float) -> float:
    elapsed = time.perf_counter() - started_at
    MODEL_INFERENCE_TOTAL.labels(endpoint=endpoint, model_type=model_type).inc()
    MODEL_INFERENCE_DURATION_SECONDS.labels(
        endpoint=endpoint, model_type=model_type
    ).observe(elapsed)
    try:
        import psutil

        process = psutil.Process(os.getpid())
        PROCESS_MEMORY_BYTES.set(process.memory_info().rss)
    except Exception:
        pass
    return elapsed


def _prepare_vlm_image_bytes(img: np.ndarray, original_bytes: bytes):
    h, w = img.shape[:2]
    longest = max(w, h)

    if longest <= VLM_MAX_EDGE:
        return original_bytes, {"resized": False, "width": w, "height": h}

    scale = VLM_MAX_EDGE / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(
        ".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), VLM_JPEG_QUALITY]
    )

    if not ok:
        return original_bytes, {"resized": False, "width": w, "height": h}

    return encoded.tobytes(), {"resized": True, "width": new_w, "height": new_h}


def _validate_upload_size(contents: bytes) -> None:
    if not contents:
        raise HTTPException(400, "빈 파일은 처리할 수 없습니다")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            413,
            f"파일이 너무 큽니다. 최대 {MAX_UPLOAD_BYTES} bytes 까지 허용됩니다",
        )


def _compact_text(text: Any, max_len: int = 220) -> str:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if not compact:
        return ""
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 3] + "..."


def _extract_pubmed_abstract(article: ET.Element) -> str:
    abstract_nodes = article.findall(".//Abstract/AbstractText")
    if not abstract_nodes:
        return ""

    sections: list[str] = []
    for node in abstract_nodes:
        section_text = "".join(node.itertext()).strip()
        if section_text:
            sections.append(section_text)
    return "\n".join(sections).strip()


def _extract_pubmed_year(article: ET.Element) -> str:
    pub_date = article.find(".//Journal/JournalIssue/PubDate")
    if pub_date is None:
        return "N/A"
    year = (pub_date.findtext("Year") or "").strip()
    if year:
        return year
    medline_date = (pub_date.findtext("MedlineDate") or "").strip()
    return medline_date[:4] if medline_date else "N/A"


def _parse_pubmed_articles(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    rows: list[dict] = []

    for entry in root.findall(".//PubmedArticle"):
        pmid = (entry.findtext(".//PMID") or "").strip()
        article = entry.find(".//Article")
        if not pmid or article is None:
            continue

        title = "".join(
            (article.find("ArticleTitle") or ET.Element("x")).itertext()
        ).strip()
        journal = (
            article.findtext(".//Journal/Title") or ""
        ).strip() or "Unknown Journal"
        abstract = _extract_pubmed_abstract(article)
        year = _extract_pubmed_year(article)

        rows.append(
            {
                "pmid": pmid,
                "title": title or "Untitled",
                "journal": journal,
                "year": year,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "abstract": abstract,
            }
        )
    return rows


def _topic_seed_by_model(model_type: str) -> str:
    if model_type == "dental":
        return (
            "dental caries periapical lesion impacted tooth panoramic x-ray guideline"
        )
    return "colorectal polyp colonoscopy surveillance guideline endoscopic management"


async def _fetch_pubmed_web_evidence(query: str, model_type: str) -> Optional[dict]:
    search_query = f"{_topic_seed_by_model(model_type)} {query[:220]}"
    timeout = httpx.Timeout(WEB_EVIDENCE_TIMEOUT_SECONDS)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            esearch_params = {
                "db": "pubmed",
                "term": search_query,
                "retmode": "json",
                "retmax": str(WEB_EVIDENCE_MAX_ARTICLES),
                "sort": "relevance",
            }
            esearch_res = await client.get(PUBMED_ESEARCH_URL, params=esearch_params)
            esearch_res.raise_for_status()
            ids = esearch_res.json().get("esearchresult", {}).get("idlist", [])
            pmids = [str(x).strip() for x in ids if str(x).strip()]
            if not pmids:
                return None

            efetch_params = {
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "xml",
            }
            efetch_res = await client.get(PUBMED_EFETCH_URL, params=efetch_params)
            efetch_res.raise_for_status()
            articles = _parse_pubmed_articles(efetch_res.text)
    except Exception:
        return None

    if not articles:
        return None

    evidence_lines = []
    sources = []
    for idx, article in enumerate(articles, start=1):
        preview = _compact_text(
            article.get("abstract") or article.get("title"), max_len=260
        )
        evidence_lines.append(
            f"{idx}) {article['title']} ({article['journal']}, {article['year']})\n- 핵심: {preview}\n- URL: {article['url']}"
        )
        sources.append(
            {
                "source_file": f"PubMed:{article['pmid']}",
                "page": "web",
                "content_preview": preview,
                "url": article["url"],
            }
        )

    answer = "\n\n".join(evidence_lines)
    return {
        "query_used": _compact_text(search_query, max_len=200),
        "query_source": "web_fallback",
        "answer": answer,
        "sources": sources,
        "num_sources": len(sources),
        "grounded": True,
        "reason": "web_pubmed_fallback",
        "disclaimer": "인터넷(PubMed) 검색 기반 참고 정보입니다. 최종 의료 판단은 전문의 상담이 필요합니다.",
    }


# ── 앱 초기화 (lifespan 패턴) ──────────────────────────────────
# @app.on_event("startup") 는 deprecated → asynccontextmanager lifespan 사용
@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 모델 초기화 (1회만 실행)"""
    global models, rag_chain, vlm_client

    # V1: YOLOv8 모델 로드 (멀티모델 — 존재하는 모델만 로드)
    for model_name, model_path in MODEL_PATHS.items():
        if Path(model_path).exists():
            models[model_name] = YOLO(model_path)
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

    if not models:
        logger.warning("no_models_loaded", stage="v1")

    # V2: RAG 체인 초기화 (ChromaDB가 있을 때만)
    rag_vectorstore = Path(__file__).parent.parent / "rag" / "vectorstore"
    if rag_vectorstore.exists():
        try:
            from rag.chain import MedicalRAGChain

            rag_chain = MedicalRAGChain()
            logger.info("rag_chain_initialized", stage="v2")
        except Exception as e:
            logger.exception("rag_chain_init_failed", stage="v2", error=str(e))
    else:
        logger.warning(
            "rag_vectorstore_missing",
            stage="v2",
            action="python rag/ingest.py",
        )

    # V4: VLM 클라이언트 초기화 (ollama + LLaVA)
    try:
        from vlm.client import MedicalVLMClient

        vlm_client = MedicalVLMClient(timeout=VLM_TIMEOUT_SECONDS)
        if await vlm_client.is_available():
            logger.info(
                "vlm_initialized",
                stage="v4",
                model=vlm_client.model,
                timeout_seconds=VLM_TIMEOUT_SECONDS,
            )
        else:
            logger.warning(
                "vlm_unavailable",
                stage="v4",
                model=vlm_client.model,
                hint="ollama serve && ollama pull model",
            )
    except Exception as e:
        logger.exception("vlm_init_failed", stage="v4", error=str(e))
        vlm_client = None

    yield  # 서버 실행

    # 서버 종료 시 VLM 클라이언트 정리
    if vlm_client is not None:
        await vlm_client.close()


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
        HTTP_REQUEST_TOTAL.labels(method=method, path=path, status_code="500").inc()
        HTTP_REQUEST_DURATION_SECONDS.labels(method=method, path=path).observe(
            elapsed_seconds
        )
        request_logger.exception("request_failed")
        raise
    else:
        elapsed_seconds = time.perf_counter() - started_at
        elapsed_ms = round(elapsed_seconds * 1000, 2)
        HTTP_REQUEST_TOTAL.labels(
            method=method, path=path, status_code=str(response.status_code)
        ).inc()
        HTTP_REQUEST_DURATION_SECONDS.labels(method=method, path=path).observe(
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


# ── 샘플 이미지 목록/서빙 ──────────────────────────────────────
# example_test/ 에 있는 의료 샘플 이미지를 데모에서 바로 써볼 수 있게 제공
SAMPLE_DIR = Path(__file__).parent.parent / "example_test"
SAMPLE_REGISTRY = {
    "colon_polyp": {
        "file": "colon_polyp.jpg",
        "model_type": "polyp",
        "label": "대장 내시경 (폴립)",
    },
    "dental_xray": {
        "file": "dental_xray.png",
        "model_type": "dental",
        "label": "치과 파노라마 X-ray",
    },
}


@app.get("/demo/samples", include_in_schema=False)
def demo_sample_list():
    """사용 가능한 샘플 이미지 목록 반환"""
    return [
        {"key": k, "label": v["label"], "model_type": v["model_type"]}
        for k, v in SAMPLE_REGISTRY.items()
        if (SAMPLE_DIR / v["file"]).exists()
    ]


@app.get("/demo/sample-image", include_in_schema=False)
def demo_sample_image(key: str = "colon_polyp"):
    """샘플 이미지 서빙 (key 파라미터로 선택)"""
    entry = SAMPLE_REGISTRY.get(key)
    if not entry:
        raise HTTPException(404, f"알 수 없는 샘플 키: {key}")
    sample_path = SAMPLE_DIR / entry["file"]
    if not sample_path.exists():
        raise HTTPException(404, f"샘플 이미지를 찾을 수 없습니다: {entry['file']}")
    media = "image/png" if sample_path.suffix == ".png" else "image/jpeg"
    return FileResponse(sample_path, media_type=media)


# ── 요청/응답 스키마 (V2 RAG) ──────────────────────────────────
class AskRequest(BaseModel):
    question: str
    k: Optional[int] = 5  # 검색할 문서 청크 수


class SourceInfo(BaseModel):
    source_file: str
    page: Optional[str] = None
    content_preview: str


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]
    num_sources: int


# ── 공통 엔드포인트 ────────────────────────────────────────────
@app.get("/health")
def health():
    """서버 상태 확인 (V1 + V2)"""
    return {
        "status": "ok",
        "v1_yolo": {
            "available_models": list(models.keys()),
            "loaded_count": len(models),
        },
        "v2_rag": {
            "loaded": rag_chain is not None,
        },
        "v4_vlm": {
            "loaded": vlm_client is not None,
            "model": vlm_client.model if vlm_client else None,
            "host": vlm_client.host if vlm_client else None,
        },
    }


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
async def predict(
    file: UploadFile = File(..., description="추론할 이미지 파일"),
    conf: float = 0.25,
    model_type: str = Form(
        "polyp", description="모델 선택: polyp(위장 폴립) 또는 dental(치과 X-ray)"
    ),
):
    """
    이미지 업로드 -> 병변 세그멘테이션 추론

    - **file**: 이미지 파일 (jpg, png 등)
    - **conf**: confidence threshold (기본 0.25)
    - **model_type**: 모델 선택 — "polyp"(위장 폴립, 기본값) 또는 "dental"(치과 X-ray)

    반환: 검출된 병변 목록 (bbox, 신뢰도, 폴리곤 좌표)
    """
    if not models:
        raise HTTPException(503, "로드된 모델 없음")

    if model_type not in models:
        available = list(models.keys())
        raise HTTPException(
            400,
            f"'{model_type}' 모델이 없습니다. 사용 가능: {available}",
        )

    model = models[model_type]

    # 이미지 타입 검증
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(400, f"이미지 파일만 가능 (받은 타입: {file.content_type})")

    # 바이트 -> numpy 배열 -> OpenCV 이미지
    contents = await file.read()
    _validate_upload_size(contents)
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "이미지 디코딩 실패")

    infer_started_at = time.perf_counter()

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

    elapsed = _observe_inference("predict", model_type, infer_started_at)
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


@app.post("/explain")
async def explain_with_heatmap(
    file: UploadFile = File(..., description="설명 가능한 분석 대상 이미지"),
    conf: float = 0.25,
    model_type: str = Form("polyp", description="YOLOv8 모델 선택: polyp / dental"),
):
    if not models:
        raise HTTPException(503, "로드된 모델 없음")

    if model_type not in models:
        available = list(models.keys())
        raise HTTPException(
            400,
            f"'{model_type}' 모델이 없습니다. 사용 가능: {available}",
        )

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(400, f"이미지 파일만 가능 (받은 타입: {file.content_type})")

    contents = await file.read()
    _validate_upload_size(contents)
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "이미지 디코딩 실패")

    model = models[model_type]
    infer_started_at = time.perf_counter()
    results = model.predict(img, conf=conf, verbose=False)
    result = results[0]

    h, w = img.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)
    detections = []

    if result.boxes is not None:
        for i, box in enumerate(result.boxes):
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

    elapsed = _observe_inference("explain", model_type, infer_started_at)
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


# ── V2: RAG 엔드포인트 ─────────────────────────────────────────
@app.post("/ask", response_model=AskResponse)
async def ask_medical_knowledge(request: AskRequest):
    """
    [V2] 의료 지식 Q&A — RAG 기반

    PubMed 논문 / WHO 가이드라인 기반으로 질문에 답변합니다.
    답변과 함께 근거 문헌(출처 파일, 페이지)을 반환합니다.

    - **question**: 의료 관련 질문
    - **k**: 검색할 문서 청크 수 (기본 5)

    예시 질문:
    - "위장 폴립이란 무엇인가요?"
    - "대장내시경 검사 전 준비사항은?"
    - "폴립 제거 후 주의사항은?"
    """
    if rag_chain is None:
        raise HTTPException(
            status_code=503,
            detail="RAG 체인이 준비되지 않았습니다. 'python rag/ingest.py'를 먼저 실행하세요.",
        )

    try:
        result = await rag_chain.query(request.question)
        return AskResponse(
            answer=result["answer"],
            sources=[SourceInfo(**s) for s in result["sources"]],
            num_sources=result["num_sources"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG 쿼리 실패: {str(e)}")


@app.get("/ask/health")
def rag_health():
    """[V2] RAG 상태 확인"""
    return {
        "rag_ready": rag_chain is not None,
        "rag_llm_provider": getattr(rag_chain, "provider", None),
        "vectorstore_path": str(Path(__file__).parent.parent / "rag" / "vectorstore"),
        "vectorstore_exists": (
            Path(__file__).parent.parent / "rag" / "vectorstore"
        ).exists(),
    }


# ── V3: 통합 분석 엔드포인트 (Vision + LLM) ───────────────────
# 핵심 설계 의도:
# - /predict (Vision)와 /ask (LLM)가 독립적이면 "같은 서버에 올린 것" 에 불과
# - /analyze는 검출 결과를 기반으로 자동 RAG 질의 → Vision과 LLM이 실제로 연동
# - 면접 포인트: "멀티모달 파이프라인 — 이미지 분석 결과가 텍스트 질의로 이어지는 구조"
@app.post("/analyze")
async def analyze_with_knowledge(
    file: UploadFile = File(..., description="분석할 의료 이미지"),
    conf: float = 0.25,
    model_type: str = Form(
        "polyp", description="모델 선택: polyp(위장 폴립) 또는 dental(치과 X-ray)"
    ),
):
    """
    [V3] Vision + LLM 통합 분석

    1단계: YOLOv8로 이미지에서 병변 검출 (polyp 또는 dental 모델)
    2단계: 검출된 병변 정보를 기반으로 RAG에 자동 질의
    3단계: 검출 결과 + 의료 지식을 함께 반환

    이 엔드포인트가 프로젝트의 핵심 — Vision과 LLM이 실제로 연동되는 파이프라인.

    - **file**: 내시경/X-ray 이미지 파일
    - **conf**: confidence threshold (기본 0.25)
    - **model_type**: 모델 선택 — "polyp"(위장 폴립, 기본값) 또는 "dental"(치과 X-ray)

    반환:
    - detections: 검출된 병변 목록 (위치, 신뢰도, 마스크)
    - medical_knowledge: 검출된 병변에 대한 의료 지식 (RAG 기반)
    """
    if not models:
        raise HTTPException(503, "로드된 모델 없음")

    if model_type not in models:
        available = list(models.keys())
        raise HTTPException(
            400,
            f"'{model_type}' 모델이 없습니다. 사용 가능: {available}",
        )

    model = models[model_type]

    # ── 1단계: YOLOv8 추론 ──
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(400, f"이미지 파일만 가능 (받은 타입: {file.content_type})")

    contents = await file.read()
    _validate_upload_size(contents)
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "이미지 디코딩 실패")

    infer_started_at = time.perf_counter()

    results = model.predict(img, conf=conf, verbose=False)
    result = results[0]

    # 검출 결과 파싱
    detections = []
    detected_classes = set()  # 검출된 클래스 수집 (RAG 질의 생성용)

    if result.boxes is not None:
        for i, box in enumerate(result.boxes):
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

    # ── 2단계: 검출 결과 기반 RAG 자동 질의 ──
    # 검출된 병변이 있고 RAG가 준비되어 있으면 → 의료 지식 자동 제공
    medical_knowledge = None
    if detected_classes and rag_chain is not None:
        # 검출된 클래스를 한국어로 변환 (클래스명이 영문일 수 있음)
        class_names_kr = {
            "polyp": "위장 폴립",
            "Impacted": "매복치",
            "Caries": "충치",
            "Deep Caries": "깊은 충치",
            "Periapical Lesion": "치근단 병변",
        }
        detected_kr = [class_names_kr.get(c) or c for c in detected_classes]
        class_text = ", ".join(detected_kr)

        # 검출 결과를 자연어 질의로 변환
        auto_question = (
            f"이 환자의 영상에서 {class_text}이(가) {len(detections)}건 검출되었습니다. "
            f"{class_text}에 대한 임상적 의미, 권장 추적 관찰 주기, "
            f"그리고 환자에게 안내할 주의사항을 알려주세요."
        )

        try:
            rag_result = await rag_chain.query(auto_question)
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

    elapsed = _observe_inference("analyze", model_type, infer_started_at)
    logger.info(
        "analyze_completed",
        model_type=model_type,
        detections=len(detections),
        rag_used=rag_chain is not None,
        duration_ms=round(elapsed * 1000, 2),
    )

    return {
        "filename": file.filename,
        "model_type": model_type,
        "image_size": {"width": img.shape[1], "height": img.shape[0]},
        "detections": detections,
        "count": len(detections),
        # Vision + LLM 통합 결과
        "medical_knowledge": medical_knowledge,
        "rag_available": rag_chain is not None,
    }


async def _process_vlm_job(
    job_id: str,
    filename: str,
    contents: bytes,
    conf: float,
    model_type: str,
    language: str,
) -> None:
    async_jobs[job_id]["status"] = "running"
    async_jobs[job_id]["started_at"] = time.time()

    upload = UploadFile(filename=filename, file=io.BytesIO(contents))

    try:
        result = await vlm_analyze_with_knowledge(
            file=upload,
            conf=conf,
            model_type=model_type,
            language=language,
        )
        async_jobs[job_id]["status"] = "completed"
        async_jobs[job_id]["result"] = result
    except Exception as e:
        async_jobs[job_id]["status"] = "failed"
        async_jobs[job_id]["error"] = str(e)
        logger.exception("async_vlm_job_failed", job_id=job_id, error=str(e))
    finally:
        async_jobs[job_id]["finished_at"] = time.time()
        await upload.close()


@app.post("/vlm-analyze/async", status_code=202)
async def vlm_analyze_async(
    file: UploadFile = File(..., description="비동기 분석할 의료 이미지"),
    conf: float = 0.25,
    model_type: str = Form("polyp", description="YOLOv8 모델 선택: polyp / dental"),
    language: str = Form("ko", description="VLM 분석 언어: ko(한국어) / en(영어)"),
):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(400, f"이미지 파일만 가능 (받은 타입: {file.content_type})")

    contents = await file.read()
    _validate_upload_size(contents)
    job_id = str(uuid.uuid4())
    async_jobs[job_id] = {
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


@app.get("/jobs/{job_id}")
def get_job_status(job_id: str):
    job = async_jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"job_id를 찾을 수 없습니다: {job_id}")
    return job


# ── V4: VLM 통합 분석 엔드포인트 (VLM + Detection + RAG) ──────
# V3와의 핵심 차이:
# - V3: YOLOv8 검출 → 클래스명 → RAG (검출 모델에 의존)
# - V4: VLM이 이미지 직접 해석 → RAG 근거 보강 + YOLOv8 정량 검출 병행
# 면접 포인트:
#   "V3는 Detection 모델의 출력에 의존하지만, V4는 VLM이 이미지를 직접 이해합니다.
#    학습 데이터에 없는 희귀 병변도 VLM의 일반적 의료 지식으로 해석 가능하고,
#    RAG가 문헌 근거를 덧붙여 신뢰성을 높입니다."
@app.post("/vlm-analyze")
async def vlm_analyze_with_knowledge(
    file: UploadFile = File(..., description="분석할 의료 이미지"),
    conf: float = 0.25,
    model_type: str = Form("polyp", description="YOLOv8 모델 선택: polyp / dental"),
    language: str = Form("ko", description="VLM 분석 언어: ko(한국어) / en(영어)"),
):
    """
    [V4] VLM + Detection + RAG 통합 분석

    3단계 파이프라인:
      1단계: VLM(LLaVA)이 이미지를 직접 해석 → 자연어 소견 생성
      2단계: YOLOv8로 정량적 병변 검출 (bbox, mask, confidence)
      3단계: VLM 소견을 기반으로 RAG에 문헌 근거 질의 → 근거 보강

    V3와의 차이:
      - V3: 검출된 클래스명 → RAG (검출 못하면 설명도 불가)
      - V4: VLM이 자유 형식으로 해석 → 학습 안 된 병변도 설명 가능

    - **file**: 내시경/X-ray 이미지 파일
    - **conf**: YOLOv8 confidence threshold (기본 0.25)
    - **model_type**: YOLOv8 모델 — "polyp" 또는 "dental"
    - **language**: VLM 분석 언어 — "ko"(한국어, 기본) 또는 "en"(영어)
    """
    # ── 이미지 읽기 ──
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(400, f"이미지 파일만 가능 (받은 타입: {file.content_type})")

    contents = await file.read()
    _validate_upload_size(contents)
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "이미지 디코딩 실패")

    infer_started_at = time.perf_counter()

    response_data = {
        "filename": file.filename,
        "image_size": {"width": img.shape[1], "height": img.shape[0]},
    }

    # ── 1단계: VLM 분석 (LLaVA가 이미지 직접 해석) ──
    vlm_analysis = None
    if vlm_client is not None:
        try:
            # YOLOv8 검출 결과를 먼저 구해서 VLM에 컨텍스트로 제공 (선택적)
            detection_hints = None
            if models and model_type in models:
                quick_results = models[model_type].predict(
                    img, conf=conf, verbose=False
                )
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

            vlm_image_bytes, vlm_image_meta = _prepare_vlm_image_bytes(img, contents)

            # VLM에 이미지 전송 → 자연어 해석
            vlm_result = await vlm_client.analyze_with_context(
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

    # ── 2단계: YOLOv8 정량 검출 (bbox, mask, confidence) ──
    detections = []
    detected_classes = set()

    if models and model_type in models:
        model = models[model_type]
        results = model.predict(img, conf=conf, verbose=False)
        result = results[0]

        if result.boxes is not None:
            for i, box in enumerate(result.boxes):
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

    # ── 3단계: RAG 문헌 근거 보강 ──
    # VLM 해석을 기반으로 RAG 질의 (V3과 다른 점: 클래스명이 아닌 VLM 소견 기반)
    medical_evidence = None
    if rag_chain is not None:
        # VLM 분석 결과 또는 검출 결과를 RAG 질의로 변환
        rag_query = None

        if vlm_analysis and "interpretation" in vlm_analysis:
            # V4 핵심: VLM 해석을 RAG 질의로 사용
            vlm_text = vlm_analysis["interpretation"]
            class_names_kr = {
                "polyp": "위장 폴립",
                "Impacted": "매복치",
                "Caries": "충치",
                "Deep Caries": "깊은 충치",
                "Periapical Lesion": "치근단 병변",
            }
            detected_kr = [class_names_kr.get(c) or c for c in detected_classes]
            keyword_hint = ", ".join(detected_kr) if detected_kr else "검출 클래스 없음"
            rag_query = (
                f"다음 의료 영상 분석 소견에 대한 임상 근거와 "
                f"권장 조치를 문헌에서 찾아주세요. "
                f"핵심 키워드: {keyword_hint}\n\n{vlm_text[:500]}"
            )
        elif detected_classes:
            # VLM 실패 시 폴백: V3 방식 (검출 클래스 기반)
            class_names_kr = {
                "polyp": "위장 폴립",
                "Impacted": "매복치",
                "Caries": "충치",
                "Deep Caries": "깊은 충치",
                "Periapical Lesion": "치근단 병변",
            }
            detected_kr = [class_names_kr.get(c) or c for c in detected_classes]
            class_text = ", ".join(detected_kr)
            rag_query = (
                f"{class_text}이(가) 검출되었습니다. "
                f"임상적 의미와 권장 추적 관찰 주기를 알려주세요."
            )

        if rag_query:
            try:
                rag_result = await rag_chain.query(rag_query)
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
                    web_evidence = await _fetch_pubmed_web_evidence(
                        query=rag_query,
                        model_type=model_type,
                    )
                    if web_evidence:
                        answer_text = web_evidence.get("answer", NO_EVIDENCE_TEXT)
                        sources_payload = web_evidence.get("sources", [])
                        fallback_reason = web_evidence.get(
                            "reason", "web_pubmed_fallback"
                        )
                        has_evidence = web_evidence.get("num_sources", 0) > 0
                    else:
                        answer_text = NO_EVIDENCE_TEXT
                        fallback_reason = "no_evidence"

                medical_evidence = {
                    "query_used": rag_query[:200] + "..."
                    if len(rag_query) > 200
                    else rag_query,
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
    response_data["rag_available"] = rag_chain is not None
    response_data["vlm_available"] = vlm_client is not None

    elapsed = _observe_inference("vlm_analyze", model_type, infer_started_at)
    logger.info(
        "vlm_analyze_completed",
        model_type=model_type,
        detections=len(detections),
        vlm_available=vlm_client is not None,
        rag_available=rag_chain is not None,
        duration_ms=round(elapsed * 1000, 2),
    )

    return response_data
