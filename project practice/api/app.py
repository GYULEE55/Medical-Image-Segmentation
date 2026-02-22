"""
Medical AI Assistant API — V3 (YOLOv8 + RAG + Multi-Model)

엔드포인트:
  GET  /health          — 서버 상태 확인
  POST /predict         — 병변 세그멘테이션 (YOLOv8) [V1] — model 파라미터로 polyp/dental 선택
  POST /ask             — 의료 지식 Q&A (RAG) [V2]
  POST /analyze         — 이미지 검출 + 의료 지식 통합 분석 [V3 핵심]
  GET  /ask/health      — RAG 상태 확인

V1: YOLOv8n-seg로 병변 검출 + 세그멘테이션 (polyp/dental 멀티모델)
V2: LangChain + ChromaDB + GPT-4o-mini로 의료 지식 Q&A (출처 포함)
V3: Vision(YOLOv8) + LLM(RAG) 통합 — 검출 결과를 기반으로 의료 지식 자동 제공
"""

import os
import cv2
import numpy as np
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

# .env 파일에서 환경변수 로드 (OPENAI_API_KEY 등)
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from pydantic import BaseModel
from ultralytics import YOLO

# ── 경로 설정 ──────────────────────────────────────────────────
# 멀티모델 지원: polyp(위장 폴립) + dental(치과 X-ray)
MODEL_PATHS: Dict[str, str] = {
    "polyp": os.getenv("MODEL_PATH", str(Path(__file__).parent.parent / "best.pt")),
    "dental": os.getenv("DENTAL_MODEL_PATH", str(Path(__file__).parent.parent / "best_dentex.pt")),
}

# ── 전역 상태 ──────────────────────────────────────────────────
models: Dict[str, YOLO] = {}  # 멀티모델 딕셔너리 {"polyp": YOLO, "dental": YOLO}
rag_chain = None   # RAG 체인 (V2)


# ── 앱 초기화 (lifespan 패턴) ──────────────────────────────────
# @app.on_event("startup") 는 deprecated → asynccontextmanager lifespan 사용
@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 모델 초기화 (1회만 실행)"""
    global models, rag_chain

    # V1: YOLOv8 모델 로드 (멀티모델 — 존재하는 모델만 로드)
    for model_name, model_path in MODEL_PATHS.items():
        if Path(model_path).exists():
            models[model_name] = YOLO(model_path)
            print(f"[V1] {model_name} 모델 로드 완료: {model_path}")
        else:
            print(f"[V1] {model_name} 모델 파일 없음 (스킵): {model_path}")

    if not models:
        print("[V1] 경고: 로드된 모델이 없습니다!")

    # V2: RAG 체인 초기화 (ChromaDB가 있을 때만)
    rag_vectorstore = Path(__file__).parent.parent / "rag" / "vectorstore"
    if rag_vectorstore.exists():
        try:
            from rag.chain import MedicalRAGChain
            rag_chain = MedicalRAGChain()
            print("[V2] RAG 체인 초기화 완료")
        except Exception as e:
            print(f"[V2] RAG 초기화 실패 (계속 진행): {e}")
    else:
        print("[V2] RAG 벡터스토어 없음 — 'python rag/ingest.py' 먼저 실행 필요")

    yield  # 서버 실행


app = FastAPI(
    title="Medical AI Assistant API",
    description="""
## Medical AI Assistant — Vision + LLM 통합 서비스

### V1: 폴립 세그멘테이션
- `POST /predict` — 내시경 이미지 업로드 → 폴립 검출 + 마스크

### V2: 의료 지식 RAG
- `POST /ask` — 의료 질문 → 근거 문헌 기반 답변 + 출처

### V3: 통합 분석 (Vision + LLM)
- `POST /analyze` — 이미지 검출 → 검출 결과 기반 자동 RAG 질의 → 의료 지식 포함 응답
    """,
    version="3.0.0",
    lifespan=lifespan,
)


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
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(..., description="추론할 이미지 파일"),
    conf: float = 0.25,
    model_type: str = Form("polyp", description="모델 선택: polyp(위장 폴립) 또는 dental(치과 X-ray)"),
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
        "model_type": model_type,
        "image_size": {"width": img.shape[1], "height": img.shape[0]},
        "detections": detections,
        "count": len(detections),
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
        "vectorstore_path": str(
            Path(__file__).parent.parent / "rag" / "vectorstore"
        ),
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
    model_type: str = Form("polyp", description="모델 선택: polyp(위장 폴립) 또는 dental(치과 X-ray)"),
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
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "이미지 디코딩 실패")

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
        detected_kr = [class_names_kr.get(c, c) for c in detected_classes]
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
