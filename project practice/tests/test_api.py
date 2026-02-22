"""
FastAPI 엔드포인트 테스트 (pytest)

실행:
    cd "project practice"
    python -m pytest tests/test_api.py -v

테스트 구조:
    - test_health: 서버 상태 확인 (멀티모델 + RAG)
    - test_predict_*: YOLOv8 세그멘테이션 추론 API (polyp/dental 멀티모델)
    - test_ask_*: RAG 의료 지식 Q&A API
    - test_input_validation_*: 잘못된 입력 처리 검증
    - test_analyze_*: V3 Vision + LLM 통합 분석 (멀티모델)

면접 포인트:
    - TestClient: 서버를 실제로 띄우지 않고 HTTP 요청 테스트 (단위 테스트)
    - fixture: 테스트 간 공유 자원 관리 (client, 이미지 등)
    - parametrize: 동일 로직을 여러 입력으로 반복 테스트
"""

import io
import pytest
import numpy as np
import cv2
from fastapi.testclient import TestClient

# ── app 임포트 ──────────────────────────────────────────────────
# TestClient는 실제 uvicorn 없이 FastAPI 앱을 직접 호출
from api.app import app


# ── Fixtures (테스트 공용 자원) ──────────────────────────────────
@pytest.fixture(scope="module")
def client():
    """
    TestClient 생성 (모듈 단위 재사용)
    scope="module": 이 파일의 모든 테스트가 같은 client 공유
    → 모델 로드가 1번만 일어남 (테스트 속도 향상)
    """
    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_image_bytes():
    """
    테스트용 더미 이미지 생성 (640x480 검정 이미지)
    실제 폴립 이미지가 아니므로 검출 결과는 0개가 정상
    """
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    _, buffer = cv2.imencode(".jpg", img)
    return buffer.tobytes()


@pytest.fixture
def sample_polyp_image_bytes():
    """
    테스트용 컬러 이미지 (실제 폴립은 아니지만 유효한 이미지)
    흰색 원 → 폴립과 유사한 형태
    """
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(img, (320, 240), 100, (200, 150, 180), -1)  # 분홍색 원
    _, buffer = cv2.imencode(".jpg", img)
    return buffer.tobytes()


# ══════════════════════════════════════════════════════════════════
# 1. Health 엔드포인트 테스트
# ══════════════════════════════════════════════════════════════════
class TestHealth:
    """서버 상태 확인 테스트"""

    def test_health_returns_ok(self, client):
        """GET /health → status: ok"""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "ok"

    def test_health_v1_model_info(self, client):
        """V1 멀티모델 정보가 health 응답에 포함되는지"""
        data = client.get("/health").json()
        assert "v1_yolo" in data
        assert "available_models" in data["v1_yolo"]
        assert "loaded_count" in data["v1_yolo"]
        # 모델 목록은 리스트, best.pt가 있으면 "polyp"이 포함
        assert isinstance(data["v1_yolo"]["available_models"], list)
        assert isinstance(data["v1_yolo"]["loaded_count"], int)

    def test_health_v2_rag_info(self, client):
        """V2 RAG 정보가 health 응답에 포함되는지"""
        data = client.get("/health").json()
        assert "v2_rag" in data
        assert "loaded" in data["v2_rag"]

    def test_rag_health_endpoint(self, client):
        """GET /ask/health → RAG 상태 정보 반환"""
        response = client.get("/ask/health")
        assert response.status_code == 200

        data = response.json()
        assert "rag_ready" in data
        assert "vectorstore_path" in data
        assert "vectorstore_exists" in data


# ══════════════════════════════════════════════════════════════════
# 2. Predict 엔드포인트 테스트 (V1: YOLOv8 세그멘테이션)
# ══════════════════════════════════════════════════════════════════
class TestPredict:
    """POST /predict — YOLOv8 병변 세그멘테이션 추론 (멀티모델)"""

    def test_predict_valid_image(self, client, sample_image_bytes):
        """유효한 이미지 업로드 → 200 응답 + 결과 형식 확인"""
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )
        assert response.status_code == 200

        data = response.json()
        # 응답 필드 존재 확인
        assert "filename" in data
        assert "model_type" in data
        assert "image_size" in data
        assert "detections" in data
        assert "count" in data
        # 타입 확인
        assert isinstance(data["detections"], list)
        assert isinstance(data["count"], int)
        assert data["count"] >= 0

    def test_predict_default_model_type(self, client, sample_image_bytes):
        """model_type 미지정 시 기본값 'polyp'"""
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )
        data = response.json()
        assert data["model_type"] == "polyp"

    def test_predict_explicit_polyp_model(self, client, sample_image_bytes):
        """model_type='polyp' 명시적 전달"""
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
            data={"model_type": "polyp"},
        )
        assert response.status_code == 200
        assert response.json()["model_type"] == "polyp"

    def test_predict_invalid_model_type(self, client, sample_image_bytes):
        """존재하지 않는 model_type → 400"""
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
            data={"model_type": "nonexistent"},
        )
        assert response.status_code == 400

    def test_predict_response_image_size(self, client, sample_image_bytes):
        """이미지 크기가 정확히 반환되는지"""
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )
        data = response.json()
        assert data["image_size"]["width"] == 640
        assert data["image_size"]["height"] == 480

    def test_predict_with_custom_conf(self, client, sample_image_bytes):
        """confidence threshold 커스텀 값 전달"""
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
            data={"conf": 0.5},
        )
        assert response.status_code == 200

    def test_predict_filename_preserved(self, client, sample_image_bytes):
        """업로드한 파일명이 응답에 그대로 반환되는지"""
        response = client.post(
            "/predict",
            files={"file": ("my_polyp_image.jpg", sample_image_bytes, "image/jpeg")},
        )
        data = response.json()
        assert data["filename"] == "my_polyp_image.jpg"

    def test_predict_png_format(self, client):
        """PNG 이미지도 처리 가능한지"""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buffer = cv2.imencode(".png", img)

        response = client.post(
            "/predict",
            files={"file": ("test.png", buffer.tobytes(), "image/png")},
        )
        assert response.status_code == 200


# ══════════════════════════════════════════════════════════════════
# 3. 입력 검증 테스트 (에러 케이스)
# ══════════════════════════════════════════════════════════════════
class TestInputValidation:
    """잘못된 입력에 대한 에러 처리 검증"""

    def test_predict_no_file(self, client):
        """파일 없이 요청 → 422 (Validation Error)"""
        response = client.post("/predict")
        assert response.status_code == 422

    def test_predict_invalid_image_data(self, client):
        """유효하지 않은 바이트 → 400"""
        response = client.post(
            "/predict",
            files={"file": ("bad.jpg", b"not an image", "image/jpeg")},
        )
        assert response.status_code == 400

    def test_predict_non_image_content_type(self, client):
        """이미지가 아닌 파일 타입 → 400"""
        response = client.post(
            "/predict",
            files={"file": ("doc.pdf", b"fake pdf content", "application/pdf")},
        )
        assert response.status_code == 400

    def test_ask_empty_question(self, client):
        """빈 질문은 422 (Pydantic 검증 실패) 반환"""
        response = client.post("/ask", json={})
        assert response.status_code == 422


# ══════════════════════════════════════════════════════════════════
# 4. RAG 엔드포인트 테스트 (V2)
# ══════════════════════════════════════════════════════════════════
class TestRAG:
    """
    POST /ask — RAG 의료 지식 Q&A

    주의: RAG 체인이 초기화되지 않은 환경에서는 503 반환이 정상.
    OpenAI API 키가 없거나 벡터스토어가 없으면 RAG는 로드되지 않음.
    """

    def test_ask_returns_valid_status(self, client):
        """
        RAG 상태에 따라 적절한 HTTP 상태코드 반환 확인
        - 200: RAG 정상 동작 (답변 성공)
        - 500: RAG 로드됐지만 LLM API 실패 (토큰 소진, 네트워크 등)
        - 503: RAG 미초기화 (벡터스토어 없음)
        """
        response = client.post(
            "/ask",
            json={"question": "폴립이란 무엇인가요?"},
        )
        assert response.status_code in (200, 500, 503), (
            f"예상치 못한 상태코드: {response.status_code}"
        )

    def test_ask_response_format_when_available(self, client):
        """
        RAG가 사용 가능하고 LLM API도 정상일 때 응답 형식 검증
        (RAG 미로드 또는 API 실패 시 skip)
        """
        health = client.get("/ask/health").json()
        if not health["rag_ready"]:
            pytest.skip("RAG 체인 미초기화 (API 키 또는 벡터스토어 없음)")

        response = client.post(
            "/ask",
            json={"question": "위장 폴립이란 무엇인가요?"},
        )
        if response.status_code == 500:
            pytest.skip("LLM API 호출 실패 (토큰 소진 또는 네트워크 문제)")

        assert response.status_code == 200

        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "num_sources" in data
        assert isinstance(data["sources"], list)
        assert isinstance(data["num_sources"], int)

    def test_ask_with_custom_k(self, client):
        """k 파라미터(검색 문서 수) 전달 테스트"""
        response = client.post(
            "/ask",
            json={"question": "test", "k": 3},
        )
        # RAG 상태와 무관하게 서버가 크래시 없이 응답하는지 확인
        assert response.status_code in (200, 500, 503)


# ══════════════════════════════════════════════════════════════════
# 5. 통합 분석 엔드포인트 테스트 (V3: Vision + LLM)
# ══════════════════════════════════════════════════════════════════
class TestAnalyze:
    """
    POST /analyze — Vision + LLM 통합 분석

    이 엔드포인트가 프로젝트의 핵심:
    YOLOv8 검출 결과 → 자동 RAG 질의 → 의료 지식 포함 응답
    """

    def test_analyze_returns_valid_response(self, client, sample_image_bytes):
        """유효한 이미지 → 200 + 검출 결과 + medical_knowledge 필드"""
        response = client.post(
            "/analyze",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )
        assert response.status_code == 200

        data = response.json()
        # 기본 검출 결과 필드
        assert "filename" in data
        assert "model_type" in data
        assert "detections" in data
        assert "count" in data
        # V3 통합 필드
        assert "medical_knowledge" in data
        assert "rag_available" in data

    def test_analyze_default_model_type(self, client, sample_image_bytes):
        """model_type 미지정 시 기본값 'polyp'"""
        response = client.post(
            "/analyze",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )
        data = response.json()
        assert data["model_type"] == "polyp"

    def test_analyze_no_detection_no_rag(self, client, sample_image_bytes):
        """검출 결과 0건이면 medical_knowledge는 None"""
        response = client.post(
            "/analyze",
            files={"file": ("blank.jpg", sample_image_bytes, "image/jpeg")},
        )
        data = response.json()
        # 검정 이미지 → 폴립 검출 0건 → RAG 질의 안 함
        if data["count"] == 0:
            assert data["medical_knowledge"] is None

    def test_analyze_invalid_file(self, client):
        """유효하지 않은 파일 → 400"""
        response = client.post(
            "/analyze",
            files={"file": ("bad.jpg", b"not an image", "image/jpeg")},
        )
        assert response.status_code == 400

    def test_analyze_with_custom_conf(self, client, sample_image_bytes):
        """confidence threshold 커스텀 값"""
        response = client.post(
            "/analyze",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
            data={"conf": 0.8},
        )
        assert response.status_code == 200

    def test_analyze_invalid_model_type(self, client, sample_image_bytes):
        """존재하지 않는 model_type → 400"""
        response = client.post(
            "/analyze",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
            data={"model_type": "nonexistent"},
        )
        assert response.status_code == 400
