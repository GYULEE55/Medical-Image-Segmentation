"""
Medical VLM Client — ollama + LLaVA 연동 모듈

역할:
    의료 이미지를 ollama의 LLaVA 모델에 전송하여 자연어 해석을 받는 클라이언트.
    V4 아키텍처의 핵심 — VLM이 이미지를 직접 이해하여 의료 소견을 생성.

사용법:
    from vlm.client import MedicalVLMClient
    vlm = MedicalVLMClient()
    result = await vlm.analyze_image(image_bytes)

면접 포인트:
    - VLM(Vision Language Model): 이미지 + 텍스트를 동시에 이해하는 멀티모달 모델
    - LLaVA: Visual Instruction Tuning으로 학습된 오픈소스 VLM (7B 파라미터)
    - ollama: Llama.cpp 기반 로컬 LLM 서빙 엔진 (Apple Silicon Metal 가속 지원)
    - V3→V4 차이: Detection 의존 → VLM 직접 해석 (학습 안 된 병변도 설명 가능)
"""

import asyncio
import base64
import os
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env", override=True)


# ── 설정 ──────────────────────────────────────────────────────────
# ollama 기본 포트: 11434
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
VLM_MODEL = os.getenv("VLM_MODEL", "llava")
VLM_NUM_PREDICT = int(os.getenv("VLM_NUM_PREDICT", "512"))

# ── 의료 영상 해석 전용 프롬프트 ──────────────────────────────────
# 핵심 설계 원칙:
# 1. 의료 전문가 역할 부여 → 의학적으로 의미 있는 소견 생성
# 2. 구조화된 출력 요구 → 파싱 가능한 응답
# 3. 확신도 표현 요구 → 의료 정보 신뢰성 확보
# 4. 진단 금지 → 법적/윤리적 안전장치
MEDICAL_VLM_PROMPT = """You are a medical imaging AI assistant analyzing a clinical image.

Provide a structured analysis following this format:

1. **Image Type**: What type of medical image is this? (endoscopy, X-ray, CT, etc.)
2. **Observations**: Describe visible anatomical structures and any abnormalities.
3. **Potential Findings**: List possible clinical findings with confidence level (high/medium/low).
4. **Recommended Actions**: Suggest follow-up examinations or clinical considerations.

Important rules:
- Be specific about locations, sizes, and characteristics of findings.
- Always express uncertainty levels — never state a definitive diagnosis.
- Use standard medical terminology.
- If the image quality is poor or unclear, state that explicitly.
- This is for educational/research purposes only — not for clinical diagnosis.
- Keep the final answer short (3-5 sentences).
- Use plain, easy-to-understand language.

Analyze the provided medical image:"""

# 한국어 프롬프트 (선택적)
MEDICAL_VLM_PROMPT_KR = """당신은 의료 영상을 분석하는 AI 어시스턴트입니다.

다음 형식으로 구조화된 분석을 제공하세요:

1. **영상 종류**: 이 의료 영상의 종류는? (내시경, X-ray, CT 등)
2. **관찰 소견**: 보이는 해부학적 구조물과 이상 소견을 기술하세요.
3. **추정 소견**: 가능한 임상적 소견을 확신도(높음/중간/낮음)와 함께 나열하세요.
4. **권장 조치**: 추가 검사나 임상적 고려사항을 제안하세요.

규칙:
- 소견의 위치, 크기, 특성을 구체적으로 기술하세요.
- 확정 진단은 절대 하지 마세요 — 항상 불확실성을 표현하세요.
- 의학 용어를 쓰더라도 쉬운 한국어로 풀어서 설명하세요.
- 영상 품질이 좋지 않으면 명시하세요.
- 이 분석은 교육/연구 목적이며, 임상 진단용이 아닙니다.
- 최종 답변은 3~5문장으로 짧게 작성하세요.
- 불필요한 장문 설명은 피하고 핵심만 말하세요.

제공된 의료 영상을 분석하세요:"""


class MedicalVLMClient:
    """
    의료 VLM 클라이언트 (ollama + LLaVA)

    ollama의 REST API를 통해 LLaVA 모델에 이미지를 전송하고
    의료 영상 해석을 받는 비동기 클라이언트.

    면접 포인트:
        - httpx.AsyncClient 사용 → FastAPI async endpoint와 자연스럽게 통합
        - 타임아웃 설정 → VLM 추론은 수십 초 걸릴 수 있어 충분한 타임아웃 필요
        - 연결 상태 확인 → 서버 시작 시 ollama 서버 상태를 미리 체크
    """

    def __init__(
        self,
        host: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 120.0,
        language: str = "ko",
    ):
        """
        Args:
            host: ollama 서버 주소 (기본: http://localhost:11434)
            model: 사용할 VLM 모델명 (기본: llava)
            timeout: API 호출 타임아웃 초 (VLM 추론은 느릴 수 있음)
            language: 프롬프트 언어 ("ko" 한국어, "en" 영어)
        """
        self.host = host or OLLAMA_HOST
        self.model = model or VLM_MODEL
        self.timeout = timeout

        # 언어별 프롬프트 선택
        self.system_prompt = MEDICAL_VLM_PROMPT_KR if language == "ko" else MEDICAL_VLM_PROMPT

        # httpx 비동기 클라이언트 (재사용 — 매 요청마다 생성하면 비효율)
        self._client = httpx.AsyncClient(
            base_url=self.host,
            timeout=httpx.Timeout(self.timeout, connect=10.0),
        )

    async def close(self):
        """HTTP 클라이언트 정리 (서버 종료 시 호출)"""
        await self._client.aclose()

    async def is_available(self) -> bool:
        """
        ollama 서버 연결 상태 확인

        서버 시작 시 호출하여 VLM 사용 가능 여부를 미리 파악.
        연결 실패해도 서버는 정상 실행 — VLM 기능만 비활성화.
        """
        try:
            response = await self._client.get("/api/tags")
            if response.status_code == 200:
                # 모델 목록에서 사용할 모델이 있는지 확인
                data = response.json()
                model_names = [m["name"] for m in data.get("models", [])]
                # "llava" 또는 "llava:latest" 등 부분 매칭
                return any(self.model in name for name in model_names)
            return False
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    async def analyze_image(
        self,
        image_bytes: bytes,
        custom_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        이미지를 VLM에 전송하여 의료 영상 해석을 받습니다.

        Args:
            image_bytes: 이미지 바이트 데이터 (JPEG/PNG)
            custom_prompt: 커스텀 프롬프트 (기본: 의료 영상 분석 프롬프트)

        Returns:
            {
                "analysis": str,       # VLM의 영상 해석 텍스트
                "model": str,          # 사용된 모델명
                "prompt_used": str,    # 사용된 프롬프트 (축약)
                "total_duration_ms": int,  # 총 소요 시간 (밀리초)
            }

        Raises:
            ConnectionError: ollama 서버 연결 실패
            RuntimeError: VLM 추론 실패
        """
        # 이미지를 base64로 인코딩 (ollama API 요구사항)
        # ollama는 이미지를 base64 문자열로 받음
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        prompt = custom_prompt or self.system_prompt

        # ollama REST API: POST /api/chat
        # 참고: https://github.com/ollama/ollama/blob/main/docs/api.md
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_b64],  # base64 인코딩된 이미지 리스트
                }
            ],
            "stream": False,  # 스트리밍 비활성화 — 전체 응답을 한 번에 받음
            "options": {
                "temperature": 0.2,  # 의료 분석은 일관성이 중요 → 낮은 temperature
                "num_predict": VLM_NUM_PREDICT,
            },
        }

        try:
            response = await self._client.post("/api/chat", json=payload)
            response.raise_for_status()
        except httpx.ConnectError:
            raise ConnectionError(
                f"ollama 서버에 연결할 수 없습니다: {self.host}\n"
                "'ollama serve'가 실행 중인지 확인하세요."
            )
        except httpx.TimeoutException:
            raise RuntimeError(
                f"VLM 추론 타임아웃 ({self.timeout}초 초과). "
                "이미지가 너무 크거나 서버 부하가 높을 수 있습니다."
            )
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"ollama API 오류: {e.response.status_code} — {e.response.text}")

        result = response.json()

        # 응답 파싱
        # ollama /api/chat 응답 형식:
        # {
        #   "model": "llava",
        #   "message": {"role": "assistant", "content": "..."},
        #   "total_duration": 12345678900 (나노초),
        #   ...
        # }
        analysis_text = result.get("message", {}).get("content", "")
        total_duration_ns = result.get("total_duration", 0)

        return {
            "analysis": analysis_text,
            "model": result.get("model", self.model),
            "prompt_used": prompt[:100] + "..." if len(prompt) > 100 else prompt,
            "total_duration_ms": total_duration_ns // 1_000_000,  # 나노초 → 밀리초
        }

    async def analyze_with_context(
        self,
        image_bytes: bytes,
        detection_results: Optional[list[dict[str, Any]]] = None,
        model_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        YOLOv8 검출 결과를 컨텍스트로 제공하여 VLM 분석 정확도를 높입니다.

        V4 핵심 기능:
        - YOLOv8이 검출한 병변 정보를 VLM에 힌트로 제공
        - VLM은 힌트를 참고하되, 이미지 전체를 자유롭게 해석
        - Detection(정량) + VLM(정성) 상호 보완

        Args:
            image_bytes: 이미지 바이트
            detection_results: YOLOv8 검출 결과 리스트 (optional)
                [{class: "polyp", confidence: 0.92, bbox: [...]}, ...]

        Returns:
            analyze_image()와 동일한 형식
        """
        modality_hint = None
        if model_type == "polyp":
            modality_hint = "endoscopy"
        elif model_type == "dental":
            modality_hint = "dental panoramic X-ray"

        modality_constraint = ""
        if modality_hint is not None:
            modality_constraint = (
                "\n\n영상 도메인 힌트:\n"
                f"- model_type={model_type}\n"
                f"- expected modality: {modality_hint}\n"
                "- 다른 modality(예: CT/MRI)로 단정하지 말고, 불확실하면 '확인 불가'라고 명시하세요."
            )

        # YOLOv8 검출 결과가 있으면 프롬프트에 컨텍스트 추가
        if detection_results:
            # 검출 결과를 자연어로 변환
            det_summary = []
            for det in detection_results:
                cls = det.get("class", "unknown")
                conf = det.get("confidence", 0)
                det_summary.append(f"- {cls} (confidence: {conf:.2f})")

            detection_context = "\n".join(det_summary)

            context_prompt = (
                f"{self.system_prompt}{modality_constraint}\n\n"
                f"참고: 객체 검출 모델(YOLOv8)이 이 영상에서 다음을 검출했습니다:\n"
                f"{detection_context}\n\n"
                f"위 검출 결과를 참고하되, 영상 전체를 종합적으로 분석해주세요. "
                f"검출 모델이 놓쳤을 수 있는 추가 소견도 포함해주세요."
            )
        else:
            context_prompt = f"{self.system_prompt}{modality_constraint}"

        return await self.analyze_image(image_bytes, custom_prompt=context_prompt)


# ── 로컬 테스트 ────────────────────────────────────────────────
if __name__ == "__main__":

    async def test_vlm():
        """VLM 클라이언트 테스트"""
        client = MedicalVLMClient()

        # 1. 연결 확인
        print(f"ollama 서버: {client.host}")
        print(f"VLM 모델: {client.model}")

        available = await client.is_available()
        print(f"사용 가능: {available}")

        if not available:
            print("ollama 서버가 실행 중이 아니거나 모델이 설치되지 않았습니다.")
            print("  1. ollama serve  (서버 실행)")
            print("  2. ollama pull llava  (모델 다운로드)")
            await client.close()
            return

        # 2. 테스트 이미지로 분석
        test_image_path = Path(__file__).parent.parent / "bus.jpg"
        if test_image_path.exists():
            print(f"\n테스트 이미지: {test_image_path}")
            image_bytes = test_image_path.read_bytes()

            print("VLM 분석 중... (30초~1분 소요)")
            result = await client.analyze_image(image_bytes)

            print("\n[분석 결과]")
            print(f"모델: {result['model']}")
            print(f"소요 시간: {result['total_duration_ms']}ms")
            print(f"해석:\n{result['analysis']}")
        else:
            print(f"테스트 이미지 없음: {test_image_path}")

        await client.close()

    asyncio.run(test_vlm())
