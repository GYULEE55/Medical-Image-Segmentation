# MacBook 환경 설정 가이드

> Windows에서 개발한 Medical AI 프로젝트를 MacBook(24GB RAM)으로 이전하여
> **ollama + LLaVA VLM**을 로컬에서 실행하는 가이드입니다.

---

## 왜 MacBook인가?

| 항목 | Windows (GTX 1650 4GB) | MacBook (24GB RAM) |
|------|:---:|:---:|
| YOLOv8 추론 | ✅ (CPU/GPU) | ✅ (CPU, Apple Silicon) |
| RAG (ChromaDB + GPT-4o-mini) | ✅ | ✅ |
| **VLM (LLaVA 7B)** | ❌ (VRAM 부족) | **✅ (ollama + Metal)** |
| 학습 (Training) | ❌ → Colab | ❌ → Colab |

**핵심**: LLaVA 7B 모델은 최소 8GB RAM 필요. MacBook 24GB면 넉넉하게 실행 가능.
ollama는 Apple Silicon의 **Metal GPU 가속**을 자동으로 사용합니다.

---

## 1단계: 프로젝트 클론

```bash
# GitHub에서 클론
git clone https://github.com/GYULEE55/Medical-Image-Segmentation.git
cd Medical-Image-Segmentation/project\ practice/
```

---

## 2단계: Python 환경 설정

```bash
# Python 3.10+ 설치 (Homebrew)
brew install python@3.11

# 가상환경 생성 + 활성화
python3.11 -m venv venv
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

---

## 3단계: ollama 설치 + LLaVA 모델 다운로드

### ollama 설치

```bash
# Homebrew로 설치 (가장 간단)
brew install ollama

# 또는 공식 사이트에서 다운로드
# https://ollama.ai/download/mac
```

### ollama 서버 시작

```bash
# 터미널 1: ollama 서버 실행 (백그라운드)
ollama serve
```

### LLaVA 모델 다운로드

```bash
# 터미널 2: LLaVA 7B 다운로드 (~4.7GB)
ollama pull llava

# 다운로드 확인
ollama list
```
알ㅇ
### 모델 테스트

```bash
# 텍스트 질의 테스트
ollama run llava "What is a polyp in gastroenterology?"

# 이미지 질의 테스트 (로컬 이미지)
# → 이건 Python API로 테스트합니다 (아래 참조)
```

**면접 포인트**: 
- ollama는 Llama.cpp 기반 — C++로 최적화된 추론 엔진
- Apple Silicon Metal GPU 가속 자동 적용
- 모델을 GGUF 양자화 포맷으로 변환하여 메모리 효율적 실행
- LLaVA = **L**arge **L**anguage **a**nd **V**ision **A**ssistant (Visual Instruction Tuning)

---

## 4단계: 환경 변수 설정

```bash
# .env 파일 생성 (project practice/ 안에)
cat > .env << 'EOF'
# OpenAI API (RAG용)
OPENAI_API_KEY=sk-your-key-here

# VLM 설정 (ollama 로컬)
OLLAMA_HOST=http://localhost:11434
VLM_MODEL=llava
EOF
```

---

## 5단계: RAG 벡터스토어 구축

```bash
# PDF → ChromaDB 인덱싱 (최초 1회, BGE-M3 임베딩 모델 다운로드 포함)
python rag/ingest.py
```

> ⚠️ 첫 실행 시 BGE-M3 모델(~2.3GB) 다운로드됩니다. Wi-Fi 환경에서 실행하세요.

---

## 6단계: 서버 실행

```bash
# 터미널 1: ollama 서버 (이미 실행 중이면 스킵)
ollama serve

# 터미널 2: FastAPI 서버
cd "project practice/"
uvicorn api.app:app --reload --port 8000
```

서버 실행 후: http://localhost:8000/docs 에서 Swagger UI로 테스트

---

## 7단계: VLM 테스트

```bash
# 헬스 체크
curl http://localhost:8000/health

# VLM 분석 테스트 (이미지 + VLM 해석 + RAG 근거)
curl -X POST http://localhost:8000/vlm-analyze \
  -F "file=@bus.jpg"
```

### Python으로 테스트

```python
import requests

# VLM 분석
with open("bus.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/vlm-analyze",
        files={"file": ("test.jpg", f, "image/jpeg")},
    )
    print(response.json())
```

---

## 트러블슈팅

### ollama 서버 연결 실패

```
ConnectionError: Cannot connect to ollama at http://localhost:11434
```

**해결**: `ollama serve`가 실행 중인지 확인. 별도 터미널에서 실행 필요.

### 메모리 부족 (LLaVA 로드 실패)

```
Error: model requires more memory than available
```

**해결**: 
- 다른 메모리 사용 앱 종료
- LLaVA 7B는 ~8GB 필요, 24GB RAM이면 충분하지만 Chrome 등이 많이 사용할 수 있음
- 최악의 경우: `ollama pull llava:7b-v1.6-mistral-q4_0` (더 작은 양자화 버전)

### Apple Silicon vs Intel Mac

- **Apple Silicon (M1/M2/M3)**: Metal GPU 가속 자동 적용 → 빠름
- **Intel Mac**: CPU만 사용 → 느리지만 동작은 함

---

## 아키텍처 비교: V3 vs V4

### V3 (현재): YOLOv8 → RAG

```
이미지 → [YOLOv8] 병변 검출 → 클래스명("polyp") → [RAG] 문헌 검색 → 답변
```

- YOLOv8이 **검출한 것만** RAG에 질의 가능
- 학습하지 않은 병변은 아예 설명 불가

### V4 (VLM 추가): VLM → RAG

```
이미지 → [LLaVA VLM] 이미지 직접 해석 → 자연어 설명 → [RAG] 문헌 근거 보강 → 답변
```

- VLM이 이미지를 **자유 형식으로 해석** (학습 안 된 병변도 설명 가능)
- RAG가 VLM 해석에 대한 **문헌 근거를 보강**
- YOLOv8 검출도 함께 제공 → **정량적(bbox/mask) + 정성적(VLM 해석) 이중 분석**

**면접 포인트**:
> "V3는 Detection 모델의 출력에 의존하지만, V4는 VLM이 이미지를 직접 이해합니다.
> 이는 의료 영상에서 특히 중요한데, 학습 데이터에 없는 희귀 병변도
> VLM의 일반적 의료 지식으로 해석할 수 있기 때문입니다.
> RAG는 이 해석에 대한 문헌 근거를 덧붙여 신뢰성을 높입니다."
