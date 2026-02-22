FROM python:3.10-slim

WORKDIR /app

# OpenCV headless가 필요로 하는 시스템 라이브러리
# curl: docker-compose healthcheck에서 사용
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# CPU 전용 PyTorch 먼저 설치 (CUDA 없이 ~200MB, GPU 버전은 ~2GB)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 나머지 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y opencv-python \
    && pip install --no-cache-dir --force-reinstall opencv-python-headless==4.10.0.84

# 앱 코드 복사
# V1: API 서버 + YOLOv8 멀티모델 (polyp + dental)
COPY api/ api/
COPY best.pt .
# DENTEX 모델 (있으면 복사, 없으면 무시 — .dockerignore에서 제어)
COPY best_dentex.pt* ./

# V2: RAG 모듈 (chain.py, ingest.py, __init__.py)
# vectorstore/와 docs/는 docker-compose에서 볼륨 마운트
COPY rag/*.py rag/
COPY rag/__init__.py rag/

# vectorstore, docs 디렉토리 생성 (볼륨 마운트 대상)
RUN mkdir -p rag/vectorstore rag/docs

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
