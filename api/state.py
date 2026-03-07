import time
from typing import Any, Dict, Optional

from prometheus_client import Counter, Gauge, Histogram
from ultralytics import YOLO

models: Dict[str, YOLO] = {}
rag_chain: Optional[Any] = None
vlm_client: Optional[Any] = None
async_jobs: Dict[str, Dict[str, Any]] = {}

MAX_JOBS = 100
JOB_TTL_SECONDS = 3600


def cleanup_expired_jobs() -> None:
    """완료된 job을 TTL 기반으로 자동 정리 및 최대 저장 건수 제한"""
    now = time.time()
    expired = [k for k, v in async_jobs.items() if v.get("created_at", now) + JOB_TTL_SECONDS < now]
    for k in expired:
        del async_jobs[k]

    if len(async_jobs) > MAX_JOBS:
        oldest = sorted(async_jobs.keys(), key=lambda k: async_jobs[k].get("created_at", 0))
        for k in oldest[: len(async_jobs) - MAX_JOBS]:
            del async_jobs[k]


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
