from typing import Any, Dict, Optional

from prometheus_client import Counter, Gauge, Histogram
from ultralytics import YOLO

models: Dict[str, YOLO] = {}
rag_chain: Optional[Any] = None
vlm_client: Optional[Any] = None
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
