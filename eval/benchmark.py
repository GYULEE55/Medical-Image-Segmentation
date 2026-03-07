import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def _list_images(images_dir: Path, limit: int) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = [p for p in images_dir.rglob("*") if p.suffix.lower() in exts]
    files.sort()
    return files[:limit] if limit > 0 else files


def _benchmark_inference(model: YOLO, image_paths: list[Path], conf: float) -> dict:
    if not image_paths:
        return {
            "num_images": 0,
            "avg_latency_ms": None,
            "p95_latency_ms": None,
            "fps": None,
        }

    latencies = []
    for path in image_paths:
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        started = time.perf_counter()
        model.predict(img, conf=conf, verbose=False)
        latencies.append((time.perf_counter() - started) * 1000.0)

    if not latencies:
        return {
            "num_images": 0,
            "avg_latency_ms": None,
            "p95_latency_ms": None,
            "fps": None,
        }

    avg_ms = statistics.mean(latencies)
    sorted_lat = sorted(latencies)
    p95_idx = min(len(sorted_lat) - 1, int(len(sorted_lat) * 0.95))
    p95_ms = sorted_lat[p95_idx]
    fps = 1000.0 / avg_ms if avg_ms > 0 else None
    return {
        "num_images": len(latencies),
        "avg_latency_ms": round(avg_ms, 3),
        "p95_latency_ms": round(p95_ms, 3),
        "fps": round(fps, 3) if fps else None,
    }


def _safe_model_val(model: YOLO, data_yaml: str, imgsz: int, conf: float) -> dict:
    try:
        metrics = model.val(data=data_yaml, imgsz=imgsz, conf=conf, verbose=False)
        return {
            "box_map50": float(getattr(metrics.box, "map50", 0.0)),
            "box_map50_95": float(getattr(metrics.box, "map", 0.0)),
            "seg_map50": float(getattr(metrics.seg, "map50", 0.0)),
            "seg_map50_95": float(getattr(metrics.seg, "map", 0.0)),
            "box_precision": float(getattr(metrics.box, "p", 0.0)),
            "box_recall": float(getattr(metrics.box, "r", 0.0)),
            "seg_precision": float(getattr(metrics.seg, "p", 0.0)),
            "seg_recall": float(getattr(metrics.seg, "r", 0.0)),
        }
    except Exception as e:
        return {"error": str(e)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--images-dir", default="example_test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output", default="eval/benchmark_result.json")
    args = parser.parse_args()

    model = YOLO(args.model)
    image_paths = _list_images(Path(args.images_dir), args.limit)
    report = {
        "model_path": str(Path(args.model).resolve()),
        "data_yaml": str(Path(args.data).resolve()),
        "images_dir": str(Path(args.images_dir).resolve()),
        "inference": _benchmark_inference(model, image_paths, args.conf),
        "validation": _safe_model_val(model, args.data, args.imgsz, args.conf),
        "config": {
            "imgsz": args.imgsz,
            "conf": args.conf,
            "limit": args.limit,
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
