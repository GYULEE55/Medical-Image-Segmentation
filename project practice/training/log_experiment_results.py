"""
results.csv -> SQLite 실험 DB 적재 스크립트

실무 포인트:
- 실험 끝나고 수치를 말로만 설명하지 않고 DB/리포트로 남김
- 비교 가능한 형태로 적재해 before/after를 증명

예시:
python training/log_experiment_results.py \
  --name dentex-baseline-v2 \
  --model yolov8n-seg \
  --dataset dentex \
  --results-csv training/results_dentex.csv \
  --epochs 100 --batch-size 16 --imgsz 640 --lr 0.01 --optimizer SGD \
  --notes "baseline reproduction"
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from db.experiment_db import ExperimentDB


METRIC_MAP = {
    "map50": "metrics/mAP50(B)",
    "map50_95": "metrics/mAP50-95(B)",
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "map50_mask": "metrics/mAP50(M)",
    "map50_95_mask": "metrics/mAP50-95(M)",
}


def _to_float(value: str) -> float:
    return float(str(value).strip())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Log YOLO results.csv into experiment DB")
    p.add_argument("--name", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--results-csv", required=True)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--optimizer", default=None)
    p.add_argument("--notes", default=None)
    p.add_argument("--db-path", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_path = Path(args.results_csv)
    if not results_path.exists():
        raise FileNotFoundError(f"results.csv not found: {results_path}")

    db = ExperimentDB(db_path=args.db_path)
    exp_id = db.add_experiment(
        name=args.name,
        model=args.model,
        dataset=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        lr=args.lr,
        optimizer=args.optimizer,
        notes=args.notes,
    )

    last_row = None
    with results_path.open() as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            reader.fieldnames = [h.strip() for h in reader.fieldnames]

        for row in reader:
            row = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
            epoch_str = row.get("epoch")
            if not epoch_str:
                continue

            epoch = int(float(epoch_str))
            payload = {}
            for out_key, csv_key in METRIC_MAP.items():
                if csv_key in row and row[csv_key] != "":
                    payload[out_key] = _to_float(row[csv_key])

            if payload:
                db.add_metrics(exp_id, epoch=epoch, **payload)
                last_row = row

    if last_row:
        final_payload = {}
        for out_key, csv_key in METRIC_MAP.items():
            if csv_key in last_row and last_row[csv_key] != "":
                final_payload[out_key] = _to_float(last_row[csv_key])
        if final_payload:
            db.add_metrics(exp_id, **final_payload)

    print(f"[DONE] experiment_id={exp_id}, name={args.name}")
    db.close()


if __name__ == "__main__":
    main()
