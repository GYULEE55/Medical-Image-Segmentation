import argparse
from ultralytics import YOLO


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8n-seg.pt")
    parser.add_argument("--data", default="/content/dentex.yaml")
    parser.add_argument("--project", default="/content/drive/MyDrive/yolo-results")
    parser.add_argument("--name", default="dentex-v2-augmented")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=args.device,
        seed=args.seed,
        workers=args.workers,
        patience=20,
        cos_lr=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        fl_gamma=2.5,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.15,
        degrees=3.0,
        translate=0.08,
        scale=0.3,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.5,
        mixup=0.1,
        copy_paste=0.3,
        erasing=0.2,
        close_mosaic=15,
        amp=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
