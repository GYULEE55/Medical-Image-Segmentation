"""Export YOLOv8 model to ONNX format for edge deployment."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Export YOLOv8 to ONNX format")
    parser.add_argument("--model", required=True, help="Path to .pt model file")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version (default: 12)")
    parser.add_argument(
        "--output", type=str, default=None, help="Output path (default: same dir as model)"
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    export_path = model.export(format="onnx", imgsz=args.imgsz, opset=args.opset)
    print(f"Exported to: {export_path}")


if __name__ == "__main__":
    main()
