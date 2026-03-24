"""
로컬 YOLOv8 세그멘테이션 학습 CLI
GPU가 있는 로컬 머신에서 학습을 실행하기 위한 스크립트입니다.

사용 예시:
    python3 training/train.py --data datasets/kvasir-seg/kvasir-seg.yaml --epochs 50 --seed 42
"""

import argparse
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    재현성을 위해 모든 난수 생성기의 시드를 설정합니다.

    Args:
        seed: 설정할 시드 값
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    """메인 학습 함수"""
    parser = argparse.ArgumentParser(
        description="Local YOLOv8 Segmentation Training CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 training/train.py --data datasets/kvasir-seg/kvasir-seg.yaml --epochs 50
  python3 training/train.py --data datasets/kvasir-seg/kvasir-seg.yaml --epochs 100 --batch 32 --seed 42
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n-seg.pt",
        help="Base model path or name (default: yolov8n-seg.pt)",
    )
    parser.add_argument("--data", type=str, required=True, help="Dataset YAML path (required)")
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs (default: 50)"
    )
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size (default: 640)")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    # 시드 설정 (재현성 보장)
    set_seed(args.seed)

    # YOLOv8 모델 로드 및 학습
    from ultralytics import YOLO

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    print(f"Starting training with seed={args.seed}")
    print(f"  Dataset: {args.data}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch}")
    print(f"  Image size: {args.imgsz}")

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        seed=args.seed,
        device=0 if torch.cuda.is_available() else "cpu",
    )

    print("Training completed!")
    return results


if __name__ == "__main__":
    main()
