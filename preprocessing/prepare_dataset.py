"""
Kvasir-SEG → YOLOv8 세그멘테이션 포맷 변환 스크립트

원본 구조:
  data/kvasir-seg/Kvasir-SEG/images/*.jpg   (원본 이미지)
  data/kvasir-seg/Kvasir-SEG/masks/*.jpg     (바이너리 마스크)

변환 후 구조:
  datasets/kvasir-seg/images/train/  &  val/
  datasets/kvasir-seg/labels/train/  &  val/   (YOLO seg 포맷 txt)

YOLO 세그멘테이션 라벨 포맷:
  <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
  좌표는 0~1로 정규화 (x/width, y/height)
"""

import cv2
import numpy as np
import shutil
import random
from pathlib import Path

SEED = 42
VAL_RATIO = 0.2

BASE_DIR = Path(__file__).parent
SRC_IMAGES = BASE_DIR / "data" / "kvasir-seg" / "Kvasir-SEG" / "images"
SRC_MASKS = BASE_DIR / "data" / "kvasir-seg" / "Kvasir-SEG" / "masks"

DST_DIR = BASE_DIR / "datasets" / "kvasir-seg"
DST_IMAGES_TRAIN = DST_DIR / "images" / "train"
DST_IMAGES_VAL = DST_DIR / "images" / "val"
DST_LABELS_TRAIN = DST_DIR / "labels" / "train"
DST_LABELS_VAL = DST_DIR / "labels" / "val"

CLASS_ID = 0  # polyp = class 0


def mask_to_yolo_segments(mask_path: Path, min_area: int = 100) -> list[str]:
    """바이너리 마스크 → YOLO seg 라벨 문자열. contour 추출 후 좌표 정규화."""
    # NOTE: cv2.imread()가 한글 경로 못 읽어서 numpy로 우회
    buf = np.fromfile(str(mask_path), dtype=np.uint8)
    mask = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"  [경고] 마스크 읽기 실패: {mask_path}")
        return []

    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    h, w = binary.shape

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:  # 노이즈 제거
            continue

        # 폴리곤 단순화 — 점이 너무 많으면 학습에 비효율적
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) < 3:  # 삼각형 미만은 유효하지 않음
            continue

        points = approx.reshape(-1, 2)
        normalized = []
        for x, y in points:
            normalized.append(f"{x / w:.6f}")
            normalized.append(f"{y / h:.6f}")

        line = f"{CLASS_ID} " + " ".join(normalized)
        lines.append(line)

    return lines


def main():
    # 출력 폴더 생성
    for d in [DST_IMAGES_TRAIN, DST_IMAGES_VAL, DST_LABELS_TRAIN, DST_LABELS_VAL]:
        d.mkdir(parents=True, exist_ok=True)
    print(f"출력 폴더 생성 완료: {DST_DIR}")

    # Train/Val 분할
    image_files = sorted(SRC_IMAGES.glob("*.jpg"))
    print(f"전체 이미지 수: {len(image_files)}")

    random.seed(SEED)
    random.shuffle(image_files)

    val_count = int(len(image_files) * VAL_RATIO)
    val_files = image_files[:val_count]
    train_files = image_files[val_count:]
    print(f"Train: {len(train_files)}장 | Val: {len(val_files)}장")

    # 마스크 → YOLO 라벨 변환 & 이미지 복사
    stats = {"success": 0, "skip": 0, "fail": 0}

    for split_name, file_list, img_dst, lbl_dst in [
        ("train", train_files, DST_IMAGES_TRAIN, DST_LABELS_TRAIN),
        ("val", val_files, DST_IMAGES_VAL, DST_LABELS_VAL),
    ]:
        print(f"\n--- {split_name} 변환 중 ---")
        for img_path in file_list:
            stem = img_path.stem
            mask_path = SRC_MASKS / img_path.name

            if not mask_path.exists():
                print(f"  [스킵] 마스크 없음: {stem}")
                stats["skip"] += 1
                continue

            label_lines = mask_to_yolo_segments(mask_path)
            if not label_lines:
                print(f"  [스킵] 유효한 폴리곤 없음: {stem}")
                stats["skip"] += 1
                continue

            shutil.copy2(img_path, img_dst / img_path.name)
            label_path = lbl_dst / f"{stem}.txt"
            label_path.write_text("\n".join(label_lines))

            stats["success"] += 1

    print(f"\n===== 변환 완료 =====")
    print(f"성공: {stats['success']}장")
    print(f"스킵: {stats['skip']}장")
    print(f"실패: {stats['fail']}장")
    print(f"\n데이터셋 경로: {DST_DIR}")
    print(f"  images/train: {len(list(DST_IMAGES_TRAIN.glob('*')))}장")
    print(f"  images/val:   {len(list(DST_IMAGES_VAL.glob('*')))}장")
    print(f"  labels/train: {len(list(DST_LABELS_TRAIN.glob('*')))}개")
    print(f"  labels/val:   {len(list(DST_LABELS_VAL.glob('*')))}개")


if __name__ == "__main__":
    main()
