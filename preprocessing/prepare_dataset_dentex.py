"""
DENTEX (치과 X-ray) → YOLOv8 세그멘테이션 포맷 변환 스크립트

Colab에서 실행하도록 설계됨.
Google Drive에 DENTEX 데이터가 있어야 합니다.

원본 구조 (HuggingFace ibrahimhamamci/DENTEX):
  DENTEX/training_data.zip       → 훈련 이미지 (705장)
  DENTEX/validation_data.zip     → 검증 이미지 (51장)
  DENTEX/test_data.zip           → 테스트 이미지 (250장, 어노테이션 없음)
  DENTEX/validation_triple.json  → COCO 포맷 어노테이션

어노테이션 포맷 (COCO JSON):
  categories_3: 4개 클래스
    0: Impacted       (매복치 — 뼈/잇몸 속에 묻힌 치아)
    1: Caries          (충치)
    2: Periapical Lesion (치근단병변 — 치아 뿌리 끝 감염)
    3: Deep Caries     (깊은충치 — 신경 근처까지 진행)

  annotation 예시:
    {"image_id": 1, "category_id_3": 0, "bbox": [x,y,w,h],
     "segmentation": [[x1,y1,x2,y2,...]], "area": 37377}

변환 후 구조 (YOLO seg 포맷):
  datasets/dentex/images/train/  &  val/
  datasets/dentex/labels/train/  &  val/  (.txt, YOLO polygon 좌표)

YOLO 세그멘테이션 라벨 포맷:
  <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
  좌표는 0~1로 정규화 (x/width, y/height)

사용법 (Colab):
  # 셀 1: Google Drive 마운트
  from google.colab import drive
  drive.mount('/content/drive')

  # 셀 2: 스크립트 실행
  from pathlib import Path
  script_path = Path(
      "/content/drive/MyDrive/Python.Algrithm/project practice/"
      "preprocessing/prepare_dataset_dentex.py"
  )
  %run {script_path}
"""

import json
import os
import random
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────
# 설정 (Colab 환경에 맞게 수정 가능)
# ─────────────────────────────────────────────────
SEED = 42
VAL_RATIO = 0.2  # 훈련 데이터에서 20%를 검증용으로 분할

# Google Drive 경로 (Colab 마운트 기준)
DRIVE_BASE = Path("/content/drive/MyDrive/DENTEX/DENTEX")

# 로컬 작업 디렉토리 (Colab /content/)
WORK_DIR = Path("/content/dentex_work")

# 출력 경로 (Google Drive에 저장 — 세션 종료 시 유지)
OUTPUT_DIR = Path("/content/drive/MyDrive/datasets/dentex")

# DENTEX 4개 클래스 (categories_3 기준)
CLASS_NAMES = {
    0: "Impacted",  # 매복치
    1: "Caries",  # 충치
    2: "Periapical Lesion",  # 치근단병변
    3: "Deep Caries",  # 깊은충치
}
NUM_CLASSES = len(CLASS_NAMES)


def extract_zip(zip_path: Path, extract_to: Path) -> Optional[Path]:
    """
    ZIP 파일 압축 해제.
    이미지가 들어있는 폴더 경로를 반환합니다.
    """
    if not zip_path.exists():
        print(f"  [경고] ZIP 파일 없음: {zip_path}")
        return None

    print(f"  압축 해제 중: {zip_path.name} → {extract_to}")
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(extract_to))

    # ZIP 내부에 폴더가 있을 수 있음 — 이미지 파일 탐색
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    for root, dirs, files in os.walk(str(extract_to)):
        image_files = [f for f in files if Path(f).suffix.lower() in image_exts]
        if image_files:
            print(f"  → 이미지 {len(image_files)}장 발견: {root}")
            return Path(root)

    print(f"  [경고] 이미지를 찾을 수 없음: {extract_to}")
    return None


def find_annotation_json(search_dir: Path) -> Optional[Path]:
    """
    디렉토리에서 COCO 포맷 JSON 어노테이션 파일을 탐색합니다.
    training_data.zip 내부에 어노테이션이 포함되어 있을 수 있음.
    """
    for root, dirs, files in os.walk(str(search_dir)):
        for f in files:
            if f.endswith(".json") and "triple" in f.lower():
                return Path(root) / f
            if f.endswith(".json") and ("train" in f.lower() or "annotation" in f.lower()):
                return Path(root) / f

    # 아무 JSON이라도 찾기
    for root, dirs, files in os.walk(str(search_dir)):
        for f in files:
            if f.endswith(".json"):
                json_path = Path(root) / f
                # COCO 포맷인지 간단 확인
                try:
                    with open(json_path) as jf:
                        data = json.load(jf)
                    if "annotations" in data and "images" in data:
                        return json_path
                except Exception:
                    pass
    return None


def load_coco_annotations(json_path: Path):
    """
    COCO JSON을 로드하고, image_id별 어노테이션 그룹핑.

    Returns:
        images_dict: {image_id: {"file_name": str, "width": int, "height": int}}
        annotations_by_image: {image_id: [annotation, ...]}
        categories: {category_id: category_name}
    """
    with open(json_path) as f:
        data = json.load(f)

    # 이미지 정보
    images_dict = {}
    for img in data["images"]:
        images_dict[img["id"]] = {
            "file_name": img["file_name"],
            "width": img["width"],
            "height": img["height"],
        }

    # 카테고리 (categories_3 = 4클래스, categories_2 = 7클래스, categories_1 = 2클래스)
    # 우리는 4클래스(categories_3) 사용
    categories = {}
    cat_key = "categories_3" if "categories_3" in data else "categories"
    for cat in data[cat_key]:
        categories[cat["id"]] = cat["name"]

    # 이미지별 어노테이션 그룹핑
    annotations_by_image = defaultdict(list)
    cat_id_key = "category_id_3" if "category_id_3" in data["annotations"][0] else "category_id"

    for ann in data["annotations"]:
        annotations_by_image[ann["image_id"]].append(
            {
                "category_id": ann[cat_id_key],
                "segmentation": ann["segmentation"],
                "bbox": ann["bbox"],
                "area": ann.get("area", 0),
            }
        )

    print(f"  JSON 로드 완료: {json_path.name}")
    print(f"  → 이미지: {len(images_dict)}장")
    print(f"  → 어노테이션: {sum(len(v) for v in annotations_by_image.values())}개")
    print(f"  → 카테고리: {categories}")

    return images_dict, annotations_by_image, categories


def coco_segmentation_to_yolo(segmentation, img_width, img_height, category_id, min_points=3):
    """
    COCO segmentation 폴리곤 → YOLO seg 라벨 문자열 변환.

    COCO 포맷: [[x1, y1, x2, y2, ..., xn, yn]]  (절대 좌표, flat list)
    YOLO 포맷: <class_id> <x1/w> <y1/h> <x2/w> <y2/h> ...  (정규화 좌표)

    Args:
        segmentation: COCO segmentation (list of polygon lists)
        img_width: 이미지 너비
        img_height: 이미지 높이
        category_id: 클래스 ID
        min_points: 최소 꼭짓점 수 (3 미만이면 유효하지 않음)

    Returns:
        list[str]: YOLO seg 라벨 문자열 리스트
    """
    lines = []
    for polygon in segmentation:
        # COCO polygon은 [x1, y1, x2, y2, ...] flat list
        if len(polygon) < min_points * 2:
            continue

        # 좌표 정규화 (0~1)
        normalized = []
        for i in range(0, len(polygon), 2):
            x = polygon[i] / img_width
            y = polygon[i + 1] / img_height
            # 범위 클리핑 (간혹 어노테이션이 이미지 밖으로 나가는 경우)
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            normalized.append(f"{x:.6f}")
            normalized.append(f"{y:.6f}")

        line = f"{category_id} " + " ".join(normalized)
        lines.append(line)

    return lines


def convert_dataset(
    images_dict,
    annotations_by_image,
    image_dir,
    output_dir,
    split_name="all",
    val_ratio=0.0,
) -> tuple[dict[str, int], dict[int, int]]:
    """
    COCO 어노테이션을 YOLO seg 포맷으로 변환하고, train/val 분할.

    Args:
        images_dict: {image_id: {file_name, width, height}}
        annotations_by_image: {image_id: [annotations]}
        image_dir: 원본 이미지가 있는 디렉토리
        output_dir: YOLO 데이터셋 출력 경로
        split_name: "all" (자동 분할) 또는 "train"/"val" (지정 분할)
        val_ratio: 검증 데이터 비율 (split_name="all"일 때만 사용)
    """
    # 출력 폴더 생성
    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 어노테이션이 있는 이미지만 처리
    valid_image_ids = [img_id for img_id in images_dict if img_id in annotations_by_image]

    if not valid_image_ids:
        print("  [경고] 유효한 이미지가 없습니다!")
        return {"train": 0, "val": 0, "skip": 0}, {}

    # Train/Val 분할
    random.seed(SEED)
    random.shuffle(valid_image_ids)

    if split_name == "all" and val_ratio > 0:
        val_count = int(len(valid_image_ids) * val_ratio)
        val_ids = set(valid_image_ids[:val_count])
    elif split_name == "val":
        val_ids = set(valid_image_ids)
    else:  # train
        val_ids = set()

    stats: dict[str, int] = {"train": 0, "val": 0, "skip": 0}
    class_counts: defaultdict[int, int] = defaultdict(int)

    for img_id in valid_image_ids:
        img_info = images_dict[img_id]
        file_name = img_info["file_name"]
        width = img_info["width"]
        height = img_info["height"]

        # 원본 이미지 경로 찾기
        src_path = image_dir / file_name
        if not src_path.exists():
            # 파일명만으로 검색 (디렉토리 구조가 다를 수 있음)
            candidates = list(image_dir.glob(f"**/{file_name}"))
            if candidates:
                src_path = candidates[0]
            else:
                stats["skip"] += 1
                continue

        # 어노테이션 → YOLO 라벨 변환
        annotations = annotations_by_image[img_id]
        label_lines = []
        for ann in annotations:
            lines = coco_segmentation_to_yolo(
                ann["segmentation"], width, height, ann["category_id"]
            )
            label_lines.extend(lines)
            class_counts[ann["category_id"]] += 1

        if not label_lines:
            stats["skip"] += 1
            continue

        # split 결정
        split = "val" if img_id in val_ids else "train"

        # 이미지 복사 + 라벨 저장
        dst_img = output_dir / "images" / split / file_name
        dst_lbl = output_dir / "labels" / split / (Path(file_name).stem + ".txt")

        shutil.copy2(str(src_path), str(dst_img))
        dst_lbl.write_text("\n".join(label_lines))

        stats[split] += 1

    return stats, dict(class_counts)


def create_yaml(output_dir: Path, class_names: dict[int, str]):
    """
    YOLOv8 데이터셋 YAML 설정 파일 생성.
    Colab에서 학습 시 이 YAML을 사용합니다.
    """
    yaml_content = f"""# DENTEX 치과 X-ray 데이터셋 설정
# 4개 클래스: 매복치, 충치, 치근단병변, 깊은충치

path: {output_dir}
train: images/train
val: images/val

nc: {len(class_names)}
names:
"""
    for idx in sorted(class_names.keys()):
        yaml_content += f"  {idx}: {class_names[idx]}\n"

    yaml_path = output_dir / "dentex.yaml"
    yaml_path.write_text(yaml_content)
    print(f"  YAML 설정 저장: {yaml_path}")
    return yaml_path


def main():
    print("=" * 60)
    print("DENTEX → YOLOv8 세그멘테이션 포맷 변환")
    print("=" * 60)

    # ── 1단계: ZIP 압축 해제 ─────────────────────────────────
    print("\n[1/4] ZIP 압축 해제")
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    # 훈련 데이터 압축 해제
    train_zip = DRIVE_BASE / "training_data.zip"
    train_img_dir = extract_zip(train_zip, WORK_DIR / "training")

    # 검증 데이터 압축 해제
    val_zip = DRIVE_BASE / "validation_data.zip"
    val_img_dir = extract_zip(val_zip, WORK_DIR / "validation")

    # ── 2단계: 어노테이션 JSON 로드 ──────────────────────────
    print("\n[2/4] 어노테이션 로드")

    # 검증 데이터 어노테이션 (확실히 존재)
    val_json = DRIVE_BASE / "validation_triple.json"
    if val_json.exists():
        val_images, val_anns, _ = load_coco_annotations(val_json)
        print("  ✅ 검증 어노테이션 로드 완료")
    else:
        print("  ❌ validation_triple.json 없음!")
        val_images, val_anns = {}, {}

    # 훈련 데이터 어노테이션 (ZIP 내부 또는 별도 파일 탐색)
    train_json = None
    train_images, train_anns, train_cats = {}, {}, {}

    # 1) ZIP 해제된 디렉토리에서 탐색
    if train_img_dir:
        train_json = find_annotation_json(WORK_DIR / "training")

    # 2) Drive 루트에서 탐색
    if train_json is None:
        for name in ["training_triple.json", "train_triple.json", "train.json"]:
            candidate = DRIVE_BASE / name
            if candidate.exists():
                train_json = candidate
                break

    if train_json:
        train_images, train_anns, train_cats = load_coco_annotations(train_json)
        print(f"  ✅ 훈련 어노테이션 로드 완료: {train_json.name}")
    else:
        print("  ⚠️ 훈련 어노테이션 미발견 — 검증 데이터만으로 진행합니다")
        print("     (HuggingFace에서 training annotation을 별도 다운로드해야 할 수 있음)")

    # ── 3단계: YOLO 포맷 변환 ────────────────────────────────
    print("\n[3/4] YOLO seg 포맷 변환")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_stats: dict[str, int] = {"train": 0, "val": 0, "skip": 0}
    total_class_counts: defaultdict[int, int] = defaultdict(int)

    # 훈련 데이터가 있으면 train/val 분할
    if train_images and train_anns and train_img_dir:
        print("\n  --- 훈련 데이터 변환 (train/val 자동 분할) ---")
        stats, cc = convert_dataset(
            train_images,
            train_anns,
            train_img_dir,
            OUTPUT_DIR,
            split_name="all",
            val_ratio=VAL_RATIO,
        )
        for k in stats:
            total_stats[k] += stats[k]
        for k, v in cc.items():
            total_class_counts[k] += v

    # 검증 데이터 (train annotation이 없으면 이것을 train/val로 분할)
    if val_images and val_anns and val_img_dir:
        if not train_images:
            # 훈련 어노테이션 없음 → 검증 데이터를 train/val로 분할
            print("\n  --- 검증 데이터 → train/val 분할 (훈련 어노테이션 없으므로) ---")
            stats, cc = convert_dataset(
                val_images,
                val_anns,
                val_img_dir,
                OUTPUT_DIR,
                split_name="all",
                val_ratio=VAL_RATIO,
            )
        else:
            # 훈련 데이터 있음 → 검증 데이터는 val로만 추가
            print("\n  --- 검증 데이터 → val에 추가 ---")
            stats, cc = convert_dataset(
                val_images,
                val_anns,
                val_img_dir,
                OUTPUT_DIR,
                split_name="val",
                val_ratio=0,
            )
        for k in stats:
            total_stats[k] += stats[k]
        for k, v in cc.items():
            total_class_counts[k] += v

    # ── 4단계: YAML 설정 파일 생성 ───────────────────────────
    print("\n[4/4] YAML 설정 파일 생성")
    yaml_path = create_yaml(OUTPUT_DIR, CLASS_NAMES)

    # ── 결과 요약 ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("변환 완료!")
    print("=" * 60)
    print(f"  Train: {total_stats['train']}장")
    print(f"  Val:   {total_stats['val']}장")
    print(f"  Skip:  {total_stats['skip']}장")
    print("\n  클래스별 어노테이션 수:")
    for cls_id in sorted(total_class_counts.keys()):
        name = CLASS_NAMES.get(cls_id, f"unknown_{cls_id}")
        print(f"    [{cls_id}] {name}: {total_class_counts[cls_id]}개")
    print(f"\n  출력 경로: {OUTPUT_DIR}")
    print(f"  YAML 설정: {yaml_path}")

    # 폴더별 파일 수 확인
    for split in ["train", "val"]:
        img_count = len(list((OUTPUT_DIR / "images" / split).glob("*")))
        lbl_count = len(list((OUTPUT_DIR / "labels" / split).glob("*")))
        print(f"  {split}: images={img_count}, labels={lbl_count}")


if __name__ == "__main__":
    main()
