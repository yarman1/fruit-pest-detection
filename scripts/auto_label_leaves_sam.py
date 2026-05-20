"""
scripts/auto_label_leaves_sam.py

Автоматично/напівавтоматично розмічає leaf-зображення через Ultralytics SAM/SAM2.

Вхід:
  data/interim/leaves_inaturalist/
  ├── apple/
  ├── pear/
  ├── plum/
  ├── sour_cherry/
  ├── sweet_cherry/
  ├── apricot/
  └── _metadata.csv

Вихід:
  data/interim/leaves_sam/
  ├── images/<tree_key>/
  ├── labels/<tree_key>/
  ├── preview/<tree_key>/        optional
  └── dataset_stats.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import random
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from ultralytics import SAM
except ImportError:
    SAM = None


IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp",
    ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP",
}


@dataclass(frozen=True)
class SourceImage:
    image_path: Path
    tree_key: str


@dataclass(frozen=True)
class Box:
    x1: int
    y1: int
    x2: int
    y2: int
    area_ratio: float
    green_ratio: float = 0.0
    fruit_ratio: float = 0.0
    solidity: float = 1.0
    circularity: float = 1.0

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def aspect_ratio(self) -> float:
        """width / height. Leaves: ~0.3 .. 5.0"""
        return self.width / max(self.height, 1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix in IMAGE_EXTS


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9а-яіїєґ_-]+", "_", value, flags=re.IGNORECASE)
    value = re.sub(r"_+", "_", value)
    return value.strip("_") or "item"


def short_hash(value: str, length: int = 10) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def make_output_stem(src: SourceImage) -> str:
    original = slugify(src.image_path.stem)
    original = original[:60].strip("_") or "image"
    suffix = short_hash(src.image_path.as_posix())
    return f"sam_{slugify(src.tree_key)}_{original}_{suffix}"


def collect_source_images(input_root: Path) -> list[SourceImage]:
    if not input_root.exists():
        return []

    result: list[SourceImage] = []

    for tree_dir in sorted(input_root.iterdir()):
        if not tree_dir.is_dir():
            continue
        if tree_dir.name.startswith("_"):
            continue
        tree_key = tree_dir.name
        for image_path in sorted(tree_dir.rglob("*")):
            if is_image_file(image_path):
                result.append(SourceImage(image_path=image_path, tree_key=tree_key))

    return result

def select_source_images(
    source_images: list[SourceImage],
    max_images: int | None,
    max_images_per_tree: int | None,
    sample_seed: int,
) -> list[SourceImage]:
    """
    Обирає підмножину зображень для тестового прогону.

    --max-images:
      глобальний ліміт на кількість зображень.

    --max-images-per-tree:
      ліміт на кількість зображень з кожної папки дерева:
      apple, pear, plum, sour_cherry, sweet_cherry, apricot.
    """
    rng = random.Random(sample_seed)

    if max_images_per_tree is not None:
        grouped: dict[str, list[SourceImage]] = {}

        for item in source_images:
            grouped.setdefault(item.tree_key, []).append(item)

        selected: list[SourceImage] = []

        for tree_key in sorted(grouped.keys()):
            items = grouped[tree_key][:]
            rng.shuffle(items)
            selected.extend(items[:max_images_per_tree])

        rng.shuffle(selected)

        if max_images is not None:
            selected = selected[:max_images]

        return selected

    selected = source_images[:]

    if max_images is not None:
        selected = selected[:max_images]

    return selected

# ---------------------------------------------------------------------------
# Mask → Box
# ---------------------------------------------------------------------------

def mask_to_box(
    mask: np.ndarray,
    image_width: int,
    image_height: int,
    min_solidity: float = 0.35,
) -> Box | None:
    """
    Конвертує binary mask у bbox.

    НОВИНКА: обчислює solidity через convex hull cv2.
    Якщо солідність нижче min_solidity — це найчастіше гілка або фоновий шум,
    а не листок, і функція повертає None.
    """
    if mask.ndim != 2:
        return None

    mask_bool = mask.astype(bool)

    if not mask_bool.any():
        return None

    ys, xs = np.where(mask_bool)

    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1

    mask_h, mask_w = mask_bool.shape

    if mask_w != image_width or mask_h != image_height:
        scale_x = image_width / mask_w
        scale_y = image_height / mask_h
        x1 = int(round(x1 * scale_x))
        x2 = int(round(x2 * scale_x))
        y1 = int(round(y1 * scale_y))
        y2 = int(round(y2 * scale_y))

    x1 = max(0, min(x1, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))
    x2 = max(1, min(x2, image_width))
    y2 = max(1, min(y2, image_height))

    if x2 <= x1 or y2 <= y1:
        return None

    area_ratio = ((x2 - x1) * (y2 - y1)) / float(image_width * image_height)

    # --- Solidity + Circularity з реального SAM-контуру маски ---
    # Circularity = 4π·area / perimeter²
    #   Коло (яблуко, слива) → ~0.80-0.95
    #   Листок                → ~0.25-0.60
    #   Широкий овальний лист → ~0.60-0.75
    solidity = 1.0
    circularity = 0.0
    if cv2 is not None:
        mask_uint8 = mask_bool.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            cnt_area = cv2.contourArea(largest)

            hull = cv2.convexHull(largest)
            hull_area = cv2.contourArea(hull)
            solidity = cnt_area / hull_area if hull_area > 0 else 0.0

            perimeter = cv2.arcLength(largest, True)
            circularity = (4 * np.pi * cnt_area / (perimeter ** 2)) if perimeter > 0 else 0.0

            if min_solidity > 0.0 and solidity < min_solidity:
                return None

    return Box(x1=x1, y1=y1, x2=x2, y2=y2, area_ratio=area_ratio,
               solidity=solidity, circularity=circularity)


# ---------------------------------------------------------------------------
# IoU + Containment
# ---------------------------------------------------------------------------

def box_iou(a: Box, b: Box) -> float:
    inter_x1 = max(a.x1, b.x1)
    inter_y1 = max(a.y1, b.y1)
    inter_x2 = min(a.x2, b.x2)
    inter_y2 = min(a.y2, b.y2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = a.width * a.height
    area_b = b.width * b.height
    union = area_a + area_b - inter_area

    return inter_area / union if union > 0 else 0.0


def box_is_contained_in(small: Box, large: Box, threshold: float = 0.85) -> bool:
    """
    Повертає True, якщо small здебільшого перекривається large.
    """
    inter_x1 = max(small.x1, large.x1)
    inter_y1 = max(small.y1, large.y1)
    inter_x2 = min(small.x2, large.x2)
    inter_y2 = min(small.y2, large.y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    small_area = small.width * small.height

    return (inter_area / small_area) > threshold if small_area > 0 else False


# ---------------------------------------------------------------------------
# Color metrics
# ---------------------------------------------------------------------------

def compute_color_metrics(image_rgb: np.ndarray, box: Box) -> tuple[float, float]:
    """
    Рахує частку green-like і fruit-like пікселів у bbox через HSV.
    """
    if cv2 is None:
        # Fallback до старої RGB-евристики, якщо cv2 недоступний
        return _compute_color_metrics_rgb_fallback(image_rgb, box)

    crop = image_rgb[box.y1:box.y2, box.x1:box.x2]

    if crop.size == 0:
        return 0.0, 0.0

    # cv2 очікує BGR; конвертуємо RGB→BGR→HSV
    bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]

    # Зелений лист: hue 35°..85° → OpenCV 17.5..42.5
    green_mask = (
        (h >= 15) & (h <= 50) &   # ~30°..100° реального hue
        (s >= 25) &                # не сірий
        (v >= 30)                  # не чорний
    )

    # Червоні плоди: hue 0..10 або 160..180 → OpenCV 0..5 або 160..180
    red_mask = (
        ((h <= 10) | (h >= 160)) &
        (s >= 50) &
        (v >= 50)
    )

    # Жовто-помаранчеві плоди: hue ~10°..25° → OpenCV 5..12
    yellow_orange_mask = (
        (h > 5) & (h < 20) &
        (s >= 60) &
        (v >= 60)
    )

    fruit_mask = red_mask | yellow_orange_mask

    return float(np.mean(green_mask)), float(np.mean(fruit_mask))


def _compute_color_metrics_rgb_fallback(image_rgb: np.ndarray, box: Box) -> tuple[float, float]:
    """RGB-евристика як запасний варіант (без cv2)."""
    crop = image_rgb[box.y1:box.y2, box.x1:box.x2]

    if crop.size == 0:
        return 0.0, 0.0

    rgb = crop.astype(np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    max_c = np.maximum(np.maximum(r, g), b)
    saturation = (max_c - np.minimum(np.minimum(r, g), b)) / (max_c + 1e-6)
    brightness = (r + g + b) / 3.0

    green_mask = (g > r * 1.05) & (g > b * 1.03) & (saturation > 0.10) & (brightness > 35)
    red_fruit_mask = (r > g * 1.10) & (r > b * 1.20) & (saturation > 0.16) & (brightness > 45)
    yellow_fruit_mask = (
        (r > 80) & (g > 70) &
        (b < np.minimum(r, g) * 0.78) &
        (saturation > 0.12) & (brightness > 55)
    )

    return float(np.mean(green_mask)), float(np.mean(red_fruit_mask | yellow_fruit_mask))


def add_color_metrics_to_boxes(image_rgb: np.ndarray, boxes: list[Box]) -> list[Box]:
    result: list[Box] = []
    for box in boxes:
        green_ratio, fruit_ratio = compute_color_metrics(image_rgb, box)
        result.append(replace(box, green_ratio=green_ratio, fruit_ratio=fruit_ratio))
    return result


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_boxes(
    boxes: list[Box],
    image_width: int,
    image_height: int,
    min_area_ratio: float,
    max_area_ratio: float,
    min_box_width_ratio: float,
    min_box_height_ratio: float,
    max_box_width_ratio: float,
    max_box_height_ratio: float,
    min_aspect_ratio: float,
    max_aspect_ratio: float,
    max_circularity: float,
    iou_threshold: float,
    containment_threshold: float,
    max_boxes_per_image: int,
    enable_color_filter: bool,
    min_green_ratio: float,
    max_fruit_ratio: float,
    fruit_dominance_ratio: float,
) -> list[Box]:
    filtered: list[Box] = []

    for box in boxes:
        width_ratio = box.width / image_width
        height_ratio = box.height / image_height

        # 1. Геометричні фільтри
        if box.area_ratio < min_area_ratio:
            continue

        if box.area_ratio > max_area_ratio:
            continue

        if width_ratio < min_box_width_ratio or height_ratio < min_box_height_ratio:
            continue

        if width_ratio > max_box_width_ratio or height_ratio > max_box_height_ratio:
            continue

        # 2. Aspect ratio
        if box.aspect_ratio < min_aspect_ratio or box.aspect_ratio > max_aspect_ratio:
            continue

        # 3. Circularity — відкидає круглі плоди
        if box.circularity > max_circularity:
            continue

        # 4. Колірні фільтри
        if enable_color_filter:
            if box.green_ratio < min_green_ratio:
                continue

            fruit_dominates = box.fruit_ratio > box.green_ratio * fruit_dominance_ratio

            if fruit_dominates:
                continue

            if box.fruit_ratio > max_fruit_ratio:
                continue

        # ВАЖЛИВО: якщо bbox пройшов усі фільтри — додаємо його
        filtered.append(box)

    # Спочатку більші об'єкти
    filtered.sort(key=lambda b: b.area_ratio, reverse=True)

    deduped: list[Box] = []

    for box in filtered:
        if any(box_is_contained_in(box, existing, containment_threshold) for existing in deduped):
            continue

        if all(box_iou(box, existing) < iou_threshold for existing in deduped):
            deduped.append(box)

        if len(deduped) >= max_boxes_per_image:
            break

    return deduped


def get_image_rejection_reason(
    raw_boxes_count: int,
    kept_boxes_count: int,
    max_boxes_per_image: int,
    reject_if_raw_boxes_more_than: int | None,
    reject_if_hit_max_boxes: bool,
    reject_hit_max_min_raw_boxes: int,
) -> str | None:
    """
    Відкидає підозрілі зображення на рівні всього image.
    """
    reasons: list[str] = []

    if (
        reject_if_raw_boxes_more_than is not None
        and raw_boxes_count > reject_if_raw_boxes_more_than
    ):
        reasons.append(f"too_many_raw_boxes_{raw_boxes_count}")

    hit_max_boxes = kept_boxes_count >= max_boxes_per_image

    if (
        reject_if_hit_max_boxes
        and hit_max_boxes
        and raw_boxes_count >= reject_hit_max_min_raw_boxes
    ):
        reasons.append(f"hit_max_boxes_{kept_boxes_count}_from_raw_{raw_boxes_count}")

    if not reasons:
        return None

    return "__".join(reasons)

# ---------------------------------------------------------------------------
# YOLO format
# ---------------------------------------------------------------------------

def box_to_yolo_line(box: Box, image_width: int, image_height: int) -> str:
    x_center = ((box.x1 + box.x2) / 2.0) / image_width
    y_center = ((box.y1 + box.y2) / 2.0) / image_height
    width = (box.x2 - box.x1) / image_width
    height = (box.y2 - box.y1) / image_height
    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


# ---------------------------------------------------------------------------
# SAM inference
# ---------------------------------------------------------------------------

def extract_masks_from_result(result) -> list[np.ndarray]:
    if result is None:
        return []
    masks = getattr(result, "masks", None)
    if masks is None:
        return []
    data = getattr(masks, "data", None)
    if data is None:
        return []
    try:
        data_np = data.detach().cpu().numpy()
    except AttributeError:
        data_np = np.asarray(data)
    if data_np.ndim == 2:
        data_np = data_np[None, :, :]
    return [data_np[i] for i in range(data_np.shape[0])]


def run_sam_on_image(
    model,
    image_path: Path,
    device: str,
    imgsz: int,
    sam_conf: float,
    sam_iou: float,
    verbose: bool,
) -> list[np.ndarray]:
    """
    Запускає SAM/SAM2 на одному зображенні.
    """
    results = model(
        str(image_path),
        device=device,
        imgsz=imgsz,
        retina_masks=True,
        conf=sam_conf,
        iou=sam_iou,
        verbose=verbose,
    )

    if not results:
        return []

    return extract_masks_from_result(results[0])


def run_sam_tiled(
    model,
    image_path: Path,
    image_rgb: np.ndarray,
    device: str,
    imgsz: int,
    sam_conf: float,
    sam_iou: float,
    verbose: bool,
    tile_size: int,
    overlap: float,
    min_solidity: float,
) -> list[Box]:
    """
    Тайлінг для великих зображень
    """
    H, W = image_rgb.shape[:2]
    step = int(tile_size * (1 - overlap))
    all_boxes: list[Box] = []

    for y0 in range(0, H, step):
        for x0 in range(0, W, step):
            y1_t = min(y0 + tile_size, H)
            x1_t = min(x0 + tile_size, W)
            tile_h = y1_t - y0
            tile_w = x1_t - x0

            tile_arr = image_rgb[y0:y1_t, x0:x1_t]
            tile_pil = Image.fromarray(tile_arr)

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                tile_pil.save(f.name, quality=92)
                tmp_path = Path(f.name)

            masks = run_sam_on_image(
                model=model,
                image_path=tmp_path,
                device=device,
                imgsz=imgsz,
                sam_conf=sam_conf,
                sam_iou=sam_iou,
                verbose=verbose,
            )
            tmp_path.unlink(missing_ok=True)

            for mask in masks:
                box = mask_to_box(mask, tile_w, tile_h, min_solidity=min_solidity)
                if box is None:
                    continue
                # Зсуваємо bbox назад у повне зображення
                all_boxes.append(
                    replace(
                        box,
                        x1=box.x1 + x0,
                        y1=box.y1 + y0,
                        x2=box.x2 + x0,
                        y2=box.y2 + y0,
                        area_ratio=((box.x2 - box.x1) * (box.y2 - box.y1)) / float(W * H),
                    )
                )

    return all_boxes


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------

def save_preview(image_path: Path, boxes: list[Box], preview_path: Path) -> None:
    if cv2 is None:
        return

    image = cv2.imread(str(image_path))
    if image is None:
        return

    for idx, box in enumerate(boxes):
        green_intensity = int(min(255, 100 + box.green_ratio * 500))
        color = (0, green_intensity, 50)

        cv2.rectangle(image, (box.x1, box.y1), (box.x2, box.y2), color, 2)

        label = (
            f"#{idx} "
            f"g:{box.green_ratio:.2f} "
            f"f:{box.fruit_ratio:.2f} "
            f"c:{box.circularity:.2f}"
        )
        y_text = max(15, box.y1 - 5)

        cv2.putText(
            image, label,
            (box.x1, y_text),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA,
        )

    preview_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(preview_path), image)


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def copy_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def write_label(label_path: Path, boxes: list[Box], image_width: int, image_height: int) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [box_to_yolo_line(box, image_width, image_height) for box in boxes]
    label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--input", type=Path, default=Path("data/interim/leaves_inaturalist"))
    parser.add_argument("--output", type=Path, default=Path("data/interim/leaves_sam"))
    parser.add_argument("--model", default="sam2_b.pt")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--imgsz", type=int, default=1024)

    # SAM-level filters
    parser.add_argument(
        "--sam-conf", type=float, default=0.40,
        help="SAM confidence threshold. Вище → менше шумних масок",
    )
    parser.add_argument(
        "--sam-iou", type=float, default=0.70,
        help="SAM internal NMS IoU threshold",
    )

    # Geometry filters
    parser.add_argument("--min-area-ratio", type=float, default=0.003)
    parser.add_argument("--max-area-ratio", type=float, default=0.40)
    parser.add_argument("--min-box-width-ratio", type=float, default=0.03)
    parser.add_argument("--min-box-height-ratio", type=float, default=0.03)
    parser.add_argument("--max-box-width-ratio", type=float, default=0.80)
    parser.add_argument("--max-box-height-ratio", type=float, default=0.80)

    # Aspect ratio
    parser.add_argument(
        "--min-aspect-ratio", type=float, default=0.20,
        help="Мін. співвідношення ширини до висоти bbox (відсіює вертикальні гілки)",
    )
    parser.add_argument(
        "--max-aspect-ratio", type=float, default=5.0,
        help="Макс. співвідношення ширини до висоти bbox (відсіює горизонтальні гілки)",
    )

    # Circularity
    parser.add_argument(
        "--max-circularity", type=float, default=0.72,
        help="Макс. circularity маски (4π·area/perimeter²). "
             "Яблука/сливи ~0.80-0.95, листки ~0.25-0.65. "
             "0.72 відкидає більшість плодів і зберігає овальні листки. "
             "Знизь до 0.65 для суворішого фільтра, підніми до 0.80 якщо губяться листки.",
    )

    # Solidity
    parser.add_argument(
        "--min-solidity", type=float, default=0.35,
        help="Мін. solidity маски (area/convex_hull_area). Гілки мають ~0.1-0.25",
    )

    # Dedup
    parser.add_argument("--iou-threshold", type=float, default=0.55)
    parser.add_argument(
        "--containment-threshold", type=float, default=0.85,
        help="Якщо малий bbox перекривається на X з більшим — видалити малий",
    )
    parser.add_argument("--max-boxes-per-image", type=int, default=10)

    # Image-level reject filters
    parser.add_argument(
        "--reject-if-raw-boxes-more-than",
        type=int,
        default=None,
        help=(
            "Відкинути все зображення, якщо SAM створив більше raw boxes, ніж цей поріг. "
            "Корисно для захаращених фото з травою, доріжками, гілками, фоном."
        ),
    )

    parser.add_argument(
        "--reject-if-hit-max-boxes",
        action="store_true",
        help=(
            "Відкинути зображення, якщо kept boxes вперлися в --max-boxes-per-image. "
            "Це часто означає, що фото шумне і скрипт просто набрав максимум кандидатів."
        ),
    )

    parser.add_argument(
        "--reject-hit-max-min-raw-boxes",
        type=int,
        default=25,
        help=(
            "Застосовувати --reject-if-hit-max-boxes тільки якщо raw boxes не менше цього числа. "
            "Це захищає нормальні прості фото, де raw boxes мало."
        ),
    )

    parser.add_argument(
        "--save-rejected-preview",
        action="store_true",
        help="Зберігати preview для відкинутих image-level reject фото в preview_rejected/",
    )

    # Color filters
    parser.add_argument("--disable-color-filter", action="store_true")
    parser.add_argument(
        "--min-green-ratio", type=float, default=0.08,
        help="Мінімальна частка green-like пікселів у bbox (HSV hue 30-100°). "
             "Підвищено до 0.08 щоб відкидати bbox без помітного листя.",
    )
    parser.add_argument(
        "--max-fruit-ratio", type=float, default=0.38,
        help="Абсолютна стеля fruit-like пікселів. Відкидається НЕЗАЛЕЖНО від green. "
             "Підвищено до 0.38 — перевіряємо тільки очевидні плоди.",
    )
    parser.add_argument(
        "--fruit-dominance-ratio", type=float, default=0.80,
        help="fruit_ratio / green_ratio поріг домінування (OR-логіка з max_fruit_ratio). "
             "0.80 означає: відкидаємо якщо fruit > green * 0.80 — "
             "тобто плід займає майже стільки ж пікселів, скільки листок.",
    )

    # Tiling (НОВИНКА)
    parser.add_argument(
        "--use-tiling", action="store_true",
        help="Тайлінг для великих зображень (рекомендується якщо листки дрібні відносно кадру)",
    )
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--tile-overlap", type=float, default=0.20)

    # Run control
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Глобальний максимум зображень для тестового запуску",
    )

    parser.add_argument(
        "--max-images-per-tree",
        type=int,
        default=None,
        help=(
            "Максимум зображень з кожної папки дерева. "
            "Корисно для preview, щоб не тестувати тільки apple."
        ),
    )

    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Seed для випадкової, але відтворюваної вибірки зображень",
    )

    parser.add_argument("--skip-no-boxes", action="store_true", default=True)
    parser.add_argument("--save-preview", action="store_true")
    parser.add_argument("--clear-output", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    print(f"\n{'=' * 72}")
    print("  Auto-label leaves with SAM/SAM2  [v3 — circularity filter]")
    print(f"{'=' * 72}")
    print(f"  Input:                {args.input}")
    print(f"  Output:               {args.output}")
    print(f"  Model:                {args.model}")
    print(f"  Device:               {args.device}")
    print(f"  Image size:           {args.imgsz}")
    print(f"  SAM conf/iou:         {args.sam_conf} / {args.sam_iou}")
    print(f"  Max images:           {args.max_images}")
    print(f"  Max images/tree:      {args.max_images_per_tree}")
    print(f"  Sample seed:          {args.sample_seed}")
    print(f"  Save preview:         {args.save_preview}")
    print(f"  Color filter:         {not args.disable_color_filter}")
    print(f"  Min green ratio:      {args.min_green_ratio}  (HSV-based)")
    print(f"  Max fruit ratio:      {args.max_fruit_ratio}")
    print(f"  Fruit dominance:      {args.fruit_dominance_ratio}")
    print(f"  Min solidity:         {args.min_solidity}")
    print(f"  Max circularity:      {args.max_circularity}  (яблуко~0.85, листок~0.35-0.65)")
    print(f"  Aspect ratio:         {args.min_aspect_ratio} .. {args.max_aspect_ratio}")
    print(f"  IoU threshold:        {args.iou_threshold}")
    print(f"  Containment thresh:   {args.containment_threshold}")
    print(f"  Reject raw >:         {args.reject_if_raw_boxes_more_than}")
    print(f"  Reject hit max:       {args.reject_if_hit_max_boxes}")
    print(f"  Reject hit max raw≥:  {args.reject_hit_max_min_raw_boxes}")
    print(f"  Tiling:               {args.use_tiling}" +
          (f" (tile={args.tile_size}, overlap={args.tile_overlap})" if args.use_tiling else ""))
    print(f"  Dry run:              {args.dry_run}")
    print(f"{'=' * 72}\n")

    source_images = collect_source_images(args.input)

    source_images = select_source_images(
        source_images=source_images,
        max_images=args.max_images,
        max_images_per_tree=args.max_images_per_tree,
        sample_seed=args.sample_seed,
    )

    if not source_images:
        print("Не знайдено зображень.")
        print(f"Перевір папку: {args.input}")
        return 1

    counts_by_tree: dict[str, int] = {}
    for item in source_images:
        counts_by_tree[item.tree_key] = counts_by_tree.get(item.tree_key, 0) + 1

    print("Знайдено зображень:")
    for tree_key, count in sorted(counts_by_tree.items()):
        print(f"  - {tree_key:<20} {count:>6}")
    print(f"\nВсього: {len(source_images)}")

    if args.dry_run:
        print("\n[dry-run] SAM не запускався, файли не створені.")
        return 0

    if SAM is None:
        print("Помилка: ultralytics не встановлено.")
        print("  python -m pip install ultralytics")
        return 1

    if cv2 is None:
        print("Увага: OpenCV не встановлено — solidity filter і HSV відключені.")
        print("  python -m pip install opencv-python-headless")

    output_root = args.output

    if output_root.exists() and any(output_root.iterdir()):
        if not args.clear_output:
            print(f"\nOutput директорія вже існує: {output_root}")
            print("Щоб перезаписати: додай --clear-output")
            return 1
        shutil.rmtree(output_root)

    images_out_root = output_root / "images"
    labels_out_root = output_root / "labels"
    preview_out_root = output_root / "preview"
    preview_rejected_out_root = output_root / "preview_rejected"

    images_out_root.mkdir(parents=True, exist_ok=True)
    labels_out_root.mkdir(parents=True, exist_ok=True)

    print("\nЗавантаження SAM/SAM2 моделі...")
    model = SAM(args.model)

    processed = 0
    written = 0
    skipped_no_boxes = 0
    rejected_suspicious = 0
    failed = 0
    total_boxes = 0
    total_raw_boxes = 0
    total_filtered_out_boxes = 0

    per_tree_written: dict[str, int] = {}
    per_tree_boxes: dict[str, int] = {}
    per_tree_skipped_no_boxes: dict[str, int] = {}
    per_tree_rejected_suspicious: dict[str, int] = {}
    rejection_reasons: dict[str, int] = {}
    failures: list[dict] = []

    for index, src in enumerate(source_images, start=1):
        print(f"[{index}/{len(source_images)}] {src.tree_key}: {src.image_path.name}")

        try:
            with Image.open(src.image_path) as img:
                image_width, image_height = img.size
                image_rgb = np.asarray(img.convert("RGB"))

            if args.use_tiling and (image_width > args.tile_size or image_height > args.tile_size):
                raw_boxes = run_sam_tiled(
                    model=model,
                    image_path=src.image_path,
                    image_rgb=image_rgb,
                    device=args.device,
                    imgsz=args.imgsz,
                    sam_conf=args.sam_conf,
                    sam_iou=args.sam_iou,
                    verbose=args.verbose,
                    tile_size=args.tile_size,
                    overlap=args.tile_overlap,
                    min_solidity=args.min_solidity,
                )
            else:
                masks = run_sam_on_image(
                    model=model,
                    image_path=src.image_path,
                    device=args.device,
                    imgsz=args.imgsz,
                    sam_conf=args.sam_conf,
                    sam_iou=args.sam_iou,
                    verbose=args.verbose,
                )
                raw_boxes = []
                for mask in masks:
                    box = mask_to_box(
                        mask,
                        image_width=image_width,
                        image_height=image_height,
                        min_solidity=args.min_solidity,
                    )
                    if box is not None:
                        raw_boxes.append(box)

            raw_boxes = add_color_metrics_to_boxes(image_rgb, raw_boxes)

            boxes = filter_boxes(
                boxes=raw_boxes,
                image_width=image_width,
                image_height=image_height,
                min_area_ratio=args.min_area_ratio,
                max_area_ratio=args.max_area_ratio,
                min_box_width_ratio=args.min_box_width_ratio,
                min_box_height_ratio=args.min_box_height_ratio,
                max_box_width_ratio=args.max_box_width_ratio,
                max_box_height_ratio=args.max_box_height_ratio,
                min_aspect_ratio=args.min_aspect_ratio,
                max_aspect_ratio=args.max_aspect_ratio,
                max_circularity=args.max_circularity,
                iou_threshold=args.iou_threshold,
                containment_threshold=args.containment_threshold,
                max_boxes_per_image=args.max_boxes_per_image,
                enable_color_filter=not args.disable_color_filter,
                min_green_ratio=args.min_green_ratio,
                max_fruit_ratio=args.max_fruit_ratio,
                fruit_dominance_ratio=args.fruit_dominance_ratio,
            )

            processed += 1
            total_raw_boxes += len(raw_boxes)
            total_filtered_out_boxes += max(0, len(raw_boxes) - len(boxes))

            rejection_reason = get_image_rejection_reason(
                raw_boxes_count=len(raw_boxes),
                kept_boxes_count=len(boxes),
                max_boxes_per_image=args.max_boxes_per_image,
                reject_if_raw_boxes_more_than=args.reject_if_raw_boxes_more_than,
                reject_if_hit_max_boxes=args.reject_if_hit_max_boxes,
                reject_hit_max_min_raw_boxes=args.reject_hit_max_min_raw_boxes,
            )

            print(
                f"  raw boxes: {len(raw_boxes):>3}, "
                f"kept: {len(boxes):>2}  "
                f"(green≥{args.min_green_ratio:.2f}, "
                f"solid≥{args.min_solidity:.2f})"
                + (f"  REJECT: {rejection_reason}" if rejection_reason else "")
            )

            stem = make_output_stem(src)

            if not boxes and args.skip_no_boxes:
                skipped_no_boxes += 1
                per_tree_skipped_no_boxes[src.tree_key] = (
                    per_tree_skipped_no_boxes.get(src.tree_key, 0) + 1
                )
                continue

            if rejection_reason is not None:
                rejected_suspicious += 1
                per_tree_rejected_suspicious[src.tree_key] = (
                    per_tree_rejected_suspicious.get(src.tree_key, 0) + 1
                )
                rejection_reasons[rejection_reason] = rejection_reasons.get(rejection_reason, 0) + 1

                if args.save_rejected_preview:
                    preview_rejected_dst = (
                        preview_rejected_out_root
                        / src.tree_key
                        / f"{stem}_{slugify(rejection_reason)}.jpg"
                    )
                    save_preview(src.image_path, boxes, preview_rejected_dst)

                continue

            image_dst = images_out_root / src.tree_key / f"{stem}{src.image_path.suffix.lower()}"
            label_dst = labels_out_root / src.tree_key / f"{stem}.txt"

            copy_image(src.image_path, image_dst)
            write_label(label_dst, boxes, image_width=image_width, image_height=image_height)

            if args.save_preview:
                preview_dst = preview_out_root / src.tree_key / f"{stem}.jpg"
                save_preview(src.image_path, boxes, preview_dst)

            written += 1
            total_boxes += len(boxes)
            per_tree_written[src.tree_key] = per_tree_written.get(src.tree_key, 0) + 1
            per_tree_boxes[src.tree_key] = per_tree_boxes.get(src.tree_key, 0) + len(boxes)

        except Exception as ex:
            failed += 1
            failures.append({
                "image": str(src.image_path),
                "tree_key": src.tree_key,
                "error": repr(ex),
            })
            print(f"  ERROR: {ex}")

    stats = {
        "version": "v2",
        "input": str(args.input),
        "output": str(args.output),
        "model": args.model,
        "device": args.device,
        "imgsz": args.imgsz,
        "filters": {
            "sam_conf": args.sam_conf,
            "sam_iou": args.sam_iou,
            "min_area_ratio": args.min_area_ratio,
            "max_area_ratio": args.max_area_ratio,
            "min_aspect_ratio": args.min_aspect_ratio,
            "max_aspect_ratio": args.max_aspect_ratio,
            "min_solidity": args.min_solidity,
            "max_circularity": args.max_circularity,
            "iou_threshold": args.iou_threshold,
            "containment_threshold": args.containment_threshold,
            "max_boxes_per_image": args.max_boxes_per_image,
            "color_filter": not args.disable_color_filter,
            "min_green_ratio": args.min_green_ratio,
            "max_fruit_ratio": args.max_fruit_ratio,
            "fruit_dominance_ratio": args.fruit_dominance_ratio,
            "reject_if_raw_boxes_more_than": args.reject_if_raw_boxes_more_than,
            "reject_if_hit_max_boxes": args.reject_if_hit_max_boxes,
            "reject_hit_max_min_raw_boxes": args.reject_hit_max_min_raw_boxes,
        },
        "source_images": len(source_images),
        "max_images": args.max_images,
        "max_images_per_tree": args.max_images_per_tree,
        "sample_seed": args.sample_seed,
        "processed": processed,
        "written": written,
        "skipped_no_boxes": skipped_no_boxes,
        "rejected_suspicious": rejected_suspicious,
        "failed": failed,
        "total_raw_boxes": total_raw_boxes,
        "total_filtered_out_boxes": total_filtered_out_boxes,
        "total_boxes": total_boxes,
        "avg_raw_boxes_per_processed_image": total_raw_boxes / processed if processed else 0,
        "avg_boxes_per_written_image": total_boxes / written if written else 0,
        "per_tree_written": dict(sorted(per_tree_written.items())),
        "per_tree_boxes": dict(sorted(per_tree_boxes.items())),
        "per_tree_skipped_no_boxes": dict(sorted(per_tree_skipped_no_boxes.items())),
        "per_tree_rejected_suspicious": dict(sorted(per_tree_rejected_suspicious.items())),
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "failures": failures[:100],
        "failures_truncated": len(failures) > 100,
    }

    stats_path = output_root / "dataset_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'=' * 72}")
    print("  Готово")
    print(f"{'=' * 72}")
    print(f"  Source images:          {len(source_images)}")
    print(f"  Processed:              {processed}")
    print(f"  Written images:         {written}")
    print(f"  Skipped (no boxes):     {skipped_no_boxes}")
    print(f"  Rejected suspicious:    {rejected_suspicious}")
    print(f"  Failed:                 {failed}")
    print(f"  Total raw boxes:        {total_raw_boxes}")
    print(f"  Filtered out:           {total_filtered_out_boxes}")
    print(f"  Total kept boxes:       {total_boxes}")
    print(f"  Avg raw/image:          {total_raw_boxes / processed if processed else 0:.2f}")
    print(f"  Avg kept/image:         {total_boxes / written if written else 0:.2f}")
    print(f"  Output:                 {output_root}")
    print(f"  Stats:                  {stats_path}")
    print(f"{'=' * 72}\n")

    if args.save_preview:
        print(f"Preview:  open {preview_out_root}")
        if args.save_rejected_preview:
            print(f"Rejected preview:  open {preview_rejected_out_root}")

    return 0


if __name__ == "__main__":
    sys.exit(main())