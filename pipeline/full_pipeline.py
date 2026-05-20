"""
pipeline/full_pipeline.py

Двостадійний пайплайн виявлення шкідників плодових дерев:

  Стадія 1 (Model A): YOLO + SAHI → детекція leaf-кандидатів
  Стадія 2 (Model B): YOLO detection-format model → pest prediction на кожному leaf-crop

Приклади:
    python pipeline/full_pipeline.py \
        --image data/test_images/example.jpg \
        --leaf-conf 0.45 \
        --pest-conf 0.60 \
        --possible-conf 0.60 \
        --infected-conf 0.80 \
        --healthy-conf 0.70 \
        --slice-size 640 \
        --device mps \
        --save-crops \
        --clean
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Кольори для класів шкідників (BGR)
# ---------------------------------------------------------------------------

CLASS_COLORS = {
    "codling_moth":               (0,   120, 255),
    "gypsy_moth":                 (255, 100,   0),
    "fall_webworm":               (0,   200, 100),
    "apple_ermine_moth":          (200,   0, 200),
    "plum_fruit_moth":            (0,   200, 200),
    "cherry_fruit_fly":           (255,  50,  50),
    "apple_blossom_weevil":       (50,  200, 255),
    "brown_marmorated_stink_bug": (100, 100, 255),
    "lackey_moth":                (255, 200,   0),
    "winter_moth":                (180, 255,  50),
    "healthy":                    (50,  220,  50),
}

STATUS_COLORS = {
    "infected": (0, 0, 220),       # red
    "possible": (0, 180, 255),     # orange/yellow
    "healthy":  (50, 220, 50),     # green
    "unknown":  (150, 150, 150),   # gray
}

DEFAULT_COLOR = (200, 200, 200)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_device(device: str) -> str:
    if device != "auto":
        return device

    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass

    return "cpu"


def clamp_bbox_xyxy(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    return x1, y1, x2, y2


def bbox_area_ratio(bbox: dict[str, int], width: int, height: int) -> float:
    """Частка площі bbox відносно crop/image."""
    area = max(0, bbox["x2"] - bbox["x1"]) * max(0, bbox["y2"] - bbox["y1"])
    denom = max(1, width * height)
    return round(area / denom, 4)


def choose_best_prediction(pests: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not pests:
        return None
    return max(pests, key=lambda p: float(p["conf"]))


def classify_status(
    pests: list[dict[str, Any]],
    healthy_conf: float,
    possible_conf: float,
    infected_conf: float,
) -> tuple[str, dict[str, Any] | None]:
    best = choose_best_prediction(pests)
    if best is None:
        return "unknown", None

    cls_name = best["class"]
    conf = float(best["conf"])

    if cls_name == "healthy":
        if conf >= healthy_conf:
            return "healthy", best
        return "unknown", best

    if conf >= infected_conf:
        return "infected", best

    if conf >= possible_conf:
        return "possible", best

    return "unknown", best


def aggregate_image_status(result: dict[str, Any]) -> str:
    statuses = [leaf.get("status", "unknown") for leaf in result.get("leaves", [])]

    fallback = result.get("image_level_prediction")
    if fallback and fallback.get("status"):
        statuses.append(fallback["status"])

    if "infected" in statuses:
        return "infected"
    if "possible" in statuses:
        return "possible"
    if "healthy" in statuses:
        return "healthy"
    return "unknown"


# ---------------------------------------------------------------------------
# Стадія 1 — leaf detector через SAHI
# ---------------------------------------------------------------------------

class LeafDetector:
    def __init__(
        self,
        weights: str | Path,
        conf: float = 0.45,
        slice_size: int = 640,
        overlap: float = 0.2,
        device: str = "cpu",
        match_threshold: float = 0.5,
    ):
        from sahi import AutoDetectionModel

        self.slice_size = slice_size
        self.overlap = overlap
        self.match_threshold = match_threshold

        self.model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=str(weights),
            confidence_threshold=conf,
            device=device,
        )

    def predict(self, image_path: str | Path) -> list[dict[str, Any]]:
        """
        Повертає список bbox leaf-кандидатів:
        [{'x1': int, 'y1': int, 'x2': int, 'y2': int, 'conf': float}, ...]
        """
        from sahi.predict import get_sliced_prediction

        result = get_sliced_prediction(
            str(image_path),
            self.model,
            slice_height=self.slice_size,
            slice_width=self.slice_size,
            overlap_height_ratio=self.overlap,
            overlap_width_ratio=self.overlap,
            postprocess_type="NMS",
            postprocess_match_threshold=self.match_threshold,
            verbose=0,
        )

        leaves = []
        for pred in result.object_prediction_list:
            leaves.append({
                "x1": int(round(pred.bbox.minx)),
                "y1": int(round(pred.bbox.miny)),
                "x2": int(round(pred.bbox.maxx)),
                "y2": int(round(pred.bbox.maxy)),
                "conf": round(float(pred.score.value), 4),
            })

        return leaves


# ---------------------------------------------------------------------------
# Стадія 2 — pest model
# ---------------------------------------------------------------------------

class PestDetector:
    def __init__(
        self,
        weights: str | Path,
        conf: float = 0.60,
        device: str = "cpu",
    ):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.conf = conf
        self.device = device

    def predict(self, image_bgr: np.ndarray) -> list[dict[str, Any]]:
        """
        Приймає BGR image/crop, повертає список pest predictions:
        [
          {
            'class': str,
            'conf': float,
            'bbox': {'x1','y1','x2','y2'},  # координати в межах crop/image
            'area_ratio': float
          },
          ...
        ]
        """
        h, w = image_bgr.shape[:2]

        results = self.model(
            image_bgr,
            conf=self.conf,
            device=self.device,
            verbose=False,
        )

        pests: list[dict[str, Any]] = []
        boxes = getattr(results[0], "boxes", None)
        if boxes is None:
            return pests

        for box in boxes:
            cls_id = int(box.cls)
            cls_name = self.model.names[cls_id]
            conf = round(float(box.conf), 4)

            xyxy = box.xyxy[0].detach().cpu().numpy().tolist()
            x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
            x1, y1, x2, y2 = clamp_bbox_xyxy(x1, y1, x2, y2, w, h)

            bbox = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

            pests.append({
                "class": cls_name,
                "conf": conf,
                "bbox": bbox,
                "area_ratio": bbox_area_ratio(bbox, w, h),
            })

        return pests


# ---------------------------------------------------------------------------
# Повний пайплайн
# ---------------------------------------------------------------------------

class FruitPestPipeline:
    def __init__(
        self,
        leaf_weights: str | Path,
        pest_weights: str | Path,
        leaf_conf: float = 0.45,
        pest_conf: float = 0.60,
        possible_conf: float = 0.60,
        infected_conf: float = 0.80,
        healthy_conf: float = 0.70,
        slice_size: int = 640,
        overlap: float = 0.2,
        device: str = "cpu",
        leaf_match_threshold: float = 0.5,
        fallback_full_image: bool = True,
    ):
        self.device = resolve_device(device)
        self.possible_conf = possible_conf
        self.infected_conf = infected_conf
        self.healthy_conf = healthy_conf
        self.fallback_full_image = fallback_full_image

        print("  Завантаження Model A (leaf detector)...")
        self.leaf_detector = LeafDetector(
            weights=leaf_weights,
            conf=leaf_conf,
            slice_size=slice_size,
            overlap=overlap,
            device=self.device,
            match_threshold=leaf_match_threshold,
        )

        print("  Завантаження Model B (pest model)...")
        self.pest_detector = PestDetector(
            weights=pest_weights,
            conf=pest_conf,
            device=self.device,
        )

    def _predict_crop_pests(
        self,
        crop: np.ndarray,
        offset_x: int,
        offset_y: int,
    ) -> tuple[list[dict[str, Any]], str, dict[str, Any] | None]:
        pests = self.pest_detector.predict(crop)

        # Додаємо абсолютні координати pest bbox на оригінальному зображенні.
        for pest in pests:
            b = pest["bbox"]
            pest["bbox_abs"] = {
                "x1": b["x1"] + offset_x,
                "y1": b["y1"] + offset_y,
                "x2": b["x2"] + offset_x,
                "y2": b["y2"] + offset_y,
            }

        status, best = classify_status(
            pests,
            healthy_conf=self.healthy_conf,
            possible_conf=self.possible_conf,
            infected_conf=self.infected_conf,
        )

        return pests, status, best

    def _predict_image_level(self, image: np.ndarray) -> dict[str, Any]:
        pests, status, best = self._predict_crop_pests(image, offset_x=0, offset_y=0)

        return {
            "used": True,
            "status": status,
            "best": best,
            "pests": pests,
        }

    def predict(self, image_path: str | Path) -> dict[str, Any]:
        """
        Запускає повний пайплайн на одному зображенні.

        Основні поля:
          - leaves: leaf-level predictions;
          - possible/infected/healthy/unknown: кількість leaf-кандидатів за статусами;
          - image_level_prediction: fallback на повне зображення;
          - image_status: загальний статус зображення.
        """
        image_path = Path(image_path)
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Не вдалося прочитати зображення: {image_path}")

        H, W = image.shape[:2]

        # ── Стадія 1: leaf detection ─────────────────────────────────
        t0 = time.perf_counter()
        leaves_raw = self.leaf_detector.predict(image_path)
        t_leaf = time.perf_counter() - t0

        # ── Стадія 2: pest prediction на leaf-crops ──────────────────
        t1 = time.perf_counter()
        leaves_out: list[dict[str, Any]] = []

        for leaf in leaves_raw:
            x1, y1, x2, y2 = clamp_bbox_xyxy(
                int(leaf["x1"]),
                int(leaf["y1"]),
                int(leaf["x2"]),
                int(leaf["y2"]),
                W,
                H,
            )

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image[y1:y2, x1:x2]
            pests, status, best = self._predict_crop_pests(crop, offset_x=x1, offset_y=y1)

            leaves_out.append({
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "leaf_conf": leaf["conf"],
                "pests": pests,
                "best_pest": best,
                "status": status,
            })

        t_pest = time.perf_counter() - t1

        # ── Fallback: якщо leaf-crops нічого не підтвердили ──────────
        # Корисно для test_images, де фото може бути крупним pest-shot без листя.
        image_level_prediction = {
            "used": False,
            "status": "unknown",
            "best": None,
            "pests": [],
        }

        leaf_has_signal = any(
            leaf["status"] in {"infected", "possible", "healthy"}
            for leaf in leaves_out
        )

        if self.fallback_full_image and not leaf_has_signal:
            t_fb0 = time.perf_counter()
            image_level_prediction = self._predict_image_level(image)
            t_fallback = time.perf_counter() - t_fb0
        else:
            t_fallback = 0.0

        t_total = time.perf_counter() - t0

        result = {
            "image": str(image_path),
            "resolution": f"{W}×{H}",
            "leaf_count": len(leaves_out),
            "infected": sum(1 for l in leaves_out if l["status"] == "infected"),
            "possible": sum(1 for l in leaves_out if l["status"] == "possible"),
            "healthy": sum(1 for l in leaves_out if l["status"] == "healthy"),
            "unknown": sum(1 for l in leaves_out if l["status"] == "unknown"),
            "image_level_prediction": image_level_prediction,
            "time": {
                "leaf_s": round(t_leaf, 3),
                "pest_s": round(t_pest, 3),
                "fallback_s": round(t_fallback, 3),
                "total_s": round(t_total, 3),
            },
            "thresholds": {
                "possible_conf": self.possible_conf,
                "infected_conf": self.infected_conf,
                "healthy_conf": self.healthy_conf,
            },
            "leaves": leaves_out,
        }

        result["image_status"] = aggregate_image_status(result)
        return result

    def visualize(
        self,
        image_path: str | Path,
        result: dict[str, Any],
        out_path: str | Path,
        show_conf: bool = True,
        show_pest_boxes: bool = True,
    ) -> np.ndarray:
        """Малює результати на оригінальному зображенні і зберігає."""
        image = cv2.imread(str(image_path))
        if image is None:
            return np.zeros((100, 100, 3), dtype=np.uint8)

        for i, leaf in enumerate(result["leaves"]):
            b = leaf["bbox"]
            x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]

            status = leaf.get("status", "unknown")
            leaf_color = STATUS_COLORS.get(status, STATUS_COLORS["unknown"])

            cv2.rectangle(image, (x1, y1), (x2, y2), leaf_color, 2)

            leaf_label = f"#{i + 1} {status}"
            if show_conf:
                leaf_label += f" {leaf['leaf_conf']:.2f}"

            _draw_label(image, leaf_label, x1, y1, leaf_color)

            # Pest bbox-и всередині leaf-crop у координатах оригінального зображення.
            if show_pest_boxes:
                for p in leaf.get("pests", []):
                    pb = p.get("bbox_abs")
                    if not pb:
                        continue

                    pest_color = CLASS_COLORS.get(p["class"], DEFAULT_COLOR)
                    cv2.rectangle(
                        image,
                        (pb["x1"], pb["y1"]),
                        (pb["x2"], pb["y2"]),
                        pest_color,
                        1,
                    )

            # Текстові pest predictions біля leaf bbox.
            if leaf.get("pests"):
                seen: dict[str, float] = {}
                for p in leaf["pests"]:
                    if p["class"] not in seen or p["conf"] > seen[p["class"]]:
                        seen[p["class"]] = p["conf"]

                y_offset = y2 + 16
                for cls_name, conf in seen.items():
                    color = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)
                    label = f"{cls_name} {conf:.2f}" if show_conf else cls_name
                    _draw_label(image, label, x1, y_offset, color, bg=True)
                    y_offset += 20

        # Fallback image-level prediction.
        fallback = result.get("image_level_prediction", {})
        if fallback.get("used") and fallback.get("best"):
            best = fallback["best"]
            status = fallback.get("status", "unknown")
            color = STATUS_COLORS.get(status, STATUS_COLORS["unknown"])
            label = f"image-level: {status} | {best['class']} {best['conf']:.2f}"
            _draw_label(image, label, 10, image.shape[0] - 10, color, bg=True)

            if show_pest_boxes:
                for p in fallback.get("pests", []):
                    pb = p.get("bbox_abs")
                    if not pb:
                        continue
                    pest_color = CLASS_COLORS.get(p["class"], DEFAULT_COLOR)
                    cv2.rectangle(
                        image,
                        (pb["x1"], pb["y1"]),
                        (pb["x2"], pb["y2"]),
                        pest_color,
                        1,
                    )

        _draw_summary(image, result)

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), image)
        return image

    def save_crops(
        self,
        image_path: str | Path,
        result: dict[str, Any],
        crops_dir: str | Path,
        clean_dir: bool = True,
    ) -> None:
        """Зберігає crop-и кожного leaf-кандидата окремо."""
        image = cv2.imread(str(image_path))
        if image is None:
            return

        crops_dir = Path(crops_dir)
        if clean_dir and crops_dir.exists():
            shutil.rmtree(crops_dir)
        crops_dir.mkdir(parents=True, exist_ok=True)

        stem = Path(image_path).stem

        for i, leaf in enumerate(result["leaves"]):
            b = leaf["bbox"]
            crop = image[b["y1"]:b["y2"], b["x1"]:b["x2"]]
            status = leaf["status"]

            best = leaf.get("best_pest")
            if best:
                suffix = f"{status}_{best['class']}_{best['conf']:.2f}"
            else:
                suffix = status

            out_name = f"{stem}_leaf{i + 1:02d}_{suffix}.jpg"
            cv2.imwrite(str(crops_dir / out_name), crop)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_label(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
    bg: bool = True,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1

    h_img, w_img = image.shape[:2]
    x = max(0, min(w_img - 1, x))
    y = max(16, min(h_img - 4, y))

    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

    # Якщо label виходить за правий край — зсуваємо вліво.
    if x + tw + 8 > w_img:
        x = max(0, w_img - tw - 8)

    if bg:
        cv2.rectangle(
            image,
            (x, y - th - baseline - 4),
            (x + tw + 6, y + baseline),
            color,
            -1,
        )
        cv2.putText(
            image,
            text,
            (x + 3, y - 3),
            font,
            scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            image,
            text,
            (x + 3, y - 3),
            font,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )


def _draw_summary(image: np.ndarray, result: dict[str, Any]) -> None:
    lines = [
        f"Image:    {result.get('image_status', 'unknown')}",
        f"Leaves:   {result['leaf_count']}",
        f"Infected: {result['infected']}",
        f"Possible: {result.get('possible', 0)}",
        f"Healthy:  {result['healthy']}",
        f"Unknown:  {result['unknown']}",
        f"Time:     {result['time']['total_s']:.2f}s",
    ]

    pad = 10
    line_h = 21
    box_w = 205
    box_h = pad * 2 + line_h * len(lines)

    overlay = image.copy()
    cv2.rectangle(
        overlay,
        (pad, pad),
        (pad + box_w, pad + box_h),
        (30, 30, 30),
        -1,
    )
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

    for i, line in enumerate(lines):
        y = pad * 2 + i * line_h
        cv2.putText(
            image,
            line,
            (pad + 8, y + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Двостадійний пайплайн виявлення шкідників плодових дерев.",
    )

    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--leaf-weights", type=Path, default=Path("runs/leaf_detector/v1/weights/best.pt"))
    parser.add_argument("--pest-weights", type=Path, default=Path("runs/pest_detector/v1/weights/best.pt"))
    parser.add_argument("--out-dir", type=Path, default=Path("pipeline/output"))

    parser.add_argument("--leaf-conf", type=float, default=0.45)
    parser.add_argument("--pest-conf", type=float, default=0.60)
    parser.add_argument("--possible-conf", type=float, default=0.60)
    parser.add_argument("--infected-conf", type=float, default=0.80)
    parser.add_argument("--healthy-conf", type=float, default=0.70)

    parser.add_argument("--slice-size", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--leaf-match-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="mps")

    parser.add_argument("--save-crops", action="store_true")
    parser.add_argument("--no-vis", action="store_true")
    parser.add_argument("--hide-pest-boxes", action="store_true")
    parser.add_argument("--no-fallback-full-image", action="store_true")
    parser.add_argument("--clean", action="store_true")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.device = resolve_device(args.device)

    if not args.image.exists():
        print(f"Зображення не знайдено: {args.image}")
        return
    if not args.leaf_weights.exists():
        print(f"Ваги leaf detector не знайдено: {args.leaf_weights}")
        return
    if not args.pest_weights.exists():
        print(f"Ваги pest model не знайдено: {args.pest_weights}")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)

    stem = args.image.stem
    if args.clean:
        for path in [
            args.out_dir / f"{stem}_result.json",
            args.out_dir / f"{stem}_annotated.jpg",
            args.out_dir / f"{stem}_crops",
        ]:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()

    print(f'\n{"=" * 72}')
    print("  Fruit Pest Detection Pipeline")
    print(f'{"=" * 72}')
    print(f"  image:           {args.image}")
    print(f"  leaf model:      {args.leaf_weights}")
    print(f"  pest model:      {args.pest_weights}")
    print(f"  leaf conf:       {args.leaf_conf}")
    print(f"  pest conf:       {args.pest_conf}")
    print(f"  possible conf:   {args.possible_conf}")
    print(f"  infected conf:   {args.infected_conf}")
    print(f"  healthy conf:    {args.healthy_conf}")
    print(f"  slice size:      {args.slice_size}")
    print(f"  device:          {args.device}")
    print(f"  fallback image:  {not args.no_fallback_full_image}")
    print(f'{"=" * 72}\n')

    pipeline = FruitPestPipeline(
        leaf_weights=args.leaf_weights,
        pest_weights=args.pest_weights,
        leaf_conf=args.leaf_conf,
        pest_conf=args.pest_conf,
        possible_conf=args.possible_conf,
        infected_conf=args.infected_conf,
        healthy_conf=args.healthy_conf,
        slice_size=args.slice_size,
        overlap=args.overlap,
        device=args.device,
        leaf_match_threshold=args.leaf_match_threshold,
        fallback_full_image=not args.no_fallback_full_image,
    )

    print(f"\nОбробка: {args.image.name}")
    result = pipeline.predict(args.image)

    print(f'\n{"=" * 72}')
    print("  Результат")
    print(f'{"=" * 72}')
    print(f'  Роздільна здатність: {result["resolution"]}')
    print(f'  Image status:         {result["image_status"]}')
    print(f'  Знайдено листків:     {result["leaf_count"]}')
    print(f'  Уражені:              {result["infected"]}')
    print(f'  Можливі:              {result["possible"]}')
    print(f'  Здорові:              {result["healthy"]}')
    print(f'  Невизначені:          {result["unknown"]}')
    print(f'  Час (leaf):           {result["time"]["leaf_s"]}s')
    print(f'  Час (pest):           {result["time"]["pest_s"]}s')
    print(f'  Час (fallback):       {result["time"]["fallback_s"]}s')
    print(f'  Час (total):          {result["time"]["total_s"]}s')
    print(f'{"=" * 72}')

    if result["leaf_count"] > 0:
        print("\n  Деталі по leaf-кандидатах:")
        for i, leaf in enumerate(result["leaves"], start=1):
            b = leaf["bbox"]
            best = leaf.get("best_pest")
            if best:
                pests_str = f"{best['class']} ({best['conf']:.2f})"
            else:
                pests_str = "—"

            print(
                f'  #{i:>2} [{b["x1"]},{b["y1"]},{b["x2"]},{b["y2"]}] '
                f'{leaf["status"]:<9} leaf={leaf["leaf_conf"]:.2f}  {pests_str}'
            )

    fallback = result.get("image_level_prediction", {})
    if fallback.get("used"):
        best = fallback.get("best")
        best_str = f"{best['class']} ({best['conf']:.2f})" if best else "—"
        print(f"\n  Fallback full-image: {fallback['status']}  {best_str}")

    json_path = args.out_dir / f"{stem}_result.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  JSON: {json_path}")

    if not args.no_vis:
        vis_path = args.out_dir / f"{stem}_annotated.jpg"
        pipeline.visualize(
            args.image,
            result,
            vis_path,
            show_pest_boxes=not args.hide_pest_boxes,
        )
        print(f"  Vis:  {vis_path}")

    if args.save_crops:
        crops_dir = args.out_dir / f"{stem}_crops"
        pipeline.save_crops(args.image, result, crops_dir, clean_dir=True)
        print(f"  Crops: {crops_dir}/")

    print()


if __name__ == "__main__":
    main()
