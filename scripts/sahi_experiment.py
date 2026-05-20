"""
scripts/sahi_experiment.py

Порівнює детекцію листя:
  1. Без SAHI — пряме передбачення YOLO на повному зображенні
  2. З SAHI — sliced inference з різними розмірами тайлів

Результат:
  - results.json               — всі сирі результати
  - table_for_report.md        — агрегована таблиця для звіту
  - per_image_results.md       — детальні результати по кожному зображенню
  - visualizations/            — preview з bbox

Приклади запуску:
    python scripts/sahi_experiment.py

    python scripts/sahi_experiment.py \
        --weights runs/leaf_detector/v1/weights/best.pt \
        --images data/interim/leaves_inaturalist \
        --n-images 10 \
        --conf 0.35 \
        --device mps

    python scripts/sahi_experiment.py --slice-sizes 512,640,1024 --overlap 0.2
    python scripts/sahi_experiment.py --n-images 30 --seed 123
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class SahiConfig:
    slice_size: int
    overlap: float


def parse_slice_sizes(value: str) -> list[int]:
    """Парсить рядок типу '512,640,1024' у список int."""
    try:
        sizes = [int(x.strip()) for x in value.split(",") if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "slice-sizes має бути списком чисел, наприклад: 512,640,1024"
        ) from exc

    if not sizes:
        raise argparse.ArgumentTypeError("Потрібно вказати хоча б один slice size.")

    invalid = [s for s in sizes if s <= 0]
    if invalid:
        raise argparse.ArgumentTypeError(f"Некоректні slice sizes: {invalid}")

    return sizes


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


def get_test_images(
    src: Path,
    n: int,
    seed: int = 42,
    max_images_per_tree: int = 0,
) -> list[Path]:
    if not src.exists():
        raise FileNotFoundError(f"Папку із зображеннями не знайдено: {src}")
    if not src.is_dir():
        raise NotADirectoryError(f"Очікувалася папка із зображеннями: {src}")

    rng = random.Random(seed)
    tree_dirs = [p for p in sorted(src.iterdir()) if p.is_dir()]

    # Якщо зображення лежать прямо в src без підпапок.
    if not tree_dirs:
        images = [p for p in sorted(src.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
        rng.shuffle(images)
        return images if n <= 0 else images[:n]

    buckets: dict[str, list[Path]] = {}
    for tree_dir in tree_dirs:
        images = [p for p in sorted(tree_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
        if not images:
            continue

        rng.shuffle(images)
        if max_images_per_tree > 0:
            images = images[:max_images_per_tree]

        buckets[tree_dir.name] = images

    selected: list[Path] = []
    tree_names = list(buckets.keys())

    # Round-robin: беремо по одному з кожної папки, щоб не перекосити вибірку.
    while tree_names and (n <= 0 or len(selected) < n):
        progressed = False

        for tree_name in list(tree_names):
            if n > 0 and len(selected) >= n:
                break

            bucket = buckets[tree_name]
            if bucket:
                selected.append(bucket.pop())
                progressed = True

            if not bucket:
                tree_names.remove(tree_name)

        if not progressed:
            break

    return selected


def relative_image_name(image_path: Path, root: Path) -> str:
    """Повертає відносний шлях image відносно root, якщо це можливо."""
    try:
        return str(image_path.relative_to(root))
    except ValueError:
        return str(image_path)


def safe_stem(image_path: Path, root: Path) -> str:
    rel = relative_image_name(image_path, root)
    return Path(rel).with_suffix("").as_posix().replace("/", "__")


def yolo_boxes_to_stats(result_obj: Any) -> tuple[int, float]:
    boxes = getattr(result_obj, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return 0, 0.0

    confidences = boxes.conf.tolist()
    avg_conf = float(np.mean(confidences)) if confidences else 0.0
    return len(boxes), avg_conf


def run_no_sahi(
    model: Any,
    image_path: Path,
    images_root: Path,
    conf: float,
    imgsz: int,
    device: str,
) -> dict[str, Any]:
    """Інференс без SAHI — напряму на повному зображенні."""
    t0 = time.perf_counter()

    results = model(
        str(image_path),
        conf=conf,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )

    elapsed = time.perf_counter() - t0
    result_obj = results[0]
    n_det, avg_conf = yolo_boxes_to_stats(result_obj)

    return {
        "method": "no_sahi",
        "tree": image_path.parent.name,
        "image": relative_image_name(image_path, images_root),
        "slice_size": None,
        "overlap": None,
        "detections": int(n_det),
        "avg_conf": round(avg_conf, 4),
        "time_s": round(elapsed, 3),
        "result_obj": result_obj,
    }


def run_sahi(
    sahi_model: Any,
    image_path: Path,
    images_root: Path,
    config: SahiConfig,
    postprocess_type: str,
    match_threshold: float,
) -> dict[str, Any]:
    """Інференс через SAHI з нарізкою на тайли."""
    from sahi.predict import get_sliced_prediction

    t0 = time.perf_counter()

    result = get_sliced_prediction(
        str(image_path),
        sahi_model,
        slice_height=config.slice_size,
        slice_width=config.slice_size,
        overlap_height_ratio=config.overlap,
        overlap_width_ratio=config.overlap,
        postprocess_type=postprocess_type,
        postprocess_match_threshold=match_threshold,
        verbose=0,
    )

    elapsed = time.perf_counter() - t0
    preds = result.object_prediction_list
    confidences = [float(p.score.value) for p in preds]
    avg_conf = float(np.mean(confidences)) if confidences else 0.0

    return {
        "method": f"sahi_{config.slice_size}",
        "tree": image_path.parent.name,
        "image": relative_image_name(image_path, images_root),
        "slice_size": config.slice_size,
        "overlap": config.overlap,
        "detections": int(len(preds)),
        "avg_conf": round(avg_conf, 4),
        "time_s": round(elapsed, 3),
        "result_obj": result,
    }


def save_no_sahi_vis(result_obj: Any, dst: Path) -> None:
    """Зберігає візуалізацію YOLO без SAHI."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    img = result_obj.plot()
    cv2.imwrite(str(dst), img)


def draw_label(image: np.ndarray, text: str, x: int, y: int) -> None:
    """Малює підпис над bbox так, щоб він був читабельний."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1

    y = max(y, 14)
    (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)

    cv2.rectangle(
        image,
        (x, y - h - baseline - 4),
        (x + w + 4, y + baseline),
        (0, 200, 80),
        -1,
    )
    cv2.putText(
        image,
        text,
        (x + 2, y - 3),
        font,
        scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


def save_sahi_vis(result_obj: Any, image_path: Path, dst: Path) -> None:
    """Зберігає візуалізацію SAHI результату через OpenCV."""
    dst.parent.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"  WARN: не вдалося прочитати image для preview: {image_path}")
        return

    for pred in result_obj.object_prediction_list:
        x1 = int(round(pred.bbox.minx))
        y1 = int(round(pred.bbox.miny))
        x2 = int(round(pred.bbox.maxx))
        y2 = int(round(pred.bbox.maxy))
        conf = float(pred.score.value)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 200, 80), 2)
        draw_label(image, f"{conf:.2f}", x1, y1 - 4)

    cv2.imwrite(str(dst), image)


def strip_result_obj(row: dict[str, Any]) -> dict[str, Any]:
    """Видаляє важкий Python-об'єкт перед збереженням у JSON."""
    clean = dict(row)
    clean.pop("result_obj", None)
    return clean


def summarize_by_method(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Агрегує результати по method."""
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)

    summary: list[dict[str, Any]] = []
    for method, method_rows in sorted(by_method.items()):
        detections = [r["detections"] for r in method_rows]
        confidences = [r["avg_conf"] for r in method_rows]
        times = [r["time_s"] for r in method_rows]

        summary.append(
            {
                "method": method,
                "images": len(method_rows),
                "avg_detections": round(float(np.mean(detections)), 2),
                "median_detections": round(float(np.median(detections)), 2),
                "avg_conf": round(float(np.mean(confidences)), 4),
                "avg_time_s": round(float(np.mean(times)), 3),
                "total_time_s": round(float(np.sum(times)), 3),
            }
        )

    return summary


def print_summary_table(rows: list[dict[str, Any]]) -> None:
    """Виводить зведену таблицю по методах."""
    summary = summarize_by_method(rows)

    print(f"\n{'=' * 82}")
    print("  SAHI Experiment — зведена таблиця")
    print(f"{'=' * 82}")
    print(
        f"  {'Метод':<16} {'Imgs':>5} {'Avg det':>9} "
        f"{'Med det':>9} {'Avg conf':>10} {'Avg time':>10}"
    )
    print(f"  {'-'*16} {'-'*5} {'-'*9} {'-'*9} {'-'*10} {'-'*10}")

    for row in summary:
        print(
            f"  {row['method']:<16} "
            f"{row['images']:>5} "
            f"{row['avg_detections']:>9.2f} "
            f"{row['median_detections']:>9.2f} "
            f"{row['avg_conf']:>10.4f} "
            f"{row['avg_time_s']:>9.3f}s"
        )

    print(f"{'=' * 82}\n")


def write_outputs(rows: list[dict[str, Any]], out_dir: Path) -> None:
    """Зберігає JSON і markdown-таблиці."""
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize_by_method(rows)

    json_path = out_dir / "results.json"
    json_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    md_lines = [
        "| Метод | К-сть зображень | Avg детекцій | Median детекцій | Avg confidence | Avg час (с) | Total час (с) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for row in summary:
        md_lines.append(
            f"| {row['method']} "
            f"| {row['images']} "
            f"| {row['avg_detections']:.2f} "
            f"| {row['median_detections']:.2f} "
            f"| {row['avg_conf']:.4f} "
            f"| {row['avg_time_s']:.3f} "
            f"| {row['total_time_s']:.3f} |"
        )

    (out_dir / "table_for_report.md").write_text("\n".join(md_lines), encoding="utf-8")

    per_image_lines = [
        "| Зображення | Дерево | Метод | Slice | Overlap | Det | Avg conf | Час (с) |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]

    for row in rows:
        slice_size = "" if row["slice_size"] is None else row["slice_size"]
        overlap = "" if row["overlap"] is None else row["overlap"]
        per_image_lines.append(
            f"| {row['image']} "
            f"| {row['tree']} "
            f"| {row['method']} "
            f"| {slice_size} "
            f"| {overlap} "
            f"| {row['detections']} "
            f"| {row['avg_conf']:.4f} "
            f"| {row['time_s']:.3f} |"
        )

    (out_dir / "per_image_results.md").write_text(
        "\n".join(per_image_lines),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Порівняння YOLO leaf detector без SAHI та з SAHI.",
    )

    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("runs/leaf_detector/v1/weights/best.pt"),
        help="Шлях до YOLO .pt weights.",
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("data/interim/leaves_inaturalist"),
        help="Папка з тестовими зображеннями. Може містити підпапки класів/дерев.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("report/sahi_experiment"),
        help="Папка для результатів.",
    )
    parser.add_argument(
        "--n-images",
        type=int,
        default=10,
        help="Скільки зображень взяти. Якщо <= 0 — взяти всі.",
    )
    parser.add_argument(
        "--max-images-per-tree",
        type=int,
        default=0,
        help="Максимум зображень з однієї підпапки/дерева. 0 — без обмеження.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed для повторюваного вибору зображень.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Confidence threshold для YOLO/SAHI.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Розмір inference для no_sahi.",
    )
    parser.add_argument(
        "--device",
        default="mps",
        help="Device: mps, cpu, cuda або auto.",
    )
    parser.add_argument(
        "--slice-sizes",
        type=parse_slice_sizes,
        default=parse_slice_sizes("512,640,1024"),
        help="SAHI tile sizes через кому.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.2,
        help="Overlap ratio для SAHI.",
    )
    parser.add_argument(
        "--postprocess-type",
        choices=["NMS", "NMM", "GREEDYNMM"],
        default="NMS",
        help="SAHI postprocess type.",
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.5,
        help="Threshold для об'єднання/фільтрації bbox після slicing.",
    )
    parser.add_argument(
        "--no-save-vis",
        action="store_true",
        help="Не зберігати preview-зображення.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    args.device = resolve_device(args.device)

    if not args.weights.exists():
        print(f"Ваги не знайдено: {args.weights}")
        print("Вкажи правильний шлях через --weights")
        return

    if not args.images.exists():
        print(f"Папку із зображеннями не знайдено: {args.images}")
        print("Вкажи правильний шлях через --images")
        return

    args.out.mkdir(parents=True, exist_ok=True)
    vis_dir = args.out / "visualizations"
    if not args.no_save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 72}")
    print("  SAHI Experiment — Leaf Detector")
    print(f"{'=' * 72}")
    print(f"  weights:            {args.weights}")
    print(f"  images:             {args.images}")
    print(f"  out:                {args.out}")
    print(f"  n-images:           {args.n_images}")
    print(f"  max-images/tree:    {args.max_images_per_tree}")
    print(f"  seed:               {args.seed}")
    print(f"  conf:               {args.conf}")
    print(f"  imgsz no_sahi:      {args.imgsz}")
    print(f"  device:             {args.device}")
    print(f"  slice sizes:        {args.slice_sizes}")
    print(f"  overlap:            {args.overlap}")
    print(f"  postprocess:        {args.postprocess_type}")
    print(f"  match threshold:    {args.match_threshold}")
    print(f"  save vis:           {not args.no_save_vis}")
    print(f"{'=' * 72}\n")

    print("Завантаження YOLO...")
    from ultralytics import YOLO

    yolo_model = YOLO(str(args.weights))

    print("Завантаження SAHI wrapper...")
    from sahi import AutoDetectionModel

    sahi_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(args.weights),
        confidence_threshold=args.conf,
        device=args.device,
    )

    try:
        test_images = get_test_images(
            args.images,
            n=args.n_images,
            seed=args.seed,
            max_images_per_tree=args.max_images_per_tree,
        )
    except Exception as exc:
        print(f"Помилка під час вибору тестових зображень: {exc}")
        return

    if not test_images:
        print(f"Зображення не знайдено у {args.images}")
        return

    print(f"Вибрано {len(test_images)} зображень\n")

    sahi_configs = [
        SahiConfig(slice_size=s, overlap=args.overlap)
        for s in args.slice_sizes
    ]

    all_rows: list[dict[str, Any]] = []

    for i, img_path in enumerate(test_images, 1):
        print(f"[{i}/{len(test_images)}] {relative_image_name(img_path, args.images)}")

        # --- Без SAHI ---
        r = run_no_sahi(
            yolo_model,
            img_path,
            images_root=args.images,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
        )

        print(f"  no_sahi:     {r['detections']:>3} det  {r['avg_conf']:.3f} conf  {r['time_s']:.2f}s")

        if not args.no_save_vis:
            vis_path = vis_dir / f"{i:02d}_{safe_stem(img_path, args.images)}__no_sahi.jpg"
            save_no_sahi_vis(r["result_obj"], vis_path)

        all_rows.append(strip_result_obj(r))

        # --- З SAHI ---
        for cfg in sahi_configs:
            r = run_sahi(
                sahi_model,
                img_path,
                images_root=args.images,
                config=cfg,
                postprocess_type=args.postprocess_type,
                match_threshold=args.match_threshold,
            )

            print(
                f"  sahi_{cfg.slice_size:<4}: "
                f"{r['detections']:>3} det  "
                f"{r['avg_conf']:.3f} conf  "
                f"{r['time_s']:.2f}s"
            )

            if not args.no_save_vis:
                vis_path = vis_dir / (
                    f"{i:02d}_{safe_stem(img_path, args.images)}"
                    f"__sahi_{cfg.slice_size}.jpg"
                )
                save_sahi_vis(r["result_obj"], img_path, vis_path)

            all_rows.append(strip_result_obj(r))

        print()

    print_summary_table(all_rows)
    write_outputs(all_rows, args.out)

    print(f"Результати збережено: {args.out}/")
    print("  results.json          ← summary + всі дані")
    print("  table_for_report.md   ← агрегована таблиця для звіту")
    print("  per_image_results.md  ← деталізація по кожному зображенню")
    if not args.no_save_vis:
        print("  visualizations/       ← preview-картинки з bbox")
    print()


if __name__ == "__main__":
    main()
