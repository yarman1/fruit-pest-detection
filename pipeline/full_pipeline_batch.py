"""
pipeline/full_pipeline_batch.py

Batch runner для pipeline/full_pipeline.py.

Проганяє всі картинки з папки, завантажуючи моделі один раз.

Приклад:
    python pipeline/full_pipeline_batch.py \
        --images-dir data/test_images \
        --leaf-conf 0.45 \
        --pest-conf 0.60 \
        --possible-conf 0.60 \
        --infected-conf 0.80 \
        --healthy-conf 0.70 \
        --slice-size 640 \
        --device mps \
        --save-crops \
        --clean \
        --out-dir pipeline/output_test_images
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

try:
    from full_pipeline import FruitPestPipeline, resolve_device
except ImportError:
    from pipeline.full_pipeline import FruitPestPipeline, resolve_device


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def collect_images(images_dir: Path, recursive: bool = False) -> list[Path]:
    if not images_dir.exists():
        raise FileNotFoundError(f"Папку не знайдено: {images_dir}")
    if not images_dir.is_dir():
        raise NotADirectoryError(f"Очікувалася папка: {images_dir}")

    iterator = images_dir.rglob("*") if recursive else images_dir.iterdir()
    return sorted(
        p for p in iterator
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def safe_name(image_path: Path, images_root: Path) -> str:
    try:
        rel = image_path.relative_to(images_root)
    except ValueError:
        rel = image_path.name

    if isinstance(rel, Path):
        return rel.with_suffix("").as_posix().replace("/", "__")

    return Path(str(rel)).with_suffix("").as_posix().replace("/", "__")


def make_summary_row(image_path: Path, result: dict[str, Any]) -> dict[str, Any]:
    top_predictions = []

    for i, leaf in enumerate(result.get("leaves", []), start=1):
        best = leaf.get("best_pest")
        if not best:
            continue

        top_predictions.append({
            "scope": "leaf",
            "leaf": i,
            "status": leaf.get("status"),
            "class": best.get("class"),
            "conf": best.get("conf"),
        })

    fallback = result.get("image_level_prediction", {})
    if fallback.get("used") and fallback.get("best"):
        best = fallback["best"]
        top_predictions.append({
            "scope": "image",
            "leaf": None,
            "status": fallback.get("status"),
            "class": best.get("class"),
            "conf": best.get("conf"),
        })

    return {
        "image": str(image_path),
        "resolution": result.get("resolution", ""),
        "image_status": result.get("image_status", "unknown"),
        "leaf_count": result.get("leaf_count", 0),
        "infected": result.get("infected", 0),
        "possible": result.get("possible", 0),
        "healthy": result.get("healthy", 0),
        "unknown": result.get("unknown", 0),
        "fallback_used": bool(result.get("image_level_prediction", {}).get("used", False)),
        "fallback_status": result.get("image_level_prediction", {}).get("status", "unknown"),
        "leaf_s": result.get("time", {}).get("leaf_s", 0.0),
        "pest_s": result.get("time", {}).get("pest_s", 0.0),
        "fallback_s": result.get("time", {}).get("fallback_s", 0.0),
        "total_s": result.get("time", {}).get("total_s", 0.0),
        "top_predictions": top_predictions,
    }


def write_batch_summary(rows: list[dict[str, Any]], out_dir: Path, wall_time_s: float) -> None:
    total_images = len(rows)
    ok_rows = [r for r in rows if "error" not in r]
    failed_rows = [r for r in rows if "error" in r]

    total_leaves = sum(r["leaf_count"] for r in ok_rows)
    total_infected = sum(r["infected"] for r in ok_rows)
    total_possible = sum(r["possible"] for r in ok_rows)
    total_healthy = sum(r["healthy"] for r in ok_rows)
    total_unknown = sum(r["unknown"] for r in ok_rows)
    fallback_used = sum(1 for r in ok_rows if r["fallback_used"])

    image_status_counts = {
        "infected": sum(1 for r in ok_rows if r["image_status"] == "infected"),
        "possible": sum(1 for r in ok_rows if r["image_status"] == "possible"),
        "healthy": sum(1 for r in ok_rows if r["image_status"] == "healthy"),
        "unknown": sum(1 for r in ok_rows if r["image_status"] == "unknown"),
    }

    avg_time = sum(r["total_s"] for r in ok_rows) / len(ok_rows) if ok_rows else 0.0

    batch_json = {
        "summary": {
            "total_images": total_images,
            "ok_images": len(ok_rows),
            "failed_images": len(failed_rows),
            "image_status_counts": image_status_counts,
            "total_leaf_candidates": total_leaves,
            "total_infected_leaf_candidates": total_infected,
            "total_possible_leaf_candidates": total_possible,
            "total_healthy_leaf_candidates": total_healthy,
            "total_unknown_leaf_candidates": total_unknown,
            "fallback_used_images": fallback_used,
            "avg_image_time_s": round(avg_time, 3),
            "wall_time_s": round(wall_time_s, 3),
        },
        "rows": rows,
    }

    (out_dir / "batch_results.json").write_text(
        json.dumps(batch_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    md_lines = [
        "# Full pipeline batch summary",
        "",
        f"- Total images: **{total_images}**",
        f"- OK images: **{len(ok_rows)}**",
        f"- Failed images: **{len(failed_rows)}**",
        f"- Image status infected: **{image_status_counts['infected']}**",
        f"- Image status possible: **{image_status_counts['possible']}**",
        f"- Image status healthy: **{image_status_counts['healthy']}**",
        f"- Image status unknown: **{image_status_counts['unknown']}**",
        f"- Total leaf candidates: **{total_leaves}**",
        f"- Leaf infected: **{total_infected}**",
        f"- Leaf possible: **{total_possible}**",
        f"- Leaf healthy: **{total_healthy}**",
        f"- Leaf unknown: **{total_unknown}**",
        f"- Fallback used images: **{fallback_used}**",
        f"- Avg image time: **{avg_time:.3f}s**",
        f"- Wall time: **{wall_time_s:.3f}s**",
        "",
        "| # | Image | Image status | Leaves | Infected | Possible | Healthy | Unknown | Fallback | Time (s) | Top predictions |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|---:|---|",
    ]

    for idx, row in enumerate(rows, start=1):
        if "error" in row:
            md_lines.append(
                f"| {idx} | `{Path(row['image']).name}` | error | 0 | 0 | 0 | 0 | 0 | — | 0.000 | {row['error']} |"
            )
            continue

        preds = []
        for p in row["top_predictions"]:
            if p["scope"] == "leaf":
                preds.append(f"leaf#{p['leaf']} {p['class']} {p['conf']:.2f} [{p['status']}]")
            else:
                preds.append(f"image {p['class']} {p['conf']:.2f} [{p['status']}]")

        preds_str = ", ".join(preds) or "—"
        fallback = f"{row['fallback_status']}" if row["fallback_used"] else "—"

        md_lines.append(
            f"| {idx} "
            f"| `{Path(row['image']).name}` "
            f"| {row['image_status']} "
            f"| {row['leaf_count']} "
            f"| {row['infected']} "
            f"| {row['possible']} "
            f"| {row['healthy']} "
            f"| {row['unknown']} "
            f"| {fallback} "
            f"| {row['total_s']:.3f} "
            f"| {preds_str} |"
        )

    (out_dir / "batch_summary.md").write_text(
        "\n".join(md_lines),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Проганяє full pipeline на всіх зображеннях з папки.",
    )

    parser.add_argument("--images-dir", type=Path, default=Path("data/test_images"))
    parser.add_argument("--leaf-weights", type=Path, default=Path("runs/leaf_detector/v1/weights/best.pt"))
    parser.add_argument("--pest-weights", type=Path, default=Path("runs/pest_detector/v1/weights/best.pt"))
    parser.add_argument("--out-dir", type=Path, default=Path("pipeline/output_test_images"))

    parser.add_argument("--leaf-conf", type=float, default=0.45)
    parser.add_argument("--pest-conf", type=float, default=0.60)
    parser.add_argument("--possible-conf", type=float, default=0.60)
    parser.add_argument("--infected-conf", type=float, default=0.80)
    parser.add_argument("--healthy-conf", type=float, default=0.70)

    parser.add_argument("--slice-size", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--leaf-match-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="mps")

    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--save-crops", action="store_true")
    parser.add_argument("--no-vis", action="store_true")
    parser.add_argument("--hide-pest-boxes", action="store_true")
    parser.add_argument("--no-fallback-full-image", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--limit", type=int, default=0)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.device = resolve_device(args.device)

    if not args.leaf_weights.exists():
        print(f"Leaf weights не знайдено: {args.leaf_weights}")
        return

    if not args.pest_weights.exists():
        print(f"Pest weights не знайдено: {args.pest_weights}")
        return

    try:
        image_paths = collect_images(args.images_dir, recursive=args.recursive)
    except Exception as exc:
        print(f"Помилка: {exc}")
        return

    if args.limit > 0:
        image_paths = image_paths[:args.limit]

    if not image_paths:
        print(f"У папці не знайдено зображень: {args.images_dir}")
        print(f"Підтримувані розширення: {sorted(IMAGE_EXTS)}")
        return

    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 72}")
    print("  Fruit Pest Detection Pipeline — Batch")
    print(f"{'=' * 72}")
    print(f"  images dir:      {args.images_dir}")
    print(f"  found:           {len(image_paths)}")
    print(f"  out dir:         {args.out_dir}")
    print(f"  leaf conf:       {args.leaf_conf}")
    print(f"  pest conf:       {args.pest_conf}")
    print(f"  possible conf:   {args.possible_conf}")
    print(f"  infected conf:   {args.infected_conf}")
    print(f"  healthy conf:    {args.healthy_conf}")
    print(f"  slice size:      {args.slice_size}")
    print(f"  device:          {args.device}")
    print(f"  fallback image:  {not args.no_fallback_full_image}")
    print(f"{'=' * 72}\n")

    print("Завантаження моделей один раз...")
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

    rows: list[dict[str, Any]] = []
    wall_t0 = time.perf_counter()

    for idx, image_path in enumerate(image_paths, start=1):
        print(f"\n[{idx}/{len(image_paths)}] {image_path.name}")

        stem = safe_name(image_path, args.images_dir)
        image_out_dir = args.out_dir / stem

        if image_out_dir.exists():
            shutil.rmtree(image_out_dir)
        image_out_dir.mkdir(parents=True, exist_ok=True)

        try:
            result = pipeline.predict(image_path)

            json_path = image_out_dir / f"{stem}_result.json"
            json_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            if not args.no_vis:
                vis_path = image_out_dir / f"{stem}_annotated.jpg"
                pipeline.visualize(
                    image_path,
                    result,
                    vis_path,
                    show_pest_boxes=not args.hide_pest_boxes,
                )

            if args.save_crops:
                crops_dir = image_out_dir / f"{stem}_crops"
                pipeline.save_crops(image_path, result, crops_dir, clean_dir=True)

            row = make_summary_row(image_path, result)
            rows.append(row)

            print(
                f"  image_status={row['image_status']} "
                f"leaves={row['leaf_count']} "
                f"infected={row['infected']} "
                f"possible={row['possible']} "
                f"healthy={row['healthy']} "
                f"unknown={row['unknown']} "
                f"fallback={row['fallback_status'] if row['fallback_used'] else '-'} "
                f"time={row['total_s']:.3f}s"
            )

        except Exception as exc:
            print(f"  ERROR: {exc}")
            rows.append({
                "image": str(image_path),
                "error": str(exc),
            })

    wall_time_s = time.perf_counter() - wall_t0
    write_batch_summary(rows, args.out_dir, wall_time_s)

    ok_count = sum(1 for r in rows if "error" not in r)
    failed_count = len(rows) - ok_count

    print(f"\n{'=' * 72}")
    print("  Batch завершено")
    print(f"{'=' * 72}")
    print(f"  Images:        {len(rows)}")
    print(f"  OK:            {ok_count}")
    print(f"  Failed:        {failed_count}")
    print(f"  Wall time:     {wall_time_s:.3f}s")
    print(f"  Summary JSON:  {args.out_dir / 'batch_results.json'}")
    print(f"  Summary MD:    {args.out_dir / 'batch_summary.md'}")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()
