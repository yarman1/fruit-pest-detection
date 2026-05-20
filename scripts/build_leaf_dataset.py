#!/usr/bin/env python3
"""
scripts/build_leaf_dataset.py

Збирає фінальний YOLO-датасет для Model A (детектор листя)
із вже почищеного data/interim/leaves_sam/.

Вхід:
    data/interim/leaves_sam/
    ├── images/<tree>/
    └── labels/<tree>/

Вихід:
    data/processed/leaf_detector/
    ├── images/train/
    ├── images/val/
    ├── labels/train/
    ├── labels/val/
    └── data.yaml

Запуск:
    python scripts/build_leaf_dataset.py
    python scripts/build_leaf_dataset.py --val-split 0.15 --seed 42
    python scripts/build_leaf_dataset.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


@dataclass
class Sample:
    image_path: Path
    label_path: Path
    tree: str


# ---------------------------------------------------------------------------
# Collect
# ---------------------------------------------------------------------------

def collect_samples(src: Path) -> list[Sample]:
    images_root = src / 'images'
    labels_root = src / 'labels'
    samples: list[Sample] = []

    if not images_root.exists():
        return samples

    for tree_dir in sorted(images_root.iterdir()):
        if not tree_dir.is_dir():
            continue
        tree = tree_dir.name
        for img in sorted(tree_dir.iterdir()):
            if img.suffix.lower() not in IMAGE_EXTS:
                continue
            lbl = labels_root / tree / f'{img.stem}.txt'
            if not lbl.exists():
                continue
            # Пропускаємо порожні лейбли
            if lbl.stat().st_size == 0:
                continue
            samples.append(Sample(image_path=img, label_path=lbl, tree=tree))

    return samples


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------

def stratified_split(
    samples: list[Sample],
    val_ratio: float,
    seed: int,
) -> tuple[list[Sample], list[Sample]]:
    """
    Stratified split по дереву — щоб кожне дерево було
    рівномірно представлено в train і val.
    """
    rng = random.Random(seed)

    by_tree: dict[str, list[Sample]] = {}
    for s in samples:
        by_tree.setdefault(s.tree, []).append(s)

    train_samples: list[Sample] = []
    val_samples:   list[Sample] = []

    for tree, items in sorted(by_tree.items()):
        shuffled = items[:]
        rng.shuffle(shuffled)
        n_val = max(1, round(len(shuffled) * val_ratio))
        val_samples.extend(shuffled[:n_val])
        train_samples.extend(shuffled[n_val:])

    return train_samples, val_samples


# ---------------------------------------------------------------------------
# Copy
# ---------------------------------------------------------------------------

def copy_samples(samples: list[Sample], dst_img: Path, dst_lbl: Path) -> int:
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)
    copied = 0
    for s in samples:
        shutil.copy2(s.image_path, dst_img / s.image_path.name)
        shutil.copy2(s.label_path, dst_lbl / s.label_path.name)
        copied += 1
    return copied


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def count_boxes(samples: list[Sample]) -> int:
    total = 0
    for s in samples:
        text = s.label_path.read_text(encoding='utf-8').strip()
        total += len([l for l in text.splitlines() if l.strip()])
    return total


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--src',  type=Path, default=Path('data/interim/leaves_sam'))
    parser.add_argument('--dst',  type=Path, default=Path('data/processed/leaf_detector'))
    parser.add_argument('--val-split', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--clear-output', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    print(f'\n{"=" * 60}')
    print('  Build Leaf Detector Dataset (Model A)')
    print(f'{"=" * 60}')
    print(f'  src:       {args.src}')
    print(f'  dst:       {args.dst}')
    print(f'  val split: {args.val_split:.0%}')
    print(f'  seed:      {args.seed}')
    print(f'  dry run:   {args.dry_run}')
    print(f'{"=" * 60}\n')

    # Collect
    samples = collect_samples(args.src)
    if not samples:
        print(f'Не знайдено семплів у {args.src}')
        return

    # Per-tree stats
    by_tree: dict[str, int] = {}
    for s in samples:
        by_tree[s.tree] = by_tree.get(s.tree, 0) + 1

    print('Знайдено семплів:')
    for tree, cnt in sorted(by_tree.items()):
        print(f'  {tree:<20} {cnt:>5}')
    print(f'  {"TOTAL":<20} {len(samples):>5}\n')

    # Split
    train, val = stratified_split(samples, val_ratio=args.val_split, seed=args.seed)

    train_boxes = count_boxes(train)
    val_boxes   = count_boxes(val)

    print(f'Split (stratified per tree):')
    print(f'  train: {len(train):>5} images  {train_boxes:>6} boxes')
    print(f'  val:   {len(val):>5} images  {val_boxes:>6} boxes')
    print(f'  total: {len(samples):>5} images  {train_boxes+val_boxes:>6} boxes\n')

    if args.dry_run:
        print('[dry-run] Файли не записані.')
        return

    # Clear
    if args.dst.exists() and any(args.dst.iterdir()):
        if not args.clear_output:
            print(f'Директорія {args.dst} вже існує. Додай --clear-output щоб перезаписати.')
            return
        shutil.rmtree(args.dst)

    # Copy
    copy_samples(train, args.dst / 'images' / 'train', args.dst / 'labels' / 'train')
    copy_samples(val,   args.dst / 'images' / 'val',   args.dst / 'labels' / 'val')

    # data.yaml
    data_yaml = f"""\
path: {args.dst.resolve()}
train: images/train
val:   images/val
nc: 1
names: ['leaf']
"""
    (args.dst / 'data.yaml').write_text(data_yaml, encoding='utf-8')

    # stats.json
    stats = {
        'src': str(args.src),
        'dst': str(args.dst),
        'val_split': args.val_split,
        'seed': args.seed,
        'total_images': len(samples),
        'train_images': len(train),
        'val_images': len(val),
        'train_boxes': train_boxes,
        'val_boxes': val_boxes,
        'per_tree_total': dict(sorted(by_tree.items())),
        'per_tree_train': {
            t: sum(1 for s in train if s.tree == t)
            for t in sorted(by_tree)
        },
        'per_tree_val': {
            t: sum(1 for s in val if s.tree == t)
            for t in sorted(by_tree)
        },
    }
    (args.dst / 'dataset_stats.json').write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'{"=" * 60}')
    print('  Готово')
    print(f'{"=" * 60}')
    print(f'  train: {len(train)} images  ({len(train)/len(samples):.0%})')
    print(f'  val:   {len(val)} images  ({len(val)/len(samples):.0%})')
    print(f'  data.yaml:  {args.dst / "data.yaml"}')
    print(f'{"=" * 60}\n')


if __name__ == '__main__':
    main()