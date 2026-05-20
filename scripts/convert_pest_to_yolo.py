"""
scripts/convert_pest_to_yolo.py

Конвертує inaturalist_train/ і inaturalist_test/ у YOLO-формат
для Model B.

Вхід:
    data/interim/inaturalist_train/<class>/
    data/interim/inaturalist_test/<class>/

Вихід:
    data/processed/pest_detector/
    ├── images/train/
    ├── images/val/
    ├── images/test/
    ├── labels/train/
    ├── labels/val/
    ├── labels/test/
    ├── data.yaml
    └── dataset_stats.json

Запуск:
    python scripts/convert_pest_to_yolo.py --dry-run
    python scripts/convert_pest_to_yolo.py --clear-output
    python scripts/convert_pest_to_yolo.py --val-split 0.15 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

# ---------------------------------------------------------------------------
# КЛАС-МАППІНГ: порядок визначає YOLO class_id
# Має відповідати class_mapping.yaml
# ---------------------------------------------------------------------------
CLASS_NAMES = [
    "codling_moth",               # 0  Cydia pomonella
    "gypsy_moth",                 # 1  Lymantria dispar
    "fall_webworm",               # 2  Hyphantria cunea
    "apple_ermine_moth",          # 3  Yponomeuta malinellus
    "plum_fruit_moth",            # 4  Cydia funebrana
    "cherry_fruit_fly",           # 5  Rhagoletis cerasi
    "apple_blossom_weevil",       # 6  Anthonomus pomorum
    "brown_marmorated_stink_bug", # 7  Halyomorpha halys
    "lackey_moth",                # 8  Malacosoma neustria
    "winter_moth",                # 9  Operophtera brumata
    "healthy",                    # 10 здоровий листок
]


@dataclass
class Sample:
    src_path: Path
    class_id: int
    class_name: str
    dst_stem: str   # унікальне ім'я файлу (з префіксом класу)


# ---------------------------------------------------------------------------
# Collect
# ---------------------------------------------------------------------------

def collect_samples(src_root: Path, class_names: list[str]) -> list[Sample]:
    samples: list[Sample] = []
    for class_id, cls in enumerate(class_names):
        cls_dir = src_root / cls
        if not cls_dir.exists():
            continue
        for img in sorted(cls_dir.iterdir()):
            if img.suffix.lower() not in IMAGE_EXTS:
                continue
            # Префікс класу гарантує унікальність імені в одній папці
            dst_stem = f"{cls}__{img.stem}"
            samples.append(Sample(
                src_path=img,
                class_id=class_id,
                class_name=cls,
                dst_stem=dst_stem,
            ))
    return samples


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------

def stratified_split(
    samples: list[Sample],
    val_ratio: float,
    seed: int,
) -> tuple[list[Sample], list[Sample]]:
    rng = random.Random(seed)
    by_class: dict[str, list[Sample]] = {}
    for s in samples:
        by_class.setdefault(s.class_name, []).append(s)

    train_list: list[Sample] = []
    val_list:   list[Sample] = []

    for cls in sorted(by_class):
        items = by_class[cls][:]
        rng.shuffle(items)
        n_val = max(1, round(len(items) * val_ratio))
        val_list.extend(items[:n_val])
        train_list.extend(items[n_val:])

    return train_list, val_list


# ---------------------------------------------------------------------------
# Copy split to dst
# ---------------------------------------------------------------------------

def copy_split(samples: list[Sample], dst: Path, split: str) -> None:
    img_out = dst / 'images' / split
    lbl_out = dst / 'labels' / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    for s in samples:
        ext = s.src_path.suffix.lower()
        shutil.copy2(s.src_path, img_out / f'{s.dst_stem}{ext}')
        (lbl_out / f'{s.dst_stem}.txt').write_text(
            f'{s.class_id} 0.5 0.5 1.0 1.0\n', encoding='utf-8'
        )


# ---------------------------------------------------------------------------
# Print stats table
# ---------------------------------------------------------------------------

def print_stats(
    train: list[Sample],
    val: list[Sample],
    test: list[Sample],
    class_names: list[str],
) -> dict:
    all_splits = {'train': train, 'val': val, 'test': test}
    stats_per_class: dict[str, dict] = {}

    print(f'\n{"Клас":<30} {"train":>6} {"val":>6} {"test":>6} {"total":>6}')
    print('-' * 54)

    for cls in class_names:
        row: dict[str, int] = {}
        for split_name, split_list in all_splits.items():
            row[split_name] = sum(1 for s in split_list if s.class_name == cls)
        row['total'] = sum(row.values())
        stats_per_class[cls] = row
        status = '⚠' if row['total'] < 150 else '✓'
        print(f'  {cls:<28} {row["train"]:>6} {row["val"]:>6} {row["test"]:>6} {row["total"]:>6}  {status}')

    totals = {
        split_name: sum(1 for s in split_list for _ in [s])
        for split_name, split_list in all_splits.items()
    }
    totals['total'] = sum(totals.values())
    print('-' * 54)
    print(f'  {"TOTAL":<28} {totals["train"]:>6} {totals["val"]:>6} {totals["test"]:>6} {totals["total"]:>6}')

    return stats_per_class


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--src-train', type=Path,
                        default=Path('data/interim/inaturalist_train'))
    parser.add_argument('--src-test',  type=Path,
                        default=Path('data/interim/inaturalist_test'))
    parser.add_argument('--dst',       type=Path,
                        default=Path('data/processed/pest_detector'))
    parser.add_argument('--val-split', type=float, default=0.15)
    parser.add_argument('--seed',      type=int,   default=42)
    parser.add_argument('--clear-output', action='store_true')
    parser.add_argument('--dry-run',   action='store_true')
    args = parser.parse_args()

    print(f'\n{"=" * 60}')
    print('  Convert Pest Dataset → YOLO (Model B)')
    print(f'{"=" * 60}')
    print(f'  src train: {args.src_train}')
    print(f'  src test:  {args.src_test}')
    print(f'  dst:       {args.dst}')
    print(f'  val split: {args.val_split:.0%}  (з train)')
    print(f'  seed:      {args.seed}')
    print(f'  bbox:      whole-image  (0 0.5 0.5 1.0 1.0)')
    print(f'  dry run:   {args.dry_run}')
    print(f'{"=" * 60}')

    # Перевіряємо наявність папок класів
    missing = []
    for cls in CLASS_NAMES:
        if cls == 'healthy':
            continue
        if not (args.src_train / cls).exists():
            missing.append(f'  TRAIN/{cls}')
    if missing:
        print('\nВідсутні папки класів:')
        for m in missing:
            print(m)
        print('Перевір шляхи або CLASS_NAMES у скрипті.')

    # Collect
    train_raw = collect_samples(args.src_train, CLASS_NAMES)
    test      = collect_samples(args.src_test,  CLASS_NAMES)

    if not train_raw:
        print(f'\nНе знайдено зображень у {args.src_train}')
        return

    # Val split зі train
    train, val = stratified_split(train_raw, val_ratio=args.val_split, seed=args.seed)

    # Stats
    stats_per_class = print_stats(train, val, test, CLASS_NAMES)

    print(f'\nСтратегія bbox: whole-image (class cx cy w h)')
    print(f'  Всі зображення → "0 0.5 0.5 1.0 1.0" (де 0 = class_id)')

    if args.dry_run:
        print('\n[dry-run] Файли не записані.')
        return

    # Clear output
    if args.dst.exists() and any(args.dst.iterdir()):
        if not args.clear_output:
            print(f'\nДиректорія {args.dst} вже існує. Додай --clear-output.')
            return
        shutil.rmtree(args.dst)

    # Copy
    copy_split(train, args.dst, 'train')
    copy_split(val,   args.dst, 'val')
    copy_split(test,  args.dst, 'test')

    # data.yaml
    data_yaml = (
        f"path: {args.dst.resolve()}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"test:  images/test\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n"
    )
    (args.dst / 'data.yaml').write_text(data_yaml, encoding='utf-8')

    # dataset_stats.json
    stats = {
        'class_names': CLASS_NAMES,
        'val_split': args.val_split,
        'seed': args.seed,
        'bbox_strategy': 'whole-image: 0 0.5 0.5 1.0 1.0',
        'splits': {
            'train': len(train),
            'val':   len(val),
            'test':  len(test),
            'total': len(train) + len(val) + len(test),
        },
        'per_class': stats_per_class,
    }
    (args.dst / 'dataset_stats.json').write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8'
    )

    print(f'\n{"=" * 60}')
    print('  Готово')
    print(f'{"=" * 60}')
    print(f'  train: {len(train)} images')
    print(f'  val:   {len(val)} images')
    print(f'  test:  {len(test)} images')
    print(f'  data.yaml: {args.dst / "data.yaml"}')
    print(f'{"=" * 60}\n')


if __name__ == '__main__':
    main()