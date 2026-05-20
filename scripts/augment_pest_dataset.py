"""
scripts/augment_pest_dataset.py

Аугментує train-спліт pest_detector датасету до рівного розподілу класів.
Генерує гістограми до/після для звіту.

Запуск:
    python scripts/augment_pest_dataset.py --dry-run
    python scripts/augment_pest_dataset.py
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import albumentations as A
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    print("pip install albumentations")


DATASET_ROOT  = Path("data/processed/pest_detector")
FIGURES_DIR   = Path("report/figures")

# Аугментаційний пайплайн для комах
# Уникаємо агресивного Cutout — може стерти єдину комаху в кадрі
def build_transform() -> "A.Compose":
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
        A.HueSaturationValue(hue_shift_limit=12, sat_shift_limit=25, val_shift_limit=20, p=0.4),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=0.2),
        A.RandomShadow(p=0.15),
        A.CLAHE(clip_limit=2.0, p=0.2),
    ])


def count_per_class(split_dir: Path) -> dict[str, int]:
    """Рахує кількість зображень по класах за префіксом імені файлу."""
    counts: dict[str, int] = {}
    for img in split_dir.glob("*.*"):
        if img.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.webp'}:
            continue
        cls = img.stem.split("__")[0]
        counts[cls] = counts.get(cls, 0) + 1
    return counts


def plot_distribution(
    counts_before: dict[str, int],
    counts_after: dict[str, int],
    save_path: Path,
    title: str = "Розподіл класів",
) -> None:
    classes = sorted(counts_before.keys())
    x = range(len(classes))
    before = [counts_before.get(c, 0) for c in classes]
    after  = [counts_after.get(c, 0)  for c in classes]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    colors_b = ['#e74c3c' if v < max(before) * 0.75 else '#3498db' for v in before]
    colors_a = ['#2ecc71' for _ in after]

    for ax, values, colors, subtitle in [
        (axes[0], before, colors_b, "До аугментації"),
        (axes[1], after,  colors_a, "Після аугментації"),
    ]:
        bars = ax.bar(x, values, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_xticks(list(x))
        ax.set_xticklabels(
            [c.replace('_', '\n') for c in classes],
            rotation=0, ha='center', fontsize=7.5,
        )
        ax.set_ylabel("Кількість зображень")
        ax.set_title(subtitle, fontsize=12)
        ax.set_ylim(0, max(max(before), max(after)) * 1.15)
        ax.axhline(y=max(values), color='gray', linestyle='--', alpha=0.4, linewidth=1)

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.01,
                str(val), ha='center', va='bottom', fontsize=8, fontweight='bold',
            )

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Збережено: {save_path}")


def augment_class(
    images: list[Path],
    needed: int,
    img_out: Path,
    lbl_out: Path,
    transform,
    rng: random.Random,
    dry_run: bool,
) -> int:
    generated = 0
    rng.shuffle(images)

    for i in range(needed):
        src = images[i % len(images)]
        aug_idx = i // len(images)  # скільки разів пройшли по всіх зображеннях

        img_bgr = cv2.imread(str(src))
        if img_bgr is None:
            continue

        augmented = transform(image=img_bgr)
        aug_img = augmented['image']

        stem    = src.stem
        new_stem = f"{stem}_aug{i:05d}"
        dst_img  = img_out / f"{new_stem}.jpg"
        dst_lbl  = lbl_out / f"{new_stem}.txt"

        src_lbl = lbl_out.parent / f"{stem}.txt"

        if not dry_run:
            cv2.imwrite(str(dst_img), aug_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if src_lbl.exists():
                shutil.copy2(src_lbl, dst_lbl)

        generated += 1

    return generated


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset',   type=Path, default=DATASET_ROOT)
    parser.add_argument('--figures',   type=Path, default=FIGURES_DIR)
    parser.add_argument('--strategy',  choices=['max', 'median'], default='max',
                        help='max — до максимального класу; median — до медіани')
    parser.add_argument('--seed',      type=int, default=42)
    parser.add_argument('--dry-run',   action='store_true')
    args = parser.parse_args()

    if not HAS_ALBUMENTATIONS:
        print("Встанови albumentations:  pip install albumentations")
        return

    train_img = args.dataset / 'images' / 'train'
    train_lbl = args.dataset / 'labels' / 'train'

    if not train_img.exists():
        print(f"Не знайдено {train_img}. Спочатку запусти convert_pest_to_yolo.py")
        return

    print(f'\n{"=" * 60}')
    print('  Augment Pest Detector — Train Split')
    print(f'{"=" * 60}')
    print(f'  dataset:  {args.dataset}')
    print(f'  strategy: {args.strategy}')
    print(f'  dry run:  {args.dry_run}')
    print(f'{"=" * 60}\n')

    # Поточний розподіл
    counts_before = count_per_class(train_img)
    classes = sorted(counts_before)

    print(f'{"Клас":<32} {"Зараз":>7} {"Ціль":>7} {"Додати":>7}')
    print('-' * 55)

    if args.strategy == 'max':
        target = max(counts_before.values())
    else:
        target = int(np.median(list(counts_before.values())))

    total_to_generate = 0
    for cls in classes:
        current = counts_before.get(cls, 0)
        add     = max(0, target - current)
        total_to_generate += add
        status = '→ ok' if add == 0 else f'→ +{add}'
        print(f'  {cls:<30} {current:>7} {target:>7} {status:>7}')

    print(f'\n  Цільова кількість/клас: {target}')
    print(f'  Всього генерувати:      {total_to_generate}')

    if total_to_generate == 0:
        print('\nВсі класи вже збалансовані.')
        return

    if args.dry_run:
        print('\n[dry-run] Файли не записані.')
        # Малюємо графік з очікуваним результатом для preview
        counts_after_expected = {cls: target for cls in classes}
        plot_distribution(
            counts_before, counts_after_expected,
            args.figures / 'class_distribution_preview.png',
            title='Pest Detector — Розподіл класів (preview після аугментації)',
        )
        return

    # Аугментація
    transform = build_transform()
    rng = random.Random(args.seed)

    print('\nАугментація...')
    for cls in classes:
        current = counts_before.get(cls, 0)
        needed  = max(0, target - current)
        if needed == 0:
            print(f'  {cls}: вже {current} — пропускаємо')
            continue

        # Знаходимо всі зображення цього класу
        images = [
            p for p in train_img.iterdir()
            if p.stem.split('__')[0] == cls
            and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}
        ]

        if not images:
            print(f'  {cls}: зображення не знайдені — пропускаємо')
            continue

        generated = augment_class(
            images=images,
            needed=needed,
            img_out=train_img,
            lbl_out=train_lbl,
            transform=transform,
            rng=rng,
            dry_run=False,
        )
        print(f'  {cls}: +{generated} зображень')

    # Розподіл після
    counts_after = count_per_class(train_img)

    print(f'\n{"=" * 60}')
    print('  Результат')
    print(f'{"=" * 60}')
    print(f'{"Клас":<32} {"До":>7} {"Після":>7}')
    print('-' * 44)
    for cls in classes:
        b = counts_before.get(cls, 0)
        a = counts_after.get(cls, 0)
        print(f'  {cls:<30} {b:>7} {a:>7}')
    print(f'  {"TOTAL":<30} {sum(counts_before.values()):>7} {sum(counts_after.values()):>7}')

    # Графіки для звіту
    print('\nГенеруємо графіки...')
    plot_distribution(
        counts_before, counts_after,
        args.figures / 'class_distribution_before_after.png',
        title='Pest Detector — Розподіл класів у train',
    )

    # Окремо "до" і "після" як окремі файли (для звіту)
    plot_distribution(
        counts_before, counts_before,
        args.figures / 'class_distribution_before.png',
        title='Pest Detector — Розподіл класів до аугментації',
    )
    plot_distribution(
        counts_after, counts_after,
        args.figures / 'class_distribution_after.png',
        title='Pest Detector — Розподіл класів після аугментації',
    )

    print(f'\n{"=" * 60}')
    print('  Готово')
    print(f'{"=" * 60}\n')


if __name__ == '__main__':
    main()