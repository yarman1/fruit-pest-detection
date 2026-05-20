#!/usr/bin/env python3
"""
scripts/scrape_leaf_inaturalist.py

Завантажує фото плодових дерев з iNaturalist

Вихід:
  data/interim/leaves_inaturalist/
  ├── apple/
  ├── pear/
  ├── plum/
  ├── cherry/
  ├── apricot/
  └── _metadata.csv

Запуск:
    uv run python scripts/scrape_leaf_inaturalist.py
    uv run python scripts/scrape_leaf_inaturalist.py --max-per-taxon 200
    uv run python scripts/scrape_leaf_inaturalist.py --dry-run
    uv run python scripts/scrape_leaf_inaturalist.py --trees apple pear
"""

import sys
import csv
import time
import argparse
import requests
from pathlib import Path

import yaml


INAT_API   = "https://api.inaturalist.org/v1/observations"
USER_AGENT = "FruitPestDetection/1.0 (university-research; non-commercial)"

METADATA_FIELDS = [
    'photo_id', 'tree_key', 'taxon_queried', 'observation_id',
    'observed_on', 'place_guess', 'latitude', 'longitude',
    'license', 'photo_url', 'file_path',
]

MAX_API_RETRIES = 3


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def load_config(path: str = 'class_mapping.yaml') -> dict:
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def parse_latin_names(name_lat: str) -> list[str]:
    """
    Витягує список наукових назв
    """
    parts = [p.strip() for p in name_lat.split('/')]
    names = []
    for part in parts:
        words = part.split()
        if len(words) >= 2:
            names.append(f"{words[0]} {words[1]}")
    return names


def photo_url_resize(url: str, size: str = 'large') -> str:
    for s in ('square', 'small', 'medium', 'large', 'original'):
        if f'/{s}.' in url:
            return url.replace(f'/{s}.', f'/{size}.')
    return url


def ext_from_url(url: str) -> str:
    name = url.split('/')[-1].split('?')[0]
    return ('.' + name.rsplit('.', 1)[-1].lower()) if '.' in name else '.jpg'


def load_downloaded_ids(log_path: Path) -> set[str]:
    if not log_path.exists():
        return set()
    ids = set()
    with open(log_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row.get('photo_id'):
                ids.add(row['photo_id'])
    return ids


def open_log(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not log_path.exists() or log_path.stat().st_size == 0
    f = open(log_path, 'a', newline='', encoding='utf-8')
    writer = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
    if is_new:
        writer.writeheader()
    return f, writer


# ──────────────────────────────────────────────
# iNaturalist API
# ──────────────────────────────────────────────

def fetch_page(session: requests.Session, taxon_name: str,
               place_id: int | None, quality_grade: str,
               date_max: str | None, page: int,
               per_page: int = 100) -> dict:

    params = [
        ('taxon_name',    taxon_name),
        ('quality_grade', quality_grade),
        ('has[]',         'photos'),
        ('per_page',      per_page),
        ('page',          page),
        ('order_by',      'id'),
        ('order',         'desc'),
    ]

    if place_id is not None:
        params.append(('place_id', place_id))
    if date_max:
        params.append(('d2', date_max))

    resp = session.get(INAT_API, params=params, timeout=30)

    if resp.status_code >= 400:
        try:
            body = resp.json()
        except Exception:
            body = resp.text
        raise RuntimeError(f"API {resp.status_code}: {body}")

    return resp.json()


def fetch_observations(session, taxon_name: str,
                       place_id: int | None, quality_grade: str,
                       date_max: str | None, max_results: int) -> tuple[list, int]:
    all_obs   = []
    page      = 1
    total     = 0
    retries   = 0

    while len(all_obs) < max_results:
        try:
            data    = fetch_page(session, taxon_name, place_id,
                                 quality_grade, date_max, page)
            retries = 0
        except RuntimeError as e:
            retries += 1
            print(f"\n      [!] {e}")
            if retries > MAX_API_RETRIES:
                break
            time.sleep(10)
            continue
        except requests.RequestException as e:
            retries += 1
            print(f"\n      [!] Мережева помилка: {e}")
            if retries > MAX_API_RETRIES:
                break
            time.sleep(10)
            continue

        results = data.get('results', [])
        total   = data.get('total_results', 0)

        if not results:
            break

        all_obs.extend(results)

        if len(all_obs) >= total:
            break

        page += 1
        time.sleep(1.0)

    return all_obs[:max_results], total


def download_photo(url: str, dst: Path, session: requests.Session) -> bool:
    if dst.exists():
        return True
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        resp = session.get(url, timeout=30, stream=True)
        resp.raise_for_status()
        with open(dst, 'wb') as f:
            for chunk in resp.iter_content(8192):
                if chunk:
                    f.write(chunk)
        return True
    except Exception as e:
        print(f"      [!] Не вдалося завантажити: {e}")
        return False


# ──────────────────────────────────────────────
# Processing
# ──────────────────────────────────────────────

def process_taxon(
    tree_key: str,
    taxon_name: str,
    place_id: int | None,
    quality_grade: str,
    date_max: str | None,
    max_results: int,
    photo_size: str,
    allowed_licenses: list[str],
    out_dir: Path,
    log_path: Path,
    dry_run: bool,
    session: requests.Session,
) -> int:
    """Скрапить один таксон і повертає кількість завантажених фото."""

    already = load_downloaded_ids(log_path)

    print(f"      {taxon_name}", end=' ', flush=True)

    observations, total = fetch_observations(
        session, taxon_name, place_id,
        quality_grade, date_max, max_results,
    )

    print(f"→ {len(observations)} / {total} на сервері")

    if dry_run:
        return len(observations)

    log_f, writer = open_log(log_path)
    downloaded = 0

    try:
        for obs in observations:
            photos = obs.get('photos', [])
            if not photos:
                continue

            photo    = photos[0]
            photo_id = str(photo.get('id', ''))
            raw_url  = photo.get('url', '')

            if not raw_url or photo_id in already:
                if photo_id in already:
                    downloaded += 1
                continue

            # Пост-фільтр ліцензії
            lic = (photo.get('license_code') or '').lower()

            url = photo_url_resize(raw_url, photo_size)
            ext = ext_from_url(url)
            dst = out_dir / tree_key / f"inat_{tree_key}_{photo_id}{ext}"

            if download_photo(url, dst, session):
                geopos = obs.get('geojson') or {}
                coords = geopos.get('coordinates', [None, None])

                writer.writerow({
                    'photo_id':      photo_id,
                    'tree_key':      tree_key,
                    'taxon_queried': taxon_name,
                    'observation_id': obs.get('id', ''),
                    'observed_on':   obs.get('observed_on', ''),
                    'place_guess':   obs.get('place_guess', ''),
                    'latitude':      coords[1] if len(coords) > 1 else '',
                    'longitude':     coords[0] if len(coords) > 0 else '',
                    'license':       lic,
                    'photo_url':     url,
                    'file_path':     str(dst),
                })
                already.add(photo_id)
                downloaded += 1

            time.sleep(0.3)

    finally:
        log_f.close()

    return downloaded


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-per-taxon', type=int, default=300,
                        help='Максимум фото на таксон (default: 300)')
    parser.add_argument('--photo-size',
                        choices=['medium', 'large', 'original'],
                        default='large')
    parser.add_argument('--output', default='data/interim/leaves_inaturalist')
    parser.add_argument('--config', default='class_mapping.yaml')
    parser.add_argument('--trees', nargs='+', default=None,
                        help='Конкретні дерева за key (default: всі)')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    config    = load_config(args.config)
    trees     = config.get('trees', {})
    inat_cfg  = config['data_sources']['inaturalist']

    place_id      = inat_cfg.get('place_id_train')
    quality_grade = inat_cfg.get('quality_grade', 'research')
    date_max      = inat_cfg.get('train_date_max')
    licenses      = inat_cfg.get('photo_license', [])

    if not trees:
        print("Помилка: секція 'trees' не знайдена у class_mapping.yaml")
        sys.exit(1)

    # Фільтр по --trees
    if args.trees:
        trees = {k: v for k, v in trees.items() if k in args.trees}

    out_dir  = Path(args.output)
    log_path = out_dir / '_metadata.csv'

    session = requests.Session()
    session.headers.update({'User-Agent': USER_AGENT})

    print(f"\n{'='*62}")
    print(f"  iNaturalist Leaf Scraper")
    print(f"{'='*62}")
    print(f"  Place ID:      {place_id} (Європа)")
    print(f"  Дата макс.:    {date_max}")
    print(f"  Макс/таксон:   {args.max_per_taxon}")
    print(f"  Розмір фото:   {args.photo_size}")
    print(f"  Дерев:         {len(trees)}")
    print(f"  Dry run:       {args.dry_run}")
    print(f"{'='*62}\n")

    total_downloaded = 0
    stats: dict[str, int] = {}

    for tree_key, tree_info in trees.items():
        name_uk  = tree_info.get('name_uk', tree_key)
        name_lat = tree_info.get('name_lat', '')
        taxa     = parse_latin_names(name_lat)

        if not taxa:
            print(f"  ⚠ {tree_key}: не вдалося розпарсити Latin name '{name_lat}'")
            continue

        print(f"  ▶ {name_uk} ({tree_key})")

        tree_total = 0
        per_taxon  = args.max_per_taxon // len(taxa) if len(taxa) > 1 else args.max_per_taxon

        for taxon in taxa:
            count = process_taxon(
                tree_key=tree_key,
                taxon_name=taxon,
                place_id=place_id,
                quality_grade=quality_grade,
                date_max=date_max,
                max_results=per_taxon,
                photo_size=args.photo_size,
                allowed_licenses=licenses,
                out_dir=out_dir,
                log_path=log_path,
                dry_run=args.dry_run,
                session=session,
            )
            tree_total += count

        stats[tree_key] = tree_total
        total_downloaded += tree_total
        print(f"      → {tree_total} зображень\n")

    # Підсумок
    print(f"{'='*62}")
    print(f"  {'Дерево':<20} {'Key':<15} {'Зображень':>10}")
    print(f"  {'─'*20} {'─'*15} {'─'*10}")
    for tree_key, count in stats.items():
        name_uk = trees[tree_key].get('name_uk', tree_key)
        print(f"  {name_uk:<20} {tree_key:<15} {count:>10}")
    print(f"  {'─'*20} {'─'*15} {'─'*10}")
    print(f"  {'ВСЬОГО':<20} {'':<15} {total_downloaded:>10}")
    print(f"{'='*62}")

    if not args.dry_run:
        print(f"\n  Збережено: {out_dir}")
        print(f"  Метадані:  {log_path}\n")


if __name__ == '__main__':
    main()