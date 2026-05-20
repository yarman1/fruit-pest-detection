import sys
import csv
import time
import argparse
import requests
from pathlib import Path

import yaml


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

INAT_API = "https://api.inaturalist.org/v1/observations"
USER_AGENT = "FruitPestDetection/1.0 (university-research; non-commercial)"

DEFAULT_PER_PAGE = 100
MAX_API_RETRIES = 3

METADATA_FIELDS = [
    'photo_id', 'class_key', 'class_id', 'taxon_name',
    'observation_id', 'observed_on', 'place_guess',
    'latitude', 'longitude', 'license', 'photo_url', 'file_path', 'split',
]


# ──────────────────────────────────────────────
# Exceptions
# ──────────────────────────────────────────────

class InatApiError(Exception):
    """Помилка відповіді iNaturalist API."""

    def __init__(self, status_code: int, message: str, url: str):
        super().__init__(f"{status_code}: {message}")
        self.status_code = status_code
        self.message = message
        self.url = url


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def load_config(path: str = 'class_mapping.yaml') -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def photo_url_resize(url: str, size: str) -> str:
    """
    Конвертує URL iNaturalist фото до потрібного розміру.
    Розміри: square(75) / small(240) / medium(500) / large(1024) / original
    """
    for s in ('square', 'small', 'medium', 'large', 'original'):
        if f'/{s}.' in url or url.endswith(f'/{s}'):
            return url.replace(f'/{s}.', f'/{size}.').replace(
                url.split('/')[-1],
                url.split('/')[-1].replace(s, size)
            )

    # Якщо не знайшли розмір у URL — повертаємо як є
    return url


def load_downloaded_ids(log_path: Path) -> set[str]:
    """Читає вже завантажені photo_id з metadata CSV."""
    if not log_path.exists():
        return set()

    ids = set()
    with open(log_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            photo_id = row.get('photo_id', '')
            if photo_id:
                ids.add(photo_id)

    return ids


def open_log(log_path: Path) -> tuple:
    """Відкриває CSV для дозапису. Повертає (file, writer)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    is_new = not log_path.exists() or log_path.stat().st_size == 0

    f = open(log_path, 'a', newline='', encoding='utf-8')
    writer = csv.DictWriter(f, fieldnames=METADATA_FIELDS)

    if is_new:
        writer.writeheader()

    return f, writer


def normalize_license_list(licenses: list[str]) -> set[str]:
    """Нормалізує список дозволених ліцензій."""
    return {license_code.lower().strip() for license_code in licenses}


# ──────────────────────────────────────────────
# iNaturalist API
# ──────────────────────────────────────────────

def fetch_page(
    session: requests.Session,
    taxon_name: str,
    place_id: int | None,
    quality_grade: str,
    date_min: str | None,
    date_max: str | None,
    page: int,
    per_page: int = DEFAULT_PER_PAGE,
) -> dict:
    """
    Завантажує одну сторінку observations з iNaturalist API.
    """

    params = [
        ('taxon_name', taxon_name),
        ('quality_grade', quality_grade),
        ('photos', 'true'),
        ('per_page', per_page),
        ('page', page),
        ('order_by', 'id'),
        ('order', 'desc'),
    ]

    if place_id is not None:
        params.append(('place_id', place_id))

    if date_min:
        params.append(('d1', date_min))

    if date_max:
        params.append(('d2', date_max))

    resp = session.get(INAT_API, params=params, timeout=30)

    if resp.status_code >= 400:
        try:
            error_body = resp.json()
        except Exception:
            error_body = resp.text

        raise InatApiError(
            status_code=resp.status_code,
            message=str(error_body),
            url=resp.url,
        )

    return resp.json()


def fetch_observations(
    session: requests.Session,
    taxon_name: str,
    place_id: int | None,
    quality_grade: str,
    date_min: str | None,
    date_max: str | None,
    max_results: int,
) -> tuple[list, int]:
    """
    Завантажує всі спостереження з пагінацією.
    Повертає (список спостережень, загальна кількість на сервері).
    """

    all_obs = []
    page = 1
    total_on_server = 0
    retry_count = 0

    while len(all_obs) < max_results:
        try:
            data = fetch_page(
                session=session,
                taxon_name=taxon_name,
                place_id=place_id,
                quality_grade=quality_grade,
                date_min=date_min,
                date_max=date_max,
                page=page,
            )

            # Якщо сторінка успішна — скидаємо лічильник retry
            retry_count = 0

        except InatApiError as e:
            print(f"\n    [!] API помилка (сторінка {page}): {e.status_code}")
            print(f"    URL: {e.url}")
            print(f"    Відповідь API: {e.message}")

            if e.status_code == 422:
                print("    [!] 422 означає, що API відхилив параметри запиту. Пропускаємо цей клас.")
                break

            if 400 <= e.status_code < 500 and e.status_code != 429:
                print("    [!] Клієнтська помилка запиту. Пропускаємо цей клас.")
                break

            retry_count += 1
            if retry_count > MAX_API_RETRIES:
                print(f"    [!] Перевищено кількість повторів ({MAX_API_RETRIES}). Пропускаємо цей клас.")
                break

            print("    [!] Тимчасова помилка — чекаємо 10с і пробуємо ще раз")
            time.sleep(10)
            continue

        except requests.RequestException as e:
            retry_count += 1

            print(f"\n    [!] Мережева помилка (сторінка {page}): {e}")

            if retry_count > MAX_API_RETRIES:
                print(f"    [!] Перевищено кількість повторів ({MAX_API_RETRIES}). Пропускаємо цей клас.")
                break

            print("    [!] Чекаємо 10с і пробуємо ще раз")
            time.sleep(10)
            continue

        results = data.get('results', [])
        total_on_server = data.get('total_results', 0)

        if not results:
            break

        all_obs.extend(results)

        if len(all_obs) >= total_on_server:
            break

        page += 1
        time.sleep(1.0)  # Rate limit: 1 запит/сек

    return all_obs[:max_results], total_on_server


# ──────────────────────────────────────────────
# Download
# ──────────────────────────────────────────────

def download_photo(url: str, dst: Path, session: requests.Session) -> bool:
    """Завантажує фото. True = успішно."""
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
        print(f"      [!] Не вдалося завантажити фото: {url}")
        print(f"          Причина: {e}")
        return False


def ext_from_url(url: str) -> str:
    """Витягує розширення файлу з URL."""
    name = url.split('/')[-1].split('?')[0]

    if '.' in name:
        return '.' + name.rsplit('.', 1)[-1].lower()

    return '.jpg'


# ──────────────────────────────────────────────
# Main processing
# ──────────────────────────────────────────────

def process_class(
    pest: dict,
    split_name: str,
    date_min: str | None,
    date_max: str | None,
    place_id: int | None,
    inat_cfg: dict,
    args,
    session: requests.Session,
) -> int:

    key = pest['key']
    class_id = pest['id']
    taxon = pest['inaturalist_taxon']

    base_dir = Path(f'data/interim/inaturalist_{split_name}')
    log_path = base_dir / '_metadata.csv'

    already = load_downloaded_ids(log_path)

    # Отримуємо список спостережень
    print(f"    Запит: {taxon} | split={split_name}", end=' ', flush=True)

    observations, total_server = fetch_observations(
        session=session,
        taxon_name=taxon,
        place_id=place_id,
        quality_grade=inat_cfg['quality_grade'],
        date_min=date_min,
        date_max=date_max,
        max_results=args.max_per_class,
    )

    print(f"→ {len(observations)} / {total_server} на сервері")

    # Fallback на global якщо test-спліт дав менше порогу
    fallback_threshold = inat_cfg.get('test_fallback_threshold', 50)
    if (split_name == 'test'
            and place_id is not None
            and len(observations) < fallback_threshold):

        print(f"      ↳ {len(observations)} < {fallback_threshold} — "
              f"fallback: глобальний пошук", end=' ', flush=True)

        observations, total_server = fetch_observations(
            session=session,
            taxon_name=taxon,
            place_id=None,
            quality_grade=inat_cfg['quality_grade'],
            date_min=date_min,
            date_max=date_max,
            max_results=args.max_per_class,
        )
        print(f"→ {len(observations)} / {total_server} на сервері")

    if args.dry_run:
        return len(observations)

    class_dir = base_dir / key
    downloaded = 0

    log_f, writer = open_log(log_path)

    try:
        for obs in observations:
            photos = obs.get('photos', [])

            if not photos:
                continue

            # Беремо перше фото спостереження
            photo = photos[0]

            photo_id = str(photo.get('id', ''))
            raw_url = photo.get('url', '')

            if not raw_url:
                continue

            if photo_id in already:
                downloaded += 1
                continue

            url = photo_url_resize(raw_url, args.photo_size)
            ext = ext_from_url(url)
            dst = class_dir / f"inat_{key}_{photo_id}{ext}"

            success = download_photo(url, dst, session)

            if success:
                geopos = obs.get('geojson') or {}
                coords = geopos.get('coordinates', [None, None])

                writer.writerow({
                    'photo_id': photo_id,
                    'class_key': key,
                    'class_id': class_id,
                    'taxon_name': taxon,
                    'observation_id': obs.get('id', ''),
                    'observed_on': obs.get('observed_on', ''),
                    'place_guess': obs.get('place_guess', ''),
                    'latitude': coords[1] if len(coords) > 1 else '',
                    'longitude': coords[0] if len(coords) > 0 else '',
                    'license': photo.get('license_code', ''),
                    'photo_url': url,
                    'file_path': str(dst),
                    'split': split_name,
                })

                log_f.flush()

                already.add(photo_id)
                downloaded += 1

            time.sleep(0.3)  # Не навантажуємо сервер

    finally:
        log_f.close()

    return downloaded


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--max-per-class', type=int, default=500)

    parser.add_argument(
        '--photo-size',
        choices=['medium', 'large', 'original'],
        default='large'
    )

    parser.add_argument(
        '--split',
        choices=['train', 'test', 'both'],
        default='train'
    )

    parser.add_argument(
        '--classes',
        nargs='+',
        default=None,
        help='Конкретні класи за key (default: всі)'
    )

    parser.add_argument(
        '--config',
        default='class_mapping.yaml'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Тільки показати кількість доступних спостережень'
    )

    parser.add_argument(
        '--no-place-filter',
        action='store_true',
        help='Не обмежувати пошук place_id, тобто шукати глобально'
    )

    args = parser.parse_args()

    config = load_config(args.config)
    inat_cfg = config['data_sources']['inaturalist']
    thresholds = config['thresholds']

    # Pest-класи без healthy
    pests = [
        pc for pc in config['pest_classes']
        if pc['key'] != 'healthy' and pc.get('inaturalist_taxon')
    ]

    if args.classes:
        requested_classes = set(args.classes)
        pests = [pc for pc in pests if pc['key'] in requested_classes]

    if not pests:
        print("Жодного класу для завантаження. Перевір --classes.")
        sys.exit(1)

    # Сплітові параметри.
    # За замовчуванням train/test беруть place_id з class_mapping.yaml.
    # Якщо передано --no-place-filter, place_id стає None для всіх сплітів,
    # тобто пошук виконується глобально.
    train_place_id = None if args.no_place_filter else inat_cfg.get('place_id_train')
    test_place_id = None if args.no_place_filter else inat_cfg.get('place_id_test')

    splits: list[tuple[str, str | None, str | None, int | None]] = []

    if args.split in ('train', 'both'):
        splits.append((
            'train',
            None,
            inat_cfg['train_date_max'],
            train_place_id,
        ))

    if args.split in ('test', 'both'):
        splits.append((
            'test',
            inat_cfg['test_date_min'],
            None,
            test_place_id,
        ))

    session = requests.Session()
    session.headers.update({'User-Agent': USER_AGENT})

    place_mode = 'global search (--no-place-filter)' if args.no_place_filter else 'configured place_id'

    print(f"\n{'=' * 65}")
    print("  iNaturalist Scraper")
    print(f"{'=' * 65}")
    print(f"  Place mode:  {place_mode}")
    print(f"  Train place: {train_place_id}")
    print(f"  Test place:  {test_place_id}")
    print(f"  Розмір:      {args.photo_size}")
    print(f"  Макс/клас:   {args.max_per_class}")
    print(f"  Класів:      {len(pests)}")
    print(f"  Сплітів:     {[s[0] for s in splits]}")
    print(f"  Dry run:     {args.dry_run}")
    print(f"{'=' * 65}\n")

    grand_total = 0
    all_stats: dict[str, dict[str, int]] = {}

    for split_name, date_min, date_max, place_id in splits:
        print(f"▶ Спліт: {split_name.upper()}")

        if date_max:
            print(f"  Дати: до {date_max}")

        if date_min:
            print(f"  Дати: з {date_min}")

        print()

        split_total = 0

        for pest in pests:
            count = process_class(
                pest=pest,
                split_name=split_name,
                date_min=date_min,
                date_max=date_max,
                place_id=place_id,
                inat_cfg=inat_cfg,
                args=args,
                session=session,
            )

            meets = count >= thresholds['min_samples_per_class']
            mark = '✓' if meets else '⚠'

            print(f"    {mark} {pest['key']:<35} {count:>4} зображень")

            all_stats.setdefault(pest['key'], {})[split_name] = count
            split_total += count

        print(f"\n  Спліт '{split_name}': {split_total} зображень\n")
        grand_total += split_total

    # Підсумок
    print(f"{'=' * 65}")
    print("  Підсумок по класах:")
    print(f"  {'Клас':<35} {'Train':>7} {'Test':>7}  Статус")
    print(f"  {'-' * 35} {'-' * 7} {'-' * 7}  {'-' * 10}")

    min_s = thresholds['min_samples_per_class']

    for key, counts in sorted(all_stats.items()):
        tr = counts.get('train', 0)
        te = counts.get('test', 0)
        status = '✓' if tr >= min_s else f'⚠ < {min_s}'

        print(f"  {key:<35} {tr:>7} {te:>7}  {status}")

    print(f"{'=' * 65}")
    print(f"  Всього завантажено: {grand_total} зображень\n")


if __name__ == '__main__':
    main()