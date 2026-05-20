"""
train_both.py

Тренує обидві моделі послідовно:
  Model A — детектор листя   (1 клас)
  Model B — детектор шкідників (11 класів)

Запуск:
    python train_both.py
    python train_both.py --model-a-only
    python train_both.py --model-b-only
"""
import argparse
import json
import time
from pathlib import Path

from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Конфіг
# ---------------------------------------------------------------------------

MODEL_A_CONFIG = dict(
    data          = "data/processed/leaf_detector/data.yaml",
    epochs        = 300,
    imgsz         = 1024,
    batch         = 8,
    device        = 0,
    patience      = 80,
    optimizer     = "AdamW",
    lr0           = 0.0003,
    lrf           = 0.01,
    cos_lr        = True,
    warmup_epochs = 5,
    weight_decay  = 0.001,
    augment       = True,
    workers       = 8,
    project       = "runs/leaf_detector",
    name          = "v2",
)

MODEL_B_CONFIG = dict(
    data       = "data/processed/pest_detector/data.yaml",
    epochs     = 150,
    imgsz      = 640,
    batch      = 32,
    device     = 0,
    patience   = 30,
    freeze     = 10,       # freeze backbone
    cls        = 0.7,      # вища вага cls loss
    box        = 7.5,
    optimizer  = "AdamW",
    lr0        = 0.001,
    cos_lr     = True,
    augment    = True,
    workers    = 8,
    project    = "runs/pest_detector",
    name       = "v1",
    exist_ok   = False,
)


# ---------------------------------------------------------------------------

def check_gpu():
    import torch
    print(f"\n{'=' * 50}")
    print("  GPU Check")
    print(f"{'=' * 50}")
    print(f"  PyTorch:        {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU:            {props.name}")
        print(f"  VRAM:           {props.total_memory / 1e9:.1f} GB")
        cc = f"{props.major}.{props.minor}"
        print(f"  Compute cap:    {cc}")
        if props.major < 8:
            print("  ⚠  Compute < 8.0 — можливі проблеми з YOLO11m")
    print(f"{'=' * 50}\n")


def train_model(label: str, weights: str, config: dict) -> dict:
    print(f"\n{'=' * 50}")
    print(f"  {label}")
    print(f"{'=' * 50}\n")

    t0 = time.time()
    model = YOLO(weights)
    results = model.train(**config)
    elapsed = time.time() - t0

    best_weights = Path(config["project"]) / config["name"] / "weights" / "best.pt"

    summary = {
        "label":       label,
        "weights_src": weights,
        "best_pt":     str(best_weights),
        "elapsed_min": round(elapsed / 60, 1),
        "map50":       round(float(results.results_dict.get("metrics/mAP50(B)", 0)), 4),
        "map50_95":    round(float(results.results_dict.get("metrics/mAP50-95(B)", 0)), 4),
    }

    print(f"\n  Час тренування: {summary['elapsed_min']} хв")
    print(f"  mAP@0.5:        {summary['map50']}")
    print(f"  mAP@0.5:0.95:   {summary['map50_95']}")
    print(f"  Ваги:           {summary['best_pt']}")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a-only", action="store_true")
    parser.add_argument("--model-b-only", action="store_true")
    args = parser.parse_args()

    check_gpu()

    summaries = []

    if not args.model_b_only:
        s = train_model(
            label    = "Model A — Leaf Detector (1 клас)",
            weights  = "yolo11m.pt",
            config   = MODEL_A_CONFIG,
        )
        summaries.append(s)

    if not args.model_a_only:
        s = train_model(
            label    = "Model B — Pest Detector (11 класів)",
            weights  = "yolo11m.pt",
            config   = MODEL_B_CONFIG,
        )
        summaries.append(s)

    # Підсумок
    print(f"\n{'=' * 50}")
    print("  ПІДСУМОК")
    print(f"{'=' * 50}")
    for s in summaries:
        print(f"\n  {s['label']}")
        print(f"    mAP@0.5:      {s['map50']}")
        print(f"    mAP@0.5:0.95: {s['map50_95']}")
        print(f"    Час:          {s['elapsed_min']} хв")
        print(f"    Ваги:         {s['best_pt']}")

    # Зберегти підсумок
    Path("runs").mkdir(exist_ok=True)
    out = Path("runs/training_summary.json")
    out.write_text(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(f"\n  Підсумок збережено: {out}")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    main()
