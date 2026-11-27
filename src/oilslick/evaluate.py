"""Evaluate trained runs on the locked test set.

For each run directory:

1. the binarisation threshold tau is chosen on the validation split (max oil IoU);
2. the test split is scored at tau, overall and per category: overlap metrics on
   oil patches, pixel FPR and image-level false alarm rate on oil-free and
   look-alike patches;
3. efficiency is reported: parameters, mean epoch time, inference time.

Writes ``<run>/eval.json`` and ``<run>/test_per_image.csv`` for every run and a
side-by-side ``results.md`` / ``results.json`` in ``--report-dir``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from oilslick.data import PatchDataset
from oilslick.engine import autocast, load_model, predict
from oilslick.metrics import FALSE_ALARM_MIN_PIXELS, Confusion, per_image_counts, threshold_counts
from oilslick.utils import count_parameters, get_device, synchronize, write_json

THRESHOLDS = torch.linspace(0.05, 0.95, 19)
COUNT_COLUMNS = ["tp", "fp", "fn", "tn"]


def loader(data_dir: Path, split: str, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(PatchDataset(data_dir, split), batch_size=batch_size, num_workers=num_workers)


def tune_threshold(model, val_loader, device, amp) -> tuple[float, list[dict]]:
    thresholds = THRESHOLDS.to(device)
    counts = torch.zeros(len(thresholds), 4, dtype=torch.long, device=device)
    for prob, target, _ in predict(model, val_loader, device, amp):
        counts += threshold_counts(prob, target, thresholds)
    sweep = [
        {"threshold": round(t.item(), 2), **Confusion(*c.tolist()).metrics()}
        for t, c in zip(thresholds, counts, strict=True)
    ]
    best = max(sweep, key=lambda r: np.nan_to_num(r["iou"], nan=-1))
    return best["threshold"], sweep


def test_per_image(model, test_loader, device, amp, threshold: float) -> pd.DataFrame:
    manifest = test_loader.dataset.manifest
    rows = []
    for prob, target, index in predict(model, test_loader, device, amp):
        counts = per_image_counts(prob >= threshold, target.bool()).cpu().numpy()
        rows.append(pd.DataFrame(counts, columns=COUNT_COLUMNS, index=index.numpy()))
    df = pd.concat(rows)
    df.insert(0, "id", manifest.loc[df.index, "id"].to_numpy())
    df.insert(1, "category", manifest.loc[df.index, "category"].to_numpy())
    df["pred_pixels"] = df["tp"] + df["fp"]
    df["iou"] = df["tp"] / (df["tp"] + df["fp"] + df["fn"]).replace(0, np.nan)
    return df.reset_index(drop=True)


def summarize(per_image: pd.DataFrame) -> dict:
    def confusion(df: pd.DataFrame) -> Confusion:
        return Confusion(*(int(df[c].sum()) for c in COUNT_COLUMNS))

    summary = {"all": confusion(per_image).metrics()}
    for category, df in per_image.groupby("category"):
        metrics = confusion(df).metrics()
        if category == "oil":
            summary[category] = metrics
        else:  # no oil in ground truth: only false positives are meaningful
            summary[category] = {
                "pixel_fpr": metrics["pixel_fpr"],
                "false_alarm_rate": float((df["pred_pixels"] > FALSE_ALARM_MIN_PIXELS).mean()),
                "images": len(df),
            }
    return summary


@torch.no_grad()
def inference_ms(model, device, amp, size: int = 512, repeats: int = 20) -> float:
    """Median single-image forward time in milliseconds."""
    x = torch.randn(1, 2, size, size, device=device)
    times = []
    for i in range(repeats + 3):
        synchronize(device)
        t0 = time.perf_counter()
        with autocast(device, amp):
            model(x)
        synchronize(device)
        if i >= 3:  # warm-up
            times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def evaluate_run(run_dir: Path, data_dir: Path, device, amp, batch_size, num_workers) -> dict:
    model, checkpoint = load_model(run_dir / "best.pt", device)
    threshold, sweep = tune_threshold(
        model, loader(data_dir, "val", batch_size, num_workers), device, amp
    )
    per_image = test_per_image(
        model, loader(data_dir, "test", batch_size, num_workers), device, amp, threshold
    )
    per_image.to_csv(run_dir / "test_per_image.csv", index=False)

    history = pd.read_csv(run_dir / "history.csv")
    result = {
        "run": run_dir.name,
        "attention": checkpoint["config"]["attention"],
        "loss": checkpoint["config"]["loss"],
        "best_epoch": checkpoint["epoch"],
        "threshold": threshold,
        "val_at_threshold": next(r for r in sweep if r["threshold"] == threshold),
        "test": summarize(per_image),
        "efficiency": {
            "parameters": count_parameters(model),
            "epochs_trained": len(history),
            "mean_epoch_time_s": float(history["train_time_s"].mean()),
            "inference_ms_per_image": inference_ms(model, device, amp),
            "device": str(device),
        },
        "threshold_sweep": sweep,
    }
    write_json(run_dir / "eval.json", result)
    return result


def results_table(results: list[dict]) -> str:
    def pct(v: float) -> str:
        return "n/a" if v is None or np.isnan(v) else f"{100 * v:.2f}"

    rows = [
        ("Threshold tau (val)", lambda r: f"{r['threshold']:.2f}"),
        ("Oil IoU (all test)", lambda r: pct(r["test"]["all"]["iou"])),
        ("Dice / F1 (all test)", lambda r: pct(r["test"]["all"]["dice"])),
        ("Precision (all test)", lambda r: pct(r["test"]["all"]["precision"])),
        ("Recall (all test)", lambda r: pct(r["test"]["all"]["recall"])),
        ("mIoU (all test)", lambda r: pct(r["test"]["all"]["miou"])),
        ("Oil IoU (oil subset)", lambda r: pct(r["test"]["oil"]["iou"])),
        ("Dice / F1 (oil subset)", lambda r: pct(r["test"]["oil"]["dice"])),
        ("Precision (oil subset)", lambda r: pct(r["test"]["oil"]["precision"])),
        ("Recall (oil subset)", lambda r: pct(r["test"]["oil"]["recall"])),
        ("Pixel FPR, oil-free", lambda r: pct(r["test"]["no_oil"]["pixel_fpr"])),
        ("False alarm rate, oil-free", lambda r: pct(r["test"]["no_oil"]["false_alarm_rate"])),
        ("Pixel FPR, look-alike", lambda r: pct(r["test"]["lookalike"]["pixel_fpr"])),
        ("False alarm rate, look-alike", lambda r: pct(r["test"]["lookalike"]["false_alarm_rate"])),
        ("Parameters", lambda r: f"{r['efficiency']['parameters']:,}"),
        ("Epochs (best)", lambda r: f"{r['efficiency']['epochs_trained']} ({r['best_epoch']})"),
        ("Mean epoch time (s)", lambda r: f"{r['efficiency']['mean_epoch_time_s']:.0f}"),
        ("Inference (ms/image)", lambda r: f"{r['efficiency']['inference_ms_per_image']:.1f}"),
    ]
    header = "| Metric | " + " | ".join(r["run"] for r in results) + " |"
    lines = [header, "|---" * (len(results) + 1) + "|"]
    lines += [f"| {name} | " + " | ".join(f(r) for r in results) + " |" for name, f in rows]
    note = (
        f"\nPercentages. False alarm: a negative image with more than "
        f"{FALSE_ALARM_MIN_PIXELS} predicted oil pixels (at 512x512).\n"
    )
    return "\n".join(lines) + "\n" + note


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="run directories, e.g. runs/E1")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    args = parser.parse_args(argv)

    device = get_device(args.device)
    results = []
    for run_dir in args.runs:
        print(f"evaluating {run_dir}")
        results.append(
            evaluate_run(
                run_dir, args.data_dir, device, args.amp, args.batch_size, args.num_workers
            )
        )

    table = results_table(results)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "results.md").write_text(table)
    write_json(
        args.report_dir / "results.json",
        [{k: v for k, v in r.items() if k != "threshold_sweep"} for r in results],
    )
    print(table)


if __name__ == "__main__":
    main()
