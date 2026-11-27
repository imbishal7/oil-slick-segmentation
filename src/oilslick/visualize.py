"""Qualitative figures comparing runs on the test set (run ``oilslick-evaluate`` first).

Example patches are picked from the per-image test results:

* ``oil_best`` / ``oil_worst``: oil patches with the highest / lowest mean IoU;
* ``oil_disagreement``: oil patches where the first and last run differ most;
* ``lookalike_false_alarms``: look-alikes with the most predicted oil pixels.

Each figure shows VV, VH and ground truth, then a TP/FP/FN error map per run,
plus the finest attention-gate coefficients for attention models.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from torch.utils.data import DataLoader, Subset

from oilslick.data import PatchDataset
from oilslick.engine import load_model, predict
from oilslick.utils import get_device, read_json

ERROR_COLORS = {
    "TP": (0.10, 0.75, 0.25),
    "FP": (0.90, 0.15, 0.15),
    "FN": (0.15, 0.45, 1.00),
    "TN": (0.08, 0.08, 0.08),
}


def select_examples(per_image: list[pd.DataFrame], n: int) -> dict[str, list[str]]:
    iou = pd.concat([df.set_index("id")["iou"] for df in per_image], axis=1)
    pred = pd.concat([df.set_index("id")["pred_pixels"] for df in per_image], axis=1)
    category = per_image[0].set_index("id")["category"]
    oil, lookalike = category == "oil", category == "lookalike"
    mean_iou = iou[oil].mean(axis=1)
    return {
        "oil_best": mean_iou.nlargest(n).index.tolist(),
        "oil_worst": mean_iou.nsmallest(n).index.tolist(),
        "oil_disagreement": (iou[oil].iloc[:, -1] - iou[oil].iloc[:, 0])
        .abs()
        .nlargest(n)
        .index.tolist(),
        "lookalike_false_alarms": pred[lookalike].max(axis=1).nlargest(n).index.tolist(),
    }


def error_map(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    rgb = np.empty((*pred.shape, 3))
    for name, sel in {
        "TP": pred & target,
        "FP": pred & ~target,
        "FN": ~pred & target,
        "TN": ~pred & ~target,
    }.items():
        rgb[sel] = ERROR_COLORS[name]
    return rgb


def stretch(channel: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(channel, [2, 98])
    return np.clip((channel - lo) / (hi - lo + 1e-6), 0, 1)


def run_predictions(run_dir: Path, dataset, indices, device) -> dict:
    model, _ = load_model(run_dir / "best.pt", device)
    threshold = read_json(run_dir / "eval.json")["threshold"]
    loader = DataLoader(Subset(dataset, indices), batch_size=4)
    out = {}
    for prob, _, index, alphas in predict(model, loader, device, return_attention=True):
        for b, i in enumerate(index.tolist()):
            out[i] = {
                "pred": (prob[b, 0] >= threshold).cpu().numpy(),
                "alpha": alphas[-1][b, 0].cpu().numpy() if alphas else None,
            }
    return out


def plot_group(title, ids, dataset, id_to_index, runs, predictions, path: Path) -> None:
    has_alpha = [any(p["alpha"] is not None for p in predictions[r].values()) for r in runs]
    columns = ["VV", "VH", "Ground truth"]
    for run, alpha in zip(runs, has_alpha, strict=True):
        columns.append(f"{run.name} errors")
        if alpha:
            columns.append(f"{run.name} attention")

    fig = Figure(figsize=(2.4 * len(columns), 2.5 * len(ids)))
    axes = fig.subplots(len(ids), len(columns), squeeze=False)
    for row, sample_id in enumerate(ids):
        index = id_to_index[sample_id]
        image, target, _ = dataset[index]
        target = target[0].numpy().astype(bool)
        panels = [stretch(image[0].numpy()), stretch(image[1].numpy()), target]
        cmaps = ["gray", "gray", "gray"]
        for run, alpha in zip(runs, has_alpha, strict=True):
            p = predictions[run][index]
            panels.append(error_map(p["pred"], target))
            cmaps.append(None)
            if alpha:
                panels.append(p["alpha"])
                cmaps.append("magma")
        for col, (panel, cmap) in enumerate(zip(panels, cmaps, strict=True)):
            ax = axes[row, col]
            ax.imshow(panel, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(columns[col], fontsize=9)
        axes[row, 0].set_ylabel(sample_id.removeprefix("test_"), fontsize=8)

    legend = [Patch(color=c, label=k) for k, c in ERROR_COLORS.items() if k != "TN"]
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(title.replace("_", " "))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="run directories, e.g. runs/E1 runs/E2")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument("-n", type=int, default=4, help="examples per figure")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    device = get_device(args.device)
    dataset = PatchDataset(args.data_dir, "test")
    id_to_index = {sample_id: i for i, sample_id in enumerate(dataset.manifest["id"])}
    groups = select_examples([pd.read_csv(r / "test_per_image.csv") for r in args.runs], args.n)
    indices = sorted({id_to_index[i] for ids in groups.values() for i in ids})
    predictions = {r: run_predictions(r, dataset, indices, device) for r in args.runs}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for title, ids in groups.items():
        path = args.out_dir / f"{title}.png"
        plot_group(title, ids, dataset, id_to_index, args.runs, predictions, path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
