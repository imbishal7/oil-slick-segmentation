"""Train one experiment (E0, E1 or E2) under the shared training protocol.

Outputs go to ``<runs-dir>/<experiment>/``: ``config.json``, ``history.csv``
(one row per epoch), ``best.pt`` (highest validation oil IoU at threshold
0.5) and ``last.pt`` (full state, for ``--resume``).
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from oilslick.data import PatchDataset
from oilslick.engine import autocast, grad_scaler, save_checkpoint
from oilslick.losses import build_loss
from oilslick.metrics import Confusion, per_image_counts
from oilslick.models import build_model
from oilslick.utils import count_parameters, get_device, seed_everything, synchronize, write_json


@dataclass
class TrainConfig:
    name: str
    attention: bool
    loss: str
    base_channels: int = 32
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    epochs: int = 50
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-4
    lr_patience: int = 4
    lr_factor: float = 0.5
    early_stop_patience: int = 10
    seed: int = 42
    num_workers: int = 2  # Colab runtimes have 2+ vCPUs; the arrays are memory-mapped
    amp: bool = True
    limit: int | None = None  # debugging: random subset of N patches per split


EXPERIMENTS = {
    "E0": {"attention": False, "loss": "bce"},
    "E1": {"attention": False, "loss": "focal_dice"},
    "E2": {"attention": True, "loss": "focal_dice"},
}

# TrainConfig fields that can be overridden from the command line.
OVERRIDABLE = {
    "epochs": int,
    "batch_size": int,
    "lr": float,
    "focal_alpha": float,
    "focal_gamma": float,
    "base_channels": int,
    "num_workers": int,
    "seed": int,
    "limit": int,
}
# Fields that may change when resuming a run; anything else must match last.pt.
RESUMABLE_CHANGES = {"epochs", "num_workers", "amp"}


def make_loader(data_dir: Path, split: str, cfg: TrainConfig, train: bool) -> DataLoader:
    dataset = PatchDataset(data_dir, split, augment=train)
    if cfg.limit:
        # random rather than the first N: the arrays are sorted by category
        perm = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(cfg.seed))
        dataset = Subset(dataset, perm[: cfg.limit].sort().values.tolist())
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=train,
        drop_last=train and len(dataset) >= cfg.batch_size,  # never an empty epoch
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(cfg.seed) if train else None,
    )


def train_one_epoch(model, loader, loss_fn, optimizer, scaler, device, amp) -> float:
    model.train()
    total, seen = 0.0, 0
    for image, target, _ in loader:
        image, target = image.to(device, non_blocking=True), target.to(device, non_blocking=True)
        with autocast(device, amp):
            logits = model(image)
        loss = loss_fn(logits.float(), target)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * len(image)
        seen += len(image)
    return total / seen


@torch.no_grad()
def validate(model, loader, loss_fn, device, amp) -> dict[str, float]:
    model.eval()
    total, counts = 0.0, torch.zeros(4, dtype=torch.long, device=device)
    for image, target, _ in loader:
        image, target = image.to(device), target.to(device)
        with autocast(device, amp):
            logits = model(image).float()
        total += loss_fn(logits, target).item() * len(image)
        counts += per_image_counts(logits > 0, target.bool()).sum(0)
    return {"loss": total / len(loader.dataset), **Confusion(*counts.tolist()).metrics()}


def run(cfg: TrainConfig, data_dir: Path, out_dir: Path, device: torch.device, resume: bool):
    if (out_dir / "last.pt").exists() and not resume:
        raise SystemExit(f"{out_dir} already has a run; pass --resume or choose another --name")
    seed_everything(cfg.seed)

    train_loader = make_loader(data_dir, "train", cfg, train=True)
    val_loader = make_loader(data_dir, "val", cfg, train=False)
    model = build_model(cfg.attention, cfg.base_channels).to(device)
    loss_fn = build_loss(cfg.loss, cfg.focal_gamma, cfg.focal_alpha)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=cfg.lr_factor, patience=cfg.lr_patience
    )
    scaler = grad_scaler(device, cfg.amp)
    config = asdict(cfg)

    start_epoch, best_iou, bad_epochs = 1, -1.0, 0
    if resume and (out_dir / "last.pt").exists():
        state = torch.load(out_dir / "last.pt", map_location=device, weights_only=False)
        _check_resumable(config, state["config"], out_dir)
        for obj, key in [
            (model, "model"),
            (optimizer, "optimizer"),
            (scheduler, "scheduler"),
            (scaler, "scaler"),
        ]:
            obj.load_state_dict(state[key])
        start_epoch, best_iou, bad_epochs = (
            state["epoch"] + 1,
            state["best_iou"],
            state["bad_epochs"],
        )
        print(f"resumed from epoch {state['epoch']} (best val IoU {best_iou:.4f})")
        if bad_epochs >= cfg.early_stop_patience or start_epoch > cfg.epochs:
            print(f"{cfg.name} already finished training -> {out_dir / 'best.pt'}")
            return
    else:
        write_json(out_dir / "config.json", {**config, "parameters": count_parameters(model)})

    print(
        f"{cfg.name}: {count_parameters(model):,} params, {len(train_loader.dataset)} train / "
        f"{len(val_loader.dataset)} val patches, device={device}"
    )
    history_path = out_dir / "history.csv"
    _truncate_history(history_path, start_epoch)
    for epoch in range(start_epoch, cfg.epochs + 1):
        synchronize(device)
        t0 = time.perf_counter()
        train_loss = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scaler, device, cfg.amp
        )
        synchronize(device)
        train_time = time.perf_counter() - t0
        val = validate(model, val_loader, loss_fn, device, cfg.amp)
        # NaN (no oil predicted or present) would never compare as an improvement
        score = 0.0 if math.isnan(val["iou"]) else val["iou"]
        scheduler.step(score)

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "train_time_s": train_time,
            **{f"val_{k}": v for k, v in val.items()},
        }
        _append_csv(history_path, row)
        print(
            f"epoch {epoch:3d} | train {train_loss:.4f} | val {val['loss']:.4f} | "
            f"IoU {val['iou']:.4f} F1 {val['dice']:.4f} P {val['precision']:.4f} "
            f"R {val['recall']:.4f} | {train_time:.0f}s"
        )

        if score > best_iou:
            best_iou, bad_epochs = score, 0
            save_checkpoint(out_dir / "best.pt", model, config, epoch=epoch, val=val)
        else:
            bad_epochs += 1
        save_checkpoint(
            out_dir / "last.pt",
            model,
            config,
            epoch=epoch,
            best_iou=best_iou,
            bad_epochs=bad_epochs,
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            scaler=scaler.state_dict(),
        )
        if bad_epochs >= cfg.early_stop_patience:
            print(f"early stopping: no val IoU improvement for {bad_epochs} epochs")
            break
    print(f"best val IoU {best_iou:.4f} -> {out_dir / 'best.pt'}")


def _check_resumable(config: dict, saved: dict, out_dir: Path) -> None:
    changed = sorted(
        k for k, v in config.items() if k not in RESUMABLE_CHANGES and saved.get(k) != v
    )
    if changed:
        raise SystemExit(
            f"cannot resume {out_dir}: settings differ from last.pt in {changed}; "
            "pass the original settings or choose another --name"
        )


def _truncate_history(path: Path, start_epoch: int) -> None:
    """Drop rows for epochs about to be (re)run, e.g. one logged just before a crash
    that happened before ``last.pt`` was written."""
    if not path.exists():
        return
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = [r for r in reader if int(r["epoch"]) < start_epoch]
    if not rows:
        path.unlink()
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _append_csv(path: Path, row: dict) -> None:
    new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=sorted(EXPERIMENTS))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--name", help="run name (default: the experiment id)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    for field, kind in OVERRIDABLE.items():  # e.g. --epochs 2 --limit 32 for a smoke test
        parser.add_argument(f"--{field.replace('_', '-')}", type=kind)
    args = parser.parse_args(argv)

    name = args.name or args.experiment
    overrides = {k: getattr(args, k) for k in OVERRIDABLE if getattr(args, k) is not None}
    cfg = TrainConfig(name=name, **EXPERIMENTS[args.experiment], amp=args.amp, **overrides)
    run(cfg, args.data_dir, args.runs_dir / name, get_device(args.device), args.resume)


if __name__ == "__main__":
    main()
