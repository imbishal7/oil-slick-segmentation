"""Pixel-level confusion counting and the derived segmentation metrics."""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Negative images with more than this many predicted oil pixels count as a false alarm.
FALSE_ALARM_MIN_PIXELS = 50


@dataclass
class Confusion:
    """Accumulated pixel counts for the oil (positive) class."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def __add__(self, other: Confusion) -> Confusion:
        return Confusion(*(a + b for a, b in zip(self.astuple(), other.astuple(), strict=True)))

    def astuple(self) -> tuple[int, int, int, int]:
        return self.tp, self.fp, self.fn, self.tn

    def metrics(self) -> dict[str, float]:
        tp, fp, fn, tn = self.astuple()
        iou_oil = _ratio(tp, tp + fp + fn)
        iou_background = _ratio(tn, tn + fn + fp)
        return {
            "iou": iou_oil,
            "dice": _ratio(2 * tp, 2 * tp + fp + fn),
            "precision": _ratio(tp, tp + fp),
            "recall": _ratio(tp, tp + fn),
            "miou": (iou_oil + iou_background) / 2,
            "pixel_fpr": _ratio(fp, fp + tn),
        }


def _ratio(num: int, den: int) -> float:
    return num / den if den else float("nan")


def per_image_counts(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """(B, 4) tensor of [tp, fp, fn, tn] for boolean (B, 1, H, W) inputs."""
    pred, target = pred.flatten(1), target.flatten(1)
    tp = (pred & target).sum(1)
    fp = (pred & ~target).sum(1)
    fn = (~pred & target).sum(1)
    tn = (~pred & ~target).sum(1)
    return torch.stack([tp, fp, fn, tn], dim=1)


def threshold_counts(
    prob: torch.Tensor, target: torch.Tensor, thresholds: torch.Tensor
) -> torch.Tensor:
    """(T, 4) [tp, fp, fn, tn] summed over the batch for each threshold."""
    target = target.bool()
    return torch.stack([per_image_counts(prob >= t, target).sum(0) for t in thresholds])
