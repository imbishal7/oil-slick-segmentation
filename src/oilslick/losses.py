"""Segmentation losses. All take raw logits and binary targets of shape (B, 1, H, W)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    """Binary focal loss (Lin et al., 2017); ``alpha`` weights the positive (oil) class."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25) -> None:
        super().__init__()
        self.gamma, self.alpha = gamma, alpha

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p_t = torch.exp(-bce)  # probability assigned to the true class
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()


class SoftDiceLoss(nn.Module):
    """Soft Dice loss computed over the whole batch.

    Batch-level sums keep the loss well defined for the many patches without
    any oil, where a per-image Dice would be degenerate.
    """

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        intersection = (prob * target).sum()
        denominator = prob.sum() + target.sum()
        return 1 - (2 * intersection + self.smooth) / (denominator + self.smooth)


class FocalDiceLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25) -> None:
        super().__init__()
        self.focal = FocalLoss(gamma, alpha)
        self.dice = SoftDiceLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.focal(logits, target) + self.dice(logits, target)


def build_loss(name: str, focal_gamma: float = 2.0, focal_alpha: float = 0.25) -> nn.Module:
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    if name == "focal_dice":
        return FocalDiceLoss(focal_gamma, focal_alpha)
    raise ValueError(f"unknown loss {name!r}")
