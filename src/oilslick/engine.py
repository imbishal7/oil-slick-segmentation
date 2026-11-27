"""Device-aware mixed precision, model checkpoints and batched inference."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from oilslick.models import UNet, build_model


def autocast(device: torch.device, enabled: bool = True) -> contextlib.AbstractContextManager:
    """Mixed precision: fp16 on CUDA (with a GradScaler), bf16 on Apple MPS.

    bf16 is as fast as fp16 on MPS and needs no loss scaling. CPU runs in fp32.
    """
    if not enabled or device.type == "cpu":
        return contextlib.nullcontext()
    dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    return torch.autocast(device.type, dtype=dtype)


def grad_scaler(device: torch.device, amp: bool) -> torch.amp.GradScaler:
    return torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")


def save_checkpoint(path: Path, model: UNet, config: dict, **extra) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "config": config, **extra}, tmp)
    tmp.replace(path)


def load_model(path: Path, device: torch.device) -> tuple[UNet, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = build_model(config["attention"], config["base_channels"])
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint


@torch.no_grad()
def predict(
    model: UNet,
    loader: DataLoader,
    device: torch.device,
    amp: bool = True,
    return_attention: bool = False,
) -> Iterator[tuple[torch.Tensor, ...]]:
    """Yield ``(prob, target, index[, attention maps])`` batches on ``device``."""
    model.eval()
    for image, target, index in loader:
        image, target = image.to(device), target.to(device)
        with autocast(device, amp):
            out = model(image, return_attention=return_attention)
        if return_attention:
            logits, alphas = out
            yield torch.sigmoid(logits.float()), target, index, [a.float() for a in alphas]
        else:
            yield torch.sigmoid(out.float()), target, index
