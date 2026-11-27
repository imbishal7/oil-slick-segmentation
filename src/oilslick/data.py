"""PyTorch dataset over the prepared split arrays (see ``oilslick.prepare``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from oilslick.utils import read_json


def load_manifest(data_dir: Path, split: str | None = None) -> pd.DataFrame:
    """The manifest, optionally restricted to one split and ordered like its arrays."""
    df = pd.read_csv(Path(data_dir) / "manifest.csv")
    if split is None:
        return df
    return df[df["split"] == split].sort_values("index", ignore_index=True)


class PatchDataset(Dataset):
    """Yields ``(image, mask, index)``: image (2, H, W) z-scored float32, mask
    (1, H, W) float32, and the row of the patch in ``self.manifest``.

    Arrays are memory-mapped, so DataLoader workers share them without copies.
    With ``augment=True`` a random element of the dihedral group (90-degree
    rotations x horizontal flip, which also covers vertical flips) is applied.
    """

    def __init__(self, data_dir: Path, split: str, augment: bool = False) -> None:
        data_dir = Path(data_dir)
        self.manifest = load_manifest(data_dir, split)
        self.images = np.load(data_dir / f"{split}_images.npy", mmap_mode="r")
        self.masks = np.load(data_dir / f"{split}_masks.npy", mmap_mode="r")
        if not len(self.manifest) == len(self.images) == len(self.masks):
            raise ValueError(f"{split}: manifest and arrays disagree in length")
        stats = read_json(data_dir / "stats.json")
        self.mean = np.asarray(stats["mean"], dtype=np.float32)[:, None, None]
        self.std = np.asarray(stats["std"], dtype=np.float32)[:, None, None]
        self.augment = augment

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        image = np.nan_to_num((self.images[index].astype(np.float32) - self.mean) / self.std)
        mask = self.masks[index][None].astype(np.float32)
        if self.augment:
            k = np.random.randint(4)
            image, mask = np.rot90(image, k, axes=(1, 2)), np.rot90(mask, k, axes=(1, 2))
            if np.random.rand() < 0.5:
                image, mask = image[:, :, ::-1], mask[:, :, ::-1]
        return (
            torch.from_numpy(np.ascontiguousarray(image)),
            torch.from_numpy(np.ascontiguousarray(mask)),
            index,
        )
