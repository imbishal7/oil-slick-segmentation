"""Pure preprocessing steps: downsampling, leakage-aware splitting, statistics, audit."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from oilslick.sources import CHANNELS

BOUNDS = ["lon_min", "lat_min", "lon_max", "lat_max"]


def block_mean(a: np.ndarray, factor: int) -> np.ndarray:
    """Average non-overlapping ``factor`` x ``factor`` blocks of the last two axes."""
    *lead, h, w = a.shape
    if h % factor or w % factor:
        raise ValueError(f"shape {a.shape} not divisible by {factor}")
    blocks = a.reshape(*lead, h // factor, factor, w // factor, factor)
    with warnings.catch_warnings():  # all-NaN blocks legitimately stay NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(blocks, axis=(-3, -1))


def _factor(shape: tuple[int, ...], size: int) -> int:
    h, w = shape[-2:]
    if h != w or h % size:
        raise ValueError(f"cannot downsample {shape} to {size}x{size}")
    return h // size


def downsample_image(image: np.ndarray, size: int) -> tuple[np.ndarray, dict]:
    """(C, H, W) dB image -> (C, size, size) float16 by block averaging, plus audit stats.

    Block averaging is area interpolation, i.e. anti-aliased bilinear for an
    integer factor. Non-finite pixels are ignored (left NaN if a whole block is).
    """
    finite = np.isfinite(image)
    image = np.where(finite, image, np.nan)
    stats = {
        "nonfinite_frac": float(1 - finite.mean()),
        **{f"{c}_mean_db": float(np.nanmean(image[i])) for i, c in enumerate(CHANNELS)},
    }
    return block_mean(image, _factor(image.shape, size)).astype(np.float16), stats


def downsample_mask(mask: np.ndarray, size: int) -> tuple[np.ndarray, dict]:
    """(H, W) mask -> (size, size) uint8; oil where at least half of the block is oil.

    This keeps the mask pixel-aligned with block-averaged images and, unlike
    nearest-neighbour sampling, does not break up thin slicks.
    """
    oil = mask > 0
    small = block_mean(oil.astype(np.float32), _factor(mask.shape, size)) >= 0.5
    stats = {
        "mask_values": sorted(int(v) for v in np.unique(mask)),
        "oil_frac_raw": float(oil.mean()),
        "oil_frac": float(small.mean()),
    }
    return small.astype(np.uint8), stats


def footprint_overlaps(df: pd.DataFrame, other: pd.DataFrame | None = None) -> np.ndarray:
    """Boolean matrix: does footprint i (in df) overlap footprint j (in other)?"""
    other = df if other is None else other
    a = df[BOUNDS].to_numpy(dtype=float)[:, None, :]
    b = other[BOUNDS].to_numpy(dtype=float)[None, :, :]
    with np.errstate(invalid="ignore"):
        return (
            (a[..., 0] < b[..., 2])
            & (b[..., 0] < a[..., 2])
            & (a[..., 1] < b[..., 3])
            & (b[..., 1] < a[..., 3])
        )


def leakage_groups(df: pd.DataFrame) -> np.ndarray:
    """Connected components of 'same source product' OR 'overlapping footprint'."""
    parent = np.arange(len(df))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        parent[find(i)] = find(j)

    for i, j in zip(*np.nonzero(np.triu(footprint_overlaps(df), k=1)), strict=True):
        union(i, j)
    for idx in df.reset_index(drop=True).groupby("product", dropna=True).indices.values():
        for j in idx[1:]:
            union(idx[0], j)
    return np.array([find(i) for i in range(len(df))])


def assign_splits(df: pd.DataFrame, val_frac: float, seed: int) -> pd.DataFrame:
    """Add ``group`` and ``split`` columns. Test patches stay test; validation is one
    fold of a category-stratified group K-fold with K = round(1 / val_frac)."""
    n_splits = round(1 / val_frac) if val_frac > 0 else 0
    if n_splits < 2:
        raise ValueError(f"val_frac must be in (0, 2/3), got {val_frac}")
    df = df.copy()
    df["group"] = -1
    df["split"] = "test"
    trainval = (df["part"] == "trainval").to_numpy()
    tv = df[trainval]
    groups = leakage_groups(tv)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    _, val_idx = next(splitter.split(tv, tv["category"], groups))
    split = np.full(len(tv), "train", dtype=object)
    split[val_idx] = "val"
    df.loc[trainval, "group"] = groups
    df.loc[trainval, "split"] = split
    return df


def channel_stats(images: np.ndarray, chunk: int = 64) -> dict:
    """Per-channel mean/std of an (N, C, H, W) array, ignoring NaNs."""
    channels = images.shape[1]
    total, total_sq, count = np.zeros(channels), np.zeros(channels), np.zeros(channels)
    for start in range(0, len(images), chunk):
        x = images[start : start + chunk].astype(np.float64)
        total += np.nansum(x, axis=(0, 2, 3))
        total_sq += np.nansum(x**2, axis=(0, 2, 3))
        count += np.isfinite(x).sum(axis=(0, 2, 3))
    mean = total / count
    std = np.sqrt(total_sq / count - mean**2)
    return {"channels": list(CHANNELS), "mean": mean.tolist(), "std": std.tolist()}


def audit(df: pd.DataFrame, problems: list[str]) -> dict:
    tv, test = df[df["part"] == "trainval"], df[df["part"] == "test"]
    negatives = df[df["category"] != "oil"]
    overlap = footprint_overlaps(test, tv).any(axis=1) if len(test) and len(tv) else []
    return {
        "counts": df.groupby(["split", "category"]).size().unstack(fill_value=0).to_dict("index"),
        "problems": problems,
        "unexpected_mask_values": df.loc[
            ~df["mask_values"].map(lambda v: set(v) <= {0, 1}), "id"
        ].tolist(),
        "negatives_with_oil_pixels": negatives.loc[negatives["oil_frac_raw"] > 0, "id"].tolist(),
        "oil_patches_without_oil": df.loc[
            (df["category"] == "oil") & (df["oil_frac"] == 0), "id"
        ].tolist(),
        "patches_with_nonfinite": int((df["nonfinite_frac"] > 0).sum()),
        "max_nonfinite_frac": float(df["nonfinite_frac"].max()),
        "oil_fraction_in_oil_patches": df.loc[df["category"] == "oil", "oil_frac"]
        .describe()
        .to_dict(),
        "trainval_missing_product_id": int(tv["product"].isna().sum()),
        "trainval_leakage_groups": int(tv["group"].nunique()),
        "largest_leakage_group": int(tv["group"].value_counts().max()) if len(tv) else 0,
        "test_patches_overlapping_trainval_footprint": int(np.sum(overlap)),
    }
