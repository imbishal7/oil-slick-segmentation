"""Reading Sentinel-1 GeoTIFF patches and their metadata.

Every reader accepts a path or a binary file object (e.g. ``io.BytesIO``), so
patches can be decoded straight out of an archive without touching the disk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
import tifffile

from oilslick.sources import CHANNELS, RAW_BAND_INDEX

Source = str | Path | BinaryIO

DIMAP_TAG = 65000  # SNAP writes its DIMAP XML into this private tag
PRODUCT_ID = re.compile(
    r"S1[A-D]_[A-Z0-9]{2}_[A-Z0-9]{4}_[A-Z0-9]{4}_\d{8}T\d{6}_\d{8}T\d{6}_\d{6}_[0-9A-F]{6}_[0-9A-F]{4}"
)


@dataclass(frozen=True)
class PatchInfo:
    # Geographic footprint (lon_min, lat_min, lon_max, lat_max) in WGS84 degrees,
    # None if the file has no geo tags.
    bounds: tuple[float, float, float, float] | None
    # Source Sentinel-1 product ID, None if the file has no DIMAP metadata.
    product: str | None


def read_image(source: Source) -> tuple[np.ndarray, PatchInfo]:
    """Return a (C, H, W) float32 dB array in ``CHANNELS`` order (VV, VH) and its metadata."""
    with tifffile.TiffFile(source) as tif:
        raw = tif.asarray()
        tags = tif.pages[0].tags
        if raw.ndim != 3 or raw.shape[-1] != len(RAW_BAND_INDEX):
            raise ValueError(f"expected an (H, W, 2) image, got {raw.shape}")
        height, width = raw.shape[:2]
        product = None
        if DIMAP_TAG in tags:
            match = PRODUCT_ID.search(str(tags[DIMAP_TAG].value)[:20000])
            product = match.group(0) if match else None
        info = PatchInfo(_bounds(tags, height, width), product)
    image = np.stack([raw[..., RAW_BAND_INDEX[c]] for c in CHANNELS]).astype(np.float32)
    return image, info


def read_mask(source: Source) -> np.ndarray:
    """Return the (H, W) mask array as stored."""
    return tifffile.imread(source)


def _bounds(tags, height: int, width: int) -> tuple[float, float, float, float] | None:
    if "ModelTransformationTag" in tags:
        m = tags["ModelTransformationTag"].value
        x0, dx, y0, dy = m[3], m[0], m[7], m[5]
    elif "ModelPixelScaleTag" in tags and "ModelTiepointTag" in tags:
        sx, sy = tags["ModelPixelScaleTag"].value[:2]
        i, j, _, x, y, _ = tags["ModelTiepointTag"].value[:6]
        x0, dx, y0, dy = x - i * sx, sx, y + j * sy, -sy
    else:
        return None
    xs = sorted((x0, x0 + dx * width))
    ys = sorted((y0, y0 + dy * height))
    return (xs[0], ys[0], xs[1], ys[1])
