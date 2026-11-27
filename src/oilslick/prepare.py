"""Download the dataset and turn it into model-ready arrays (Colab notebook 01).

Stage 1, per archive (resumable): download it into ``--scratch-dir`` (local
disk), stream every TIFF out of it in memory, downsample to ``--size`` and save
``<cache-dir>/<archive>.npz``; then delete the archive. Full-resolution
patches never touch the disk, so peak disk use is one archive (at most 41 GB).

Stage 2, assemble: pair images with masks, audit them, make the leakage-aware
train/val split and compute train-only normalization statistics. Writes to
``--out-dir``:

    {train,val,test}_images.npy   float16 (N, 2, size, size), (VV, VH) in dB
    {train,val,test}_masks.npy    uint8   (N, size, size), 1 = oil
    manifest.csv                  one row per patch; ``index`` = row in its split's arrays
    stats.json                    per-channel train mean / std
    audit.json                    data checks (see ``preprocess.audit``)
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from oilslick.download import ensure_downloaded, stream_members
from oilslick.preprocess import (
    BOUNDS,
    assign_splits,
    audit,
    channel_stats,
    downsample_image,
    downsample_mask,
)
from oilslick.sario import read_image, read_mask
from oilslick.sources import ARCHIVES, SUBSETS, Archive
from oilslick.utils import write_json

SPLITS = ("train", "val", "test")


def locate(archive: Archive, member: str) -> tuple[str, str, dict] | None:
    """Map an archive member to (sample id, "image" | "mask", identity), or None."""
    path = PurePosixPath(archive.stem) / member
    parent, stem = str(path.parent), path.stem
    for s in SUBSETS:
        if parent == s.image_dir:
            kind = "image"
        elif parent == s.mask_dir and stem.endswith(s.mask_suffix):
            kind, stem = "mask", stem.removesuffix(s.mask_suffix)
        else:
            continue
        return f"{s.part}_{s.category}_{stem}", kind, {"part": s.part, "category": s.category}
    return None


def cache_archive(
    archive: Archive, scratch_dir: Path, cache_dir: Path, size: int, keep_archive: bool = False
) -> Path:
    """Download one archive and cache its downsampled patches; skipped if cached."""
    out = cache_dir / f"{archive.stem}.npz"
    if out.exists():
        print(f"[cached] {archive.filename}")
        return out

    path = ensure_downloaded(archive, scratch_dir)
    patches: dict[str, dict] = {"image": {}, "mask": {}}
    skipped: list[str] = []

    def handle(member: str, data: io.BytesIO) -> None:
        located = locate(archive, member)
        if located is None:
            skipped.append(member)
            return
        sample_id, kind, identity = located
        if kind == "image":
            image, info = read_image(data)
            array, stats = downsample_image(image, size)
            meta = {
                **identity,
                "product": info.product,
                **dict(zip(BOUNDS, info.bounds or [np.nan] * 4, strict=True)),
                **stats,
            }
        else:
            array, stats = downsample_mask(read_mask(data), size)
            meta = {**identity, **stats}
        patches[kind][sample_id] = (array, meta)

    stream_members(path, handle)
    if skipped:
        print(f"[warn] {archive.filename}: ignored {len(skipped)} unexpected files")

    cache_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for kind, items in patches.items():
        ids = sorted(items)
        arrays[f"{kind}_ids"] = np.array(ids, dtype=str)
        arrays[f"{kind}_meta"] = np.array(json.dumps([items[i][1] for i in ids]))
        if ids:
            arrays[kind] = np.stack([items[i][0] for i in ids])
    tmp = cache_dir / f"{archive.stem}.tmp.npz"
    np.savez(tmp, **arrays)
    tmp.rename(out)
    if not keep_archive:
        path.unlink()
    return out


def load_caches(cache_dir: Path) -> dict[str, dict]:
    """Merge every archive cache into {kind: {id: (array, meta)}}."""
    missing = [a.filename for a in ARCHIVES if not (cache_dir / f"{a.stem}.npz").exists()]
    if missing:
        raise SystemExit(f"archives not cached yet: {missing}")
    merged: dict[str, dict] = {"image": {}, "mask": {}}
    for archive in ARCHIVES:
        with np.load(cache_dir / f"{archive.stem}.npz") as z:
            for kind in merged:
                ids = z[f"{kind}_ids"].tolist()
                metas = json.loads(str(z[f"{kind}_meta"]))
                arrays = z[kind] if ids else []
                merged[kind].update(zip(ids, zip(arrays, metas, strict=True), strict=True))
    return merged


def assemble(cache_dir: Path, out_dir: Path, val_frac: float, seed: int) -> dict:
    patches = load_caches(cache_dir)
    images, masks = patches["image"], patches["mask"]
    problems = [f"no mask for {i}" for i in sorted(images.keys() - masks.keys())]
    problems += [f"no image for {i}" for i in sorted(masks.keys() - images.keys())]
    ids = sorted(images.keys() & masks.keys())

    df = pd.DataFrame([{"id": i, **images[i][1], **masks[i][1]} for i in ids])
    df = assign_splits(df, val_frac, seed).sort_values(["split", "id"], ignore_index=True)
    df.insert(1, "index", df.groupby("split").cumcount())

    out_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        split_ids = df.loc[df["split"] == split, "id"]
        split_images = np.stack([images[i][0] for i in split_ids])
        np.save(out_dir / f"{split}_images.npy", split_images)
        np.save(out_dir / f"{split}_masks.npy", np.stack([masks[i][0] for i in split_ids]))
        if split == "train":
            stats = channel_stats(split_images)
        del split_images

    report = audit(df, problems)
    df.to_csv(out_dir / "manifest.csv", index=False)
    write_json(out_dir / "stats.json", stats)
    write_json(out_dir / "audit.json", report)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--scratch-dir",
        type=Path,
        default=Path("data/scratch"),
        help="local disk for archive downloads (needs ~45 GB free)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/cache"),
        help="per-archive downsampled patches (persistent, e.g. Google Drive)",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-archives", action="store_true")
    args = parser.parse_args(argv)

    for archive in ARCHIVES:
        cache_archive(archive, args.scratch_dir, args.cache_dir, args.size, args.keep_archives)
    report = assemble(args.cache_dir, args.out_dir, args.val_frac, args.seed)
    print(json.dumps({k: report[k] for k in ("counts", "problems")}, indent=2))
    print(f"full audit: {args.out_dir / 'audit.json'}")


if __name__ == "__main__":
    main()
