"""End-to-end smoke test on tiny synthetic archives laid out like the Zenodo ones."""

import json

import numpy as np
import pandas as pd
import py7zr
import pytest
import tifffile

from oilslick import evaluate, prepare, train, visualize
from oilslick.preprocess import block_mean, leakage_groups
from oilslick.sources import ARCHIVES, SUBSETS

RAW, SIZE = 128, 32
PER_SUBSET = {"trainval": 10, "test": 4}


def write_patch(image_path, mask_path, rng, oil: bool, lon: float):
    vh = rng.normal(-25, 2, (RAW, RAW))
    vv = rng.normal(-18, 2, (RAW, RAW))
    mask = np.zeros((RAW, RAW), np.uint8)
    if oil:
        r, c = rng.integers(0, RAW - 40, 2)
        mask[r : r + 40, c : c + 40] = 1
        vv[mask == 1] -= 8
        vh[mask == 1] -= 8
    image = np.stack([vh, vv], axis=-1).astype(np.float32)  # raw band order: VH, VV
    geo = [(33550, "d", 3, (0.001, 0.001, 0.0)), (33922, "d", 6, (0, 0, 0, lon, 10.0, 0))]
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    # one page with 2 samples per pixel, like the Zenodo files
    tifffile.imwrite(
        image_path, image, photometric="minisblack", planarconfig="contig", extratags=geo
    )
    tifffile.imwrite(mask_path, mask)


@pytest.fixture(scope="module")
def archives(tmp_path_factory):
    """Synthetic patches packed into 7z archives named and structured like Zenodo's."""
    raw = tmp_path_factory.mktemp("raw")
    rng = np.random.default_rng(0)
    lon = 0.0
    for s in SUBSETS:
        for k in range(PER_SUBSET[s.part]):
            lon += 1.0  # disjoint footprints...
            if s.part == "trainval" and k % 2:
                lon -= 0.9  # ...except odd patches overlap their predecessor
            write_patch(
                raw / s.image_dir / f"{k:05d}.tif",
                raw / s.mask_dir / f"{k:05d}{s.mask_suffix}.tif",
                rng,
                oil=s.category == "oil",
                lon=lon,
            )
    scratch = tmp_path_factory.mktemp("scratch")
    for archive in ARCHIVES:
        with py7zr.SevenZipFile(scratch / archive.filename, "w") as z:
            for path in sorted((raw / archive.stem).rglob("*.tif")):
                z.write(path, str(path.relative_to(raw / archive.stem)))
    return scratch


def test_block_mean_ignores_nan():
    a = np.array([[1.0, 3.0, np.nan, np.nan], [5.0, 7.0, np.nan, 2.0]])
    np.testing.assert_allclose(block_mean(a, 2), [[4.0, 2.0]])


def test_leakage_groups_join_products_and_overlaps():
    df = pd.DataFrame(
        {
            "product": ["P1", "P1", None, None, None],
            "lon_min": [0, 10, 20, 20.5, 40],
            "lon_max": [1, 11, 21, 21.5, 41],
            "lat_min": [0, 0, 0, 0, 0],
            "lat_max": [1, 1, 1, 1, 1],
        }
    )
    g = leakage_groups(df)
    assert g[0] == g[1] and g[2] == g[3]
    assert len(set(g)) == 3


def test_end_to_end(archives, tmp_path):
    data, runs, reports = tmp_path / "data", tmp_path / "runs", tmp_path / "reports"
    prepare.main(
        [
            "--scratch-dir",
            str(archives),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--out-dir",
            str(data),
            "--size",
            str(SIZE),
            "--keep-archives",
        ]
    )

    manifest = pd.read_csv(data / "manifest.csv")
    assert len(manifest) == 3 * sum(PER_SUBSET.values())
    assert set(manifest["split"]) == {"train", "val", "test"}
    tv = manifest[manifest["part"] == "trainval"]
    # overlapping patches share a group and therefore a split
    assert (tv.groupby("group")["split"].nunique() == 1).all()
    assert tv["group"].nunique() < len(tv)
    audit = json.loads((data / "audit.json").read_text())
    assert audit["problems"] == [] and audit["negatives_with_oil_pixels"] == []

    train_images = np.load(data / "train_images.npy")
    assert train_images.shape == ((manifest["split"] == "train").sum(), 2, SIZE, SIZE)
    assert train_images[:, 0].mean() > train_images[:, 1].mean()  # VV (brighter) first
    row = manifest[manifest["split"] == "train"].iloc[0]
    oil_frac = np.load(data / "train_masks.npy")[row["index"]].mean()
    assert oil_frac == pytest.approx(row["oil_frac"])

    for exp in ("E1", "E2"):
        args = [
            exp,
            "--data-dir",
            str(data),
            "--runs-dir",
            str(runs),
            "--epochs",
            "2",
            "--batch-size",
            "4",
            "--num-workers",
            "0",
            "--base-channels",
            "4",
            "--device",
            "cpu",
            "--resume",
        ]
        train.main(args)
        train.main(args)  # resuming a finished run is a no-op
        assert len(pd.read_csv(runs / exp / "history.csv")) == 2

    run_dirs = [str(runs / "E1"), str(runs / "E2")]
    evaluate.main(
        [
            *run_dirs,
            "--data-dir",
            str(data),
            "--report-dir",
            str(reports),
            "--device",
            "cpu",
            "--num-workers",
            "0",
        ]
    )
    result = json.loads((runs / "E2" / "eval.json").read_text())
    assert {"oil", "no_oil", "lookalike", "all"} <= result["test"].keys()
    assert (reports / "results.md").exists()

    visualize.main(
        [
            *run_dirs,
            "--data-dir",
            str(data),
            "--out-dir",
            str(reports / "fig"),
            "-n",
            "2",
            "--device",
            "cpu",
        ]
    )
    assert len(list((reports / "fig").glob("*.png"))) == 4
