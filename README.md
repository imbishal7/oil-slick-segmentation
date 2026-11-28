# Oil slick segmentation in Sentinel-1 SAR

Standard U-Net vs Attention U-Net on dual-polarisation (VV, VH) Sentinel-1 patches
from Trujillo-Acatitla et al. (2024). See [PLAN.md](PLAN.md) for the study design.

## Workflow (Google Colab)

Both notebooks clone (or pull) [imbishal7/oil-slick-segmentation](https://github.com/imbishal7/oil-slick-segmentation)
and install it, so they always run the latest `src/` on `main`. Everything persistent lives
in `MyDrive/oil-slick-segmentation/`.

| Notebook | | Runtime | What it does |
|---|---|---|---|
| [01_prepare_data.ipynb](notebooks/01_prepare_data.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imbishal7/oil-slick-segmentation/blob/main/notebooks/01_prepare_data.ipynb) | CPU | download → downsample → audit → split → cache on Drive |
| [02_train_evaluate.ipynb](notebooks/02_train_evaluate.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imbishal7/oil-slick-segmentation/blob/main/notebooks/02_train_evaluate.ipynb) | GPU | train U-Net (E1) and Attention U-Net (E2), evaluate, plot |

Both notebooks are resumable: after a disconnect, run all cells again.

```
MyDrive/oil-slick-segmentation/
├── cache/<archive>.npz            # 01: downsampled patches per Zenodo archive (deletable)
├── data/                          # 01: model-ready arrays, ~3.5 GB
│   ├── {train,val,test}_images.npy    float16 (N, 2, 512, 512), (VV, VH) in dB
│   ├── {train,val,test}_masks.npy     uint8 (N, 512, 512), 1 = oil
│   └── manifest.csv  stats.json  audit.json
├── runs/{E1,E2}/                  # 02: config, history.csv, best.pt, last.pt, eval.json
└── reports/                       # 02: results.md, figures/
```

The raw archives (about 96 GB) are never stored on Drive. Notebook 01 downloads one
archive at a time to Colab's local disk and streams each GeoTIFF out of it in memory.
It caches the 512×512 result and then deletes the archive, so peak local disk use is
the largest archive (41 GB).

## Code layout

| Module | Purpose |
|---|---|
| `sources.py` | Zenodo archives and their internal layout |
| `download.py` | resumable, MD5-checked download; streaming archive extraction |
| `sario.py` | GeoTIFF decoding: (VV, VH) bands, footprint, source product ID |
| `preprocess.py` | downsampling, leakage-aware split, normalization stats, audit |
| `prepare.py` | notebook 01 pipeline (`oilslick-prepare`) |
| `data.py` | `PatchDataset`: memory-mapped arrays, dihedral augmentation |
| `models.py` | `UNet(attention=False/True)`: shared backbone, optional attention gates |
| `losses.py` | BCE, Focal (Lin et al.), batch Soft Dice, Focal + Dice |
| `metrics.py` | pixel confusion counts, IoU / Dice / precision / recall / mIoU / FPR |
| `engine.py` | mixed precision, checkpoints, batched inference |
| `train.py`, `evaluate.py`, `visualize.py` | notebook 02 steps (also available as CLIs) |

## Development

```bash
uv sync
uv run pytest      # unit tests + end-to-end run on synthetic 7z archives
uv run ruff check src tests
```

## Where this differs from PLAN.md

- **Dataset size.** The archives hold 1,200 oil, 685 oil-free and 685 look-alike
  train/val patches (2,570), plus 450 test patches (150 per category).
- **Raw format.** Each patch is one 2-band float32 GeoTIFF (band 0 = VH dB,
  band 1 = VV dB). Channels are reordered to (VV, VH).
- **Leakage control.** Train/val patches are grouped by source Sentinel-1 product
  (read from the embedded SNAP metadata) and by overlapping geographic footprint.
  Groups never cross the split, which is stratified by category.
  `audit.json` also reports test patches whose footprint overlaps train/val.
- **Downsampling.** Images use 4x4 block averaging (area interpolation, which is
  anti-aliased bilinear for an integer factor). A mask pixel is oil when at least
  half of its 4x4 block is oil. This keeps masks pixel-aligned with the images,
  and nearest-neighbour sampling would break up thin slicks.
- **Model output.** The networks return logits, and the sigmoid is applied in the
  loss and at inference, which is more stable numerically.
- **Augmentation.** Flips and 90° rotations are done in NumPy rather than with
  Albumentations.
- **Mixed precision.** fp16 with GradScaler on CUDA, bf16 on Apple MPS.
- **Notebooks.** Two notebooks instead of three: evaluation lives in notebook 02.
- **Focal alpha.** Defaults to the plan's α = 0.25, which *down-weights* oil (the
  positive class). It is worth also trying `--focal-alpha 0.75`.
