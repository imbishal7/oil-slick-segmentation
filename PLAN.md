# Project Plan: Ocean Oil Slick Segmentation in Sentinel-1 SAR Imagery

## 1. Core Idea & Objective
Satellite Synthetic Aperture Radar (SAR)—specifically ESA's Sentinel-1 mission—enables all-weather, day-and-night monitoring of marine oil spills. Oil slicks dampen ocean capillary waves and appear as dark regions in SAR imagery, but automated detection faces two core challenges:
1. **Look-Alike Confusion:** Natural features (low-wind zones, biogenic films, rain cells) also appear dark, causing false positives (Trujillo-Acatitla et al., 2024; Prajapati et al., 2025).
2. **Severe Class Imbalance:** Oil pixels make up a tiny fraction of each image and form irregular, fragmented shapes.

**Objective:** Implement, train, and compare **Standard U-Net** (Ronneberger et al., 2015) and **Attention U-Net** (Oktay et al., 2018) on dual-polarization Sentinel-1 SAR data under identical training conditions. We will evaluate how accurately each model outlines oil slicks and whether Attention Gates reduce false positives on look-alikes.

---

## 2. Dataset & Preprocessing

### Primary Dataset: Trujillo-Acatitla et al. (2024)
We will use the open-access Sentinel-1 SAR oil spill dataset hosted on Zenodo (2,850 total patches at $10\text{ m}$ spatial resolution):
* **Part I (Oil Train/Val):** [Zenodo 8346860](https://zenodo.org/records/8346860) — 1,200 confirmed oil-spill patches with binary masks.
* **Part II (No-Oil & Look-Alike Train/Val):** [Zenodo 8253899](https://zenodo.org/records/8253899) — 1,200 oil-free and look-alike patches (exact file counts verified during initial audit).
* **Part III (Held-Out Test Set):** [Zenodo 13761290](https://zenodo.org/records/13761290) — 450 test patches (150 oil, 150 oil-free, 150 look-alike), kept strictly locked until final evaluation.

**Data Format:** Dual-polarization **VV** and **VH** ($\sigma^0$ backscatter in $\text{dB}$), original size $2048 \times 2048$, with binary ground-truth masks (`1` = oil, `0` = sea/look-alike).

### Preprocessing & Split Hygiene
1. **Audit & Leakage Control:** Verify channel alignment, NaN values, and filenames/metadata to ensure overlapping patches from the same spill event do not leak across train and validation splits.
2. **Resizing:** Pre-resize all patches from $2048 \times 2048$ to $512 \times 512$ (bilinear interpolation for VV/VH; nearest-neighbor for masks) to match Trujillo-Acatitla et al. (2024) and speed up training.
3. **Normalization:** Apply channel-wise z-score normalization using mean ($\mu_c$) and standard deviation ($\sigma_c$) computed strictly on the training split.
4. **Splits & Augmentation:** Split Parts I–II into **80% Train / 20% Val** (stratified across oil, oil-free, and look-alike scenes). Apply random horizontal/vertical flips and $90^\circ$ rotations (`Albumentations`) to the training set only.

---

## 3. Models & Training Strategy

Both models take a 2-channel input tensor of shape $(B, 2, 512, 512)$ for $(\text{VV}, \text{VH})$ and output a 1-channel probability map via Sigmoid.

* **Model A — Standard U-Net (Baseline):** 5-level encoder-decoder ($32 \rightarrow 64 \rightarrow 128 \rightarrow 256 \rightarrow 512$ filters) with double $3 \times 3$ `Conv-BN-ReLU` blocks and $2 \times 2$ max pooling. Uses **bilinear upsampling + $3 \times 3$ convolution** in the decoder rather than transposed convolutions, which avoids checkerboard artifacts and improved validation IoU in Trujillo-Acatitla et al. (2024).
* **Model B — Attention U-Net:** Uses the exact same backbone and hyperparameters as Model A, adding additive **Attention Gates** on the skip connections to compute spatial attention coefficients ($\alpha^l \in [0, 1]$) that suppress irrelevant dark ocean background and focus on slick boundaries (Oktay et al., 2018).

### Loss Function & Optimization
Because standard Binary Cross-Entropy (BCE) struggles with severe foreground/background imbalance, both models will be trained with a **Combined Focal + Soft Dice Loss** (Lin et al., 2017; Sudre et al., 2017):

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Focal}}(\gamma = 2.0, \alpha = 0.25) + \mathcal{L}_{\text{Dice}}$$

* **Optimizer & Schedule:** Adam ($\eta = 1 \times 10^{-4}$, weight decay $1 \times 10^{-4}$), `ReduceLROnPlateau` monitoring validation Oil IoU, and early stopping (patience $= 10$ epochs, max $40$–$50$ epochs).
* **Experiment Runs:**
  1. `E0`: Standard U-Net + BCE (quick sanity check / imbalance baseline).
  2. `E1`: Standard U-Net + Focal + Dice Loss (primary baseline).
  3. `E2`: Attention U-Net + Focal + Dice Loss (main comparison).

---

## 4. Google Colab Setup & Workflow
All development, training, and evaluation will run in **Google Colab** (using PyTorch on a T4/L4 GPU runtime):
* **Google Drive Persistence:** Mount Google Drive (`/content/drive/MyDrive/oil-slick-segmentation/`) to store dataset splits, model checkpoints, and metric logs so progress is preserved across Colab disconnects.
* **Fast I/O:** Run preprocessing once to save resized $512 \times 512$ arrays/patches (`.npy` or compressed `.npz`) to Drive, then copy them to Colab's local `/content/` SSD at the start of each session for fast DataLoader throughput.
* **Memory & Speed:** Enable Automatic Mixed Precision (`torch.cuda.amp`, FP16) and use batch size $8$–$16$ (with gradient accumulation if needed) to stay comfortably within Colab's 16 GB GPU VRAM.
* **Notebook Organization:**
  * `01_data_audit_and_preprocess.ipynb` — Download Zenodo data, audit splits, resize to $512 \times 512$, save manifests.
  * `02_train_models.ipynb` — Train U-Net (`E0`, `E1`) and Attention U-Net (`E2`) with automatic checkpointing to Drive.
  * `03_evaluate_and_visualize.ipynb` — Tune threshold on validation set, evaluate on locked Part III test set, generate plots.

---

## 5. Evaluation & Error Analysis
Final evaluation is performed on the locked **Part III Test Set (450 images)** using a binarization threshold $\tau$ selected on the validation set.

1. **Quantitative Metrics:** Oil-class **IoU**, **Dice / F1-Score**, **Precision**, **Recall**, and **Mean IoU (mIoU)**.
2. **Stratified Subset Audit:**
   * *Oil-Spill Subset (150 images):* Measure slick localization and boundary overlap (IoU, F1, Precision, Recall).
   * *Oil-Free (150) & Look-Alike (150) Subsets:* Measure **Pixel False Positive Rate (FPR)** and **Image-Level False Alarm Rate** (% of negative images with predicted oil area $> 50$ pixels) to directly test look-alike rejection.
3. **Efficiency Comparison:** Report trainable parameters, average epoch time on Colab GPU, and inference time per image.
4. **Qualitative Visual Analysis:** Generate side-by-side grids (`VV/VH Input` | `Ground Truth` | `U-Net` | `Attention U-Net` | `TP/FP/FN Error Map`) plus Attention Gate heatmaps ($\alpha^l$) to inspect successful detections, complex boundaries, missed thin slicks (FN), and look-alike false alarms (FP).

---

## 6. Execution Milestones
* **Phase 1 — Setup & Data Prep:** Configure Colab + Drive, download Zenodo Parts I–III, run data audit, and cache $512 \times 512$ patches.
* **Phase 2 — Baseline U-Net:** Implement dataset loader, U-Net, and losses; train `E0` and `E1` and verify validation convergence.
* **Phase 3 — Attention U-Net:** Implement Attention Gates, train `E2` under matched settings, and tune threshold $\tau$ on validation data.
* **Phase 4 — Evaluation & Report:** Run locked Part III test evaluation, compile stratified metric tables and visual error grids, and write the final report.

---

## 7. Related References
1. **Trujillo-Acatitla, R., Tuxpan-Vargas, J., Ovando-Vázquez, C., & Monterrubio-Martínez, E. (2024).** Marine oil spill detection and segmentation in SAR data with two steps Deep Learning framework. *Marine Pollution Bulletin*, 204, 116549. https://doi.org/10.1016/j.marpolbul.2024.116549
2. **Ronneberger, O., Fischer, P., & Brox, T. (2015).** U-Net: Convolutional Networks for Biomedical Image Segmentation. In *MICCAI*, pp. 234–241. Springer.
3. **Oktay, O., Schlemper, J., Folgoc, L. L., et al. (2018).** Attention U-Net: Learning Where to Look for the Pancreas. *arXiv preprint* arXiv:1804.03999.
4. **Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).** Focal Loss for Dense Object Detection. In *ICCV*, pp. 2980–2988.
5. **Sudre, C. H., Li, W., Vercauteren, T., Ourselin, S., & Jorge Cardoso, M. (2017).** Generalised Dice Overlap as a Deep Learning Loss Function for Highly Unbalanced Segmentations. In *Deep Learning in Medical Image Analysis*, pp. 240–248.
6. **Prajapati, K., et al. (2025).** GSCAT-UNET: Global Spatial-Channel Attention U-Net for efficient oil spill detection and discrimination from look-alikes using SAR imagery. *Marine Pollution Bulletin*, 212, 117583.