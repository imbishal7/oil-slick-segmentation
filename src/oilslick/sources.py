"""Where the Trujillo-Acatitla et al. (2024) dataset lives and how it is laid out.

Layout verified against the Zenodo archives:

* Train/val images are 2048x2048x2 float32 GeoTIFFs (band 0 = Sigma0_VH_db,
  band 1 = Sigma0_VV_db), tiled + LZW, with SNAP DIMAP metadata naming the
  source Sentinel-1 product. Masks are 2048x2048 uint8 {0, 1}.
* Test images use the same band order but carry only GeoTIFF tags.
"""

from __future__ import annotations

from dataclasses import dataclass

ZENODO_API = "https://zenodo.org/api/records/{record}"


@dataclass(frozen=True)
class Archive:
    record: int
    filename: str

    @property
    def stem(self) -> str:
        return self.filename.removesuffix(".7z")


ARCHIVES = (
    Archive(8346860, "01_Train_Val_Oil_Spill_images.7z"),
    Archive(8346860, "01_Train_Val_Oil_Spill_mask.7z"),
    Archive(8253899, "01_Train_Val_No_Oil_Images.7z"),
    Archive(8253899, "01_Train_Val_No_Oil_mask.7z"),
    Archive(8253899, "01_Train_Val_Lookalike_images.7z"),
    Archive(8253899, "01_Train_Val_Lookalike_mask.7z"),
    Archive(13761290, "02_Test_images_and_ground_truth.7z"),
)


@dataclass(frozen=True)
class Subset:
    """One (part, category) group of images and where its masks are.

    Paths are relative to the raw data root; each archive is extracted into a
    folder named after the archive.
    """

    part: str  # "trainval" or "test"
    category: str  # "oil", "no_oil" or "lookalike"
    image_dir: str
    mask_dir: str
    mask_suffix: str = ""  # appended to the image stem to get the mask stem


SUBSETS = (
    Subset(
        "trainval",
        "oil",
        "01_Train_Val_Oil_Spill_images/Oil",
        "01_Train_Val_Oil_Spill_mask/Mask_oil",
    ),
    Subset(
        "trainval",
        "no_oil",
        "01_Train_Val_No_Oil_Images/No_oil",
        "01_Train_Val_No_Oil_mask/Mask_no_oil",
    ),
    Subset(
        "trainval",
        "lookalike",
        "01_Train_Val_Lookalike_images/Lookalike",
        "01_Train_Val_Lookalike_mask/Mask_lookalike",
    ),
    Subset(
        "test",
        "oil",
        "02_Test_images_and_ground_truth/Images/Oil",
        "02_Test_images_and_ground_truth/Mask/Oil",
        "_segmentation",
    ),
    Subset(
        "test",
        "no_oil",
        "02_Test_images_and_ground_truth/Images/No oil",
        "02_Test_images_and_ground_truth/Mask/No oil",
        "_segmentation",
    ),
    Subset(
        "test",
        "lookalike",
        "02_Test_images_and_ground_truth/Images/Lookalike",
        "02_Test_images_and_ground_truth/Mask/Lookalike",
        "_segmentation",
    ),
)

# Channel order used everywhere after preprocessing.
CHANNELS = ("VV", "VH")
# Band index of each CHANNELS entry in the raw GeoTIFFs.
RAW_BAND_INDEX = {"VH": 0, "VV": 1}
