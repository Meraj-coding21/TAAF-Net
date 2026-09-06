from dataclasses import dataclass
from pathlib import Path
import numpy as np

SEED = 42

CLASS_NAMES = (
    "Black_Spot",
    "Dry_Leaf",
    "Healthy",
    "Leaf_Holes",
    "Yellow_Mosaic_Virus",
)
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS = {i: name for name, i in CLASS_TO_ID.items()}
NUM_CLASSES = len(CLASS_NAMES)

FINAL_CLASS_COUNTS = {
    "Black_Spot": 1486,
    "Dry_Leaf": 323,
    "Healthy": 2497,
    "Leaf_Holes": 1131,
    "Yellow_Mosaic_Virus": 680,
}
TRAIN_CLASS_COUNTS = {
    "Black_Spot": 1188,
    "Dry_Leaf": 258,
    "Healthy": 1999,
    "Leaf_Holes": 904,
    "Yellow_Mosaic_Virus": 544,
}
TEST_CLASS_COUNTS = {
    "Black_Spot": 298,
    "Dry_Leaf": 65,
    "Healthy": 498,
    "Leaf_Holes": 227,
    "Yellow_Mosaic_Virus": 136,
}

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
FUSION_BATCH_SIZE = 64
N_FOLDS = 5
FINAL_VALIDATION_SPLIT = 0.15

ROI_THRESHOLD = 240
ROI_PADDING = 5
MIN_LAPLACIAN_VARIANCE = 10.0
MIN_BRIGHTNESS = 20.0
MAX_BRIGHTNESS = 250.0

LBP_P = 8
LBP_R = 1
GLCM_LEVELS = 32
GLCM_DISTANCE = 1
GLCM_ANGLES = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)

DENSENET_MAX_EPOCHS = 50
EFFICIENTNET_MAX_EPOCHS = 50
CONVNEXT_FROZEN_EPOCHS = 20
CONVNEXT_FINETUNE_EPOCHS = 10
FUSION_EPOCHS = 10

DENSENET_LR = 5e-4
EFFICIENTNET_LR = 1e-4
CONVNEXT_LR = 1e-4
CONVNEXT_FINETUNE_LR = 1e-5
FUSION_LR = 1e-4
LABEL_SMOOTHING = 0.05

CI_LEVEL = 0.95
N_BOOTSTRAPS = 2000
CI_SEED = 2026

EXTERNAL_SHARED_CLASSES = (
    "Black_Spot",
    "Dry_Leaf",
    "Healthy",
    "Leaf_Holes",
)
EXTERNAL_CLASS_TO_ID = {
    name: i for i, name in enumerate(EXTERNAL_SHARED_CLASSES)
}
EXTERNAL_SOURCE_IDS = tuple(
    CLASS_TO_ID[name] for name in EXTERNAL_SHARED_CLASSES
)
EXTERNAL_NUM_CLASSES = len(EXTERNAL_SHARED_CLASSES)

FEWSHOT_K_VALUES = (5, 10, 20)
FEWSHOT_SEEDS = (42, 123, 456, 789, 2026)
FEWSHOT_PRIMARY_SEED = 42
FEWSHOT_EPOCHS = 40
FEWSHOT_LR = 5e-5
FEWSHOT_BATCH_SIZE = 16

VALID_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}

AUGMENTATION_SETTINGS = {
    "rotation_range": 20,
    "width_shift_range": 0.1,
    "height_shift_range": 0.1,
    "shear_range": 0.1,
    "zoom_range": 0.1,
    "horizontal_flip": True,
    "vertical_flip": True,
    "brightness_range": [0.85, 1.15],
    "fill_mode": "nearest",
}


@dataclass(frozen=True)
class OutputPaths:
    root: Path

    @property
    def models(self):
        return self.root / "models"

    @property
    def results(self):
        return self.root / "results"

    @property
    def figures(self):
        return self.root / "figures"

    @property
    def cache(self):
        return self.root / "cache"

    @property
    def cross_validation(self):
        return self.root / "cross_validation"

    def create(self):
        for path in (
            self.root,
            self.models,
            self.results,
            self.figures,
            self.cache,
            self.cross_validation,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self
