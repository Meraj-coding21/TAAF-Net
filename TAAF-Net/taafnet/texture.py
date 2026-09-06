from pathlib import Path

import numpy as np
from PIL import Image
from skimage.feature import (
    graycomatrix,
    graycoprops,
    local_binary_pattern,
)
from tqdm.auto import tqdm

from .config import (
    GLCM_ANGLES,
    GLCM_DISTANCE,
    GLCM_LEVELS,
    IMAGE_SIZE,
    LBP_P,
    LBP_R,
)


def texture_vector_from_gray(gray):
    gray = np.asarray(gray, dtype=np.uint8)

    lbp = local_binary_pattern(
        gray,
        P=LBP_P,
        R=LBP_R,
        method="uniform",
    )
    lbp_hist, _ = np.histogram(
        lbp,
        bins=10,
        range=(0, 10),
        density=True,
    )

    quantized = (gray // 8).astype(np.uint8)
    glcm = graycomatrix(
        quantized,
        distances=[GLCM_DISTANCE],
        angles=list(GLCM_ANGLES),
        levels=GLCM_LEVELS,
        symmetric=True,
        normed=True,
    )

    glcm_features = np.concatenate([
        graycoprops(glcm, "contrast").reshape(-1),
        graycoprops(glcm, "homogeneity").reshape(-1),
        graycoprops(glcm, "energy").reshape(-1),
        graycoprops(glcm, "correlation").reshape(-1),
    ])

    vector = np.concatenate(
        [lbp_hist, glcm_features]
    ).astype(np.float32)

    if vector.shape != (26,):
        raise ValueError(f"Expected 26 texture features, found {vector.shape}")

    return vector


def texture_vector_from_path(path):
    image = (
        Image.open(path)
        .convert("L")
        .resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    )
    return texture_vector_from_gray(
        np.asarray(image, dtype=np.uint8)
    )


def texture_matrix(dataframe, description="Texture features"):
    return np.stack([
        texture_vector_from_path(path)
        for path in tqdm(
            dataframe["path"].astype(str),
            desc=description,
        )
    ]).astype(np.float32)


def build_texture_cache(dataframe, cache_path):
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_paths = dataframe["path"].astype(str).tolist()

    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        cached_paths = cached["paths"].astype(str).tolist()

        if cached_paths == ordered_paths:
            features = cached["features"].astype(np.float32)
            if features.shape == (len(dataframe), 26):
                return features

    features = texture_matrix(dataframe)

    np.savez_compressed(
        cache_path,
        paths=np.asarray(ordered_paths, dtype=str),
        features=features,
    )
    return features


def texture_lookup(dataframe, features):
    return {
        str(path): features[index]
        for index, path in enumerate(
            dataframe["path"].astype(str).tolist()
        )
    }


def texture_for_dataframe(
    dataframe,
    lookup,
    mode="full",
):
    values = np.stack([
        lookup[str(path)]
        for path in dataframe["path"]
    ]).astype(np.float32)

    if mode == "full":
        return values
    if mode == "lbp":
        return values[:, :10]
    if mode == "glcm":
        return values[:, 10:]
    if mode == "none":
        return None

    raise ValueError(f"Unknown texture mode: {mode}")
