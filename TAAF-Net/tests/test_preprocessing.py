import numpy as np

from taafnet.preprocessing import (
    extract_leaf_roi,
    quality_metrics,
    resize_lanczos4,
)


def test_preprocessing_functions():
    image = np.full(
        (300, 400, 3),
        255,
        dtype=np.uint8,
    )
    image[40:260, 80:320] = 80

    roi, mask = extract_leaf_roi(image)

    assert roi.ndim == 3
    assert mask.shape == image.shape[:2]

    resized = resize_lanczos4(roi)
    assert resized.shape == (224, 224, 3)

    metrics = quality_metrics(resized)
    assert "brightness" in metrics
    assert "laplacian_variance" in metrics
    assert isinstance(
        metrics["quality_pass"],
        bool,
    )
