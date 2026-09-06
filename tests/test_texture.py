import numpy as np

from taafnet.texture import (
    texture_vector_from_gray,
)


def test_texture_vector_shape():
    gray = np.tile(
        np.arange(
            224,
            dtype=np.uint8,
        ),
        (224, 1),
    )

    vector = texture_vector_from_gray(
        gray
    )

    assert vector.shape == (26,)
    assert np.isfinite(vector).all()
