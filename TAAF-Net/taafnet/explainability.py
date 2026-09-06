import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.manifold import TSNE
from skimage.feature import (
    graycomatrix,
    local_binary_pattern,
)

from .augmentation import PREPROCESS_FUNCTIONS
from .config import (
    CI_SEED,
    CLASS_NAMES,
    FUSION_BATCH_SIZE,
    GLCM_ANGLES,
    GLCM_DISTANCE,
    GLCM_LEVELS,
    IMAGE_SIZE,
    LBP_P,
    LBP_R,
    N_BOOTSTRAPS,
    SEED,
)


def load_rgb(path):
    return np.asarray(
        Image.open(path)
        .convert("RGB")
        .resize(IMAGE_SIZE, Image.Resampling.LANCZOS),
        dtype=np.uint8,
    )


def lbp_and_glcm_map(path):
    gray = np.asarray(
        Image.open(path)
        .convert("L")
        .resize(IMAGE_SIZE, Image.Resampling.LANCZOS),
        dtype=np.uint8,
    )

    lbp = local_binary_pattern(
        gray,
        P=LBP_P,
        R=LBP_R,
        method="uniform",
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
    glcm_mean = glcm[:, :, 0, :].mean(axis=-1)
    return lbp, glcm_mean


def run_tsne(features):
    kwargs = {
        "n_components": 2,
        "perplexity": 30,
        "learning_rate": "auto",
        "init": "pca",
        "random_state": SEED,
    }

    try:
        reducer = TSNE(
            max_iter=1000,
            **kwargs,
        )
    except TypeError:
        reducer = TSNE(
            n_iter=1000,
            **kwargs,
        )

    embedding = reducer.fit_transform(
        features
    )
    return (
        embedding,
        float(reducer.kl_divergence_),
    )


def gate_analysis(
    fusion_model,
    fusion_inputs,
    y_true,
    class_names=CLASS_NAMES,
    n_bootstraps=N_BOOTSTRAPS,
    seed=CI_SEED + 700,
):
    from tensorflow import keras

    attention_model = keras.Model(
        fusion_model.inputs,
        fusion_model.get_layer("alpha").output,
    )
    alpha = attention_model.predict(
        fusion_inputs,
        batch_size=FUSION_BATCH_SIZE,
        verbose=0,
    )

    segments = {
        "EfficientNetV2-M": (0, 256),
        "DenseNet169": (256, 512),
        "ConvNeXt-Tiny": (512, 768),
        "Texture": (768, 832),
    }

    stream_rows = []
    deep_texture_rows = []
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=int)

    for class_id, class_name in enumerate(
        class_names
    ):
        class_alpha = alpha[
            y_true == class_id
        ]

        row = {
            "Class": class_name,
            "N": len(class_alpha),
        }

        for stream, (
            start,
            end,
        ) in segments.items():
            row[stream] = float(
                class_alpha[:, start:end].mean()
            )

        stream_rows.append(row)

        sample_deep = (
            class_alpha[:, :768]
            .mean(axis=1)
        )
        sample_texture = (
            class_alpha[:, 768:832]
            .mean(axis=1)
        )

        deep_mean = float(
            sample_deep.mean()
        )
        texture_mean = float(
            sample_texture.mean()
        )

        deep_boot = []
        texture_boot = []
        ratio_boot = []
        sample_count = len(sample_deep)

        for _ in range(n_bootstraps):
            indices = rng.choice(
                sample_count,
                size=sample_count,
                replace=True,
            )
            deep_value = float(
                sample_deep[indices].mean()
            )
            texture_value = float(
                sample_texture[indices].mean()
            )

            deep_boot.append(deep_value)
            texture_boot.append(texture_value)
            ratio_boot.append(
                texture_value / deep_value
                if deep_value > 0
                else np.nan
            )

        deep_lower, deep_upper = np.quantile(
            deep_boot,
            [0.025, 0.975],
        )
        texture_lower, texture_upper = np.quantile(
            texture_boot,
            [0.025, 0.975],
        )

        ratio_values = np.asarray(
            ratio_boot,
            dtype=float,
        )
        ratio_values = ratio_values[
            np.isfinite(ratio_values)
        ]
        ratio_lower, ratio_upper = np.quantile(
            ratio_values,
            [0.025, 0.975],
        )

        deep_texture_rows.append({
            "Class": class_name,
            "N": sample_count,
            "Deep_Gate_Mean": deep_mean,
            "Deep_CI_Lower": float(deep_lower),
            "Deep_CI_Upper": float(deep_upper),
            "Texture_Gate_Mean": texture_mean,
            "Texture_CI_Lower": float(texture_lower),
            "Texture_CI_Upper": float(texture_upper),
            "Texture_to_Deep_Ratio": (
                texture_mean / deep_mean
                if deep_mean > 0
                else np.nan
            ),
            "Ratio_CI_Lower": float(ratio_lower),
            "Ratio_CI_Upper": float(ratio_upper),
        })

    return (
        pd.DataFrame(stream_rows),
        pd.DataFrame(deep_texture_rows),
    )


def get_nested_base(model):
    from tensorflow import keras

    for layer in model.layers:
        if isinstance(layer, keras.Model):
            return layer

    raise ValueError(
        "Backbone base model was not found"
    )


def find_first_conv2d(layer):
    from tensorflow.keras import layers

    if isinstance(layer, layers.Conv2D):
        return layer

    if hasattr(layer, "layers"):
        for child in layer.layers:
            result = find_first_conv2d(child)

            if result is not None:
                return result

    return None


def preprocess_single(rgb, name):
    import tensorflow as tf

    array = PREPROCESS_FUNCTIONS[name](
        rgb.astype(np.float32)
    )
    return tf.convert_to_tensor(
        array[None, ...],
        dtype=tf.float32,
    )


def backbone_gradcam(
    model,
    name,
    rgb,
    target_class,
):
    import tensorflow as tf
    from tensorflow import keras

    base = get_nested_base(model)
    embedding_layer = model.get_layer(
        f"{name}_embedding"
    )
    logits_layer = model.get_layer(
        f"{name}_logits"
    )
    head_model = keras.Model(
        embedding_layer.input,
        logits_layer.output,
    )
    tensor = preprocess_single(
        rgb,
        name,
    )

    with tf.GradientTape() as tape:
        feature_map = base(
            tensor,
            training=False,
        )
        tape.watch(feature_map)
        logits = head_model(
            feature_map,
            training=False,
        )
        score = logits[:, target_class]

    gradients = tape.gradient(
        score,
        feature_map,
    )
    weights = tf.reduce_mean(
        gradients,
        axis=(1, 2),
        keepdims=True,
    )
    cam = tf.reduce_sum(
        feature_map * weights,
        axis=-1,
    )[0]
    cam = tf.nn.relu(cam)
    cam = cam / (
        tf.reduce_max(cam) + 1e-8
    )
    return cam.numpy()


def binary_iou(a, b, threshold=0.5):
    a = (
        cv2.resize(a, IMAGE_SIZE)
        >= threshold
    )
    b = (
        cv2.resize(b, IMAGE_SIZE)
        >= threshold
    )

    intersection = np.logical_and(
        a,
        b,
    ).sum()
    union = np.logical_or(
        a,
        b,
    ).sum()

    return (
        float(intersection / union)
        if union
        else 1.0
    )
