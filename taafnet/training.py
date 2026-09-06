import gc
from pathlib import Path

import numpy as np

from .augmentation import make_generator
from .backbones import (
    BACKBONE_DIMS,
    build_backbone,
    compile_backbone,
)
from .config import (
    CONVNEXT_FINETUNE_EPOCHS,
    CONVNEXT_FINETUNE_LR,
    CONVNEXT_FROZEN_EPOCHS,
    CONVNEXT_LR,
    DENSENET_MAX_EPOCHS,
    EFFICIENTNET_MAX_EPOCHS,
    SEED,
)
from .evaluation import class_weight_dict


def _merge_histories(*histories):
    merged = {}

    for history in histories:
        if history is None:
            continue

        for key, values in history.history.items():
            merged.setdefault(key, []).extend(
                list(values)
            )

    return merged


def train_backbone(
    name,
    train_part,
    val_part,
    save_path=None,
    verbose=1,
):
    import tensorflow as tf
    from tensorflow import keras

    tf.keras.backend.clear_session()
    gc.collect()
    tf.keras.utils.set_random_seed(SEED)

    model, base = build_backbone(name)
    train_gen = make_generator(
        train_part,
        name,
        training=True,
    )
    val_gen = make_generator(
        val_part,
        name,
        training=False,
    )
    weights = class_weight_dict(
        train_part["class_id"].values
    )

    if name == "densenet":
        compile_backbone(model, name)

        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=DENSENET_MAX_EPOCHS,
            class_weight=weights,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=10,
                    restore_best_weights=True,
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.3,
                    patience=4,
                    min_lr=1e-7,
                ),
            ],
            verbose=verbose,
        )
        merged_history = history.history

    elif name == "efficientnet":
        compile_backbone(model, name)

        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EFFICIENTNET_MAX_EPOCHS,
            class_weight=weights,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_accuracy",
                    mode="max",
                    patience=12,
                    min_delta=0.001,
                    restore_best_weights=True,
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7,
                ),
            ],
            verbose=verbose,
        )
        merged_history = history.history

    elif name == "convnext":
        compile_backbone(
            model,
            name,
            CONVNEXT_LR,
        )

        phase1 = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=CONVNEXT_FROZEN_EPOCHS,
            class_weight=weights,
            verbose=verbose,
        )

        for layer in base.layers:
            layer.trainable = False

        for layer in base.layers[-50:]:
            layer.trainable = True

        compile_backbone(
            model,
            name,
            CONVNEXT_FINETUNE_LR,
        )

        phase2 = model.fit(
            train_gen,
            validation_data=val_gen,
            initial_epoch=CONVNEXT_FROZEN_EPOCHS,
            epochs=(
                CONVNEXT_FROZEN_EPOCHS
                + CONVNEXT_FINETUNE_EPOCHS
            ),
            class_weight=weights,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=5,
                    restore_best_weights=True,
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.3,
                    patience=3,
                    min_lr=1e-7,
                ),
            ],
            verbose=verbose,
        )

        merged_history = _merge_histories(
            phase1,
            phase2,
        )

    else:
        raise ValueError(f"Unknown backbone: {name}")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        model.save(save_path)

    return model, merged_history


def predict_backbone(
    model,
    name,
    dataframe,
):
    generator = make_generator(
        dataframe,
        name,
        training=False,
    )
    return model.predict(
        generator,
        verbose=0,
    )


def extract_backbone_embeddings(
    model,
    name,
    dataframe,
):
    from tensorflow import keras

    embedding_model = keras.Model(
        model.input,
        model.get_layer(
            f"{name}_embedding"
        ).output,
        name=f"{name}_embedding_model",
    )

    generator = make_generator(
        dataframe,
        name,
        training=False,
    )
    embeddings = embedding_model.predict(
        generator,
        verbose=0,
    )

    expected = BACKBONE_DIMS[name]

    if embeddings.shape != (
        len(dataframe),
        expected,
    ):
        raise ValueError(
            f"{name}: expected embeddings with shape "
            f"({len(dataframe)}, {expected}), "
            f"found {embeddings.shape}"
        )

    return embeddings.astype(np.float32)
