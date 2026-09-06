import gc

import numpy as np
from sklearn.metrics import accuracy_score

from .config import FUSION_BATCH_SIZE, NUM_CLASSES
from .evaluation import classification_metrics
from .fusion import (
    fusion_input_list,
    train_fusion_model,
)
from .texture import texture_for_dataframe


def evaluate_fusion_variant(
    train_df,
    val_df,
    test_df,
    embeddings,
    texture_lookup,
    branches,
    texture_mode,
    fusion_type,
    verbose=0,
):
    import tensorflow as tf

    train_texture = texture_for_dataframe(
        train_df,
        texture_lookup,
        texture_mode,
    )
    val_texture = texture_for_dataframe(
        val_df,
        texture_lookup,
        texture_mode,
    )
    test_texture = texture_for_dataframe(
        test_df,
        texture_lookup,
        texture_mode,
    )

    model, _ = train_fusion_model(
        train_embeddings=embeddings["train"],
        val_embeddings=embeddings["val"],
        y_train=train_df["class_id"].to_numpy(),
        y_val=val_df["class_id"].to_numpy(),
        train_texture=train_texture,
        val_texture=val_texture,
        branches=branches,
        texture_mode=texture_mode,
        fusion_type=fusion_type,
        verbose=verbose,
    )

    test_inputs = fusion_input_list(
        embeddings["test"],
        test_texture,
        branches,
    )
    probabilities = model.predict(
        test_inputs,
        batch_size=FUSION_BATCH_SIZE,
        verbose=0,
    )
    metrics = classification_metrics(
        test_df["class_id"].to_numpy(),
        probabilities,
    )

    del model
    tf.keras.backend.clear_session()
    gc.collect()

    return metrics


def probability_fusion(
    strategy,
    base_test_probabilities,
    base_val_probabilities=None,
    val_labels=None,
):
    ordered = (
        "efficientnet",
        "densenet",
        "convnext",
    )
    stacked = np.stack([
        base_test_probabilities[name]
        for name in ordered
    ])

    if strategy == "simple_average":
        probabilities = stacked.mean(axis=0)

    elif strategy == "majority_voting":
        votes = stacked.argmax(axis=2)
        predictions = []

        for sample_index in range(
            votes.shape[1]
        ):
            values, counts = np.unique(
                votes[:, sample_index],
                return_counts=True,
            )
            predictions.append(
                values[np.argmax(counts)]
            )

        probabilities = np.eye(
            NUM_CLASSES,
            dtype=np.float32,
        )[np.asarray(predictions, dtype=int)]

    elif strategy == "accuracy_weighted":
        if (
            base_val_probabilities is None
            or val_labels is None
        ):
            raise ValueError(
                "Validation probabilities and labels are required"
            )

        val_accuracy = np.asarray([
            accuracy_score(
                val_labels,
                base_val_probabilities[name].argmax(axis=1),
            )
            for name in ordered
        ])
        weights = val_accuracy / val_accuracy.sum()

        probabilities = sum(
            weight * base_test_probabilities[name]
            for weight, name in zip(
                weights,
                ordered,
            )
        )

    else:
        raise ValueError(
            f"Unknown probability fusion strategy: {strategy}"
        )

    return probabilities
