import gc
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .backbones import BACKBONE_DISPLAY
from .config import (
    FUSION_BATCH_SIZE,
    N_FOLDS,
    SEED,
)
from .evaluation import classification_metrics
from .fusion import (
    fusion_input_list,
    train_fusion_model,
)
from .texture import texture_for_dataframe
from .training import (
    extract_backbone_embeddings,
    predict_backbone,
    train_backbone,
)


def run_cross_validation(
    train_df,
    texture_lookup,
    output_dir,
    save_models=False,
    verbose=1,
):
    import tensorflow as tf

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    splitter = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    records = []
    labels = train_df["class_id"].to_numpy()

    for fold, (train_index, val_index) in enumerate(
        splitter.split(train_df, labels),
        start=1,
    ):
        fold_train = (
            train_df
            .iloc[train_index]
            .reset_index(drop=True)
        )
        fold_val = (
            train_df
            .iloc[val_index]
            .reset_index(drop=True)
        )

        train_embeddings = {}
        val_embeddings = {}

        for name in (
            "efficientnet",
            "densenet",
            "convnext",
        ):
            save_path = (
                output_dir
                / f"fold_{fold}_{name}.keras"
                if save_models
                else None
            )

            model, _ = train_backbone(
                name,
                fold_train,
                fold_val,
                save_path=save_path,
                verbose=verbose,
            )

            val_probs = predict_backbone(
                model,
                name,
                fold_val,
            )
            metrics = classification_metrics(
                fold_val["class_id"].to_numpy(),
                val_probs,
            )
            records.append({
                "Fold": fold,
                "Model": BACKBONE_DISPLAY[name],
                **metrics,
            })

            train_embeddings[name] = (
                extract_backbone_embeddings(
                    model,
                    name,
                    fold_train,
                )
            )
            val_embeddings[name] = (
                extract_backbone_embeddings(
                    model,
                    name,
                    fold_val,
                )
            )

            del model
            tf.keras.backend.clear_session()
            gc.collect()

        train_texture = texture_for_dataframe(
            fold_train,
            texture_lookup,
            mode="full",
        )
        val_texture = texture_for_dataframe(
            fold_val,
            texture_lookup,
            mode="full",
        )

        fusion_save_path = (
            output_dir
            / f"fold_{fold}_taaf.keras"
            if save_models
            else None
        )

        fusion_model, _ = train_fusion_model(
            train_embeddings=train_embeddings,
            val_embeddings=val_embeddings,
            y_train=fold_train["class_id"].to_numpy(),
            y_val=fold_val["class_id"].to_numpy(),
            train_texture=train_texture,
            val_texture=val_texture,
            save_path=fusion_save_path,
            verbose=verbose,
        )

        val_inputs = fusion_input_list(
            val_embeddings,
            val_texture,
            ("efficientnet", "densenet", "convnext"),
        )
        fusion_probs = fusion_model.predict(
            val_inputs,
            batch_size=FUSION_BATCH_SIZE,
            verbose=0,
        )

        metrics = classification_metrics(
            fold_val["class_id"].to_numpy(),
            fusion_probs,
        )
        records.append({
            "Fold": fold,
            "Model": "TAAF-Net",
            **metrics,
        })

        del fusion_model
        tf.keras.backend.clear_session()
        gc.collect()

    results = pd.DataFrame(records)
    results.to_csv(
        output_dir / "cross_validation_folds.csv",
        index=False,
    )
    return results


def summarize_cross_validation(results):
    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "Specificity",
        "AUC",
    ]

    summary = (
        results
        .groupby("Model")[metrics]
        .agg(["mean", "std"])
    )
    return summary
