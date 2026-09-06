import gc
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .backbones import BACKBONE_DISPLAY
from .config import (
    CI_SEED,
    CLASS_NAMES,
    FINAL_VALIDATION_SPLIT,
    FUSION_BATCH_SIZE,
    SEED,
    OutputPaths,
)
from .data import load_preprocessed_dataset
from .evaluation import (
    classwise_f1_confidence_intervals,
    classification_metrics,
    metric_confidence_intervals,
)
from .fusion import (
    fusion_input_list,
    train_fusion_model,
)
from .statistical_tests import (
    cochran_q_test,
    pairwise_mcnemar_table,
)
from .texture import (
    build_texture_cache,
    texture_for_dataframe,
    texture_lookup,
)
from .training import (
    extract_backbone_embeddings,
    predict_backbone,
    train_backbone,
)


def run_final_training(
    dataset_root,
    output_root,
    verbose=1,
):
    import tensorflow as tf

    paths = OutputPaths(
        Path(output_root)
    ).create()

    train_df, test_df = (
        load_preprocessed_dataset(
            dataset_root,
            verify_counts=True,
        )
    )

    fit_train_df, fit_val_df = (
        train_test_split(
            train_df,
            test_size=FINAL_VALIDATION_SPLIT,
            stratify=train_df["class_id"],
            random_state=SEED,
        )
    )
    fit_train_df = fit_train_df.reset_index(
        drop=True
    )
    fit_val_df = fit_val_df.reset_index(
        drop=True
    )

    all_df = pd.concat(
        [train_df, test_df],
        ignore_index=True,
    )
    all_texture = build_texture_cache(
        all_df,
        paths.cache / "texture_features.npz",
    )
    lookup = texture_lookup(
        all_df,
        all_texture,
    )

    embeddings = {
        "train": {},
        "val": {},
        "test": {},
    }
    base_val_probabilities = {}
    base_test_probabilities = {}
    base_metrics = {}
    model_paths = {}

    for name in (
        "efficientnet",
        "densenet",
        "convnext",
    ):
        model_path = (
            paths.models
            / f"{name}_final.keras"
        )
        model_paths[name] = model_path

        model, history = train_backbone(
            name,
            fit_train_df,
            fit_val_df,
            save_path=model_path,
            verbose=verbose,
        )

        pd.DataFrame(history).to_csv(
            paths.results
            / f"{name}_training_history.csv",
            index=False,
        )

        embeddings["train"][name] = (
            extract_backbone_embeddings(
                model,
                name,
                fit_train_df,
            )
        )
        embeddings["val"][name] = (
            extract_backbone_embeddings(
                model,
                name,
                fit_val_df,
            )
        )
        embeddings["test"][name] = (
            extract_backbone_embeddings(
                model,
                name,
                test_df,
            )
        )

        base_val_probabilities[name] = (
            predict_backbone(
                model,
                name,
                fit_val_df,
            )
        )
        base_test_probabilities[name] = (
            predict_backbone(
                model,
                name,
                test_df,
            )
        )
        base_metrics[
            BACKBONE_DISPLAY[name]
        ] = classification_metrics(
            test_df["class_id"].to_numpy(),
            base_test_probabilities[name],
        )

        del model
        tf.keras.backend.clear_session()
        gc.collect()

    train_texture = texture_for_dataframe(
        fit_train_df,
        lookup,
        "full",
    )
    val_texture = texture_for_dataframe(
        fit_val_df,
        lookup,
        "full",
    )
    test_texture = texture_for_dataframe(
        test_df,
        lookup,
        "full",
    )

    fusion_path = (
        paths.models
        / "taaf_net_final.keras"
    )

    fusion_model, fusion_history = (
        train_fusion_model(
            train_embeddings=embeddings["train"],
            val_embeddings=embeddings["val"],
            y_train=fit_train_df[
                "class_id"
            ].to_numpy(),
            y_val=fit_val_df[
                "class_id"
            ].to_numpy(),
            train_texture=train_texture,
            val_texture=val_texture,
            save_path=fusion_path,
            verbose=verbose,
        )
    )

    pd.DataFrame(
        fusion_history
    ).to_csv(
        paths.results
        / "taaf_training_history.csv",
        index=False,
    )

    fusion_test_inputs = fusion_input_list(
        embeddings["test"],
        test_texture,
        (
            "efficientnet",
            "densenet",
            "convnext",
        ),
    )
    taaf_probabilities = (
        fusion_model.predict(
            fusion_test_inputs,
            batch_size=FUSION_BATCH_SIZE,
            verbose=0,
        )
    )

    taaf_metrics = classification_metrics(
        test_df["class_id"].to_numpy(),
        taaf_probabilities,
    )

    all_metrics = dict(base_metrics)
    all_metrics["TAAF-Net"] = taaf_metrics
    pd.DataFrame(all_metrics).T.to_csv(
        paths.results
        / "independent_test_metrics.csv"
    )

    probability_map = {
        BACKBONE_DISPLAY[name]:
            base_test_probabilities[name]
        for name in (
            "convnext",
            "efficientnet",
            "densenet",
        )
    }
    probability_map[
        "TAAF-Net"
    ] = taaf_probabilities

    y_true = test_df[
        "class_id"
    ].to_numpy()

    ci_tables = []

    for index, (
        model_name,
        probabilities,
    ) in enumerate(
        probability_map.items()
    ):
        table = metric_confidence_intervals(
            y_true,
            probabilities,
            seed=CI_SEED + index,
        )
        table.insert(
            0,
            "Model",
            model_name,
        )
        ci_tables.append(table)

    pd.concat(
        ci_tables,
        ignore_index=True,
    ).to_csv(
        paths.results
        / "independent_test_metrics_95ci.csv",
        index=False,
    )

    classwise_f1_confidence_intervals(
        y_true,
        taaf_probabilities,
        class_names=CLASS_NAMES,
    ).to_csv(
        paths.results
        / "taaf_classwise_f1_95ci.csv",
        index=False,
    )

    predictions = {
        model_name:
            probabilities.argmax(axis=1)
        for model_name, probabilities
        in probability_map.items()
    }

    correctness = np.column_stack([
        predictions[model_name] == y_true
        for model_name
        in probability_map
    ]).astype(int)

    q_statistic, q_p_value = (
        cochran_q_test(correctness)
    )
    pd.DataFrame([{
        "N": len(y_true),
        "Models": len(probability_map),
        "Q_Statistic": q_statistic,
        "p_value": q_p_value,
    }]).to_csv(
        paths.results
        / "cochran_q_test.csv",
        index=False,
    )

    pairwise_mcnemar_table(
        y_true,
        predictions,
    ).to_csv(
        paths.results
        / "mcnemar_pairwise.csv",
        index=False,
    )

    prediction_table = test_df[
        [
            "path",
            "class_name",
            "class_id",
        ]
    ].copy()

    for model_name, probabilities in (
        probability_map.items()
    ):
        safe_name = (
            model_name
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )
        prediction_table[
            f"{safe_name}_pred"
        ] = probabilities.argmax(axis=1)
        prediction_table[
            f"{safe_name}_confidence"
        ] = probabilities.max(axis=1)

    prediction_table.to_csv(
        paths.results
        / "test_predictions.csv",
        index=False,
    )

    return {
        "paths": paths,
        "train_df": train_df,
        "test_df": test_df,
        "fit_train_df": fit_train_df,
        "fit_val_df": fit_val_df,
        "embeddings": embeddings,
        "texture_lookup": lookup,
        "base_val_probabilities": (
            base_val_probabilities
        ),
        "base_test_probabilities": (
            base_test_probabilities
        ),
        "fusion_model": fusion_model,
        "fusion_test_inputs": (
            fusion_test_inputs
        ),
        "taaf_probabilities": (
            taaf_probabilities
        ),
        "model_paths": model_paths,
    }
