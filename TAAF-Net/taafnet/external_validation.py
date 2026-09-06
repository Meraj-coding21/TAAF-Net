import gc
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .augmentation import (
    PREPROCESS_FUNCTIONS,
)
from .backbones import (
    BACKBONE_DIMS,
    BACKBONE_DISPLAY,
)
from .config import (
    BATCH_SIZE,
    CLASS_NAMES,
    CLASS_TO_ID,
    CI_SEED,
    EXTERNAL_CLASS_TO_ID,
    EXTERNAL_NUM_CLASSES,
    EXTERNAL_SHARED_CLASSES,
    EXTERNAL_SOURCE_IDS,
    FEWSHOT_BATCH_SIZE,
    FEWSHOT_EPOCHS,
    FEWSHOT_K_VALUES,
    FEWSHOT_LR,
    FEWSHOT_PRIMARY_SEED,
    FEWSHOT_SEEDS,
    FUSION_BATCH_SIZE,
    IMAGE_SIZE,
    SEED,
    VALID_EXTENSIONS,
)
from .evaluation import (
    classwise_f1_confidence_intervals,
    classification_metrics,
    metric_confidence_intervals,
)
from .fusion import fusion_input_list
from .texture import texture_matrix


EXTERNAL_FOLDER_TO_SOURCE_CLASS = {
    "blackspot": "Black_Spot",
    "blacksport": "Black_Spot",
    "dryleaf": "Dry_Leaf",
    "healthy": "Healthy",
    "healthyleaf": "Healthy",
    "insecthole": "Leaf_Holes",
    "leafhole": "Leaf_Holes",
    "leafholes": "Leaf_Holes",
}


def normalize_folder_name(name):
    return "".join(
        character
        for character in name.lower()
        if character.isalnum()
    )


def collect_external_split(
    split_dir,
    split_name,
):
    split_dir = Path(split_dir)
    rows = []
    discovered = []

    for folder in sorted(
        path
        for path in split_dir.iterdir()
        if path.is_dir()
    ):
        key = normalize_folder_name(
            folder.name
        )
        discovered.append(folder.name)

        if key not in EXTERNAL_FOLDER_TO_SOURCE_CLASS:
            continue

        source_class = (
            EXTERNAL_FOLDER_TO_SOURCE_CLASS[key]
        )

        files = sorted(
            path
            for path in folder.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower()
                in VALID_EXTENSIONS
            )
        )

        for path in files:
            rows.append({
                "path": str(path),
                "external_folder": folder.name,
                "class_name": source_class,
                "class_id": (
                    EXTERNAL_CLASS_TO_ID[
                        source_class
                    ]
                ),
                "source_class_id": (
                    CLASS_TO_ID[
                        source_class
                    ]
                ),
                "split": split_name,
            })

    frame = pd.DataFrame(rows)

    if frame.empty:
        raise RuntimeError(
            f"No shared RoseLeafVision images were found in "
            f"{split_dir}. Discovered folders: {discovered}"
        )

    return frame.reset_index(drop=True)


def load_roseleafvision(external_root):
    external_root = Path(external_root)
    train_dir = external_root / "Train"
    test_dir = external_root / "Test"

    if not train_dir.is_dir() or not test_dir.is_dir():
        raise FileNotFoundError(
            "RoseLeafVision must contain Train and Test directories"
        )

    train_df = collect_external_split(
        train_dir,
        "train",
    )
    test_df = collect_external_split(
        test_dir,
        "test",
    )

    return train_df, test_df


def make_external_generator(
    dataframe,
    model_name,
    batch_size=BATCH_SIZE,
):
    from tensorflow.keras.preprocessing.image import (
        ImageDataGenerator,
    )

    datagen = ImageDataGenerator(
        preprocessing_function=(
            PREPROCESS_FUNCTIONS[
                model_name
            ]
        )
    )

    return datagen.flow_from_dataframe(
        dataframe=dataframe,
        x_col="path",
        y_col="class_name",
        classes=list(CLASS_NAMES),
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
        seed=SEED,
        validate_filenames=True,
        interpolation="lanczos",
    )


def extract_external_embeddings(
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
        name=f"{name}_external_embedding_model",
    )
    generator = make_external_generator(
        dataframe,
        name,
    )
    embeddings = embedding_model.predict(
        generator,
        verbose=0,
    )

    expected = (
        len(dataframe),
        BACKBONE_DIMS[name],
    )

    if embeddings.shape != expected:
        raise ValueError(
            f"{name}: expected {expected}, "
            f"found {embeddings.shape}"
        )

    return embeddings.astype(np.float32)


def predict_external_backbone(
    model,
    name,
    dataframe,
):
    generator = make_external_generator(
        dataframe,
        name,
    )
    probabilities = model.predict(
        generator,
        verbose=0,
    )
    return probabilities.astype(np.float32)


def restrict_to_shared_external_labels(
    source_probabilities,
):
    shared = np.asarray(
        source_probabilities,
        dtype=np.float64,
    )[:, list(EXTERNAL_SOURCE_IDS)]

    denominator = shared.sum(
        axis=1,
        keepdims=True,
    )
    denominator = np.where(
        denominator > 0,
        denominator,
        1.0,
    )

    return (
        shared / denominator
    ).astype(np.float32)


def build_external_transfer_head(
    source_fusion_model,
):
    from tensorflow import keras
    from tensorflow.keras import layers

    representation_layer = (
        source_fusion_model
        .get_layer("meta_relu_256")
    )
    representation_dim = int(
        representation_layer
        .output
        .shape[-1]
    )

    inp = keras.Input(
        shape=(representation_dim,),
        name="taaf_transfer_representation",
    )
    logits = layers.Dense(
        EXTERNAL_NUM_CLASSES,
        name="external_logits",
    )(inp)
    probabilities = layers.Softmax(
        name="external_probabilities"
    )(logits)

    head = keras.Model(
        inp,
        probabilities,
        name="TAAF_FewShot_Head",
    )

    source_weights, source_bias = (
        source_fusion_model
        .get_layer("logits")
        .get_weights()
    )

    head.get_layer(
        "external_logits"
    ).set_weights([
        source_weights[
            :, list(EXTERNAL_SOURCE_IDS)
        ],
        source_bias[
            list(EXTERNAL_SOURCE_IDS)
        ],
    ])

    return head


def sample_k_shot_support(
    dataframe,
    k,
    seed,
):
    parts = []

    for class_name in EXTERNAL_SHARED_CLASSES:
        class_df = dataframe.loc[
            dataframe["class_name"]
            == class_name
        ]

        if len(class_df) < k:
            raise ValueError(
                f"{class_name}: only {len(class_df)} "
                f"samples are available for {k}-shot"
            )

        parts.append(
            class_df.sample(
                n=k,
                replace=False,
                random_state=seed,
            )
        )

    return (
        pd.concat(parts)
        .sort_index()
    )


def run_external_validation(
    external_root,
    model_dir,
    output_dir,
    n_bootstraps=2000,
):
    import tensorflow as tf
    from tensorflow import keras

    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_df, test_df = (
        load_roseleafvision(
            external_root
        )
    )

    embeddings = {
        "train": {},
        "test": {},
    }
    base_probabilities = {}

    model_paths = {
        "efficientnet": (
            model_dir
            / "efficientnet_final.keras"
        ),
        "densenet": (
            model_dir
            / "densenet_final.keras"
        ),
        "convnext": (
            model_dir
            / "convnext_final.keras"
        ),
    }

    for name, path in model_paths.items():
        model = keras.models.load_model(
            path,
            compile=False,
        )

        embeddings["train"][name] = (
            extract_external_embeddings(
                model,
                name,
                train_df,
            )
        )
        embeddings["test"][name] = (
            extract_external_embeddings(
                model,
                name,
                test_df,
            )
        )
        base_probabilities[name] = (
            predict_external_backbone(
                model,
                name,
                test_df,
            )
        )

        del model
        gc.collect()

    train_texture = texture_matrix(
        train_df,
        "External train texture",
    )
    test_texture = texture_matrix(
        test_df,
        "External test texture",
    )

    fusion_model = keras.models.load_model(
        model_dir / "taaf_net_final.keras",
        compile=False,
    )

    train_inputs = fusion_input_list(
        embeddings["train"],
        train_texture,
        ("efficientnet", "densenet", "convnext"),
    )
    test_inputs = fusion_input_list(
        embeddings["test"],
        test_texture,
        ("efficientnet", "densenet", "convnext"),
    )

    y_test = test_df[
        "class_id"
    ].to_numpy(dtype=int)

    zero_shot = {
        BACKBONE_DISPLAY[name]:
            restrict_to_shared_external_labels(
                base_probabilities[name]
            )
        for name in (
            "convnext",
            "efficientnet",
            "densenet",
        )
    }

    source_probs = fusion_model.predict(
        test_inputs,
        batch_size=FUSION_BATCH_SIZE,
        verbose=0,
    )
    zero_shot["TAAF-Net"] = (
        restrict_to_shared_external_labels(
            source_probs
        )
    )

    metric_frames = []
    class_frames = []

    for index, (
        model_name,
        probabilities,
    ) in enumerate(zero_shot.items()):
        metrics = metric_confidence_intervals(
            y_test,
            probabilities,
            num_classes=EXTERNAL_NUM_CLASSES,
            n_bootstraps=n_bootstraps,
            seed=CI_SEED + 1000 + index,
        )
        metrics.insert(
            0,
            "Model",
            model_name,
        )
        metrics.insert(
            0,
            "Protocol",
            "Zero-shot",
        )
        metric_frames.append(metrics)

        class_table = (
            classwise_f1_confidence_intervals(
                y_test,
                probabilities,
                class_names=EXTERNAL_SHARED_CLASSES,
                n_bootstraps=n_bootstraps,
                seed=CI_SEED + 1100 + index,
            )
        )
        class_table.insert(
            0,
            "Model",
            model_name,
        )
        class_table.insert(
            0,
            "Protocol",
            "Zero-shot",
        )
        class_frames.append(class_table)

    zero_metrics = pd.concat(
        metric_frames,
        ignore_index=True,
    )
    zero_class_f1 = pd.concat(
        class_frames,
        ignore_index=True,
    )

    zero_metrics.to_csv(
        output_dir
        / "external_zero_shot_metrics_95ci.csv",
        index=False,
    )
    zero_class_f1.to_csv(
        output_dir
        / "external_zero_shot_classwise_f1_95ci.csv",
        index=False,
    )

    transfer_encoder = keras.Model(
        fusion_model.inputs,
        fusion_model
        .get_layer("meta_relu_256")
        .output,
        name="TAAF_Frozen_Transfer_Encoder",
    )
    transfer_encoder.trainable = False

    train_representation = (
        transfer_encoder.predict(
            train_inputs,
            batch_size=FUSION_BATCH_SIZE,
            verbose=0,
        )
        .astype(np.float32)
    )
    test_representation = (
        transfer_encoder.predict(
            test_inputs,
            batch_size=FUSION_BATCH_SIZE,
            verbose=0,
        )
        .astype(np.float32)
    )

    representation_lookup = {
        index: train_representation[index]
        for index in range(len(train_df))
    }

    fewshot_rows = []

    for k in FEWSHOT_K_VALUES:
        for seed in FEWSHOT_SEEDS:
            tf.keras.utils.set_random_seed(
                seed
            )

            support = sample_k_shot_support(
                train_df,
                k,
                seed,
            )
            support_indices = (
                support.index.tolist()
            )
            x_support = np.stack([
                representation_lookup[index]
                for index in support_indices
            ]).astype(np.float32)
            y_support = support[
                "class_id"
            ].to_numpy(dtype=int)

            head = (
                build_external_transfer_head(
                    fusion_model
                )
            )
            head.compile(
                optimizer=keras.optimizers.Adam(
                    FEWSHOT_LR
                ),
                loss=(
                    "sparse_categorical_crossentropy"
                ),
                metrics=["accuracy"],
            )
            head.fit(
                x_support,
                y_support,
                epochs=FEWSHOT_EPOCHS,
                batch_size=min(
                    FEWSHOT_BATCH_SIZE,
                    len(x_support),
                ),
                shuffle=True,
                callbacks=[
                    keras.callbacks.TerminateOnNaN(),
                    keras.callbacks.EarlyStopping(
                        monitor="loss",
                        mode="min",
                        patience=6,
                        min_delta=1e-4,
                        restore_best_weights=True,
                    ),
                ],
                verbose=0,
            )

            probabilities = head.predict(
                test_representation,
                batch_size=FUSION_BATCH_SIZE,
                verbose=0,
            )

            values = classification_metrics(
                y_test,
                probabilities,
                num_classes=EXTERNAL_NUM_CLASSES,
            )
            fewshot_rows.append({
                "K": k,
                "Seed": seed,
                **values,
            })

            if seed == FEWSHOT_PRIMARY_SEED:
                ci_table = metric_confidence_intervals(
                    y_test,
                    probabilities,
                    num_classes=EXTERNAL_NUM_CLASSES,
                    n_bootstraps=n_bootstraps,
                    seed=CI_SEED + 2000 + k,
                )
                ci_table.to_csv(
                    output_dir
                    / f"external_{k}shot_primary_metrics_95ci.csv",
                    index=False,
                )

            del head
            gc.collect()

    fewshot = pd.DataFrame(
        fewshot_rows
    )
    fewshot.to_csv(
        output_dir
        / "external_fewshot_all_seeds_metrics.csv",
        index=False,
    )

    return {
        "zero_shot_metrics": zero_metrics,
        "zero_shot_class_f1": zero_class_f1,
        "fewshot_metrics": fewshot,
        "external_train": train_df,
        "external_test": test_df,
    }
