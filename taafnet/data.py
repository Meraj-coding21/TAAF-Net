from pathlib import Path
import pandas as pd

from .config import (
    CLASS_NAMES,
    CLASS_TO_ID,
    TRAIN_CLASS_COUNTS,
    TEST_CLASS_COUNTS,
    VALID_EXTENSIONS,
)


def collect_split(split_dir, split_name):
    split_dir = Path(split_dir)
    rows = []

    for class_name in CLASS_NAMES:
        class_dir = split_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")

        files = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
        )

        for path in files:
            rows.append({
                "path": str(path),
                "class_name": class_name,
                "class_id": CLASS_TO_ID[class_name],
                "split": split_name,
            })

    return pd.DataFrame(rows)


def load_preprocessed_dataset(dataset_root, verify_counts=True):
    dataset_root = Path(dataset_root)
    train_df = collect_split(dataset_root / "train", "train")
    test_df = collect_split(dataset_root / "test", "test")

    if verify_counts:
        verify_source_counts(train_df, test_df)

    return train_df, test_df


def verify_source_counts(train_df, test_df):
    train_counts = (
        train_df["class_name"]
        .value_counts()
        .reindex(CLASS_NAMES)
        .fillna(0)
        .astype(int)
    )
    test_counts = (
        test_df["class_name"]
        .value_counts()
        .reindex(CLASS_NAMES)
        .fillna(0)
        .astype(int)
    )

    for class_name in CLASS_NAMES:
        expected_train = TRAIN_CLASS_COUNTS[class_name]
        expected_test = TEST_CLASS_COUNTS[class_name]

        if int(train_counts[class_name]) != expected_train:
            raise ValueError(
                f"{class_name}: expected {expected_train} training images, "
                f"found {int(train_counts[class_name])}"
            )

        if int(test_counts[class_name]) != expected_test:
            raise ValueError(
                f"{class_name}: expected {expected_test} test images, "
                f"found {int(test_counts[class_name])}"
            )

    if len(train_df) != 4893 or len(test_df) != 1224:
        raise ValueError(
            f"Expected 4893 training and 1224 test images, "
            f"found {len(train_df)} and {len(test_df)}"
        )


def dataset_summary(train_df, test_df):
    train_counts = train_df["class_name"].value_counts().reindex(CLASS_NAMES)
    test_counts = test_df["class_name"].value_counts().reindex(CLASS_NAMES)

    summary = pd.DataFrame({
        "Train": train_counts,
        "Test": test_counts,
    })
    summary["Total"] = summary.sum(axis=1)
    return summary.astype(int)
