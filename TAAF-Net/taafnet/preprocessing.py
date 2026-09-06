from pathlib import Path
import hashlib
import json
import shutil

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .config import (
    AUGMENTATION_SETTINGS,
    CLASS_NAMES,
    FINAL_CLASS_COUNTS,
    IMAGE_SIZE,
    MAX_BRIGHTNESS,
    MIN_BRIGHTNESS,
    MIN_LAPLACIAN_VARIANCE,
    ROI_PADDING,
    ROI_THRESHOLD,
    SEED,
    TEST_CLASS_COUNTS,
    TRAIN_CLASS_COUNTS,
    VALID_EXTENSIONS,
)


def source_directories(data_root):
    data_root = Path(data_root)
    return {
        "Black_Spot": [
            data_root / "Processed (.PNG)" / "BlackSpot",
            data_root / "Processed Image" / "Black Spot",
        ],
        "Dry_Leaf": [
            data_root / "Processed (.PNG)" / "DryLeaf",
        ],
        "Healthy": [
            data_root / "Processed (.PNG)" / "HealthyLeaf",
            data_root / "Processed Image" / "Healthy Leaf",
        ],
        "Leaf_Holes": [
            data_root / "Processed (.PNG)" / "LeafHoles",
            data_root / "Processed Image" / "Insect Hole",
        ],
        "Yellow_Mosaic_Virus": [
            data_root / "Processed Image" / "Yellow Mosaic Virus",
        ],
    }


def list_images(folder):
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Dataset folder was not found: {folder}")

    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    )


def read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def extract_leaf_roi(
    image,
    threshold=ROI_THRESHOLD,
    padding=ROI_PADDING,
):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(
        gray,
        threshold,
        255,
        cv2.THRESH_BINARY_INV,
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return image.copy(), mask

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    x0 = max(x - padding, 0)
    y0 = max(y - padding, 0)
    x1 = min(x + w + padding, image.shape[1])
    y1 = min(y + h + padding, image.shape[0])

    return image[y0:y1, x0:x1], mask


def quality_metrics(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    brightness = float(np.mean(gray))
    laplacian_variance = float(
        cv2.Laplacian(gray, cv2.CV_64F).var()
    )

    brightness_pass = MIN_BRIGHTNESS <= brightness <= MAX_BRIGHTNESS
    sharpness_pass = laplacian_variance >= MIN_LAPLACIAN_VARIANCE

    return {
        "brightness": brightness,
        "laplacian_variance": laplacian_variance,
        "brightness_pass": brightness_pass,
        "sharpness_pass": sharpness_pass,
        "quality_pass": brightness_pass and sharpness_pass,
    }


def resize_lanczos4(image):
    return cv2.resize(
        image,
        IMAGE_SIZE,
        interpolation=cv2.INTER_LANCZOS4,
    )


def run_quality_screen(data_root, qa_root, report_root):
    data_root = Path(data_root)
    qa_root = Path(qa_root)
    report_root = Path(report_root)

    qa_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    source_dirs = source_directories(data_root)
    rows = []

    for class_name in CLASS_NAMES:
        class_output = qa_root / class_name
        class_output.mkdir(parents=True, exist_ok=True)

        source_files = []
        for folder_index, folder in enumerate(source_dirs[class_name]):
            for path in list_images(folder):
                source_files.append((folder_index, path))

        for folder_index, path in tqdm(
            source_files,
            desc=class_name,
        ):
            image = read_image(path)
            roi, _ = extract_leaf_roi(image)
            metrics = quality_metrics(roi)

            relative_path = str(path.relative_to(data_root))
            short_hash = hashlib.sha1(
                relative_path.encode("utf-8")
            ).hexdigest()[:10]
            output_name = f"s{folder_index}_{path.stem}_{short_hash}.png"
            output_path = class_output / output_name

            if metrics["quality_pass"]:
                resized = resize_lanczos4(roi)
                written = cv2.imwrite(
                    str(output_path),
                    resized,
                    [cv2.IMWRITE_PNG_COMPRESSION, 3],
                )
                if not written:
                    raise IOError(f"Could not write image: {output_path}")

            rows.append({
                "class_name": class_name,
                "source_path": relative_path,
                "quality_pass": metrics["quality_pass"],
                "brightness": metrics["brightness"],
                "laplacian_variance": metrics["laplacian_variance"],
                "processed_path": (
                    str(output_path) if metrics["quality_pass"] else ""
                ),
            })

    table = pd.DataFrame(rows)
    table.to_csv(report_root / "quality_screening.csv", index=False)
    return table


def select_curated_images(qa_root, seed=SEED):
    qa_root = Path(qa_root)
    rng = np.random.default_rng(seed)
    rows = []

    for class_name in CLASS_NAMES:
        candidates = sorted((qa_root / class_name).glob("*.png"))
        target = FINAL_CLASS_COUNTS[class_name]

        if len(candidates) < target:
            raise ValueError(
                f"{class_name}: {len(candidates)} accepted images "
                f"for a target of {target}"
            )

        selected_indices = rng.choice(
            len(candidates),
            size=target,
            replace=False,
        )

        for index in sorted(selected_indices):
            rows.append({
                "class_name": class_name,
                "processed_path": str(candidates[index]),
            })

    curated = pd.DataFrame(rows)

    if len(curated) != 6117:
        raise ValueError(f"Expected 6117 curated images, found {len(curated)}")

    return curated


def split_curated_images(curated_df, seed=SEED):
    parts = []

    for class_index, class_name in enumerate(CLASS_NAMES):
        class_df = curated_df.loc[
            curated_df["class_name"] == class_name
        ].copy()

        class_df = class_df.sample(
            frac=1.0,
            random_state=seed + class_index,
        ).reset_index(drop=True)

        train_count = TRAIN_CLASS_COUNTS[class_name]
        test_count = TEST_CLASS_COUNTS[class_name]

        if train_count + test_count != len(class_df):
            raise ValueError(
                f"{class_name}: split counts do not match curated count"
            )

        train_part = class_df.iloc[:train_count].copy()
        test_part = class_df.iloc[
            train_count:train_count + test_count
        ].copy()

        train_part["split"] = "train"
        test_part["split"] = "test"
        parts.extend([train_part, test_part])

    return pd.concat(parts, ignore_index=True)


def save_split_dataset(split_df, output_root, report_root):
    output_root = Path(output_root)
    report_root = Path(report_root)
    report_root.mkdir(parents=True, exist_ok=True)

    for split in ("train", "test"):
        for class_name in CLASS_NAMES:
            (output_root / split / class_name).mkdir(
                parents=True,
                exist_ok=True,
            )

    rows = []

    for row in tqdm(
        split_df.itertuples(index=False),
        total=len(split_df),
        desc="Saving dataset",
    ):
        source_path = Path(row.processed_path)
        destination = (
            output_root
            / row.split
            / row.class_name
            / source_path.name
        )
        shutil.copy2(source_path, destination)

        rows.append({
            "class_name": row.class_name,
            "split": row.split,
            "file_name": destination.name,
            "relative_path": str(destination.relative_to(output_root)),
        })

    manifest = pd.DataFrame(rows)
    manifest.to_csv(report_root / "dataset_manifest.csv", index=False)

    with open(
        report_root / "augmentation_settings.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(AUGMENTATION_SETTINGS, handle, indent=2)

    return manifest


def preprocess_source_dataset(
    data_root,
    work_root,
    overwrite=False,
):
    data_root = Path(data_root)
    work_root = Path(work_root)

    if work_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"{work_root} already exists. Use overwrite=True to replace it."
            )
        shutil.rmtree(work_root)

    qa_root = work_root / "qa_pass"
    output_root = work_root / "Rose_Leaf_Preprocessed_5Class"
    report_root = work_root / "reports"

    run_quality_screen(data_root, qa_root, report_root)
    curated = select_curated_images(qa_root)
    split_df = split_curated_images(curated)
    save_split_dataset(split_df, output_root, report_root)

    return output_root, report_root
