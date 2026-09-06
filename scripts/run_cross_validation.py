import argparse
from pathlib import Path

import pandas as pd

from taafnet.config import OutputPaths
from taafnet.cross_validation import (
    run_cross_validation,
    summarize_cross_validation,
)
from taafnet.data import (
    load_preprocessed_dataset,
)
from taafnet.reproducibility import (
    set_reproducibility,
)
from taafnet.texture import (
    build_texture_cache,
    texture_lookup,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        required=True,
    )
    parser.add_argument(
        "--output-root",
        required=True,
    )
    parser.add_argument(
        "--save-models",
        action="store_true",
    )
    args = parser.parse_args()

    set_reproducibility()

    paths = OutputPaths(
        Path(args.output_root)
    ).create()

    train_df, test_df = (
        load_preprocessed_dataset(
            args.dataset_root,
            verify_counts=True,
        )
    )

    all_df = pd.concat(
        [train_df, test_df],
        ignore_index=True,
    )
    features = build_texture_cache(
        all_df,
        paths.cache / "texture_features.npz",
    )
    lookup = texture_lookup(
        all_df,
        features,
    )

    results = run_cross_validation(
        train_df,
        lookup,
        paths.cross_validation,
        save_models=args.save_models,
    )

    summary = summarize_cross_validation(
        results
    )
    summary.to_csv(
        paths.results
        / "cross_validation_summary.csv"
    )
    print(summary)


if __name__ == "__main__":
    main()
