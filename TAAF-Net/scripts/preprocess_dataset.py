import argparse

from taafnet.preprocessing import (
    preprocess_source_dataset,
)
from taafnet.reproducibility import (
    set_reproducibility,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        required=True,
    )
    parser.add_argument(
        "--work-root",
        required=True,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    args = parser.parse_args()

    set_reproducibility()

    dataset_root, report_root = (
        preprocess_source_dataset(
            data_root=args.data_root,
            work_root=args.work_root,
            overwrite=args.overwrite,
        )
    )

    print("Dataset:", dataset_root)
    print("Reports:", report_root)


if __name__ == "__main__":
    main()
