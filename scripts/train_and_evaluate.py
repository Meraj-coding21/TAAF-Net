import argparse

from taafnet.pipeline import (
    run_final_training,
)
from taafnet.reproducibility import (
    set_reproducibility,
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
        "--quiet",
        action="store_true",
    )
    args = parser.parse_args()

    set_reproducibility()

    run_final_training(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        verbose=0 if args.quiet else 1,
    )


if __name__ == "__main__":
    main()
