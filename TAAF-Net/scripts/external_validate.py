import argparse

from taafnet.external_validation import (
    run_external_validation,
)
from taafnet.reproducibility import (
    set_reproducibility,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--external-root",
        required=True,
    )
    parser.add_argument(
        "--model-dir",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
    )
    parser.add_argument(
        "--bootstraps",
        type=int,
        default=2000,
    )
    args = parser.parse_args()

    set_reproducibility()

    run_external_validation(
        external_root=args.external_root,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        n_bootstraps=args.bootstraps,
    )


if __name__ == "__main__":
    main()
