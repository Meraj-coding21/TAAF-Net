from pathlib import Path
import ast
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    failures = []

    for path in sorted(root.rglob("*.py")):
        try:
            ast.parse(
                path.read_text(
                    encoding="utf-8"
                )
            )
        except SyntaxError as error:
            failures.append(
                f"{path.relative_to(root)}: {error}"
            )

    text_files = [
        path
        for path in root.rglob("*")
        if (
            path.is_file()
            and path.suffix
            in {
                ".py",
                ".md",
                ".txt",
                ".toml",
                ".cff",
            }
        )
    ]

    for path in text_files:
        text = path.read_text(
            encoding="utf-8"
        )

        if "\u2014" in text:
            failures.append(
                f"{path.relative_to(root)} contains an em dash"
            )

    required_modules = {
        "config.py",
        "data.py",
        "preprocessing.py",
        "augmentation.py",
        "texture.py",
        "backbones.py",
        "fusion.py",
        "training.py",
        "evaluation.py",
        "statistical_tests.py",
        "cross_validation.py",
        "ablation.py",
        "explainability.py",
        "efficiency.py",
        "external_validation.py",
        "pipeline.py",
    }

    found = {
        path.name
        for path in (
            root / "taafnet"
        ).glob("*.py")
    }

    missing = required_modules - found

    if missing:
        failures.append(
            "Missing modules: "
            + ", ".join(sorted(missing))
        )

    if failures:
        print("\n".join(failures))
        sys.exit(1)

    print("Repository checks passed.")


if __name__ == "__main__":
    main()
