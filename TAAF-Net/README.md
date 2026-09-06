# TAAF-Net

Code for the experiments reported in **TAAF-Net: A Texture-Aware Attention Fusion Network for Fine-Grained Rose Disease Classification with Multi-Faceted Explainability**.

The repository separates the main parts of the method into Python modules. A demonstration notebook can be added to `demo_notebook/`.

## Repository structure

```text
TAAF-Net/
├── taafnet/
│   ├── config.py
│   ├── reproducibility.py
│   ├── data.py
│   ├── preprocessing.py
│   ├── augmentation.py
│   ├── texture.py
│   ├── backbones.py
│   ├── fusion.py
│   ├── training.py
│   ├── evaluation.py
│   ├── statistical_tests.py
│   ├── cross_validation.py
│   ├── ablation.py
│   ├── explainability.py
│   ├── efficiency.py
│   ├── external_validation.py
│   └── pipeline.py
├── scripts/
│   ├── preprocess_dataset.py
│   ├── train_and_evaluate.py
│   ├── run_cross_validation.py
│   ├── external_validate.py
│   └── check_repository.py
├── tests/
├── demo_notebook/
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

## Datasets

Three public datasets are referenced.

| Dataset | Use | Link |
| --- | --- | --- |
| RoseLeafSet, version 3 | Source data | https://data.mendeley.com/datasets/9g668bfhy5/3 |
| RoseLeafInsight, version 2 | Source data | https://data.mendeley.com/datasets/8chrjdxn79/2 |
| RoseLeafVision, version 1 | External validation | https://data.mendeley.com/datasets/yt96x47dc7/1 |

The datasets are not redistributed in this repository.

## Environment

The released configuration follows the software versions used for the paper workflow where applicable.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

The code is intended for Python 3.10 or 3.11 with a TensorFlow-compatible GPU environment.

## Source preprocessing

The source preprocessing module reproduces the following settings:

- ROI threshold: 240
- ROI padding: 5 pixels
- minimum Laplacian variance: 10
- accepted mean brightness: 20 to 250
- output size: 224 x 224
- Lanczos4 interpolation

Run:

```bash
python scripts/preprocess_dataset.py \
  --data-root /path/to/raw/source/data \
  --work-root /path/to/preprocessing_output
```

Use `--overwrite` only when the target work directory can be replaced.

The prepared dataset contains:

| Class | Train | Test | Total |
| --- | ---: | ---: | ---: |
| Black Spot | 1,188 | 298 | 1,486 |
| Dry Leaf | 258 | 65 | 323 |
| Healthy | 1,999 | 498 | 2,497 |
| Leaf Holes | 904 | 227 | 1,131 |
| Yellow Mosaic Virus | 544 | 136 | 680 |
| **Total** | **4,893** | **1,224** | **6,117** |

## Final training and independent test evaluation

Run:

```bash
python scripts/train_and_evaluate.py \
  --dataset-root /path/to/Rose_Leaf_Preprocessed_5Class \
  --output-root /path/to/run_output
```

The script trains the three backbones, extracts their embeddings, computes the 26-dimensional LBP and GLCM descriptor, trains the TAAF-Net fusion model, and evaluates the held-out test set.

The three deep embeddings have dimensions 1,280, 1,664, and 768. Each deep branch is projected to 256 dimensions. The texture vector is projected from 26 to 64 dimensions. The resulting 832-dimensional representation is gated with independent sigmoid activations before classification.

## Cross-validation

Five-fold stratified cross-validation is run only on the 4,893-image training partition.

```bash
python scripts/run_cross_validation.py \
  --dataset-root /path/to/Rose_Leaf_Preprocessed_5Class \
  --output-root /path/to/cv_output
```

## External validation

RoseLeafVision is used as a separate external dataset. Only the four labels shared with the source task are used:

- Black Spot
- Dry Leaf
- Healthy
- Leaf Holes or Insect Hole

Downy Mildew is excluded because it is not present in the source five-class label space.

The RoseLeafVision evaluation code does not apply the source blur threshold, brightness threshold, or quality-based sample rejection. Its `Train` partition is used only for few-shot support samples. Its `Test` partition is kept separate for evaluation.

```bash
python scripts/external_validate.py \
  --external-root /path/to/Original_Rose_leaf_Disease_Dataset \
  --model-dir /path/to/run_output/models \
  --output-dir /path/to/external_output
```

The module evaluates zero-shot transfer for the three backbones and TAAF-Net, followed by 5-shot, 10-shot, and 20-shot head adaptation using the predefined seeds.

## Statistical analysis

`taafnet/evaluation.py` contains:

- accuracy, macro precision, macro recall, macro F1, macro specificity, and one-vs-rest macro AUC
- Wilson 95% confidence intervals for accuracy
- stratified bootstrap 95% confidence intervals for the remaining aggregate metrics
- class-wise F1 confidence intervals

`taafnet/statistical_tests.py` contains Cochran's Q and exact pairwise McNemar tests with Bonferroni correction support.

## Explainability

`taafnet/explainability.py` contains:

- LBP and GLCM visualization helpers
- t-SNE projection
- class-wise fusion gate analysis with bootstrap confidence intervals
- first-layer kernel access
- Grad-CAM
- pairwise Grad-CAM IoU

The sigmoid gate values are treated as gate activations. They are not normalized contribution percentages.

## Reported key results

### Independent source test set

| Model | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| ConvNeXt-Tiny | 97.79% | 0.979 | 0.975 | 0.977 |
| EfficientNetV2-M | 97.96% | 0.980 | 0.976 | 0.976 |
| DenseNet169 | 98.37% | 0.983 | 0.982 | 0.982 |
| TAAF-Net | **99.59%** | **0.996** | **0.992** | **0.994** |

TAAF-Net classified 1,219 of 1,224 source test images correctly. The Wilson 95% confidence interval for accuracy is approximately 99.05% to 99.83%.

### Five-fold cross-validation on the training partition

| Model | Accuracy |
| --- | ---: |
| ConvNeXt-Tiny | 98.40% ± 0.50% |
| EfficientNetV2-M | 98.88% ± 0.14% |
| DenseNet169 | 99.06% ± 0.23% |
| TAAF-Net | **99.35% ± 0.12%** |

### External RoseLeafVision zero-shot result

For TAAF-Net, the four-class zero-shot external evaluation used 427 held-out RoseLeafVision test images and obtained 91.80% accuracy. The Wilson 95% confidence interval was 88.81% to 94.05%.

The external script recomputes all metrics from per-image model probabilities.

## Demo notebook

`demo_notebook/` is intentionally left empty except for the file required to keep the directory in Git. A demonstration notebook can be added there without changing the package modules.

## Repository check

```bash
python scripts/check_repository.py
python -m pytest -q
```

The repository does not include trained weights, datasets, generated figures, or generated result files. These are produced during execution.
