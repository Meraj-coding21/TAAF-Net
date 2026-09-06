import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .config import (
    CI_SEED,
    CLASS_NAMES,
    N_BOOTSTRAPS,
    NUM_CLASSES,
)


def macro_specificity(y_true, y_pred, num_classes=None):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    if num_classes is None:
        num_classes = int(
            max(y_true.max(), y_pred.max()) + 1
        )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
    )

    values = []

    for class_id in range(num_classes):
        tp = cm[class_id, class_id]
        fn = cm[class_id, :].sum() - tp
        fp = cm[:, class_id].sum() - tp
        tn = cm.sum() - tp - fn - fp
        values.append(
            tn / (tn + fp) if (tn + fp) else 0.0
        )

    return float(np.mean(values))


def classification_metrics(
    y_true,
    probabilities,
    num_classes=None,
):
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a 2D array")

    if probabilities.shape[0] != len(y_true):
        raise ValueError(
            "probabilities and y_true have different sample counts"
        )

    if num_classes is None:
        num_classes = probabilities.shape[1]

    y_pred = probabilities.argmax(axis=1)

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=list(range(num_classes)),
            average="macro",
            zero_division=0,
        )
    )

    y_onehot = np.eye(
        num_classes,
        dtype=np.float32,
    )[y_true]

    try:
        auc_value = roc_auc_score(
            y_onehot,
            probabilities,
            average="macro",
            multi_class="ovr",
        )
    except ValueError:
        auc_value = np.nan

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Specificity": macro_specificity(
            y_true,
            y_pred,
            num_classes,
        ),
        "AUC": auc_value,
    }


def class_weight_dict(labels, num_classes=NUM_CLASSES):
    labels = np.asarray(labels, dtype=int)
    counts = np.bincount(
        labels,
        minlength=num_classes,
    )
    total = counts.sum()

    if np.any(counts == 0):
        raise ValueError(
            "Every class must occur in the training labels"
        )

    return {
        class_id: float(
            total / (num_classes * counts[class_id])
        )
        for class_id in range(num_classes)
    }


def sample_weights(labels, num_classes=NUM_CLASSES):
    weights = class_weight_dict(
        labels,
        num_classes=num_classes,
    )
    return np.asarray(
        [weights[int(label)] for label in labels],
        dtype=np.float32,
    )


def wilson_interval(correct, total):
    if total <= 0:
        return np.nan, np.nan

    z = 1.959963984540054
    phat = correct / total
    denom = 1.0 + z ** 2 / total
    center = (
        phat + z ** 2 / (2.0 * total)
    ) / denom

    half = (
        z
        * np.sqrt(
            phat * (1.0 - phat) / total
            + z ** 2 / (4.0 * total ** 2)
        )
        / denom
    )

    return float(center - half), float(center + half)


def stratified_bootstrap_indices(
    y_true,
    rng,
    num_classes,
):
    y_true = np.asarray(y_true, dtype=int)
    sampled = []

    for class_id in range(num_classes):
        members = np.flatnonzero(y_true == class_id)

        if len(members):
            sampled.append(
                rng.choice(
                    members,
                    size=len(members),
                    replace=True,
                )
            )

    indices = np.concatenate(sampled)
    rng.shuffle(indices)
    return indices


def metric_confidence_intervals(
    y_true,
    probabilities,
    num_classes=None,
    n_bootstraps=N_BOOTSTRAPS,
    seed=CI_SEED,
):
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    if num_classes is None:
        num_classes = probabilities.shape[1]

    point = classification_metrics(
        y_true,
        probabilities,
        num_classes=num_classes,
    )
    y_pred = probabilities.argmax(axis=1)

    accuracy_lower, accuracy_upper = wilson_interval(
        int((y_pred == y_true).sum()),
        len(y_true),
    )

    rng = np.random.default_rng(seed)
    bootstrap_values = {
        metric: []
        for metric in (
            "Precision",
            "Recall",
            "F1",
            "Specificity",
            "AUC",
        )
    }

    for _ in range(n_bootstraps):
        indices = stratified_bootstrap_indices(
            y_true,
            rng,
            num_classes,
        )
        values = classification_metrics(
            y_true[indices],
            probabilities[indices],
            num_classes=num_classes,
        )

        for metric in bootstrap_values:
            if np.isfinite(values[metric]):
                bootstrap_values[metric].append(
                    values[metric]
                )

    rows = [{
        "Metric": "Accuracy",
        "Estimate": point["Accuracy"],
        "CI_Lower": accuracy_lower,
        "CI_Upper": accuracy_upper,
        "CI_Method": "Wilson",
    }]

    for metric in (
        "Precision",
        "Recall",
        "F1",
        "Specificity",
        "AUC",
    ):
        values = np.asarray(
            bootstrap_values[metric],
            dtype=float,
        )
        lower, upper = np.quantile(
            values,
            [0.025, 0.975],
        )

        rows.append({
            "Metric": metric,
            "Estimate": point[metric],
            "CI_Lower": float(lower),
            "CI_Upper": float(upper),
            "CI_Method": (
                f"Stratified bootstrap (B={n_bootstraps})"
            ),
        })

    return pd.DataFrame(rows)


def classwise_f1_confidence_intervals(
    y_true,
    probabilities,
    class_names=CLASS_NAMES,
    n_bootstraps=N_BOOTSTRAPS,
    seed=CI_SEED,
):
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )
    num_classes = len(class_names)
    y_pred = probabilities.argmax(axis=1)

    point = f1_score(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        average=None,
        zero_division=0,
    )

    rng = np.random.default_rng(seed)
    boot = np.zeros(
        (n_bootstraps, num_classes),
        dtype=np.float32,
    )

    for index in range(n_bootstraps):
        sample = stratified_bootstrap_indices(
            y_true,
            rng,
            num_classes,
        )
        pred = probabilities[sample].argmax(axis=1)

        boot[index] = f1_score(
            y_true[sample],
            pred,
            labels=list(range(num_classes)),
            average=None,
            zero_division=0,
        )

    lower = np.quantile(boot, 0.025, axis=0)
    upper = np.quantile(boot, 0.975, axis=0)

    return pd.DataFrame({
        "Class": list(class_names),
        "F1": point,
        "CI_Lower": lower,
        "CI_Upper": upper,
        "N": [
            int((y_true == class_id).sum())
            for class_id in range(num_classes)
        ],
    })


def confusion_table(
    y_true,
    probabilities,
    class_names=CLASS_NAMES,
):
    predicted = np.asarray(probabilities).argmax(axis=1)
    return confusion_matrix(
        y_true,
        predicted,
        labels=list(range(len(class_names))),
    )
