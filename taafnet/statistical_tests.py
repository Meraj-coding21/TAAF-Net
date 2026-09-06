import math

import numpy as np
from scipy.stats import binomtest, chi2


def cochran_q_test(correctness_matrix):
    matrix = np.asarray(
        correctness_matrix,
        dtype=np.int64,
    )

    if matrix.ndim != 2:
        raise ValueError(
            "correctness_matrix must be two-dimensional"
        )

    _, model_count = matrix.shape
    column_totals = matrix.sum(axis=0)
    row_totals = matrix.sum(axis=1)

    numerator = (
        (model_count - 1)
        * (
            model_count
            * np.sum(column_totals ** 2)
            - np.sum(column_totals) ** 2
        )
    )
    denominator = (
        model_count * np.sum(row_totals)
        - np.sum(row_totals ** 2)
    )

    if denominator == 0:
        return np.nan, np.nan

    statistic = numerator / denominator
    p_value = chi2.sf(
        statistic,
        model_count - 1,
    )

    return float(statistic), float(p_value)


def mcnemar_exact(
    y_true,
    pred_a,
    pred_b,
):
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)

    correct_a = pred_a == y_true
    correct_b = pred_b == y_true

    a_only = int(
        np.sum(correct_a & ~correct_b)
    )
    b_only = int(
        np.sum(~correct_a & correct_b)
    )
    discordant = a_only + b_only

    if discordant == 0:
        p_value = 1.0
    else:
        p_value = float(
            binomtest(
                min(a_only, b_only),
                n=discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )

    if a_only == 0 or b_only == 0:
        odds_ratio = (
            np.inf if b_only > a_only else 0.0
        )
        ci_low = np.nan
        ci_high = np.nan
    else:
        odds_ratio = b_only / a_only
        standard_error = math.sqrt(
            1 / a_only + 1 / b_only
        )
        ci_low = math.exp(
            math.log(odds_ratio)
            - 1.96 * standard_error
        )
        ci_high = math.exp(
            math.log(odds_ratio)
            + 1.96 * standard_error
        )

    return {
        "A_only_correct": a_only,
        "B_only_correct": b_only,
        "p_value": p_value,
        "odds_ratio_B_over_A": odds_ratio,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def pairwise_mcnemar_table(
    y_true,
    predictions,
    alpha=0.05,
):
    import pandas as pd

    names = list(predictions)
    comparisons = len(names) * (len(names) - 1) // 2
    threshold = alpha / comparisons
    rows = []

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            model_a = names[i]
            model_b = names[j]

            row = mcnemar_exact(
                y_true,
                predictions[model_a],
                predictions[model_b],
            )
            row["Model_A"] = model_a
            row["Model_B"] = model_b
            row["Bonferroni_Threshold"] = threshold
            row["Bonferroni_Significant"] = (
                row["p_value"] < threshold
            )
            rows.append(row)

    return pd.DataFrame(rows)
