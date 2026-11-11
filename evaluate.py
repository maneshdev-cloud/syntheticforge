"""Similarity and privacy metrics for synthetic tabular data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import ks_2samp


def _numeric_columns(real: pd.DataFrame, synthetic: pd.DataFrame) -> list[str]:
    return [
        column
        for column in real.columns
        if column in synthetic.columns
        and pd.api.types.is_numeric_dtype(real[column])
        and pd.api.types.is_numeric_dtype(synthetic[column])
    ]


def _categorical_columns(real: pd.DataFrame, synthetic: pd.DataFrame) -> list[str]:
    numeric = set(_numeric_columns(real, synthetic))
    return [column for column in real.columns if column in synthetic.columns and column not in numeric]


def _total_variation_distance(real: pd.Series, synthetic: pd.Series) -> float:
    real_counts = real.astype(str).value_counts(normalize=True)
    synthetic_counts = synthetic.astype(str).value_counts(normalize=True)
    categories = real_counts.index.union(synthetic_counts.index)
    real_probs = real_counts.reindex(categories, fill_value=0.0).to_numpy()
    synthetic_probs = synthetic_counts.reindex(categories, fill_value=0.0).to_numpy()
    return float(0.5 * np.abs(real_probs - synthetic_probs).sum())


def _encoded_matrix(real: pd.DataFrame, synthetic: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    numeric = _numeric_columns(real, synthetic)
    categorical = _categorical_columns(real, synthetic)
    real_parts: list[np.ndarray] = []
    synthetic_parts: list[np.ndarray] = []

    for column in numeric:
        real_values = real[column].astype(float)
        scale = float(real_values.std(ddof=0)) or 1.0
        centre = float(real_values.mean())
        real_parts.append(((real_values - centre) / scale).to_numpy().reshape(-1, 1))
        synthetic_parts.append(
            ((synthetic[column].astype(float) - centre) / scale).to_numpy().reshape(-1, 1)
        )

    for column in categorical:
        categories = sorted(set(real[column].astype(str)) | set(synthetic[column].astype(str)))
        category_index = {category: i for i, category in enumerate(categories)}
        real_one_hot = np.zeros((len(real), len(categories)), dtype=float)
        synthetic_one_hot = np.zeros((len(synthetic), len(categories)), dtype=float)
        for row, value in enumerate(real[column].astype(str)):
            real_one_hot[row, category_index[value]] = 1.0
        for row, value in enumerate(synthetic[column].astype(str)):
            synthetic_one_hot[row, category_index[value]] = 1.0
        real_parts.append(real_one_hot)
        synthetic_parts.append(synthetic_one_hot)

    if not real_parts:
        return np.empty((len(real), 0)), np.empty((len(synthetic), 0))
    return np.hstack(real_parts), np.hstack(synthetic_parts)


def _exact_match_rate(real: pd.DataFrame, synthetic: pd.DataFrame) -> float:
    columns = list(real.columns)
    real_rows = {tuple(row) for row in real[columns].astype(str).itertuples(index=False, name=None)}
    synthetic_rows = list(synthetic[columns].astype(str).itertuples(index=False, name=None))
    if not synthetic_rows:
        return 0.0
    return float(sum(row in real_rows for row in synthetic_rows) / len(synthetic_rows))


def _nearest_neighbour_summary(real: pd.DataFrame, synthetic: pd.DataFrame) -> dict[str, float | None]:
    real_matrix, synthetic_matrix = _encoded_matrix(real, synthetic)
    if real_matrix.shape[1] == 0 or len(real_matrix) < 2 or len(synthetic_matrix) == 0:
        return {
            "median_real_to_real_distance": None,
            "median_synthetic_to_real_distance": None,
            "distance_ratio": None,
        }

    real_tree = cKDTree(real_matrix)
    real_distances, _ = real_tree.query(real_matrix, k=2)
    synthetic_distances, _ = real_tree.query(synthetic_matrix, k=1)

    median_real = float(np.median(real_distances[:, 1]))
    median_synthetic = float(np.median(synthetic_distances))
    ratio = float(median_synthetic / median_real) if median_real > 0 else None
    return {
        "median_real_to_real_distance": median_real,
        "median_synthetic_to_real_distance": median_synthetic,
        "distance_ratio": ratio,
    }


def evaluate_synthetic(real: pd.DataFrame, synthetic: pd.DataFrame) -> dict:
    """Return JSON-serialisable similarity and privacy metrics."""

    if real.empty or synthetic.empty:
        raise ValueError("Both real and synthetic datasets must contain rows")

    numeric = _numeric_columns(real, synthetic)
    categorical = _categorical_columns(real, synthetic)

    ks_statistics = {
        column: float(ks_2samp(real[column].dropna(), synthetic[column].dropna()).statistic)
        for column in numeric
    }
    tvd = {
        column: _total_variation_distance(real[column], synthetic[column])
        for column in categorical
    }

    if numeric:
        real_corr = real[numeric].corr(numeric_only=True).fillna(0.0)
        synthetic_corr = synthetic[numeric].corr(numeric_only=True).fillna(0.0)
        difference = (real_corr - synthetic_corr).to_numpy(dtype=float)
        correlation_summary = {
            "frobenius_difference": float(np.linalg.norm(difference, ord="fro")),
            "mean_absolute_difference": float(np.mean(np.abs(difference))),
            "max_absolute_difference": float(np.max(np.abs(difference))),
        }
    else:
        correlation_summary = {
            "frobenius_difference": None,
            "mean_absolute_difference": None,
            "max_absolute_difference": None,
        }

    return {
        "rows": {"real": int(len(real)), "synthetic": int(len(synthetic))},
        "similarity": {
            "numeric_ks_statistic": ks_statistics,
            "categorical_total_variation_distance": tvd,
            "correlation_matrix_difference": correlation_summary,
        },
        "privacy": {
            "exact_match_rate": _exact_match_rate(real, synthetic),
            "nearest_neighbour": _nearest_neighbour_summary(real, synthetic),
        },
        "interpretation": {
            "ks_statistic": "Lower is more similar; 0 means identical empirical distributions.",
            "total_variation_distance": "Lower is more similar; 0 means identical category frequencies.",
            "exact_match_rate": "Lower means fewer synthetic rows copied exactly from the source.",
            "nearest_neighbour_distance_ratio": (
                "Synthetic-to-real median nearest-neighbour distance divided by the "
                "real-to-real equivalent. Values near or below 1 deserve privacy review."
            ),
        },
    }
