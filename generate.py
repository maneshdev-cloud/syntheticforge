"""Synthetic tabular-data generators.

The default generator uses a Gaussian copula for numeric columns so linear
correlations in the source data have a chance to survive in the synthetic
data.  A marginal sampler is also included as a transparent baseline.
Categorical columns are sampled from their observed frequencies.

Only NumPy, pandas and SciPy are required.  SDV is optional: if it is
installed, :func:`generate_with_sdv` can be used, but no part of the normal
workflow depends on it.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import norm

Method = Literal["copula", "marginal"]


def _column_kinds(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [column for column in df.columns if column not in numeric]
    return numeric, categorical


def _sample_categorical(series: pd.Series, n_rows: int, rng: np.random.Generator) -> np.ndarray:
    values, counts = np.unique(series.dropna().astype(str), return_counts=True)
    probabilities = counts / counts.sum()
    return rng.choice(values, size=n_rows, p=probabilities)


def _sample_marginals(df: pd.DataFrame, n_rows: int, rng: np.random.Generator) -> pd.DataFrame:
    numeric, categorical = _column_kinds(df)
    output: dict[str, np.ndarray] = {}

    for column in numeric:
        values = df[column].dropna().to_numpy(dtype=float)
        output[column] = rng.choice(values, size=n_rows, replace=True)

    for column in categorical:
        output[column] = _sample_categorical(df[column], n_rows, rng)

    return pd.DataFrame(output, columns=df.columns)


def _empirical_inverse_cdf(samples: np.ndarray, sorted_values: np.ndarray) -> np.ndarray:
    """Map uniform samples through an empirical quantile function."""

    if len(sorted_values) == 1:
        return np.full(len(samples), sorted_values[0], dtype=float)
    positions = samples * (len(sorted_values) - 1)
    lower = np.floor(positions).astype(int)
    upper = np.ceil(positions).astype(int)
    fraction = positions - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def _sample_gaussian_copula(df: pd.DataFrame, n_rows: int, rng: np.random.Generator) -> pd.DataFrame:
    numeric, categorical = _column_kinds(df)
    output: dict[str, np.ndarray] = {}

    if numeric:
        clean = df[numeric].apply(lambda col: col.fillna(col.median()))
        ranks = clean.rank(method="average").to_numpy(dtype=float)
        uniforms = ranks / (len(clean) + 1.0)
        uniforms = np.clip(uniforms, 1e-6, 1 - 1e-6)
        latent = norm.ppf(uniforms)

        correlation = np.corrcoef(latent, rowvar=False)
        correlation = np.atleast_2d(correlation)
        correlation = np.nan_to_num(correlation, nan=0.0)
        np.fill_diagonal(correlation, 1.0)

        # A tiny diagonal jitter keeps the matrix usable when columns are
        # nearly collinear.
        correlation += np.eye(correlation.shape[0]) * 1e-8
        eigenvalues, eigenvectors = np.linalg.eigh(correlation)
        eigenvalues = np.clip(eigenvalues, 1e-10, None)
        psd_correlation = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        d = np.sqrt(np.diag(psd_correlation))
        psd_correlation = psd_correlation / np.outer(d, d)

        latent_samples = rng.multivariate_normal(
            mean=np.zeros(psd_correlation.shape[0]),
            cov=psd_correlation,
            size=n_rows,
        )
        synthetic_uniforms = norm.cdf(latent_samples)

        for index, column in enumerate(numeric):
            sorted_values = np.sort(clean[column].to_numpy(dtype=float))
            output[column] = _empirical_inverse_cdf(synthetic_uniforms[:, index], sorted_values)

    for column in categorical:
        output[column] = _sample_categorical(df[column], n_rows, rng)

    result = pd.DataFrame(output, columns=df.columns)

    # Restore obvious integer-valued source columns as integers.
    for column in numeric:
        values = df[column].dropna().to_numpy(dtype=float)
        if len(values) and np.allclose(values, np.round(values)):
            result[column] = np.round(result[column]).astype(int)

    return result


def generate_synthetic(
    df: pd.DataFrame,
    n_rows: int | None = None,
    method: Method = "copula",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate ``n_rows`` synthetic rows from a pandas DataFrame."""

    if df.empty:
        raise ValueError("Cannot generate synthetic data from an empty DataFrame")
    if n_rows is None:
        n_rows = len(df)
    if n_rows <= 0:
        raise ValueError("n_rows must be positive")

    rng = np.random.default_rng(seed)
    if method == "marginal":
        return _sample_marginals(df, n_rows, rng)
    if method == "copula":
        return _sample_gaussian_copula(df, n_rows, rng)
    raise ValueError(f"Unknown method: {method}")


def generate_with_sdv(df: pd.DataFrame, n_rows: int, seed: int = 42) -> pd.DataFrame:
    """Optional SDV path.

    This function deliberately raises a clear error when SDV is absent.  The
    Gaussian-copula and marginal generators above remain fully usable.
    """

    try:
        from sdv.metadata import SingleTableMetadata
        from sdv.single_table import GaussianCopulaSynthesizer
    except ImportError as exc:
        raise ImportError("SDV is not installed; use method='copula' or 'marginal'") from exc

    metadata = SingleTableMetadata.load_from_dataframe(df)
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(df)
    return synthesizer.sample(num_rows=n_rows)
