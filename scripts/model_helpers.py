"""Helpers for modeling notebooks (genre encoding, cleaning, metrics)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import normalize

NO_GENRE = "(no genres listed)"


def split_genre_string(genres: str) -> list[str]:
    if pd.isna(genres) or not str(genres).strip():
        return []
    return [g for g in str(genres).split("|") if g and g != NO_GENRE]


def genre_vocabulary(genres: pd.Series) -> list[str]:
    vocab: set[str] = set()
    for raw in genres.dropna():
        vocab.update(split_genre_string(raw))
    return sorted(vocab)


def encode_genres_multihot(
    genres: pd.Series,
    vocabulary: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Multi-hot encode pipe-separated MovieLens genres (multi-label, not single-label)."""
    if vocabulary is None:
        vocabulary = genre_vocabulary(genres)

    matrix = pd.DataFrame(0, index=genres.index, columns=vocabulary, dtype=np.float32)
    for idx, raw in genres.items():
        for genre in split_genre_string(raw):
            if genre in matrix.columns:
                matrix.at[idx, genre] = 1.0
    return matrix, vocabulary


def build_movie_content_features(
    movies: pd.DataFrame,
    *,
    genre_col: str = "genres",
    vocabulary: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Genre multi-hot + normalized release year → L2-normalized movie feature matrix."""
    genre_matrix, vocabulary = encode_genres_multihot(movies[genre_col], vocabulary)
    year = pd.to_numeric(movies["release_year"], errors="coerce")
    year = year.fillna(year.median())
    year_norm = ((year - year.mean()) / (year.std() + 1e-8)).astype(np.float32)

    features = np.hstack([genre_matrix.to_numpy(), year_norm.to_numpy().reshape(-1, 1)])
    features = normalize(features, norm="l2", axis=1)
    return features, vocabulary


def rmse_mae(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return rmse, mae


def audit_parquet_missing(
    path: Path,
    *,
    check_cols: tuple[str, ...] = (
        "userId",
        "movieId",
        "rating",
        "timestamp",
        "genres",
        "release_year",
        "director",
        "cast",
    ),
    batch_size: int = 2_000_000,
) -> dict:
    """Count missing values per column by scanning parquet in batches."""
    pf = pq.ParquetFile(path)
    available = [c for c in check_cols if c in pf.schema.names]
    na_counts: Counter[str] = Counter()

    for batch in pf.iter_batches(batch_size=batch_size, columns=available):
        df = batch.to_pandas()
        for col in available:
            na_counts[col] += int(df[col].isna().sum())

    return {"rows": pf.metadata.num_rows, "na": dict(na_counts)}


def clean_ratings_df(
    df: pd.DataFrame,
    *,
    required_cols: tuple[str, ...] = (
        "userId",
        "movieId",
        "rating",
        "genres",
        "release_year",
    ),
) -> tuple[pd.DataFrame, dict]:
    """Drop rows with missing required fields and duplicate user–movie pairs."""
    n_start = len(df)
    missing_mask = df[list(required_cols)].isna().any(axis=1)
    n_na = int(missing_mask.sum())
    out = df.loc[~missing_mask].copy()
    n_before_dedup = len(out)
    out = out.drop_duplicates(subset=["userId", "movieId"], keep="last").reset_index(drop=True)
    stats = {
        "rows_in": n_start,
        "rows_out": len(out),
        "dropped_na": n_na,
        "dropped_duplicates": n_before_dedup - len(out),
    }
    return out, stats


def assert_clean_ratings(
    df: pd.DataFrame,
    *,
    required_cols: tuple[str, ...] = (
        "userId",
        "movieId",
        "rating",
        "genres",
        "release_year",
    ),
) -> None:
    """Verify no missing required values and no duplicate user–movie keys."""
    na = df[list(required_cols)].isna().sum()
    if na.sum():
        raise ValueError(f"NA remaining after clean:\n{na[na > 0]}")
    dup = df.duplicated(subset=["userId", "movieId"]).sum()
    if dup:
        raise ValueError(f"{dup} duplicate user–movie rows remain after clean")


REQUIRED_RATING_COLS = ("userId", "movieId", "rating", "genres", "release_year")


def load_clean_parquet(
    path: Path,
    *,
    required_cols: tuple[str, ...] = REQUIRED_RATING_COLS,
    dedup: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Load a split parquet file, drop NAs and duplicate user–movie pairs."""
    cols = list(required_cols)
    if dedup and "timestamp" not in cols:
        cols = cols + ["timestamp"]
    available = pq.ParquetFile(path).schema.names
    cols = [c for c in cols if c in available]
    df = pd.read_parquet(path, columns=cols)
    return clean_ratings_df(df, required_cols=required_cols)


def load_unique_movies(path: Path) -> pd.DataFrame:
    """One row per movieId with genre/year columns for content features."""
    cols = ["movieId", "genres", "release_year"]
    df = pd.read_parquet(path, columns=cols)
    return df.drop_duplicates("movieId").reset_index(drop=True)


def sample_parquet(
    path: Path,
    n: int,
    columns: list[str],
    *,
    seed: int = 42,
    batch_size: int = 1_000_000,
) -> pd.DataFrame:
    """Random row sample without loading the entire parquet file."""
    pf = pq.ParquetFile(path)
    total = pf.metadata.num_rows
    if n >= total:
        return pd.read_parquet(path, columns=columns)

    rng = np.random.default_rng(seed)
    target_idx = set(rng.choice(total, size=n, replace=False))

    chunks: list[pd.DataFrame] = []
    offset = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        df = batch.to_pandas()
        pick = [i for i in range(len(df)) if (offset + i) in target_idx]
        if pick:
            chunks.append(df.iloc[pick])
        offset += len(df)

    return pd.concat(chunks, ignore_index=True)
