"""Reusable helpers for joining MovieLens ratings with TMDb metadata."""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Callable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

MOVIE_COLUMNS = [
    "movieId",
    "title",
    "genres",
    "tmdbId",
    "director",
    "cast",
    "release_year",
    "poster_url",
    "overview",
]


def project_root() -> Path:
    """Repo root — local dev (data/), HF Space (app.py + artifacts/), or Notebooks/."""
    cwd = Path.cwd().resolve()
    for candidate in (cwd, cwd.parent):
        if (candidate / "data").is_dir():
            return candidate
        if (candidate / "app.py").is_file():
            return candidate
        if (candidate / "artifacts").is_dir() and (candidate / "scripts").is_dir():
            return candidate

    root_from_file = Path(__file__).resolve().parent.parent
    if (root_from_file / "app.py").is_file() or (root_from_file / "artifacts").is_dir():
        return root_from_file

    raise FileNotFoundError(
        "Could not find project root. Run from the CineMatch-AI repo or deploy app.py with artifacts/."
    )


def is_lfs_pointer(path: Path) -> bool:
    """True when Git LFS left a pointer file instead of the real binary."""
    if not path.is_file():
        return False
    try:
        with path.open("rb") as handle:
            return handle.read(64).startswith(b"version https://git-lfs")
    except OSError:
        return False


def resolve_artifact(
    relative_path: str,
    *,
    readable_check: Callable[[Path], bool] | None = None,
) -> Path | None:
    """Return a usable artifact path locally or via Hub download on HF Spaces."""
    rel = relative_path.replace("\\", "/")
    local = project_root() / rel
    if local.exists() and (readable_check is None or readable_check(local)):
        return local

    space_id = os.environ.get("SPACE_ID")
    if not space_id:
        return None

    try:
        from huggingface_hub import hf_hub_download

        downloaded = Path(
            hf_hub_download(
                repo_id=space_id,
                filename=rel,
                repo_type="space",
            )
        )
        if readable_check is None or readable_check(downloaded):
            return downloaded
    except Exception:
        return None
    return None


def load_metadata(data_dir: Path) -> pd.DataFrame:
    parquet_path = data_dir / "metadata_df.parquet"
    csv_path = data_dir / "metadata_df.csv"

    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)

    raise FileNotFoundError(
        "metadata_df not found. Run: python scripts/fetch_tmdb_metadata.py --only-rated"
    )


def load_ratings(ratings_path: Path) -> pd.DataFrame:
    return pd.read_csv(
        ratings_path,
        usecols=["userId", "movieId", "rating", "timestamp"],
        dtype={
            "userId": "int32",
            "movieId": "int32",
            "rating": "float32",
            "timestamp": "int64",
        },
    )


def optimize_ratings_with_movies(df: pd.DataFrame) -> pd.DataFrame:
    if "release_year" in df.columns:
        df["release_year"] = pd.to_numeric(df["release_year"], downcast="integer")
    for col in ("title", "genres", "director", "cast", "poster_url", "overview"):
        if col in df.columns:
            df[col] = df[col].astype("string")
    return df


def build_movies_catalog(movies_path: Path, data_dir: Path) -> pd.DataFrame:
    movies = pd.read_csv(movies_path)
    metadata = load_metadata(data_dir)
    ok = metadata[metadata["fetch_status"] == "ok"].copy()
    catalog = movies.merge(ok, on="movieId", how="inner")
    return catalog.reindex(columns=[c for c in MOVIE_COLUMNS if c in catalog.columns])


def _movie_columns(catalog: pd.DataFrame, slim: bool) -> list[str]:
    cols = [c for c in MOVIE_COLUMNS if c in catalog.columns]
    if slim:
        cols = [c for c in cols if c != "overview"]
    return cols


def join_ratings_with_catalog(
    ratings: pd.DataFrame,
    catalog: pd.DataFrame,
    *,
    slim: bool = True,
) -> pd.DataFrame:
    movie_cols = _movie_columns(catalog, slim)
    merged = ratings.merge(catalog[movie_cols], on="movieId", how="inner")
    return optimize_ratings_with_movies(merged)


def build_ratings_with_movies(
    ratings_path: Path,
    catalog: pd.DataFrame,
    *,
    slim: bool = True,
) -> pd.DataFrame:
    return join_ratings_with_catalog(load_ratings(ratings_path), catalog, slim=slim)


def temporal_split_cutoffs(
    timestamps: pd.Series,
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
) -> tuple[float, float]:
    val_cutoff = timestamps.quantile(train_ratio)
    test_cutoff = timestamps.quantile(train_ratio + val_ratio)
    return val_cutoff, test_cutoff


def temporal_train_val_test_split(
    df: pd.DataFrame,
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to 1, got {total}")

    val_cutoff, test_cutoff = temporal_split_cutoffs(
        df["timestamp"], train_ratio=train_ratio, val_ratio=val_ratio
    )
    train = df[df["timestamp"] < val_cutoff].copy()
    val = df[(df["timestamp"] >= val_cutoff) & (df["timestamp"] < test_cutoff)].copy()
    test = df[df["timestamp"] >= test_cutoff].copy()
    return train, val, test


def save_parquet_batched(
    df: pd.DataFrame,
    path: Path,
    *,
    batch_size: int = 1_000_000,
    compression: str = "zstd",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    try:
        for start in range(0, len(df), batch_size):
            batch = df.iloc[start : start + batch_size]
            table = pa.Table.from_pandas(batch, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema, compression=compression)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def build_and_save_splits(
    ratings_path: Path,
    catalog: pd.DataFrame,
    output_dir: Path,
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    slim: bool = True,
    batch_size: int = 1_000_000,
) -> dict[str, float | dict[str, float]]:
    """Split ratings by time, join one split at a time, write parquet in batches."""
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to 1, got {total}")

    ratings = load_ratings(ratings_path)
    val_cutoff, test_cutoff = temporal_split_cutoffs(
        ratings["timestamp"], train_ratio=train_ratio, val_ratio=val_ratio
    )

    split_masks = [
        ("train", ratings["timestamp"] < val_cutoff),
        (
            "val",
            (ratings["timestamp"] >= val_cutoff) & (ratings["timestamp"] < test_cutoff),
        ),
        ("test", ratings["timestamp"] >= test_cutoff),
    ]

    sizes_mb: dict[str, float] = {}
    row_counts: dict[str, int] = {}

    for name, mask in split_masks:
        print(f"Building {name}...")
        subset = ratings.loc[mask]
        merged = join_ratings_with_catalog(subset, catalog, slim=slim)
        del subset

        path = output_dir / f"{name}.parquet"
        save_parquet_batched(merged, path, batch_size=batch_size)
        row_counts[name] = len(merged)
        sizes_mb[name] = path.stat().st_size / (1024 * 1024)
        print(f"  {path.name}: {row_counts[name]:,} rows, {sizes_mb[name]:.1f} MB")

        del merged
        gc.collect()

    del ratings
    gc.collect()

    return {
        "val_cutoff": val_cutoff,
        "test_cutoff": test_cutoff,
        "rows": row_counts,
        "sizes_mb": sizes_mb,
    }
