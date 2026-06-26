"""Movie catalog helpers for the Streamlit recommender."""

from __future__ import annotations

import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from scripts.data_helpers import project_root, resolve_artifact


CATALOG_COLS = [
    "movieId",
    "title",
    "genres",
    "director",
    "cast",
    "release_year",
    "poster_url",
]


def _catalog_cache_path() -> Path:
    return project_root() / "artifacts" / "movies_catalog.parquet"


def is_readable_parquet(path: Path) -> bool:
    """True when path is a real Parquet file (not missing, empty, or a Git LFS pointer)."""
    if not path.exists():
        return False
    try:
        with path.open("rb") as handle:
            if handle.read(4) != b"PAR1":
                return False
        pq.ParquetFile(path)
        return True
    except Exception:
        return False


def _ensure_search_text(catalog: pd.DataFrame) -> pd.DataFrame:
    if "search_text" in catalog.columns:
        return catalog
    out = catalog.copy()
    out["search_text"] = (
        out["title"].fillna("").astype(str)
        + " "
        + out["genres"].fillna("").astype(str)
        + " "
        + out["director"].fillna("").astype(str)
    ).str.lower()
    return out


def _processed_split_paths() -> list[Path]:
    root = project_root()
    processed = root / "data" / "processed"
    paths = [processed / p for p in ("train.parquet", "val.parquet", "test.parquet")]
    existing = [p for p in paths if is_readable_parquet(p)]
    if not existing:
        raise FileNotFoundError(
            "Could not find any readable processed split files under data/processed/. "
            "Create train.parquet, val.parquet, and/or test.parquet first, "
            "or ship artifacts/movies_catalog.parquet."
        )
    return existing


def _catalog_cache_is_valid(cache_path: Path, paths: list[Path]) -> bool:
    if not cache_path.exists():
        return False
    latest_mtime = max(p.stat().st_mtime for p in paths)
    if cache_path.stat().st_mtime < latest_mtime:
        return False

    try:
        cached_ids = set(pd.read_parquet(cache_path, columns=["movieId"])["movieId"].astype(int).unique())
        split_ids: set[int] = set()
        for path in paths:
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=2_000_000, columns=["movieId"]):
                split_ids.update(batch.to_pandas()["movieId"].astype(int).tolist())
        return split_ids.issubset(cached_ids)
    except Exception:
        return False


@lru_cache(maxsize=1)
def load_movie_catalog() -> pd.DataFrame:
    """One row per movie from processed splits, enriched with overview when available."""
    catalog_path = resolve_artifact(
        "artifacts/movies_catalog.parquet",
        readable_check=is_readable_parquet,
    )
    if catalog_path is not None:
        return _ensure_search_text(pd.read_parquet(catalog_path))

    paths = _processed_split_paths()

    pf = pq.ParquetFile(paths[0])
    cols = [c for c in CATALOG_COLS if c in pf.schema.names]
    dfs = [pd.read_parquet(path, columns=cols) for path in paths]
    df = pd.concat(dfs, ignore_index=True)
    catalog = df.drop_duplicates("movieId").reset_index(drop=True)

    root = project_root()
    meta_path = root / "data" / "metadata_df.parquet"
    if meta_path.exists():
        meta = pd.read_parquet(meta_path, columns=["movieId", "overview", "fetch_status"])
        meta = meta[meta["fetch_status"] == "ok"][["movieId", "overview"]]
        catalog = catalog.merge(meta, on="movieId", how="left")
    else:
        catalog["overview"] = pd.NA

    catalog["search_text"] = (
        catalog["title"].fillna("").astype(str)
        + " "
        + catalog["genres"].fillna("").astype(str)
        + " "
        + catalog["director"].fillna("").astype(str)
    ).str.lower()

    counts = _movie_rating_counts()
    if counts is not None:
        catalog = catalog.merge(counts, on="movieId", how="left")
        catalog["rating_count"] = catalog["rating_count"].fillna(0).astype("int32")

    cache_path = _catalog_cache_path()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_parquet(cache_path, index=False)
    return _ensure_search_text(catalog)


def _format_year(year: float | int | None) -> str:
    if year is None or pd.isna(year):
        return "?"
    return str(int(year))


def movie_label(row) -> str:
    year = _format_year(getattr(row, "release_year", None))
    genres = getattr(row, "genres", "") or ""
    return f"{row.title} ({year}) — {genres}"


def search_movies(
    query: str,
    catalog: pd.DataFrame | None = None,
    limit: int = 20,
    *,
    min_chars: int = 1,
) -> pd.DataFrame:
    catalog = load_movie_catalog() if catalog is None else catalog
    q = query.strip().lower()
    if len(q) < min_chars:
        return catalog.iloc[0:0]

    tokens = [t for t in q.split() if t]
    if not tokens:
        return catalog.iloc[0:0]

    mask = pd.Series(True, index=catalog.index)
    for token in tokens:
        mask &= catalog["search_text"].str.contains(token, na=False, regex=False)
    if not mask.any():
        return catalog.iloc[0:0]

    hits = catalog.loc[mask].copy()
    title_lower = hits["title"].str.lower()
    hits["_rank"] = 0
    hits["_rank"] += title_lower.str.startswith(tokens[0]).astype(int) * 10
    hits["_rank"] += title_lower.str.contains(q, na=False, regex=False).astype(int) * 5
    for token in tokens:
        hits["_rank"] += title_lower.str.contains(token, na=False, regex=False).astype(int)
    return hits.sort_values("_rank", ascending=False).drop(columns="_rank").head(limit)


def suggest_movies(
    query: str,
    catalog: pd.DataFrame | None = None,
    limit: int = 8,
    *,
    min_chars: int = 2,
) -> pd.DataFrame:
    """Typeahead suggestions while the user is typing."""
    return search_movies(query, catalog, limit=limit, min_chars=min_chars)


def movies_by_ids(movie_ids: list[int], catalog: pd.DataFrame | None = None) -> pd.DataFrame:
    catalog = load_movie_catalog() if catalog is None else catalog
    order = {mid: i for i, mid in enumerate(movie_ids)}
    subset = catalog[catalog["movieId"].isin(movie_ids)].copy()
    subset["_order"] = subset["movieId"].map(order)
    return subset.sort_values("_order").drop(columns="_order")


# Title substring patterns → franchise label (MovieLens titles rarely include "James Bond").
FRANCHISE_PATTERNS: dict[str, list[str]] = {
    "James Bond": [
        "dr. no",
        "from russia with love",
        "goldfinger",
        "thunderball",
        "you only live twice",
        "on her majesty",
        "diamonds are forever",
        "live and let die",
        "man with the golden gun",
        "spy who loved me",
        "moonraker",
        "for your eyes only",
        "octopussy",
        "view to a kill",
        "living daylights",
        "licence to kill",
        "license to kill",
        "goldeneye",
        "tomorrow never dies",
        "world is not enough",
        "die another day",
        "casino royale",
        "quantum of solace",
        "skyfall",
        "spectre",
        "no time to die",
    ],
    "Lord of the Rings": ["lord of the rings", "hobbit"],
    "Harry Potter": ["harry potter"],
    "Star Wars": ["star wars"],
    "Mission: Impossible": ["mission: impossible", "mission impossible"],
    "Marvel": ["avengers", "iron man", "thor", "captain america", "guardians of the galaxy"],
}


def detect_franchise(title: str) -> str | None:
    t = title.lower()
    for franchise, patterns in FRANCHISE_PATTERNS.items():
        if any(p in t for p in patterns):
            return franchise
    return None


def franchise_suggestions_for_selection(
    selected_ids: list[int],
    catalog: pd.DataFrame | None = None,
    *,
    limit: int = 8,
) -> list[tuple[pd.Series, str]]:
    """Other movies in the same detected franchise(es) as the user's picks."""
    catalog = load_movie_catalog() if catalog is None else catalog
    selected = set(selected_ids)
    selected_titles = movies_by_ids(list(selected_ids), catalog)["title"].astype(str)

    franchises: set[str] = set()
    for title in selected_titles:
        name = detect_franchise(title)
        if name:
            franchises.add(name)
    if not franchises:
        return []

    title_lower = catalog["title"].str.lower()
    results: list[tuple[pd.Series, str]] = []
    seen: set[int] = set(selected)

    for franchise in sorted(franchises):
        patterns = FRANCHISE_PATTERNS[franchise]
        mask = pd.Series(False, index=catalog.index)
        for pattern in patterns:
            mask |= title_lower.str.contains(pattern, na=False, regex=False)
        hits = catalog[mask].sort_values("release_year", na_position="last")
        for _, row in hits.iterrows():
            mid = int(row["movieId"])
            if mid in seen:
                continue
            results.append((row, franchise))
            seen.add(mid)
            if len(results) >= limit:
                return results
    return results


def latest_movies(
    limit: int = 12,
    catalog: pd.DataFrame | None = None,
    *,
    eligible_ids: set[int] | frozenset[int] | None = None,
) -> pd.DataFrame:
    """Popular recent catalog titles with posters (year taken from title when possible)."""
    catalog = load_movie_catalog() if catalog is None else catalog
    df = catalog.copy()

    def _title_year(title: str) -> float | None:
        match = re.search(r"\((\d{4})\)", str(title))
        return float(match.group(1)) if match else None

    df["title_year"] = df["title"].map(_title_year)
    current_year = datetime.now().year
    min_year = max(2015, current_year - 8)
    df = df[df["title_year"].notna()]
    df = df[df["title_year"].between(min_year, current_year + 1)]
    df = df[df["poster_url"].notna() & (df["poster_url"].astype(str).str.len() > 0)]

    counts = _movie_rating_counts()
    if "rating_count" in df.columns:
        df["rating_count"] = pd.to_numeric(df["rating_count"], errors="coerce").fillna(0)
        sort_cols = ["title_year", "rating_count", "movieId"]
        ascending = [False, False, False]
    elif counts is not None:
        df = df.merge(counts, on="movieId", how="left")
        df["rating_count"] = df["rating_count"].fillna(0)
        sort_cols = ["title_year", "rating_count", "movieId"]
        ascending = [False, False, False]
    else:
        sort_cols = ["title_year", "movieId"]
        ascending = [False, False]

    df = df.sort_values(sort_cols, ascending=ascending)
    if eligible_ids is not None:
        df = df[df["movieId"].isin(eligible_ids)]
    return df.drop_duplicates("movieId").head(limit).reset_index(drop=True)


@lru_cache(maxsize=1)
def _movie_rating_counts() -> pd.DataFrame | None:
    """Rating counts from processed splits when available (local dev / full deploy)."""
    try:
        paths = _processed_split_paths()
    except FileNotFoundError:
        return None

    parts: list[pd.DataFrame] = []
    for path in paths:
        parts.append(pd.read_parquet(path, columns=["movieId"]))
    if not parts:
        return None

    counts = (
        pd.concat(parts, ignore_index=True)
        .groupby("movieId", as_index=False)
        .size()
        .rename(columns={"size": "rating_count"})
    )
    return counts
