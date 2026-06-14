"""Movie catalog helpers for the Streamlit recommender."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from scripts.data_helpers import project_root


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


@lru_cache(maxsize=1)
def load_movie_catalog() -> pd.DataFrame:
    """One row per movie from train.parquet, enriched with overview when available."""
    root = project_root()
    train_path = root / "data" / "processed" / "train.parquet"
    cache_path = _catalog_cache_path()
    if cache_path.exists() and cache_path.stat().st_mtime >= train_path.stat().st_mtime:
        return pd.read_parquet(cache_path)

    pf = pq.ParquetFile(train_path)
    cols = [c for c in CATALOG_COLS if c in pf.schema.names]
    df = pd.read_parquet(train_path, columns=cols)
    catalog = df.drop_duplicates("movieId").reset_index(drop=True)

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

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_parquet(cache_path, index=False)
    return catalog


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


def latest_movies(limit: int = 12, catalog: pd.DataFrame | None = None) -> pd.DataFrame:
    """Recent catalog titles with posters (year taken from title when possible)."""
    catalog = load_movie_catalog() if catalog is None else catalog
    df = catalog.copy()
    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")

    def _title_year(title: str) -> float | None:
        match = re.search(r"\((\d{4})\)", str(title))
        return float(match.group(1)) if match else None

    df["title_year"] = df["title"].map(_title_year)
    df = df[df["title_year"].notna()]
    df = df[df["title_year"].between(2015, 2022)]
    df = df[df["poster_url"].notna() & (df["poster_url"].astype(str).str.len() > 0)]
    df = df.sort_values(["title_year", "title"], ascending=[False, True])
    return df.drop_duplicates("movieId").head(limit).reset_index(drop=True)
