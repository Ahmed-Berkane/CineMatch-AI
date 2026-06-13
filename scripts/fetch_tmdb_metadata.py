"""
Fetch TMDb metadata for MovieLens movies and cache as metadata_df.

Requires TMDB_API_KEY in the environment or in a .env file at the project root.
Get a free key at https://www.themoviedb.org/settings/api
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

TMDB_BASE = "https://api.themoviedb.org/3/movie"
POSTER_BASE = "https://image.tmdb.org/t/p/w500"
DEFAULT_LINKS = "data/links.csv"
DEFAULT_RATINGS = "data/ratings.csv"
DEFAULT_MOVIES = "data/movies.csv"
DEFAULT_OUTPUT_DIR = "data"
METADATA_STEM = "metadata_df"
MERGED_STEM = "movies_with_metadata"
TOP_CAST = 5
MAX_RETRIES = 3


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_api_key() -> str:
    load_dotenv(project_root() / ".env")
    api_key = os.getenv("TMDB_API_KEY", "").strip()
    if not api_key:
        print(
            "Error: TMDB_API_KEY is not set.\n"
            "Copy .env.example to .env and add your key from "
            "https://www.themoviedb.org/settings/api",
            file=sys.stderr,
        )
        sys.exit(1)
    return api_key


def load_links(links_path: Path, only_rated: bool, ratings_path: Path) -> pd.DataFrame:
    links = pd.read_csv(links_path)
    links = links.dropna(subset=["tmdbId"]).copy()
    links["tmdbId"] = links["tmdbId"].astype(int)

    if only_rated:
        ratings = pd.read_csv(ratings_path, usecols=["movieId"])
        rated_ids = set(ratings["movieId"].unique())
        links = links[links["movieId"].isin(rated_ids)].copy()

    return links.drop_duplicates(subset=["movieId"]).reset_index(drop=True)


def load_existing(output_dir: Path) -> pd.DataFrame:
    parquet_path = output_dir / f"{METADATA_STEM}.parquet"
    csv_path = output_dir / f"{METADATA_STEM}.csv"

    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def extract_cast(cast: list[dict], limit: int = TOP_CAST) -> str:
    names = [person.get("name", "").strip() for person in cast[:limit]]
    return ", ".join(name for name in names if name)


def extract_director(crew: list[dict]) -> str:
    for person in crew:
        if person.get("job") == "Director":
            return person.get("name", "").strip()
    return ""


def parse_release_year(release_date: str | None) -> int | None:
    if not release_date or len(release_date) < 4:
        return None
    try:
        return int(release_date[:4])
    except ValueError:
        return None


def fetch_movie_metadata(
    session: requests.Session,
    api_key: str,
    tmdb_id: int,
) -> dict:
    url = f"{TMDB_BASE}/{tmdb_id}"
    params = {"api_key": api_key, "append_to_response": "credits"}

    last_error = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(url, params=params, timeout=30)
            if response.status_code == 404:
                return {"fetch_status": "not_found", "fetch_error": "Movie not found on TMDb"}
            response.raise_for_status()
            payload = response.json()
            credits = payload.get("credits", {})
            poster_path = payload.get("poster_path") or ""
            return {
                "overview": payload.get("overview") or "",
                "cast": extract_cast(credits.get("cast", [])),
                "director": extract_director(credits.get("crew", [])),
                "release_year": parse_release_year(payload.get("release_date")),
                "poster_path": poster_path,
                "poster_url": f"{POSTER_BASE}{poster_path}" if poster_path else "",
                "fetch_status": "ok",
                "fetch_error": "",
            }
        except requests.RequestException as exc:
            last_error = str(exc)
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)

    return {"fetch_status": "error", "fetch_error": last_error}


def load_movies(movies_path: Path) -> pd.DataFrame:
    return pd.read_csv(movies_path)


def build_movies_with_metadata(movies_path: Path, output_dir: Path) -> pd.DataFrame:
    movies = load_movies(movies_path)
    metadata = load_existing(output_dir)
    if metadata.empty:
        print(
            f"Error: no metadata cache found in {output_dir}. Run the fetch first.",
            file=sys.stderr,
        )
        sys.exit(1)
    return movies.merge(metadata, on="movieId", how="inner")


def save_movies_with_metadata(df: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{MERGED_STEM}.parquet"
    df.to_parquet(output_path, index=False)
    return output_path


def save_metadata(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / f"{METADATA_STEM}.parquet"
    csv_path = output_dir / f"{METADATA_STEM}.csv"

    ordered_cols = [
        "movieId",
        "tmdbId",
        "overview",
        "cast",
        "director",
        "release_year",
        "poster_path",
        "poster_url",
        "fetch_status",
        "fetch_error",
    ]
    df = df.reindex(columns=ordered_cols)
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)


def run(args: argparse.Namespace) -> None:
    root = project_root()
    links_path = root / args.links
    ratings_path = root / args.ratings
    output_dir = root / args.output_dir

    api_key = load_api_key()
    links = load_links(links_path, args.only_rated, ratings_path)
    existing = load_existing(output_dir)

    done_ids: set[int] = set()
    if not existing.empty and "movieId" in existing.columns:
        done_ids = set(existing["movieId"].astype(int).tolist())

    pending = links[~links["movieId"].isin(done_ids)].copy()
    if args.limit is not None:
        pending = pending.head(args.limit)

    total = len(pending)
    if total == 0:
        print(f"No new movies to fetch. Cache already has {len(existing)} rows.")
        save_metadata(existing, output_dir)
        return

    print(f"Fetching metadata for {total} movies ({len(done_ids)} already cached)...")

    rows: list[dict] = existing.to_dict("records") if not existing.empty else []
    session = requests.Session()

    for index, row in enumerate(pending.itertuples(index=False), start=1):
        metadata = fetch_movie_metadata(session, api_key, int(row.tmdbId))
        rows.append(
            {
                "movieId": int(row.movieId),
                "tmdbId": int(row.tmdbId),
                **metadata,
            }
        )

        if index % args.batch_size == 0 or index == total:
            save_metadata(pd.DataFrame(rows), output_dir)
            ok_count = sum(1 for r in rows if r.get("fetch_status") == "ok")
            print(f"Saved checkpoint: {index}/{total} fetched ({ok_count} ok total)")

        if index < total:
            time.sleep(args.sleep)

    final = pd.DataFrame(rows)
    save_metadata(final, output_dir)
    ok = (final["fetch_status"] == "ok").sum()
    print(f"Done. metadata_df saved to {output_dir} ({ok}/{len(final)} successful).")


def merge_and_save(args: argparse.Namespace) -> None:
    root = project_root()
    movies_path = root / args.movies
    output_dir = root / args.output_dir

    merged = build_movies_with_metadata(movies_path, output_dir)
    output_path = save_movies_with_metadata(merged, output_dir)
    print(f"Saved {len(merged):,} rows to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch and cache TMDb movie metadata.")
    parser.add_argument("--links", default=DEFAULT_LINKS, help="Path to links.csv")
    parser.add_argument("--ratings", default=DEFAULT_RATINGS, help="Path to ratings.csv")
    parser.add_argument("--movies", default=DEFAULT_MOVIES, help="Path to movies.csv")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Skip fetch; merge movies.csv with cached metadata_df and save parquet",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="After fetching, also save movies_with_metadata.parquet",
    )
    parser.add_argument(
        "--only-rated",
        action="store_true",
        help="Fetch only movies that appear in ratings.csv (~84k instead of ~87k)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Fetch at most N movies (useful for testing)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Save cache every N movies",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.26,
        help="Seconds to wait between API calls (rate limiting)",
    )
    return parser


if __name__ == "__main__":
    cli_args = build_parser().parse_args()
    if cli_args.merge_only:
        merge_and_save(cli_args)
    else:
        run(cli_args)
        if cli_args.merge:
            merge_and_save(cli_args)
