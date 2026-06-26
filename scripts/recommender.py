"""Hybrid movie recommender with explainable scores (Phases 4–5)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.metrics.pairwise import cosine_similarity

from scripts.catalog import is_readable_parquet, load_movie_catalog, movies_by_ids
from scripts.data_helpers import is_lfs_pointer, project_root, resolve_artifact
from scripts.feedback import apply_feedback_to_scores
from scripts.explainability import explain_recommendation
from scripts.model_helpers import split_genre_string
from scripts import neural_models as nm


@dataclass
class Recommendation:
    movie_id: int
    title: str
    genres: str
    release_year: float | int | None
    poster_url: str | None
    score: float
    content_score: float
    collaborative_score: float
    headline: str
    explanations: list[str]
    theme_tags: list[str]


class RecommenderEngine:
    """Content + collaborative hybrid recommendations using the saved HybridNet checkpoint."""

    def __init__(
        self,
        checkpoint_path: Path | None = None,
        *,
        content_weight: float = 0.6,
        collaborative_weight: float = 0.4,
        device: str | None = None,
    ):
        root = project_root()

        def _loadable_checkpoint(path: Path) -> bool:
            return path.stat().st_size > 1024 and not is_lfs_pointer(path)

        resolved: Path | None = None
        if checkpoint_path is not None:
            rel = str(checkpoint_path.relative_to(root)).replace("\\", "/")
            resolved = resolve_artifact(rel, readable_check=_loadable_checkpoint)

        if resolved is None:
            for rel in ("artifacts/best_model_full.pt", "artifacts/best_model.pt"):
                resolved = resolve_artifact(rel, readable_check=_loadable_checkpoint)
                if resolved is not None:
                    break

        if resolved is None:
            raise FileNotFoundError(
                "Missing model checkpoint under artifacts/. "
                "Expected best_model.pt or best_model_full.pt."
            )

        self.checkpoint_path = resolved

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.content_weight = content_weight
        self.collaborative_weight = collaborative_weight

        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        self.ckpt = ckpt
        self.model = nm.build_model_from_checkpoint(ckpt)
        self.model.eval()
        self.mappings = nm.id_mappings_from_checkpoint(ckpt)
        self.content_lookup = np.asarray(ckpt["content_lookup"], dtype=np.float32)
        self.catalog = load_movie_catalog()
        processed = root / "data" / "processed"
        self.processed_paths = [
            processed / p
            for p in ("train.parquet", "val.parquet", "test.parquet")
            if is_readable_parquet(processed / p)
        ]

        with torch.no_grad():
            self.movie_embeddings = self.model.movie_emb.weight.cpu().numpy()

        self._genre_by_movie_id: dict[int, set[str]] = {}
        for mid, genres in zip(self.catalog["movieId"], self.catalog["genres"].fillna("")):
            self._genre_by_movie_id[int(mid)] = set(split_genre_string(str(genres)))

        catalog_index = self.catalog.drop_duplicates("movieId").set_index("movieId")
        self._catalog_rows = catalog_index

    def _movie_indices(self, movie_ids: list[int]) -> list[int]:
        indices = []
        for mid in movie_ids:
            if mid not in self.mappings.movie_to_idx:
                raise ValueError(f"movieId {mid} was not seen during model training.")
            indices.append(self.mappings.movie_to_idx[mid])
        return indices

    def recommend(
        self,
        seed_movie_ids: list[int],
        *,
        top_k: int = 10,
        min_rating_cohort: float = 4.0,
        include_cohort_explanations: bool = True,
        liked_ids: list[int] | None = None,
        disliked_ids: list[int] | None = None,
        diversity: float = 0.35,
        explain: bool = True,
    ) -> list[Recommendation]:
        if len(seed_movie_ids) < 1:
            raise ValueError("Select at least one favorite movie.")
        if len(seed_movie_ids) > 10:
            raise ValueError("Select at most 10 favorite movies.")

        liked_ids = liked_ids or []
        disliked_ids = disliked_ids or []
        blocked = set(disliked_ids)

        seed_idx = self._movie_indices(seed_movie_ids)
        seed_set = set(seed_movie_ids) | set(liked_ids)

        emb_parts = [self.movie_embeddings[seed_idx]]
        content_parts = [self.content_lookup[seed_idx]]
        liked_idx = [
            self.mappings.movie_to_idx[mid]
            for mid in liked_ids
            if mid in self.mappings.movie_to_idx
        ]
        if liked_idx:
            emb_parts.append(self.movie_embeddings[liked_idx])
            content_parts.append(self.content_lookup[liked_idx])

        seed_emb = np.vstack(emb_parts).mean(axis=0, keepdims=True)
        seed_content = np.vstack(content_parts).mean(axis=0, keepdims=True)

        content_sims = cosine_similarity(seed_content, self.content_lookup).flatten()
        collab_sims = cosine_similarity(seed_emb, self.movie_embeddings).flatten()
        final_scores = (
            self.content_weight * content_sims + self.collaborative_weight * collab_sims
        )
        final_scores = apply_feedback_to_scores(
            self.mappings.movie_ids,
            final_scores,
            mappings=self.mappings,
            movie_embeddings=self.movie_embeddings,
            liked_ids=liked_ids,
            disliked_ids=disliked_ids,
        )

        cohort_notes: dict[int, str] = {}
        if explain and include_cohort_explanations and self.processed_paths:
            cohort_notes = self._cohort_notes(seed_movie_ids, min_rating=min_rating_cohort)

        seed_titles: list[str] = []
        seed_genres: list[str] = []
        seed_years: list[float | int | None] = []
        if explain:
            seeds_df = movies_by_ids(seed_movie_ids, self.catalog)
            seed_titles = seeds_df["title"].astype(str).tolist()
            seed_genres = seeds_df["genres"].fillna("").astype(str).tolist()
            seed_years = seeds_df["release_year"].tolist()
        seed_genre_sets = [
            self._genre_by_movie_id.get(mid, set())
            for mid in seed_movie_ids
        ]

        ranked = self._rank_candidates(
            final_scores,
            content_sims,
            collab_sims,
            seed_set=seed_set,
            blocked=blocked,
            top_k=top_k,
            diversity=diversity,
            seed_genre_sets=seed_genre_sets,
        )

        recs: list[Recommendation] = []
        for movie_id, score, c_score, cf_score in ranked:
            row = self._catalog_rows.loc[movie_id]
            if explain:
                explained = explain_recommendation(
                    rec_title=str(row.title),
                    rec_genres=str(getattr(row, "genres", "")),
                    rec_year=getattr(row, "release_year", None),
                    seed_titles=seed_titles,
                    seed_genres=seed_genres,
                    seed_years=seed_years,
                    content_similarity=c_score,
                    collaborative_similarity=cf_score,
                    cohort_note=cohort_notes.get(movie_id),
                )
                headline = str(explained["headline"])
                bullets = list(explained["bullets"])
                theme_tags = list(explained["theme_tags"])
            else:
                headline = ""
                bullets = []
                theme_tags = []
            recs.append(
                Recommendation(
                    movie_id=movie_id,
                    title=str(row.title),
                    genres=str(getattr(row, "genres", "")),
                    release_year=getattr(row, "release_year", None),
                    poster_url=getattr(row, "poster_url", None),
                    score=score,
                    content_score=c_score,
                    collaborative_score=cf_score,
                    headline=headline,
                    explanations=bullets,
                    theme_tags=theme_tags,
                )
            )
        return recs

    def _genre_overlap_penalty(self, candidate_genres: set[str], picked_genres: list[set[str]]) -> float:
        if not picked_genres or not candidate_genres:
            return 0.0
        overlaps = [len(candidate_genres & pg) / max(len(candidate_genres | pg), 1) for pg in picked_genres]
        return max(overlaps) if overlaps else 0.0

    def _rank_candidates(
        self,
        final_scores: np.ndarray,
        content_sims: np.ndarray,
        collab_sims: np.ndarray,
        *,
        seed_set: set[int],
        blocked: set[int],
        top_k: int,
        diversity: float,
        seed_genre_sets: list[set[str]],
    ) -> list[tuple[int, float, float, float]]:
        """Greedy ranking with optional diversity to avoid samey recommendations."""
        diversity = float(max(0.0, min(1.0, diversity)))
        pool_limit = max(top_k * 30, 100)

        candidates: list[tuple[int, float, float, float]] = []
        if len(final_scores) <= pool_limit:
            ordered = np.argsort(final_scores)[::-1]
        else:
            part = np.argpartition(final_scores, -pool_limit)[-pool_limit:]
            ordered = part[np.argsort(final_scores[part])[::-1]]

        for idx in ordered:
            movie_id = int(self.mappings.movie_ids[idx])
            if movie_id in seed_set or movie_id in blocked:
                continue
            candidates.append(
                (
                    movie_id,
                    float(final_scores[idx]),
                    float(content_sims[idx]),
                    float(collab_sims[idx]),
                )
            )
            if len(candidates) >= pool_limit:
                break

        if diversity <= 0.05 or top_k <= 1:
            return candidates[:top_k]

        catalog_genres = {
            movie_id: self._genre_by_movie_id.get(movie_id, set())
            for movie_id, _, _, _ in candidates
        }

        picked: list[tuple[int, float, float, float]] = []
        picked_genre_sets = list(seed_genre_sets)
        pool = list(candidates)

        while pool and len(picked) < top_k:
            best_i = 0
            best_adj = -1e9
            for i, (movie_id, score, c_score, cf_score) in enumerate(pool):
                penalty = self._genre_overlap_penalty(catalog_genres.get(movie_id, set()), picked_genre_sets)
                adj = (1.0 - diversity) * score - diversity * penalty * 0.35
                if adj > best_adj:
                    best_adj = adj
                    best_i = i
            choice = pool.pop(best_i)
            picked.append(choice)
            picked_genre_sets.append(catalog_genres.get(choice[0], set()))

        return picked

    @lru_cache(maxsize=32)
    def _cohort_notes_cached(self, seed_key: tuple[int, ...], min_rating: float) -> tuple[tuple[int, str], ...]:
        """Find movies highly rated by users who liked the seed movies."""
        seed_ids = set(seed_key)
        fan_users: set[int] = set()
        cohort_counts: Counter[int] = Counter()

        cols = ["userId", "movieId", "rating"]
        for path in self.processed_paths:
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=2_000_000, columns=cols):
                df = batch.to_pandas()
                liked_seeds = df[(df["movieId"].isin(seed_ids)) & (df["rating"] >= min_rating)]
                fan_users.update(liked_seeds["userId"].astype(int).tolist())

        if not fan_users:
            return tuple()

        fan_users_frozen = frozenset(fan_users)
        for path in self.processed_paths:
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=2_000_000, columns=cols):
                df = batch.to_pandas()
                subset = df[
                    df["userId"].isin(fan_users_frozen)
                    & (df["rating"] >= min_rating)
                    & (~df["movieId"].isin(seed_ids))
                ]
                cohort_counts.update(subset["movieId"].astype(int).tolist())

        if not cohort_counts:
            return tuple()

        seed_titles = movies_by_ids(list(seed_ids), self.catalog)["title"].astype(str).tolist()
        anchor = seed_titles[0] if seed_titles else "your picks"
        notes: dict[int, str] = {}
        for movie_id, count in cohort_counts.most_common(200):
            notes[movie_id] = (
                f"Highly rated by MovieLens users who also liked {anchor} "
                f"({count:,} high ratings)"
            )
        return tuple(notes.items())

    def _cohort_notes(self, seed_movie_ids: list[int], *, min_rating: float) -> dict[int, str]:
        items = self._cohort_notes_cached(tuple(sorted(seed_movie_ids)), min_rating)
        return dict(items)


_ENGINE: RecommenderEngine | None = None


def get_engine() -> RecommenderEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = RecommenderEngine()
    return _ENGINE


def pick_page_suggestions(
    seed_movie_ids: list[int],
    *,
    limit: int = 8,
    liked_ids: list[int] | None = None,
    disliked_ids: list[int] | None = None,
) -> list[dict]:
    """Franchise completions + hybrid picks for the favorites picker."""
    from scripts.catalog import franchise_suggestions_for_selection, load_movie_catalog

    catalog = load_movie_catalog()
    selected = set(seed_movie_ids)
    blocked = set(disliked_ids or [])
    suggestions: list[dict] = []

    for row, franchise in franchise_suggestions_for_selection(
        seed_movie_ids, catalog, limit=limit
    ):
        mid = int(row["movieId"])
        if mid in blocked:
            continue
        suggestions.append(
            {
                "movie_id": int(row["movieId"]),
                "title": str(row["title"]),
                "genres": str(row.get("genres", "")),
                "release_year": row.get("release_year"),
                "poster_url": row.get("poster_url"),
                "reason": f"More in the {franchise} series",
            }
        )

    seen = selected | {s["movie_id"] for s in suggestions}
    if len(suggestions) < limit:
        engine = get_engine()
        recs = engine.recommend(
            seed_movie_ids,
            top_k=limit,
            include_cohort_explanations=False,
            liked_ids=liked_ids,
            disliked_ids=disliked_ids,
            diversity=0.0,
            explain=False,
        )
        for rec in recs:
            if rec.movie_id in seen or rec.movie_id in blocked:
                continue
            suggestions.append(
                {
                    "movie_id": rec.movie_id,
                    "title": rec.title,
                    "genres": rec.genres,
                    "release_year": rec.release_year,
                    "poster_url": rec.poster_url,
                    "reason": "Similar to your picks (HybridNet)",
                }
            )
            seen.add(rec.movie_id)
            if len(suggestions) >= limit:
                break
    return suggestions[:limit]
