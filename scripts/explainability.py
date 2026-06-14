"""Human-readable explanations for movie recommendations (Phase 5)."""

from __future__ import annotations

from collections import Counter

import pandas as pd

from scripts.model_helpers import split_genre_string
from scripts.taste_utils import THEME_TAGS, decade_label, decade_summary, genre_summary


def _theme_phrase(genres: set[str]) -> str | None:
    for needed, phrase in THEME_TAGS:
        if needed <= genres:
            return phrase
    shared = sorted(genres)[:3]
    if shared:
        return f"{', '.join(shared).lower()} storytelling"
    return None


def _best_seed_match(
    rec_genres: set[str],
    rec_year: float | int | None,
    seed_titles: list[str],
    seed_genres: list[str],
    seed_years: list[float | int | None],
) -> tuple[str, set[str], float]:
    best_title = seed_titles[0] if seed_titles else "your picks"
    best_genres: set[str] = set()
    best_score = -1.0

    for title, raw_genres, year in zip(seed_titles, seed_genres, seed_years):
        seed_set = set(split_genre_string(raw_genres))
        overlap = len(seed_set & rec_genres)
        score = float(overlap)
        if rec_year is not None and year is not None and not pd.isna(rec_year) and not pd.isna(year):
            if decade_label(rec_year) == decade_label(year):
                score += 0.5
        if score > best_score:
            best_score = score
            best_title = title
            best_genres = seed_set

    return best_title, best_genres, best_score


def explain_recommendation(
    *,
    rec_title: str,
    rec_genres: str,
    rec_year: float | int | None,
    seed_titles: list[str],
    seed_genres: list[str],
    seed_years: list[float | int | None],
    content_similarity: float,
    collaborative_similarity: float,
    cohort_note: str | None = None,
    max_bullets: int = 5,
) -> dict[str, str | list[str]]:
    """Build headline + bullets + theme tags for one recommendation."""
    rec_genre_list = split_genre_string(rec_genres)
    rec_genre_set = set(rec_genre_list)

    best_seed, best_seed_genres, _ = _best_seed_match(
        rec_genre_set, rec_year, seed_titles, seed_genres, seed_years
    )
    shared = sorted(best_seed_genres & rec_genre_set)[:4]
    theme = _theme_phrase(rec_genre_set)

    headline_parts = [f"Because you liked **{best_seed}**"]
    if theme:
        headline_parts.append(f"this shares **{theme}**")
    elif shared:
        headline_parts.append(f"this shares **{', '.join(shared).lower()}** DNA")
    else:
        headline_parts.append("this fits your taste fingerprint")
    headline = ", ".join(headline_parts[:2]) + "."

    bullets: list[str] = []
    if shared:
        bullets.append(f"Genre overlap with *{best_seed}*: {', '.join(shared)}")

    rec_decade = decade_label(rec_year)
    seed_decade_counts = Counter(d for y in seed_years if (d := decade_label(y)))
    if rec_decade and seed_decade_counts and rec_decade == seed_decade_counts.most_common(1)[0][0]:
        bullets.append(f"Same era ({rec_decade}) as several of your favorites")

    if content_similarity >= 0.40:
        bullets.append(
            f"Content match {content_similarity:.0%} — similar genres, era, and metadata profile"
        )

    if collaborative_similarity >= 0.40:
        bullets.append(
            f"HybridNet taste match {collaborative_similarity:.0%} — "
            "users with your picks tend to rate this highly"
        )

    if cohort_note:
        bullets.append(cohort_note)

    theme_tags: list[str] = []
    for needed, tag in THEME_TAGS:
        if needed <= rec_genre_set:
            theme_tags.append(tag)
        if len(theme_tags) >= 3:
            break

    if not bullets:
        bullets.append(f"Recommended based on fit with {', '.join(seed_titles[:2])}")

    return {
        "headline": headline,
        "bullets": bullets[:max_bullets],
        "theme_tags": theme_tags[:3],
    }
