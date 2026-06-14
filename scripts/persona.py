"""Rule-based moviegoer personas and Taste DNA (Phase 6)."""

from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd

from scripts.model_helpers import split_genre_string
from scripts.taste_utils import THEME_TAGS, decade_summary

PERSONA_RULES: list[dict[str, Any]] = [
    {
        "title": "The High-Stakes Adrenaline Junkie",
        "emoji": "💥",
        "genres": {"Action", "Adventure", "Thriller"},
        "min_overlap": 2,
        "blurb": (
            "You crave massive set pieces, international stakes, and heroes who save the day "
            "when the clock hits zero. Relaxing is optional — adrenaline is not."
        ),
    },
    {
        "title": "The Covert Operative",
        "emoji": "🕵️",
        "genres": {"Mystery", "Thriller", "Crime"},
        "min_overlap": 2,
        "blurb": (
            "You love intelligence games, slow-burn tension, and plots where trust is the "
            "real weapon. Every conversation might be a trap."
        ),
    },
    {
        "title": "The Future Architect",
        "emoji": "🚀",
        "genres": {"Sci-Fi", "Fantasy", "Adventure"},
        "min_overlap": 2,
        "blurb": (
            "You build worlds in your head — futuristic tech, mythic quests, and ideas bigger "
            "than the screen. Reality is just the default setting."
        ),
    },
    {
        "title": "The Character Study Enthusiast",
        "emoji": "🎭",
        "genres": {"Drama", "Romance", "War"},
        "min_overlap": 2,
        "blurb": (
            "You watch for human depth — moral gray zones, relationships under pressure, "
            "and performances that linger after the credits."
        ),
    },
    {
        "title": "The Midnight Thrill Seeker",
        "emoji": "👻",
        "genres": {"Horror", "Thriller", "Mystery"},
        "min_overlap": 2,
        "blurb": (
            "You want atmosphere, dread, and stories that keep you checking the locks on "
            "your doors. Comfort is overrated."
        ),
    },
    {
        "title": "The Feel-Good Curator",
        "emoji": "😄",
        "genres": {"Comedy", "Romance", "Family"},
        "min_overlap": 2,
        "blurb": (
            "You pick films that lift the mood — warmth, laughs, and endings that leave "
            "you smiling. Life is heavy enough already."
        ),
    },
    {
        "title": "The Epic Questor",
        "emoji": "⚔️",
        "genres": {"Fantasy", "Adventure", "Action"},
        "min_overlap": 2,
        "blurb": (
            "You follow heroes on impossible journeys — kingdoms, battles, and destinies "
            "written in prophecy and steel."
        ),
    },
]

STYLE_RULES: list[tuple[frozenset[str], str]] = [
    (frozenset({"Mystery", "Thriller", "Crime"}), "slow-burn + plot-heavy"),
    (frozenset({"Action", "Adventure"}), "fast-paced + spectacle-driven"),
    (frozenset({"Drama", "Romance"}), "character-driven + emotional"),
    (frozenset({"Sci-Fi", "Fantasy"}), "world-building + imaginative"),
    (frozenset({"Horror", "Thriller"}), "suspense + atmosphere"),
    (frozenset({"Comedy"}), "lighthearted + witty"),
]

def _genre_sets(selected: pd.DataFrame) -> list[set[str]]:
    return [set(split_genre_string(g)) for g in selected["genres"].fillna("")]


def _genre_counter(selected: pd.DataFrame) -> Counter[str]:
    counts: Counter[str] = Counter()
    for genres in _genre_sets(selected):
        counts.update(genres)
    return counts


def infer_persona(selected: pd.DataFrame) -> dict[str, str]:
    all_genres: set[str] = set()
    for genres in _genre_sets(selected):
        all_genres |= genres

    best: dict[str, Any] | None = None
    best_score = 0
    for rule in PERSONA_RULES:
        overlap = len(all_genres & rule["genres"])
        if overlap >= rule["min_overlap"] and overlap > best_score:
            best_score = overlap
            best = rule

    if best:
        return {
            "title": str(best["title"]),
            "emoji": str(best["emoji"]),
            "blurb": str(best["blurb"]),
        }

    return {
        "title": "The Eclectic Cinephile",
        "emoji": "🎬",
        "blurb": (
            "Your picks span genres and eras — you refuse one lane and chase variety, "
            "from blockbusters to hidden gems."
        ),
    }


def taste_dna(selected: pd.DataFrame) -> dict[str, Any]:
    counts = _genre_counter(selected)
    total = sum(counts.values()) or 1
    top = counts.most_common(6)
    genre_pct = [(g, round(100 * c / total)) for g, c in top]

    all_genres = set(counts.keys())
    style = "balanced + genre-mixing"
    for needed, label in STYLE_RULES:
        if needed <= all_genres:
            style = label
            break

    years = [y for y in selected["release_year"].tolist() if pd.notna(y)]
    era = "Mixed eras"
    if years:
        y_min, y_max = int(min(years)), int(max(years))
        if y_max - y_min <= 12:
            era = f"{y_min}–{y_max}"
        else:
            top_decades = decade_summary(years, top_n=1)
            if top_decades:
                era = f"{top_decades[0][0]} (peak)"

    themes: list[str] = []
    for needed, tag in THEME_TAGS:
        if needed <= all_genres:
            themes.append(tag)
        if len(themes) >= 3:
            break
    if not themes:
        themes = ["variety across story and tone"]

    return {
        "genre_pct": genre_pct,
        "primary_genre": genre_pct[0] if genre_pct else ("Mixed", 0),
        "secondary_genre": genre_pct[1] if len(genre_pct) > 1 else None,
        "style": style,
        "era": era,
        "themes": themes,
    }
