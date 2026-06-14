"""Viewer taste profiles from favorite picks (Phase 6)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from scripts.taste_utils import decade_summary, genre_summary
from scripts.model_helpers import split_genre_string
from scripts.persona import infer_persona, taste_dna


def _rule_based_profile(selected: pd.DataFrame, persona: dict[str, str], dna: dict[str, Any]) -> str:
    genre_lists = [split_genre_string(g) for g in selected["genres"].fillna("")]
    top_genres = genre_summary(genre_lists, top_n=4)
    top_decades = decade_summary(selected["release_year"].tolist(), top_n=2)

    genre_phrase = ", ".join(g for g, _ in top_genres) if top_genres else "varied genres"
    decade_phrase = ", ".join(d for d, _ in top_decades) if top_decades else "many eras"

    primary = dna["primary_genre"]
    secondary = dna.get("secondary_genre")
    pct_line = f"{primary[0]} ({primary[1]}%)"
    if secondary:
        pct_line += f", with a strong {secondary[0]} streak ({secondary[1]}%)"

    return (
        f"As **{persona['emoji']} {persona['title']}**, you lean toward {pct_line}. "
        f"Your style reads as **{dna['style']}**, especially from the {decade_phrase}, "
        f"with {genre_phrase} showing up most often."
    )


def taste_profile_stats(selected: pd.DataFrame) -> dict[str, Any]:
    genre_lists = [split_genre_string(g) for g in selected["genres"].fillna("")]
    return {
        "top_genres": genre_summary(genre_lists, top_n=6),
        "top_decades": decade_summary(selected["release_year"].tolist(), top_n=4),
        "count": len(selected),
    }


def generate_taste_profile(selected: pd.DataFrame) -> dict[str, Any]:
    """Return persona, Taste DNA stats, and a written summary."""
    stats = taste_profile_stats(selected)
    persona = infer_persona(selected)
    dna = taste_dna(selected)
    return {
        "stats": stats,
        "persona": persona,
        "dna": dna,
        "summary": _rule_based_profile(selected, persona, dna),
    }
