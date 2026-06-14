"""Shared genre/decade helpers and theme tags (no circular imports)."""

from __future__ import annotations

from collections import Counter

import pandas as pd

THEME_TAGS: list[tuple[frozenset[str], str]] = [
    (frozenset({"Mystery", "Thriller", "Crime"}), "espionage & betrayal plots"),
    (frozenset({"Action", "Adventure"}), "high-stakes action set pieces"),
    (frozenset({"Drama"}), "morally complex characters"),
    (frozenset({"Mystery", "Thriller"}), "slow pacing & tension"),
    (frozenset({"Sci-Fi"}), "speculative ideas & future worlds"),
    (frozenset({"Horror"}), "dread & atmosphere"),
    (frozenset({"Romance", "Comedy"}), "warmth & human connection"),
    (frozenset({"War"}), "conflict & sacrifice"),
]


def decade_label(year: float | int | None) -> str | None:
    if year is None or pd.isna(year):
        return None
    y = int(year)
    return f"{(y // 10) * 10}s"


def genre_summary(genre_lists: list[list[str]], top_n: int = 5) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter()
    for genres in genre_lists:
        counts.update(genres)
    return counts.most_common(top_n)


def decade_summary(years: list[float | int | None], top_n: int = 3) -> list[tuple[str, int]]:
    decades: Counter[str] = Counter()
    for year in years:
        label = decade_label(year)
        if label:
            decades[label] += 1
    return decades.most_common(top_n)
