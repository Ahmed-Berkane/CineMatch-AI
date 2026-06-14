"""Session feedback (thumbs up/down) to personalize recommendations within a visit."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def init_feedback_state(session_state: Any) -> None:
    if "movie_feedback" not in session_state:
        session_state.movie_feedback: dict[int, int] = {}


def feedback_key(session_state: Any) -> tuple[tuple[int, int], ...]:
    """Hashable snapshot of current feedback for cache keys."""
    fb: dict[int, int] = session_state.movie_feedback
    return tuple(sorted(fb.items()))


def record_feedback(
    session_state: Any,
    *,
    movie_id: int,
    title: str,
    vote: int,
    context: str = "",
) -> bool:
    """Store thumbs feedback for this session. Returns True if vote was cleared (toggle off)."""
    init_feedback_state(session_state)
    new_vote = 1 if vote > 0 else -1
    current = session_state.movie_feedback.get(int(movie_id), 0)
    if current == new_vote:
        session_state.movie_feedback.pop(int(movie_id), None)
        return True
    session_state.movie_feedback[int(movie_id)] = new_vote
    return False


def liked_and_disliked(session_state: Any) -> tuple[list[int], list[int]]:
    init_feedback_state(session_state)
    liked = [mid for mid, v in session_state.movie_feedback.items() if v > 0]
    disliked = [mid for mid, v in session_state.movie_feedback.items() if v < 0]
    return liked, disliked


def feedback_counts(session_state: Any) -> tuple[int, int]:
    liked, disliked = liked_and_disliked(session_state)
    return len(liked), len(disliked)


def apply_feedback_to_scores(
    movie_ids: np.ndarray,
    base_scores: np.ndarray,
    *,
    mappings,
    movie_embeddings: np.ndarray,
    liked_ids: list[int],
    disliked_ids: list[int],
    like_boost: float = 0.18,
    dislike_penalty: float = 0.22,
    similarity_weight: float = 0.15,
) -> np.ndarray:
    """Adjust recommendation scores from in-session thumbs up/down."""
    scores = base_scores.copy()
    id_to_idx = mappings.movie_to_idx

    for mid in disliked_ids:
        if mid in id_to_idx:
            scores[id_to_idx[mid]] -= 0.55

    for mid in liked_ids:
        if mid in id_to_idx:
            scores[id_to_idx[mid]] += like_boost

    if liked_ids:
        liked_idx = [id_to_idx[mid] for mid in liked_ids if mid in id_to_idx]
        if liked_idx:
            liked_emb = movie_embeddings[liked_idx].mean(axis=0, keepdims=True)
            sims = cosine_similarity(liked_emb, movie_embeddings).flatten()
            scores += similarity_weight * sims

    if disliked_ids:
        disliked_idx = [id_to_idx[mid] for mid in disliked_ids if mid in id_to_idx]
        if disliked_idx:
            disliked_emb = movie_embeddings[disliked_idx].mean(axis=0, keepdims=True)
            sims = cosine_similarity(disliked_emb, movie_embeddings).flatten()
            scores -= similarity_weight * sims

    return scores
