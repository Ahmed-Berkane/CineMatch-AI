"""
CineMatch-AI Streamlit app — Phases 5–7.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from st_keyup import st_keyup

ROOT = Path(__file__).resolve().parent
LOGO_PATH = ROOT / "Logo.png"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from scripts.catalog import latest_movies, load_movie_catalog, movies_by_ids, suggest_movies
from scripts.recommender import get_engine, pick_page_suggestions
from scripts.feedback import (
    feedback_counts,
    feedback_key,
    init_feedback_state,
    record_feedback,
)
from scripts.taste_profile import generate_taste_profile

st.set_page_config(
    page_title="CineMatch AI",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAV_PAGES = ["Home", "Pick favorites", "Recommendations", "Taste profile", "Model"]

# Shown when a catalog title is not in the HybridNet checkpoint (not user-facing "training set").
NOT_IN_MODEL_HELP = "This title isn't in our recommendation model yet — try another pick."
NOT_IN_MODEL_BTN = "Unavailable"

POSTER_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* CineMatch — cinematic red on deep charcoal */
:root {
    --cm-accent: #E50914;
    --cm-accent-hover: #F40612;
    --cm-accent-dim: #B20710;
    --cm-accent-deep: #831010;
    --cm-accent-soft: rgba(229, 9, 20, 0.18);
    --cm-accent-glow: rgba(229, 9, 20, 0.35);
    --cm-bg-deep: #0a0a0a;
    --cm-bg-card: #181818;
    --cm-bg-navy: #141414;
}
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}
.stApp {
    background:
        radial-gradient(ellipse 85% 50% at 50% -10%, rgba(229, 9, 20, 0.08) 0%, transparent 55%),
        radial-gradient(ellipse 50% 35% at 100% 90%, rgba(131, 16, 16, 0.1) 0%, transparent 50%),
        linear-gradient(168deg, #141414 0%, var(--cm-bg-navy) 45%, var(--cm-bg-deep) 100%);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}
.cinematch-bg {
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    overflow: hidden;
}
.cinematch-bg span {
    position: absolute;
    font-size: clamp(2rem, 5vw, 4.5rem);
    opacity: 0.04;
    color: var(--cm-accent);
    user-select: none;
    text-shadow: 0 0 40px var(--cm-accent-glow);
}
.cinematch-bg .s1 { top: 8%;  left: 6%;  transform: rotate(-12deg); }
.cinematch-bg .s2 { top: 22%; right: 8%; transform: rotate(8deg);  }
.cinematch-bg .s3 { top: 55%; left: 4%;  transform: rotate(6deg);  }
.cinematch-bg .s4 { top: 70%; right: 12%; transform: rotate(-6deg); }
.cinematch-bg .s5 { top: 40%; left: 45%; transform: rotate(15deg); font-size: 3rem; }
.cinematch-bg .s6 { bottom: 8%; left: 35%; transform: rotate(-8deg); }
[data-testid="stAppViewContainer"] {
    position: relative;
    z-index: 1;
}
header[data-testid="stHeader"] {
    background: rgba(11, 11, 11, 0.94) !important;
    border-bottom: 1px solid rgba(229, 9, 20, 0.15);
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b0b0b 0%, var(--cm-bg-deep) 100%) !important;
    border-right: 1px solid rgba(229, 9, 20, 0.12);
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] label {
    color: #e5e5e5 !important;
}
h1, h2, h3, h4 {
    font-weight: 600 !important;
    letter-spacing: -0.02em;
    font-family: 'Inter', sans-serif !important;
}
h2 {
    border-bottom: none;
    padding-bottom: 0.35rem;
    margin-bottom: 0.75rem !important;
    background: linear-gradient(90deg, var(--cm-accent) 0%, var(--cm-accent-hover) 50%, var(--cm-accent-deep) 100%);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent !important;
    position: relative;
}
h2::after {
    content: "";
    display: block;
    height: 2px;
    width: 100%;
    max-width: 220px;
    margin-top: 0.4rem;
    border-radius: 2px;
    background: linear-gradient(90deg, var(--cm-accent) 0%, var(--cm-accent-hover) 70%, transparent 100%);
    box-shadow: 0 0 12px var(--cm-accent-glow);
}
.stCaption, small {
    color: #b3b3b3 !important;
    letter-spacing: 0.01em;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(145deg, #1c1c1c 0%, var(--cm-bg-card) 100%) !important;
    border: 1px solid rgba(229, 9, 20, 0.12) !important;
    border-radius: 12px !important;
    padding: 0.85rem 1rem 1rem 1rem !important;
    margin-bottom: 0.65rem !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 28px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(229, 9, 20, 0.08) !important;
    border-color: rgba(229, 9, 20, 0.28) !important;
}
.stButton > button {
    transition: transform 0.15s ease, background 0.15s ease, box-shadow 0.15s ease !important;
}
.stButton > button:hover:not(:disabled) {
    transform: translateY(-1px);
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.35) !important;
}
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, var(--cm-accent-dim) 0%, var(--cm-accent-hover) 100%) !important;
    box-shadow: 0 0 10px var(--cm-accent-glow);
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--cm-accent-dim) 0%, var(--cm-accent) 50%, var(--cm-accent-hover) 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 10px rgba(229, 9, 20, 0.3) !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, var(--cm-accent) 0%, var(--cm-accent-hover) 100%) !important;
    color: #fff !important;
    box-shadow: 0 4px 16px var(--cm-accent-glow) !important;
}
.stButton > button[kind="secondary"] {
    background: #333 !important;
    color: #fff !important;
    border: 1px solid #555 !important;
    border-radius: 4px !important;
}
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div,
div[data-testid="stTextInput"] > div > div {
    background-color: #222 !important;
    border: 1px solid #333 !important;
    border-radius: 8px !important;
}
div[data-baseweb="input"] input,
div[data-testid="stTextInput"] input {
    background-color: #222 !important;
    color: #E5E5E5 !important;
    caret-color: var(--cm-accent-hover) !important;
}
div[data-baseweb="input"] > div:focus-within,
div[data-testid="stTextInput"] > div > div:focus-within {
    border-color: var(--cm-accent) !important;
    box-shadow: 0 0 0 1px var(--cm-accent-soft) !important;
}
.cinematch-poster {
    border: 1px solid rgba(229, 9, 20, 0.2);
    border-radius: 8px;
    overflow: hidden;
    background: #111;
    margin: 0 auto 0.75rem auto;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.45);
    transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease;
}
.cinematch-poster:hover {
    transform: scale(1.03);
    box-shadow: 0 12px 28px rgba(0, 0, 0, 0.55), 0 0 20px var(--cm-accent-soft);
    border-color: rgba(229, 9, 20, 0.45);
}
.cinematch-poster-xs {
    max-width: 56px;
    height: 84px;
}
.cinematch-poster-xs img {
    max-height: 84px;
}
.cinematch-poster img {
    display: block;
    width: 100%;
    object-fit: contain;
    object-position: center;
}
.cinematch-poster-sm {
    width: 120px;
    max-width: 100%;
    height: 180px;
}
.cinematch-poster-sm img {
    max-height: 180px;
}
.cinematch-poster-fav {
    width: 72px !important;
    max-width: 72px !important;
    height: 108px !important;
    margin: 0 auto 0.35rem auto;
}
.cinematch-poster-fav img {
    max-height: 108px !important;
    height: 100% !important;
    object-fit: cover;
}
.cinematch-poster-grid {
    width: 120px;
    max-width: 100%;
    height: 180px;
    margin-left: auto;
    margin-right: auto;
}
.cinematch-poster-grid img {
    max-height: 180px;
    height: 100%;
    object-fit: cover;
}
.cinematch-poster-md {
    max-width: 140px;
    height: 210px;
}
.cinematch-poster-md img {
    max-height: 210px;
}
.cinematch-suggest-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.35rem 0;
    border-bottom: 1px solid rgba(229, 9, 20, 0.1);
}
.cinematch-suggest-row:last-child {
    border-bottom: none;
}
.cinematch-card-title {
    min-height: 3.4rem;
    max-height: 3.4rem;
    line-height: 1.35;
    margin-bottom: 0.25rem;
    overflow: hidden;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
}
.cinematch-card-meta {
    min-height: 2.5rem;
    max-height: 2.5rem;
    line-height: 1.3;
    margin-bottom: 0.5rem;
    color: #b3b3b3 !important;
    font-size: 0.875rem;
    overflow: hidden;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
}
/* Equal-height movie cards in catalog/suggestion grids only */
[data-testid="stHorizontalBlock"]:has(.cinematch-poster-grid) {
    align-items: stretch !important;
}
[data-testid="stHorizontalBlock"]:has(.cinematch-poster-grid) > [data-testid="column"] {
    display: flex !important;
    flex-direction: column !important;
    align-self: stretch !important;
}
[data-testid="column"]:has(.cinematch-poster-grid) > div[data-testid="stVerticalBlock"] {
    flex: 1 1 auto !important;
    display: flex !important;
    flex-direction: column !important;
    min-height: 0 !important;
    height: auto !important;
}
[data-testid="column"]:has(.cinematch-poster-grid) {
    text-align: center !important;
}
[data-testid="column"]:has(.cinematch-poster-grid) [data-testid="stVerticalBlockBorderWrapper"] {
    flex: 1 1 auto !important;
    display: flex !important;
    flex-direction: column !important;
    width: 100% !important;
    min-height: 360px !important;
    height: auto !important;
    box-sizing: border-box !important;
}
[data-testid="column"]:has(.cinematch-poster-grid) [data-testid="stVerticalBlockBorderWrapper"] > [data-testid="stVerticalBlock"] {
    flex: 1 1 auto !important;
    display: flex !important;
    flex-direction: column !important;
    min-height: 0 !important;
    height: auto !important;
}
[data-testid="column"]:has(.cinematch-poster-grid) [data-testid="stVerticalBlockBorderWrapper"] .stButton:first-of-type {
    margin-top: auto !important;
    padding-top: 0.35rem !important;
}
.cinematch-favorite-card [data-testid="stVerticalBlockBorderWrapper"] {
    min-height: unset !important;
    height: auto !important;
    padding: 0.65rem 0.75rem !important;
}
[data-testid="column"]:has(.cinematch-poster-fav) [data-testid="stVerticalBlockBorderWrapper"] {
    min-height: unset !important;
    height: auto !important;
}
.cinematch-favorite-title {
    font-size: 0.95rem;
    line-height: 1.3;
    margin-bottom: 0.15rem;
}
.cinematch-favorite-meta {
    color: #b3b3b3 !important;
    font-size: 0.78rem;
    line-height: 1.25;
    margin-bottom: 0.45rem;
}
.cinematch-persona-card {
    border-left: 4px solid var(--cm-accent);
    background: linear-gradient(90deg, rgba(229, 9, 20, 0.12) 0%, transparent 100%);
    padding: 1rem 1.1rem;
    border-radius: 8px;
    margin-bottom: 1rem;
}

/* --- Responsive: tablet & phone (Hugging Face, mobile browsers) --- */
.main .block-container {
    padding-top: 1.25rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}
[data-testid="stDataFrame"], [data-testid="stTable"] {
    overflow-x: auto !important;
    -webkit-overflow-scrolling: touch;
}
[data-testid="stMetric"] {
    min-width: 0;
}
[data-testid="stMetricValue"] {
    font-size: clamp(1.1rem, 4vw, 1.75rem) !important;
}

@media (max-width: 992px) {
    .main .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    [data-testid="column"]:has(.cinematch-poster-grid) [data-testid="stVerticalBlockBorderWrapper"] {
        min-height: 340px !important;
    }
    [data-testid="stHorizontalBlock"]:has(.cinematch-poster-grid) {
        flex-wrap: wrap !important;
        gap: 0.5rem !important;
    }
    [data-testid="stHorizontalBlock"]:has(.cinematch-poster-grid) > [data-testid="column"] {
        flex: 1 1 calc(50% - 0.5rem) !important;
        min-width: calc(50% - 0.5rem) !important;
        width: calc(50% - 0.5rem) !important;
    }
}

@media (max-width: 768px) {
    /* Stack recommendation + search rows vertically */
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.cinematch-poster-md) [data-testid="stHorizontalBlock"],
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.cinematch-suggest-row) [data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
        align-items: stretch !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.cinematch-poster-md) [data-testid="column"],
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.cinematch-suggest-row) [data-testid="column"] {
        width: 100% !important;
        min-width: 100% !important;
        flex: 1 1 100% !important;
    }
    .cinematch-poster-md {
        max-width: 140px;
        margin: 0 auto 0.75rem auto !important;
    }
}

@media (max-width: 640px) {
    .main .block-container {
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
    }
    h1 { font-size: 1.5rem !important; }
    h2 { font-size: 1.25rem !important; }
    h3 { font-size: 1.1rem !important; }
    .cinematch-bg { display: none; }
    [data-testid="column"]:has(.cinematch-poster-grid) [data-testid="stVerticalBlockBorderWrapper"] {
        min-height: 320px !important;
    }
    .cinematch-card-title {
        min-height: 3rem;
        max-height: 3rem;
    }
    .cinematch-card-meta {
        min-height: 2.25rem;
        max-height: 2.25rem;
    }
    .cinematch-poster-sm {
        max-width: 100px;
        height: 150px;
    }
    .cinematch-poster-sm img {
        max-height: 150px;
    }
    .cinematch-poster-md {
        max-width: 110px;
        height: 165px;
    }
    .cinematch-poster-md img {
        max-height: 165px;
    }
    .cinematch-persona-card {
        padding: 0.85rem;
        font-size: 0.95rem;
    }
    [data-testid="stHorizontalBlock"]:has(.cinematch-poster-grid) > [data-testid="column"] {
        flex: 1 1 100% !important;
        min-width: 100% !important;
        width: 100% !important;
    }
    .stButton > button {
        min-height: 2.75rem !important;
        font-size: 0.95rem !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        padding: 0.65rem 0.75rem !important;
    }
    section[data-testid="stSidebar"] {
        min-width: min(85vw, 280px) !important;
    }
}

@media (hover: none) {
    div[data-testid="stVerticalBlockBorderWrapper"]:hover,
    .cinematch-poster:hover {
        transform: none !important;
    }
}
</style>
"""


def render_page_title(title: str, subtitle: str = "") -> None:
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)


def render_taste_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, *, y_label: str) -> None:
    """Bar chart with horizontal x-axis labels (Streamlit default rotates them)."""
    plot_df = df.copy()
    fig, ax = plt.subplots(figsize=(10, 4.2))
    fig.patch.set_facecolor("#141414")
    ax.set_facecolor("#141414")
    ax.bar(plot_df[x_col].astype(str), plot_df[y_col], color="#E50914", width=0.62)
    ax.set_ylabel(y_label, color="#b3b3b3")
    ax.tick_params(axis="x", rotation=0, colors="#b3b3b3", labelsize=10)
    ax.tick_params(axis="y", colors="#b3b3b3")
    ax.grid(axis="y", color="#333333", linestyle="-", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color("#444444")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


@st.cache_data(show_spinner=False)
def load_logo_bytes() -> bytes | None:
    """Read logo once; bytes work reliably on HF Docker (path strings can fail)."""
    if not LOGO_PATH.exists():
        return None
    data = LOGO_PATH.read_bytes()
    # Reject Git LFS pointer files or corrupt assets.
    if len(data) < 8 or data[:4] != b"\x89PNG":
        return None
    return data


def render_sidebar_logo() -> None:
    logo = load_logo_bytes()
    if logo:
        st.sidebar.image(logo, width=160)
    else:
        st.sidebar.markdown("### 🎬")


def render_poster(
    url: str | None,
    *,
    size: str = "md",
    alt: str = "Movie poster",
    grid: bool = False,
    eager: bool = False,
) -> None:
    """Render a poster inside a fixed-size framed box."""
    if not url or (isinstance(url, float) and pd.isna(url)):
        if size != "xs":
            st.caption("No poster")
        return
    css_class = {
        "xs": "cinematch-poster-xs",
        "sm": "cinematch-poster-sm",
        "md": "cinematch-poster-md",
        "fav": "cinematch-poster-fav",
    }.get(size, "cinematch-poster-md")
    grid_class = " cinematch-poster-grid" if grid else ""
    loading = "eager" if eager or grid else "lazy"
    safe_url = html.escape(str(url), quote=True)
    safe_alt = html.escape(alt, quote=True)
    st.markdown(
        f'<div class="cinematch-poster {css_class}{grid_class}">'
        f'<img src="{safe_url}" alt="{safe_alt}" loading="{loading}" decoding="async" />'
        f"</div>",
        unsafe_allow_html=True,
    )


def _add_button(
    movie_id: int,
    title: str,
    *,
    key: str,
    disabled: bool = False,
    disabled_help: str | None = None,
) -> None:
    already = movie_id in st.session_state.selected_ids
    full = len(st.session_state.selected_ids) >= 10
    if disabled:
        label = NOT_IN_MODEL_BTN
        button_disabled = True
        kind = "secondary"
    else:
        label = "Added" if already else "Add"
        button_disabled = already or full
        kind = "primary" if not already and not full else "secondary"
    if st.button(
        label,
        key=key,
        disabled=button_disabled,
        type=kind,
        use_container_width=True,
        help=disabled_help if disabled and disabled_help else None,
    ):
        if not disabled:
            _add_favorite(movie_id, title)
            st.rerun()


def render_search_hit_row(
    row,
    *,
    key_prefix: str = "search",
    supported: bool = True,
    disabled_help: str | None = None,
) -> None:
    with st.container(border=True):
        c1, c2, c3 = st.columns([1, 5, 1], vertical_alignment="center")
        with c1:
            render_poster(
                row.poster_url if pd.notna(row.poster_url) else None,
                size="xs",
                alt=str(row.title),
            )
        with c2:
            st.markdown(f"**{row.title}**")
            year = int(row.release_year) if pd.notna(row.release_year) else "?"
            st.caption(f"{row.genres} · {year}")
            if not supported and disabled_help:
                st.caption(disabled_help)
        with c3:
            _add_button(
                int(row.movieId),
                str(row.title),
                key=f"{key_prefix}_{row.movieId}",
                disabled=not supported,
                disabled_help=disabled_help,
            )


def render_grid_movie_card(
    *,
    movie_id: int,
    title: str,
    genres: str,
    year: str | int | float | None,
    poster_url: str | None,
    key_prefix: str,
    show_feedback: bool = False,
    feedback_context: str = "picker_suggestion",
    disabled: bool = False,
    disabled_help: str | None = None,
) -> None:
    """Equal-height movie card for grid layouts."""
    year_label = int(year) if year is not None and year != "?" and pd.notna(year) else "?"
    with st.container(border=True):
        render_poster(
            poster_url if poster_url and pd.notna(poster_url) else None,
            size="sm",
            alt=title,
            grid=True,
            eager=True,
        )
        st.markdown(
            f'<div class="cinematch-card-title"><strong>{html.escape(title)}</strong></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="cinematch-card-meta">{html.escape(str(genres))} · {year_label}</div>',
            unsafe_allow_html=True,
        )
        _add_button(
            movie_id,
            title,
            key=f"{key_prefix}_{movie_id}",
            disabled=disabled,
            disabled_help=disabled_help,
        )
        if disabled and disabled_help:
            st.caption(disabled_help)
        if show_feedback:
            render_feedback_buttons(movie_id, title, key_prefix=f"{key_prefix}_fb", context=feedback_context)


def render_favorite_card(row) -> None:
    """Compact favorite tile: small poster beside title + remove."""
    with st.container(border=True):
        st.markdown('<div class="cinematch-favorite-card">', unsafe_allow_html=True)
        c1, c2 = st.columns([1, 2.2], vertical_alignment="center", gap="small")
        with c1:
            render_poster(
                row.poster_url if pd.notna(row.poster_url) else None,
                size="fav",
                alt=str(row.title),
            )
        with c2:
            st.markdown(
                f'<div class="cinematch-favorite-title"><strong>{html.escape(str(row.title))}</strong></div>',
                unsafe_allow_html=True,
            )
            year = int(row.release_year) if pd.notna(row.release_year) else "?"
            st.markdown(
                f'<div class="cinematch-favorite-meta">{html.escape(str(row.genres))} · {year}</div>',
                unsafe_allow_html=True,
            )
            if st.button("Remove", key=f"rm_{row.movieId}", use_container_width=True):
                st.session_state.selected_ids = [
                    mid for mid in st.session_state.selected_ids if mid != row.movieId
                ]
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

if "selected_ids" not in st.session_state:
    st.session_state.selected_ids: list[int] = []
if "last_added_title" not in st.session_state:
    st.session_state.last_added_title = None
if "search_input_id" not in st.session_state:
    st.session_state.search_input_id = 0
init_feedback_state(st.session_state)


def render_feedback_buttons(movie_id: int, title: str, *, key_prefix: str, context: str) -> None:
    """Thumbs up/down — stored for this session until the app is closed."""
    current = st.session_state.movie_feedback.get(movie_id, 0)
    c1, c2, c3 = st.columns([1, 1, 6])
    with c1:
        if st.button(
            "👍",
            key=f"{key_prefix}_up_{movie_id}",
            help="More like this (click again to undo)",
            type="primary" if current == 1 else "secondary",
        ):
            cleared = record_feedback(
                st.session_state,
                movie_id=movie_id,
                title=title,
                vote=1,
                context=context,
            )
            if cleared:
                st.toast(f"Removed 👍 for {title}")
            else:
                st.toast(f"👍 Noted — we'll favor titles like {title}")
            st.rerun()
    with c2:
        if st.button(
            "👎",
            key=f"{key_prefix}_down_{movie_id}",
            help="Not for me (click again to undo)",
            type="primary" if current == -1 else "secondary",
        ):
            cleared = record_feedback(
                st.session_state,
                movie_id=movie_id,
                title=title,
                vote=-1,
                context=context,
            )
            if cleared:
                st.toast(f"Removed 👎 for {title}")
            else:
                st.toast(f"👎 Noted — we'll down-rank similar titles")
            st.rerun()
    with c3:
        if current == 1:
            st.caption("👍 Active — boosts similar titles in this session.")
        elif current == -1:
            st.caption("👎 Active — hides this title and penalizes similar ones.")


def _add_favorite(movie_id: int, title: str) -> None:
    if movie_id in st.session_state.selected_ids:
        st.session_state.last_added_title = None
        return
    if len(st.session_state.selected_ids) >= 10:
        return
    st.session_state.selected_ids.append(movie_id)
    st.session_state.last_added_title = title


# Bump when RecommenderEngine API changes so Streamlit cache refreshes.
ENGINE_CACHE_VERSION = 5
# Bump when movies_catalog.parquet or catalog helpers change.
CATALOG_CACHE_VERSION = 3


@st.cache_resource(show_spinner="Loading HybridNet (once per session)…")
def load_engine(_cache_version: int = ENGINE_CACHE_VERSION):
    return get_engine()


@st.cache_data(show_spinner=False)
def get_catalog(_cache_version: int = CATALOG_CACHE_VERSION):
    return load_movie_catalog()


@st.cache_data(show_spinner=False)
def cached_movies_by_ids(seed_key: tuple[int, ...]) -> pd.DataFrame:
    return movies_by_ids(list(seed_key), get_catalog())


@st.cache_data(show_spinner=False)
def _get_supported_movie_ids() -> frozenset[int]:
    engine = load_engine()
    return frozenset(engine.mappings.movie_to_idx.keys())


def _prune_unsupported_selected_ids() -> list[int]:
    supported_ids = _get_supported_movie_ids()
    selected = st.session_state.selected_ids
    unsupported = [mid for mid in selected if mid not in supported_ids]
    if unsupported:
        st.warning(
            "Some favorites aren't covered by our recommendation model and were removed. "
            "Please pick another title from search or the home page."
        )
        st.session_state.selected_ids = [mid for mid in selected if mid in supported_ids]
    return unsupported


@st.cache_data(show_spinner="Building recommendations…")
def cached_recommend(
    seed_key: tuple[int, ...],
    top_k: int,
    include_cohort: bool,
    fb_key: tuple[tuple[int, int], ...],
    diversity: float,
) -> list[dict]:
    engine = load_engine()
    liked, disliked = [], []
    if fb_key:
        liked = [mid for mid, v in fb_key if v > 0]
        disliked = [mid for mid, v in fb_key if v < 0]
    recs = engine.recommend(
        list(seed_key),
        top_k=top_k,
        include_cohort_explanations=include_cohort,
        liked_ids=liked,
        disliked_ids=disliked,
        diversity=diversity,
    )
    recs.sort(key=lambda r: r.score, reverse=True)
    return [
        {
            "movie_id": r.movie_id,
            "title": r.title,
            "genres": r.genres,
            "release_year": r.release_year,
            "poster_url": r.poster_url,
            "score": r.score,
            "content_score": r.content_score,
            "collaborative_score": r.collaborative_score,
            "headline": r.headline,
            "explanations": r.explanations,
            "theme_tags": r.theme_tags,
        }
        for r in recs
    ]


THEME_CSS_VERSION = "nav-v5"


def inject_app_theme() -> None:
    theme_key = f"_theme_injected_{THEME_CSS_VERSION}"
    if st.session_state.get(theme_key):
        return
    st.session_state[theme_key] = True
    # st.html is reliable on hosted Streamlit; markdown CSS is often stripped.
    st.html(POSTER_CSS)


inject_app_theme()


def scroll_to_top_if_page_changed(page: str) -> None:
    """Reset scroll when switching sidebar pages (Streamlit keeps scroll position by default)."""
    if st.session_state.get("_nav_page") == page:
        return
    st.session_state._nav_page = page
    # st.html without height= — HF Streamlit rejects the height kwarg.
    st.html(
        """
        <script>
        (function () {
            function goTop() {
                try { window.scrollTo(0, 0); } catch (e) {}
                try {
                    var p = window.parent;
                    if (p && p !== window) {
                        p.scrollTo(0, 0);
                        var doc = p.document;
                        if (doc) {
                            doc.documentElement.scrollTop = 0;
                            doc.body.scrollTop = 0;
                            var main = doc.querySelector("section.main");
                            if (main) main.scrollTop = 0;
                            var view = doc.querySelector('[data-testid="stAppViewContainer"]');
                            if (view) view.scrollTop = 0;
                        }
                    }
                } catch (e) {}
            }
            goTop();
            setTimeout(goTop, 50);
            setTimeout(goTop, 200);
        })();
        </script>
        """
    )


@st.cache_data(show_spinner=False)
def cached_pick_suggestions(
    seed_key: tuple[int, ...],
    limit: int,
    fb_key: tuple[tuple[int, int], ...],
) -> list[dict]:
    liked = [mid for mid, v in fb_key if v > 0]
    disliked = [mid for mid, v in fb_key if v < 0]
    return pick_page_suggestions(
        list(seed_key),
        limit=limit,
        liked_ids=liked,
        disliked_ids=disliked,
    )


@st.fragment
def live_search_box() -> None:
    """Search that updates on every keystroke (no Tab / Enter needed)."""
    catalog = get_catalog()
    search_key = f"movie_search_live_{st.session_state.search_input_id}"
    search_col, clear_col = st.columns([11, 1], vertical_alignment="bottom")
    with search_col:
        query = st_keyup(
            "Search by title, genre, or director",
            key=search_key,
            placeholder="Start typing… e.g. Skyfall, Bond, Sci-Fi",
            debounce=300,
        )
    with clear_col:
        has_query = bool((query or "").strip())
        if st.button(
            "✕",
            key=f"clear_movie_search_{st.session_state.search_input_id}",
            help="Clear search and hide results",
            disabled=not has_query,
            use_container_width=True,
        ):
            st.session_state.search_input_id += 1
            st.rerun()

    q = (query or "").strip()
    if len(q) < 2:
        if q:
            st.caption("Keep typing — suggestions appear after 2 characters.")
        return

    supported_ids = _get_supported_movie_ids()
    hits = suggest_movies(q, catalog, limit=16)
    if not hits.empty:
        hits = hits.copy()
        hits["_eligible"] = hits["movieId"].isin(supported_ids).astype(int)
        hits = hits.sort_values("_eligible", ascending=False).drop(columns="_eligible").head(8)
    if hits.empty:
        st.warning(
            f"No movies matched **{q}**. Try another spelling or genre. "
            "TV series are not in this catalog."
        )
        return

    st.caption("Suggestions update as you type — eligible titles appear first:")
    for row in hits.itertuples(index=False):
        supported = int(row.movieId) in supported_ids
        render_search_hit_row(
            row,
            key_prefix="search",
            supported=supported,
            disabled_help=NOT_IN_MODEL_HELP if not supported else None,
        )


def render_smart_suggestions() -> None:
    """Franchise + hybrid picks shown below search."""
    if not st.session_state.selected_ids:
        return

    picks = cached_pick_suggestions(
        tuple(sorted(st.session_state.selected_ids)),
        8,
        feedback_key(st.session_state),
    )
    if not picks:
        return

    supported_ids = _get_supported_movie_ids()
    st.markdown("##### Suggested for you")
    st.caption("Same franchise or similar titles — 👍/👎 personalizes this session instantly.")
    n_cols = 2
    for row_start in range(0, len(picks), n_cols):
        cols = st.columns(n_cols, gap="medium", vertical_alignment="top")
        for j, col in enumerate(cols):
            idx = row_start + j
            if idx >= len(picks):
                break
            item = picks[idx]
            supported = int(item["movie_id"]) in supported_ids
            with col:
                render_grid_movie_card(
                    movie_id=int(item["movie_id"]),
                    title=str(item["title"]),
                    genres=str(item.get("genres") or item.get("reason", "")),
                    year=item.get("release_year", "?"),
                    poster_url=item.get("poster_url"),
                    key_prefix="suggest",
                    show_feedback=True,
                    feedback_context="picker_suggestion",
                    disabled=not supported,
                    disabled_help=NOT_IN_MODEL_HELP if not supported else None,
                )


@st.cache_data(show_spinner=False)
def cached_latest_movies(limit: int = 12, _cache_version: int = CATALOG_CACHE_VERSION) -> pd.DataFrame:
    return latest_movies(
        limit=limit,
        catalog=get_catalog(),
        eligible_ids=_get_supported_movie_ids(),
    )


def render_latest_movies_grid(latest: pd.DataFrame, *, key_prefix: str = "home_add") -> None:
    """Render a full movie grid; data must be loaded before calling (avoids partial rows)."""
    if latest.empty:
        st.info("No recent recommendation-ready titles yet — use **Pick favorites** to search the catalog.")
        return

    with st.container():
        n_cols = 4
        for row_start in range(0, len(latest), n_cols):
            cols = st.columns(n_cols, gap="medium", vertical_alignment="top")
            for j, col in enumerate(cols):
                idx = row_start + j
                if idx >= len(latest):
                    break
                row = latest.iloc[idx]
                with col:
                    year = int(row.release_year) if pd.notna(row.release_year) else "?"
                    render_grid_movie_card(
                        movie_id=int(row.movieId),
                        title=str(row.title),
                        genres=str(row.genres),
                        year=year,
                        poster_url=row.poster_url if pd.notna(row.poster_url) else None,
                        key_prefix=key_prefix,
                    )


def ensure_home_catalog_ready() -> None:
    """Warm catalog once, then rerun so Streamlit paints the full grid (not just row 1)."""
    if st.session_state.get("_home_catalog_ready"):
        return
    with st.spinner("Loading movie catalog…"):
        get_catalog()
    st.session_state._home_catalog_ready = True
    st.rerun()


def nudge_home_layout() -> None:
    """Ask the browser to reflow after the grid mounts (helps HF iframe first paint)."""
    st.html(
        """
        <script>
        (function () {
            function nudge() {
                try { window.dispatchEvent(new Event("resize")); } catch (e) {}
                try {
                    var p = window.parent;
                    if (p && p !== window) {
                        p.dispatchEvent(new Event("resize"));
                        var main = p.document.querySelector("section.main");
                        if (main) { main.style.minHeight = main.scrollHeight + "px"; }
                    }
                } catch (e) {}
            }
            nudge();
            setTimeout(nudge, 80);
            setTimeout(nudge, 300);
        })();
        </script>
        """
    )


def page_home() -> None:
    render_page_title("CineMatch AI")
    st.markdown(
        """
        Welcome to your personal movie matchmaker. Tell us a few films you love,
        and we'll recommend new titles with clear explanations — powered by **HybridNet**
        (32M MovieLens ratings + genre & era signals).

        **Get started:** open **Pick favorites** in the sidebar, search for films you enjoy,
        then explore **Recommendations** and your **Taste profile**.
        """
    )

    n_selected = len(st.session_state.selected_ids)
    if n_selected:
        st.success(f"You have **{n_selected}** favorite(s) saved — ready for recommendations.")
    else:
        st.info("No favorites yet. Head to **Pick favorites** to build your list.")

    st.markdown("---")
    st.markdown("### Latest movies in our catalog")
    st.caption("Popular recent picks you can add as favorites — all shown titles work with HybridNet.")

    latest = cached_latest_movies(12)
    render_latest_movies_grid(latest)
    nudge_home_layout()


def render_sidebar() -> None:
    """Left panel: logo, navigation, and session status."""
    render_sidebar_logo()
    st.sidebar.markdown("### CineMatch AI")
    st.sidebar.caption("Hybrid movie recommendations")
    if "nav_page" not in st.session_state:
        st.session_state.nav_page = NAV_PAGES[0]
    st.sidebar.radio(
        "Navigate",
        NAV_PAGES,
        key="nav_page",
        label_visibility="collapsed",
    )
    st.sidebar.divider()
    st.sidebar.markdown(f"**Selected:** {len(st.session_state.selected_ids)} movie(s)")
    if st.session_state.selected_ids:
        sel = cached_movies_by_ids(tuple(st.session_state.selected_ids))
        for title in sel["title"].head(5):
            st.sidebar.write(f"• {title}")
        if len(sel) > 5:
            st.sidebar.write(f"… and {len(sel) - 5} more")
    n_up, n_down = feedback_counts(st.session_state)
    if n_up or n_down:
        st.sidebar.markdown(f"**Session feedback:** 👍 {n_up}  👎 {n_down}")
        st.sidebar.caption("Thumbs adjust recommendations until you close the app.")


def page_pick_favorites() -> None:
    render_page_title(
        "Pick your favorites",
        "Search above, then review your picks and smart suggestions below.",
    )
    if st.session_state.selected_ids:
        _prune_unsupported_selected_ids()
    n_selected = len(st.session_state.selected_ids)
    st.progress(min(n_selected / 5, 1.0), text=f"{n_selected}/5 favorites selected (aim for 3–5)")

    if st.session_state.last_added_title:
        st.success(f"Added **{st.session_state.last_added_title}**.")
        st.session_state.last_added_title = None

    with st.container(border=True):
        st.markdown("##### Search movies")
        st.caption(
            "Search the full catalog — titles you can use as favorites appear first; "
            "others are browse-only until we expand the model."
        )
        live_search_box()

    if n_selected:
        with st.container(border=True):
            st.markdown("##### Your favorites")
            selected = cached_movies_by_ids(tuple(st.session_state.selected_ids))
            fav_rows = list(selected.itertuples(index=False))
            per_row = min(len(fav_rows), 5)
            for row_start in range(0, len(fav_rows), per_row):
                cols = st.columns(per_row)
                for j, col in enumerate(cols):
                    idx = row_start + j
                    if idx >= len(fav_rows):
                        break
                    with col:
                        render_favorite_card(fav_rows[idx])
            c1, c2 = st.columns([1, 4])
            with c1:
                if st.button("Clear all", use_container_width=True):
                    st.session_state.selected_ids = []
                    st.rerun()
            with c2:
                if n_selected >= 3:
                    st.info("Ready — open **Recommendations** from the menu.")

        with st.container(border=True):
            render_smart_suggestions()
    else:
        st.info("Type in the search box to find movies and build your list.")


def page_recommendations() -> None:
    render_page_title(
        "Your recommendations",
        "Updates automatically from your favorites and 👍/👎 feedback.",
    )
    if st.session_state.selected_ids:
        _prune_unsupported_selected_ids()
    ids = st.session_state.selected_ids
    if not ids:
        st.warning("Add favorites on the **Pick favorites** page first.")
        return

    top_k = st.slider("Number of recommendations", 5, 20, 10)
    diversity = st.select_slider(
        "Recommendation style",
        options=[0.0, 0.35, 0.7],
        value=0.35,
        format_func=lambda x: {
            0.0: "🎯 Similar only (safe)",
            0.35: "⚖️ Balanced",
            0.7: "🌍 Diverse (explore new genres)",
        }[x],
        help="Higher diversity surfaces titles farther from your usual genre cluster.",
    )
    include_cohort = st.checkbox(
        "Include “users who liked your picks” explanations (slower first run)",
        value=False,
    )

    with st.spinner("Finding your best matches…"):
        seed_key = tuple(sorted(ids))
        fb = feedback_key(st.session_state)
        recs = cached_recommend(seed_key, top_k, include_cohort, fb, diversity)

    if not recs:
        st.warning("No recommendations found — try different favorites.")
        return

    n_up, n_down = feedback_counts(st.session_state)
    if n_up or n_down:
        st.caption(f"Session feedback applied: 👍 {n_up} · 👎 {n_down} — list refreshed below.")

    with st.expander("How do 👍 / 👎 work?"):
        st.markdown(
            """
            - **👍** shifts your session taste vector toward that film’s HybridNet embedding (instant re-rank).
            - **👎** pushes away from that embedding direction and removes the title.
            - Click the same button again to **undo**.
            - Feedback lasts until you **close the app** — it does not retrain the full model offline.
            """
        )

    st.caption(
        "Score = 0.6 × content + 0.4 × HybridNet, adjusted by your 👍/👎 this session."
    )

    for rec in recs:
        with st.container(border=True):
            c1, c2 = st.columns([1, 4], gap="medium")
            with c1:
                render_poster(rec.get("poster_url"), size="md", alt=rec["title"])
            with c2:
                st.subheader(rec["title"])
                st.write(f"**Genres:** {rec['genres']} · **Year:** {rec['release_year']}")
                st.progress(min(max(rec["score"], 0.0), 1.0), text=f"Match score: {rec['score']:.0%}")
                render_feedback_buttons(
                    int(rec["movie_id"]),
                    str(rec["title"]),
                    key_prefix="rec_fb",
                    context="recommendation",
                )
                st.markdown("**Why this recommendation?**")
                headline = rec.get("headline") or rec["explanations"][0]
                st.markdown(headline)
                theme_tags = rec.get("theme_tags") or []
                if theme_tags:
                    st.caption("You tend to prefer films with: " + " · ".join(theme_tags))
                for bullet in rec["explanations"]:
                    st.markdown(f"- ✓ {bullet}")
                with st.expander("Score breakdown"):
                    st.write(
                        f"Content: {rec['content_score']:.0%} · "
                        f"Collaborative (HybridNet): {rec['collaborative_score']:.0%}"
                    )


@st.cache_data(show_spinner="Building your taste profile…")
def cached_taste_profile(seed_key: tuple[int, ...]) -> dict:
    selected = cached_movies_by_ids(seed_key)
    return generate_taste_profile(selected)


def page_taste_profile() -> None:
    render_page_title(
        "Your taste profile",
        "Updates automatically from your favorites — persona, Taste DNA, and charts.",
    )
    if st.session_state.selected_ids:
        _prune_unsupported_selected_ids()
    ids = st.session_state.selected_ids
    if not ids:
        st.warning("Add favorites on the **Pick favorites** page first.")
        return

    profile = cached_taste_profile(tuple(sorted(ids)))
    persona = profile.get("persona", {})
    dna = profile.get("dna", {})
    if persona:
        st.markdown(
            f"""
            <div class="cinematch-persona-card">
                <strong>{html.escape(persona.get("emoji", "🎬"))} Your Current Persona: {html.escape(persona.get("title", ""))}</strong><br/>
                <span style="color:#b3b3b3;">{html.escape(persona.get("blurb", ""))}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if dna:
        st.subheader("Taste DNA")
        d1, d2, d3, d4 = st.columns(4)
        primary = dna.get("primary_genre", ("—", 0))
        secondary = dna.get("secondary_genre")
        with d1:
            st.metric("🎭 Favorite genre", f"{primary[0]} ({primary[1]}%)")
        with d2:
            sec_label = f"{secondary[0]} ({secondary[1]}%)" if secondary else "—"
            st.metric("🕵️ Secondary", sec_label)
        with d3:
            st.metric("🧠 Style", dna.get("style", "—"))
        with d4:
            st.metric("🎬 Era", dna.get("era", "—"))
        themes = dna.get("themes") or []
        if themes:
            st.caption("Scene-type fingerprint: " + " · ".join(themes))

    stats = profile["stats"]
    if stats.get("top_decades"):
        st.subheader("Favorite decades")
        decade_df = pd.DataFrame(stats["top_decades"], columns=["Decade", "Count"])
        render_taste_bar_chart(decade_df, "Decade", "Count", y_label="Count")

    if dna and dna.get("genre_pct"):
        st.subheader("Genre mix (%)")
        pct_df = pd.DataFrame(dna["genre_pct"], columns=["Genre", "Share %"])
        render_taste_bar_chart(pct_df, "Genre", "Share %", y_label="Share %")

    summary = profile.get("summary")
    if summary:
        st.subheader("Your viewer profile")
        st.write(summary)
        st.caption("Generated from your favorite picks")


@st.cache_data(show_spinner=False)
def load_pipeline_report() -> dict | None:
    path = ROOT / "artifacts" / "pipeline_report.json"
    if not path.exists():
        return None
    import json

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def page_model() -> None:
    render_page_title(
        "Model",
        "How HybridNet powers recommendations, explainability, and live session learning.",
    )

    report = load_pipeline_report()
    best = (report or {}).get("best", {})
    val_rmse = best.get("val_RMSE")
    test_rmse = best.get("test_RMSE")
    best_name = best.get("model", "HybridNet")
    metrics_note = ""
    if val_rmse is not None and test_rmse is not None:
        metrics_note = f"Val RMSE **{val_rmse:.4f}** · Test RMSE **{test_rmse:.4f}**"
    elif val_rmse is not None:
        metrics_note = f"Val RMSE **{val_rmse:.4f}**"

    st.subheader("Best model")
    if metrics_note:
        st.success(f"**{best_name}** — {metrics_note}")
    else:
        st.info(
            "**HybridNet** (expected best after full pipeline). "
            "Run `python scripts/train_pipeline.py` locally for metrics in `pipeline_report.json`."
        )

    st.subheader("Architecture & serving")
    arch = pd.DataFrame(
        [
            {"Component": "Training data", "Detail": "MovieLens 32M ratings + TMDb metadata"},
            {"Component": "HybridNet", "Detail": "User & movie embeddings (64-d) + genre/year content → MLP → rating"},
            {"Component": "Selection", "Detail": "Lowest validation RMSE among 6 candidates (Baseline → HybridNet)"},
            {"Component": "App ranking", "Detail": "60% content cosine similarity + 40% HybridNet embedding similarity"},
            {"Component": "Diversity slider", "Detail": "Greedy re-rank penalizes genre overlap (Similar / Balanced / Diverse)"},
        ]
    )
    st.dataframe(arch, use_container_width=True, hide_index=True)

    st.subheader("Session 👍 / 👎 (live re-ranking)")
    st.markdown(
        """
        Thumbs do **not** retrain the offline model on 32M rows — they **shift your session taste vector instantly**:

        | Action | Effect |
        |--------|--------|
        | **👍** | Boost that movie’s score; add its embedding to a session “likes” pool; cosine-similarity boost toward similar titles |
        | **👎** | Hard-penalize the title; push away from its embedding direction; hide it from results |
        | **Undo** | Click the same thumb again to remove the vote |

        Weights: `like_boost ≈ 0.18`, `dislike_penalty ≈ 0.22`, embedding similarity shift `≈ 0.15`.
        Recommendations and smart suggestions refresh when feedback changes.
        """
    )

    st.subheader("Explainability & taste layers")
    layers = pd.DataFrame(
        [
            {
                "Layer": "Why this recommendation?",
                "Detail": "Headline tied to your closest favorite (e.g. “Because you liked X, this shares espionage & slow-burn tension”)",
            },
            {
                "Layer": "Score breakdown",
                "Detail": "Content match % + HybridNet collaborative match % + optional MovieLens cohort note",
            },
            {
                "Layer": "Theme tags",
                "Detail": "Scene-type fingerprint (betrayal plots, morally complex characters, etc.) from genre rules",
            },
            {
                "Layer": "AI Persona",
                "Detail": "Rule-based moviegoer title (e.g. The Covert Operative) on Taste profile page",
            },
            {
                "Layer": "Taste DNA",
                "Detail": "Genre % mix, style label, era preference, bar charts — auto-updated from favorites",
            },
        ]
    )
    st.dataframe(layers, use_container_width=True, hide_index=True)

    st.subheader("Challenges we address")
    challenges = pd.DataFrame(
        [
            {
                "Challenge": "Cold start (new picks, no user ID)",
                "Approach": "Seed from favorite movies + content similarity; HybridNet movie embeddings for collaborative signal",
            },
            {
                "Challenge": "Samey recommendations",
                "Approach": "Diversity slider + genre-overlap penalty during greedy ranking",
            },
            {
                "Challenge": "Black-box scores",
                "Approach": "Multi-bullet explanations + headline + theme tags per title",
            },
            {
                "Challenge": "Static lists",
                "Approach": "Session thumbs re-rank via embedding shifts without full retraining",
            },
            {
                "Challenge": "Scale (32M rows)",
                "Approach": "Temporal split, batched parquet writes, resumable `train_pipeline.py`, cached app inference",
            },
        ]
    )
    st.dataframe(challenges, use_container_width=True, hide_index=True)

    if report and report.get("all_models"):
        st.subheader("Full pipeline comparison")
        comparison = pd.DataFrame(report["all_models"]).sort_values("val_RMSE")
        show_cols = [c for c in ("model", "val_RMSE", "test_RMSE", "test_MAE") if c in comparison.columns]
        st.dataframe(comparison[show_cols], use_container_width=True, hide_index=True)

    st.caption("Checkpoint: `artifacts/best_model_full.pt` · Metrics: `artifacts/pipeline_report.json`")


def main() -> None:
    render_sidebar()
    page = st.session_state.nav_page
    scroll_to_top_if_page_changed(page)
    if page == "Home":
        ensure_home_catalog_ready()
        page_home()
    elif page == "Pick favorites":
        with st.spinner("Loading movie catalog…"):
            get_catalog()
        page_pick_favorites()
    elif page == "Recommendations":
        page_recommendations()
    elif page == "Model":
        page_model()
    else:
        page_taste_profile()


if __name__ == "__main__":
    main()
