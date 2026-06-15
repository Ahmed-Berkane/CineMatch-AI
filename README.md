# CineMatch-AI

**Live demo:** [CineMatch AI on Hugging Face Spaces](https://huggingface.co/spaces/Berkane-Nexus-Insights/CineMatch-AI)

**Purpose:** Help people discover movies they will actually enjoy — not just “similar titles,” but recommendations they can **understand and steer** in real time.

CineMatch combines **collaborative filtering** (what millions of MovieLens users rated) with **content signals** (genres, era, metadata) in a single hybrid model (**HybridNet**). The Streamlit app turns that model into an interactive experience: pick favorites, get explainable recommendations, refine results with 👍/👎, and view a **Persona + Taste DNA** profile.

---

## What the app can do

| Feature | Description |
|---------|-------------|
| **Pick favorites** | Live search, franchise-aware suggestions, poster grid |
| **Recommendations** | Auto-generated hybrid rankings with match scores |
| **Why this recommendation?** | Headline tied to your closest pick + bullet breakdown (content, HybridNet, cohort) |
| **Diversity slider** | Similar only → Balanced → Explore new genres |
| **Session 👍 / 👎** | Instant re-ranking via embedding shifts (undo supported) |
| **Taste profile** | Auto-updated **AI Persona**, Taste DNA metrics, genre/decade charts |
| **Model tab** | Architecture, thumbs behavior, challenges solved, benchmark table |

```powershell
pip install -r requirements.txt
streamlit run app.py
```

**Try it online:** [huggingface.co/spaces/Berkane-Nexus-Insights/CineMatch-AI](https://huggingface.co/spaces/Berkane-Nexus-Insights/CineMatch-AI)

**Requires locally:** `artifacts/best_model.pt` (from training) and `data/processed/*.parquet`.

---

## Challenges & how we address them

| Challenge | Our approach |
|-----------|--------------|
| **Cold start** | No login needed — recommendations seed from movies you pick + content similarity + HybridNet movie embeddings |
| **Black-box scores** | “Because you liked *X*…” headlines, theme tags, content vs collaborative breakdown |
| **Samey lists** | Diversity slider applies genre-overlap penalty during greedy re-ranking |
| **Static experience** | 👍 boosts similar embeddings; 👎 pushes away — list refreshes in-session |
| **Scale (~22M ratings)** | Temporal train/val/test split, batched parquet I/O, resumable full pipeline |
| **Sparse metadata** | TMDb enrichment (posters, cast, director, year) joined on `movieId` |
| **Memory / Git limits** | `overview` kept in local metadata cache, not duplicated on every rating row |

---

## Data preparation

Only **`train.parquet`**, **`val.parquet`**, and **`test.parquet`** are committed (Git LFS). Raw MovieLens CSVs and TMDb cache stay local.

### 1. Download MovieLens 32M

- [MovieLens 32M](https://grouplens.org/datasets/movielens/32m/) → extract into `data/`

| File | Description |
|------|-------------|
| `movies.csv` | Titles and pipe-separated genres |
| `ratings.csv` | User ratings (~32M rows) |
| `links.csv` | `movieId` → TMDb / IMDb IDs |
| `tags.csv` | Optional user tags |

### 2. Fetch TMDb metadata

MovieLens lacks plots, cast, directors, and posters. We enrich via [TMDb API](https://www.themoviedb.org/documentation/api).

```powershell
copy .env.example .env   # set TMDB_API_KEY
pip install -r requirements.txt
python scripts/fetch_tmdb_metadata.py --only-rated
```

Writes `data/metadata_df.parquet` (local, gitignored): `overview`, `cast`, `director`, `release_year`, `poster_url`.

### 3. Build train / val / test

Temporal **70 / 10 / 20** split on `timestamp` (oldest → newest), then inner-join each split to the movies catalog (movies with successful TMDb fetch only).

```python
from scripts.data_helpers import build_movies_catalog, build_and_save_splits, project_root

root = project_root()
catalog = build_movies_catalog(root / "data/movies.csv", root / "data")
build_and_save_splits(root / "data/ratings.csv", catalog, root / "data/processed")
```

| File | Share | Role |
|------|-------|------|
| `train.parquet` | ~70% | Fit models |
| `val.parquet` | ~10% | Pick best model (lowest RMSE) |
| `test.parquet` | ~20% | Final metrics — evaluate once |

**Why temporal?** Random splits leak future ratings. We train on the past and evaluate on the newest held-out data — matching real deployment.

---

## Column reference

Columns in `train.parquet`, `val.parquet`, and `test.parquet`.

### Interaction columns

| Column | Type | Definition |
|--------|------|------------|
| `userId` | int | Anonymized MovieLens user |
| `movieId` | int | Join key to movie metadata |
| `rating` | float | Stars 0.5–5.0 |
| `timestamp` | int | Unix UTC when rated |

### Movie metadata (joined per row)

| Column | Type | Definition |
|--------|------|------------|
| `title` | string | Title with year in parentheses |
| `genres` | string | Pipe-separated genres (e.g. `Action\|Thriller`) |
| `tmdbId` | int | TMDb ID from `links.csv` |
| `director` | string | Primary director (TMDb) |
| `cast` | string | Top 5 billed actors |
| `release_year` | int | Theatrical release year |
| `poster_url` | string | TMDb poster CDN URL |

`overview` is **not** on each rating row (too large at ~32M rows). Join from local `metadata_df.parquet` by `movieId` when needed.

---

## Models

All candidates train on the full cleaned **train** split; the winner is chosen by **lowest validation RMSE**, then reported once on **test**.

| Model | Signal | Role |
|-------|--------|------|
| **Baseline** | Global mean | Sanity check |
| **GMF** | Collaborative | Linear dot-product embeddings |
| **NeuMF** | Collaborative | GMF + MLP (He et al., 2017) |
| **Neural CF** | Collaborative | Concat embeddings → MLP |
| **ContentNet** | Content only | User + genre/year (no movie ID) |
| **HybridNet** | **Hybrid** | User + movie embeddings + content → MLP |

**Best model: HybridNet** — jointly learns *who rated what* and *what the movie is like*. Typical full-pipeline result: val RMSE **~0.82**, test RMSE **~0.83** (exact numbers in `artifacts/pipeline_report.json` after training).

**Content encoding:** multi-hot genres + normalized `release_year`, L2-normalized per movie.

### Train the pipeline

```powershell
python scripts/train_pipeline.py              # full run (~hours on CPU)
python scripts/train_pipeline.py --fresh      # restart from scratch
python scripts/train_pipeline.py --max-rows 100000 --epochs 2   # smoke test
```

**Outputs** (`artifacts/`):

| File | Contents |
|------|----------|
| `best_model.pt` | HybridNet weights, ID mappings, content lookup |
| `pipeline_report.json` | Best model + all-model metrics |
| `model_comparison.csv` | Val/test RMSE & MAE table |

```powershell
python scripts/predict.py --user-id 1 --movie-id 260
```

---

## Streamlit app architecture

```
Favorites → HybridNet + content similarity → ranked list
                ↑                    ↑
         👍/👎 session          Diversity slider
         embedding shift         (genre penalty)
                ↓
    Explainability + Persona + Taste DNA
```

| Page | Behavior |
|------|----------|
| **Home** | Latest catalog titles |
| **Pick favorites** | Search, smart suggestions, session feedback |
| **Recommendations** | Auto-refresh; headline + bullets per title |
| **Taste profile** | Auto persona, DNA metrics, charts |
| **Model** | Technical overview, thumbs math, benchmark |

**Serving formula:** `0.6 × content_similarity + 0.4 × HybridNet_embedding_similarity`, then session feedback and diversity adjustments.

### Performance notes

| What | Typical cost |
|------|----------------|
| **First Recommendations visit** | Loads HybridNet once (~5–15s on CPU) — cached for the session |
| **After that** | Recommendations recompute in ~1–3s (cached by favorites + settings) |
| **Pick favorites search** | Fast (cached catalog parquet) |
| **Cohort explanations checkbox** | Scans train.parquet — slow first time; leave off for speed |

Close other heavy apps if the laptop feels sluggish — PyTorch uses significant RAM.

### Deploy

**Live deployment (Hugging Face Spaces — Docker)**

| | |
|---|---|
| **App** | [https://huggingface.co/spaces/Berkane-Nexus-Insights/CineMatch-AI](https://huggingface.co/spaces/Berkane-Nexus-Insights/CineMatch-AI) |
| **Deploy script** | `.\scripts\deploy_hf.ps1` (from repo root, after `hf auth login`) |

The Space runs the Docker image defined in `Dockerfile`, with `artifacts/best_model.pt` and `data/processed/*.parquet` uploaded via Git LFS.

**Streamlit Community Cloud**

1. Push repo to GitHub (include `best_model.pt` via Git LFS or release asset)
2. [share.streamlit.io](https://share.streamlit.io) → New app → `app.py`

**Hugging Face Spaces (manual)**

1. Create a new **Docker** Space (see `HUGGINGFACE.md` for YAML frontmatter)
2. Copy contents of `HUGGINGFACE.md` into the Space `README.md`
3. Run `.\scripts\deploy_hf.ps1` or push the repo with LFS assets
4. Ensure `artifacts/best_model.pt` and `data/processed/*.parquet` are present (required for recommendations)
5. Optional: upload `artifacts/movies_catalog.parquet` or let the app build the catalog from `train.parquet`

**Mobile & desktop:** The app uses responsive CSS — a sticky top tab bar for navigation on phones, grids that stack on small screens, horizontally scrollable tables, and touch-sized buttons. Test with Chrome DevTools device mode or open the Space URL on your phone.

---

## Quick start (clone only)

If split parquets are already in the repo:

```powershell
git lfs install
git clone <repo>
cd CineMatch-AI
pip install -r requirements.txt
# Place artifacts/best_model.pt locally (train or download)
streamlit run app.py
```

---

## Project layout

```
CineMatch-AI/
├── app.py                    # Streamlit UI
├── HUGGINGFACE.md            # HF Space README template (used by deploy script)
├── data/processed/           # train / val / test (committed via LFS)
├── scripts/
│   ├── deploy_hf.ps1         # Push to Hugging Face Spaces
│   ├── fetch_tmdb_metadata.py
│   ├── data_helpers.py       # joins, temporal split, parquet batches
│   ├── model_helpers.py
│   ├── neural_models.py      # HybridNet + baselines
│   ├── train_pipeline.py
│   ├── predict.py
│   ├── catalog.py
│   ├── recommender.py
│   ├── explainability.py
│   ├── persona.py
│   ├── taste_profile.py
│   └── feedback.py
├── artifacts/                # local: best_model.pt, pipeline_report.json
└── requirements.txt
```

`Notebooks/` is optional local EDA (gitignored). Raw MovieLens CSVs, TMDb cache, and `.hf-publish/` deploy clone stay local too.

---

## Scripts reference

```powershell
python scripts/fetch_tmdb_metadata.py --only-rated
python scripts/train_pipeline.py
python scripts/predict.py --user-id 1 --movie-id 260
streamlit run app.py
```
