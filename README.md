# CineMatch-AI

A hybrid movie recommendation engine combining collaborative filtering and content-based signals (genres, overview, cast, director).

## Data preparation

The full pipeline runs locally in **`data_prep.ipynb`**. Only the final **train / val / test** parquet files are committed to GitHub — everything else is intermediate and gitignored.

### 1. Download MovieLens 32M

Download and extract the dataset from GroupLens:

- **Dataset page:** [MovieLens 32M](https://grouplens.org/datasets/movielens/32m/)
- **Direct download:** [ml-32m.zip](https://files.grouplens.org/datasets/movielens/ml-32m.zip)

Place these files in the `data/` folder:

| File | Description |
|------|-------------|
| `movies.csv` | Movie titles and genres |
| `ratings.csv` | User ratings (~32M rows) |
| `links.csv` | `movieId` → IMDb / TMDb IDs |
| `tags.csv` | User-generated tags (optional) |

Column definitions ship with the download in `data/README.txt`.

### 2. Fetch TMDb metadata

MovieLens does not include plot summaries, cast, directors, or posters. We enrich movies via the [TMDb API](https://www.themoviedb.org/documentation/api).

1. Create a free API key at [TMDb API settings](https://www.themoviedb.org/settings/api)
2. Copy `.env.example` to `.env` and set your key:

   ```
   TMDB_API_KEY=your_key_here
   ```

3. Install dependencies and fetch metadata (resumes if interrupted):

   ```powershell
   pip install -r requirements.txt
   python scripts/fetch_tmdb_metadata.py --only-rated
   ```

   This uses `links.csv` to map each rated movie to a TMDb ID, then caches results as `data/metadata_df.parquet` (local only, gitignored).

   Fields fetched: `overview`, `cast`, `director`, `release_year`, `poster_url`.

### 3. Build train / val / test

Open **`data_prep.ipynb`** and run the **Build processed datasets** section. The notebook joins raw inputs in memory and writes **three files** to `data/processed/`:

| File | Share | Role |
|------|-------|------|
| `train.parquet` | ~70% | Fit models (oldest ratings) |
| `val.parquet` | ~10% | Tune hyperparameters |
| `test.parquet` | ~20% | Final evaluation — use once at the end |

Each row is one rating with movie metadata attached (genres, cast, director, title, etc.). **`overview` is omitted** from these files — duplicating plot text on ~32M rows exceeds available RAM and GitHub size limits. Join `overview` locally from `metadata_df.parquet` by `movieId` when you need it for embeddings.

Intermediate tables are built one split at a time in memory and are **not** saved or committed.

### How the joins work

All joins use **`movieId`** as the key. `links.csv` and `tags.csv` are not merged in this step.

**Step 1 — movies catalog (in memory, one row per movie)**

| Left | Right | Join | Notes |
|------|-------|------|-------|
| `movies.csv` | `metadata_df.parquet` | inner on `movieId` | Only rows with `fetch_status == "ok"` |

`movies.csv` supplies `title` and `genres`. `metadata_df.parquet` supplies TMDb fields. `links.csv` was already used during fetch to map `movieId` → TMDb ID.

**Step 2 — split ratings, then join one split at a time**

| Step | Input | Action |
|------|-------|--------|
| 2a | `ratings.csv` | Temporal split on `timestamp` (70 / 10 / 20) |
| 2b | each split + movies catalog | inner join on `movieId`, write parquet in batches |

Ratings for movies without successful TMDb metadata are dropped by the inner join. Splits are written one at a time to stay within memory limits.

### Temporal split (70 / 10 / 20)

Ratings are split **by time** on `timestamp` before joining metadata:

```
|-- train (70%) --|-- val (10%) --||-- test (20%) --|
   oldest                              newest
```

- **Train** — `timestamp` below the 70th percentile cutoff
- **Val** — between the 70th and 80th percentile cutoffs
- **Test** — at or above the 80th percentile cutoff

This is a **temporal split**, not random. Random splits leak future information in recommenders (the same user can appear in both sets). Temporal splitting matches the real task: learn from the past, tune on recent held-out data, evaluate on the newest ratings.

| Set | Use for |
|-----|---------|
| **Train** | Fit model weights |
| **Val** | Pick hyperparameters (learning rate, embedding size, regularization) |
| **Test** | Report final metrics — do not tune on this set |

## Column reference

Columns in `train.parquet`, `val.parquet`, and `test.parquet`.

### Interaction columns

| Column | Type | Definition |
|--------|------|------------|
| `userId` | int | Anonymized MovieLens user ID. |
| `movieId` | int | MovieLens movie ID; join key to movie metadata. |
| `rating` | float | User rating on a 0.5–5.0 star scale. |
| `timestamp` | int | Unix UTC seconds when the rating was recorded. |

### Movie metadata columns

Joined onto every row during data prep.

| Column | Type | Definition |
|--------|------|------------|
| `title` | string | Movie title with release year in parentheses. |
| `genres` | string | Pipe-separated genre list from MovieLens (e.g. `Comedy\|Drama`). |
| `tmdbId` | int | [TMDb](https://www.themoviedb.org) movie ID from `links.csv`. |
| `director` | string | Primary director from TMDb credits. |
| `cast` | string | Top 5 billed actors, comma-separated. |
| `release_year` | int | Theatrical release year from TMDb. |
| `poster_url` | string | Full URL to the poster image on TMDb’s CDN. |

`overview` is **not** stored on each rating row (too large at ~32M rows). Join it from local `metadata_df.parquet` by `movieId` when needed for text embeddings.

## Load in Python / notebook

```python
import pandas as pd

train = pd.read_parquet("data/processed/train.parquet")
val = pd.read_parquet("data/processed/val.parquet")
test = pd.read_parquet("data/processed/test.parquet")

print(train.shape, val.shape, test.shape)
train.head()
```

## What is committed to GitHub

| Tracked | Ignored (local only) |
|---------|----------------------|
| `data/processed/train.parquet` | Raw MovieLens `data/*.csv` |
| `data/processed/val.parquet` | TMDb cache `data/metadata_df.*` |
| `data/processed/test.parquet` | `data/movies_with_metadata.parquet` |
| Scripts, notebooks, `.env.example` | Other files in `data/processed/` |
| | `.env` (API keys) |

Clone the repo and you only need the three split files to start modeling. To rebuild them from scratch, download MovieLens locally, fetch TMDb metadata, and run `data_prep.ipynb`.

> **Size note:** Split files may exceed GitHub's 100 MB per-file limit. If push fails, use [Git LFS](https://git-lfs.github.com/) or host on [Hugging Face Datasets](https://huggingface.co/datasets).

## Project layout

```
CineMatch-AI/
├── data/
│   ├── *.csv              # raw MovieLens (gitignored, download locally)
│   └── processed/         # only train / val / test are committed
│       ├── train.parquet
│       ├── val.parquet
│       └── test.parquet
├── scripts/
│   ├── fetch_tmdb_metadata.py   # TMDb enrichment (CLI)
│   └── data_helpers.py          # join/split helpers used by notebook
├── data_prep.ipynb              # EDA + build train/val/test
├── requirements.txt
└── .env.example
```

## Scripts reference

```powershell
# Fetch TMDb metadata for rated movies (~84k, several hours)
python scripts/fetch_tmdb_metadata.py --only-rated

# Rebuild movies + metadata merge only
python scripts/fetch_tmdb_metadata.py --merge-only

# Build train / val / test → run data_prep.ipynb
```
