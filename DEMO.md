# CineMatch AI — Demo guide

**Live app:** [huggingface.co/spaces/Berkane-Nexus-Insights/CineMatch-AI](https://huggingface.co/spaces/Berkane-Nexus-Insights/CineMatch-AI)

Use this script for a **~5 minute** walkthrough. Check off each step as you go.

---

## What “Match score” means

**Match score** is how well a movie fits **your current taste profile** — built from the favorites you picked (and any 👍 this session).

| Range | Meaning |
|-------|---------|
| **85–100%** | Very strong fit — similar genres, era, and viewing patterns |
| **70–84%** | Good fit — worth watching |
| **50–69%** | Moderate fit — more exploratory |
| **Below 50%** | Weaker fit — often surfaced when diversity is high |

### How it is calculated

1. **Your taste profile** — CineMatch averages the movies you selected (plus 👍 thumbs) into one profile.
2. **Content match (60%)** — How similar is the candidate in **genres, release year, and metadata**?
3. **HybridNet match (40%)** — How similar is it in **collaborative taste** — patterns learned from ~22M MovieLens ratings?
4. **Session feedback** — 👍 boosts similar titles; 👎 pushes them down or hides them.
5. **Diversity slider** — Changes **order** (more variety vs. safer picks), not the number on each card.

**Formula (simplified):**

```
Match score ≈ 0.6 × content similarity + 0.4 × HybridNet similarity
             + 👍/👎 adjustments this session
```

### Score breakdown (expand on each card)

| Field | What it tells you |
|-------|-------------------|
| **Content match** | Genre/era/metadata overlap with your picks |
| **HybridNet taste match** | “People with similar taste rated this highly” signal |
| **Headline** | Plain-English link to your closest favorite (e.g. *“Because you liked Bourne Identity…”*) |

> **One-liner for the audience:**  
> *“89% means this film is very close to your taste on both what it **is** (genres, era) and what **similar viewers** enjoyed.”*

---

## Before you start

- [ ] Open the live Space (link above).
- [ ] Visit **Recommendations** once before the demo — first load can take **10–15 seconds** (model warms up).
- [ ] Leave **cohort explanations** unchecked (faster).
- [ ] Set **Recommendation style** to **Balanced**.
- [ ] On **phone**, you should see a **sticky button bar** at the top (Home · Picks · Recs · Taste · Model) — sidebar is hidden on small screens.

---

## Demo script (~5 min)

### 1. Home — set the scene (30 sec)

**Tab:** `Home`

**Say:**

> “CineMatch helps you discover films you’ll actually enjoy — with clear explanations, not a black-box list. No login: you just pick a few movies you love.”

**Show:**

- Logo and welcome text
- Latest movies grid (optional — “you can add any of these”)

---

### 2. Pick favorites — build a taste profile (1 min)

**Tab:** `Pick favorites`

**Say:**

> “I’ll pick 3–5 films to teach the system my taste. Search is live — no need to press Enter.”

**Add these (spy/thriller theme — strong explanations):**

| Search | Film |
|--------|------|
| `Bourne` | Bourne Identity, The (2002) |
| `Skyfall` | Skyfall (2012) |
| `Gone Girl` | Gone Girl (2014) |

**Point out:**

- Progress bar: *“3–5 favorites is the sweet spot”*
- Smart suggestions below search (franchise + similar picks)
- Sidebar: selected count updates live

---

### 3. Recommendations — the core demo (2 min)

**Tab:** `Recommendations`

**Say:**

> “Now HybridNet ranks thousands of titles. Each card shows **why** — not just a score.”

**Show one card in detail (e.g. Munich):**

1. **Match score** — *“89% = very strong fit with my Bourne / thriller picks”*
2. **Headline** — *“Because you liked Bourne Identity, this shares morally complex characters”*
3. **Theme tags** — scene-type fingerprint
4. **Bullet explanations** — genre overlap, content %, HybridNet %
5. **Score breakdown** expander — split between content vs collaborative

**Live interactions:**

| Action | What to say |
|--------|-------------|
| 👍 on a recommendation | *“Instant re-rank — thumbs shift my taste vector this session”* |
| 👎 on something off-theme | *“That title drops and similar ones are penalized”* |
| Move **Diversity** slider | *“Similar only = safe; Diverse = explore new genres”* |

**Caption under the list:**

> *“Score = 0.6 × content + 0.4 × HybridNet, adjusted by your 👍/👎 this session.”*

---

### 4. Taste profile — your viewer DNA (1 min)

**Tab:** `Taste profile`

**Say:**

> “From those same picks, we auto-build a persona and Taste DNA — no extra input.”

**Show:**

- **AI Persona** card (title + blurb)
- **Taste DNA** metrics (favorite genre, era, style)
- Genre mix and decade charts

---

### 5. Model — close with credibility (30 sec)

**Tab:** `Model`

**Say:**

> “Under the hood: HybridNet trained on MovieLens 32M, lowest validation RMSE among six candidates. Thumbs don’t retrain the full model — they re-rank live via embeddings.”

**Optional:** scroll to the pipeline comparison table if the audience is technical.

---

## Closing line

> “CineMatch is explainable, steerable, and hybrid — content plus collaborative intelligence — so you always know **why** a film was recommended and can shape the list in real time.”

---

## Quick FAQ (if someone asks)

| Question | Answer |
|----------|--------|
| Do I need an account? | No — favorites are your profile for this session. |
| Does 👍 retrain the model? | No — it adjusts rankings instantly; full training is offline on 22M rows. |
| Why Munich after Bourne? | High content match (genres/era) + HybridNet taste match + theme overlap. |
| What if scores feel low? | Try more favorites, use Balanced diversity, or 👍 films you want more of. |
| Is data private? | Session state only — nothing stored after you close the app. |

---

## Local run (backup)

```powershell
cd C:\Users\ahmed\Desktop\Projects\CineMatch-AI
.\venv\Scripts\activate
streamlit run app.py
```

Requires `artifacts/best_model.pt` and `data/processed/*.parquet` locally.
