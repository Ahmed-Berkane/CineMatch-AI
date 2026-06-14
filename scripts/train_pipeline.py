"""
Full CineMatch-AI training pipeline.

Trains all candidate neural models on train.parquet, compares on val.parquet,
selects the best by validation RMSE, evaluates once on test.parquet, and saves
the winning checkpoint plus comparison metrics.

Resume support (for long CPU runs):
  - Caches cleaned train data and ID mappings after first load
  - Saves progress after each epoch and each completed model
  - Re-run the same command to resume from the last checkpoint
  - Use --fresh to discard checkpoints and start over
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.data_helpers import project_root
from scripts.model_helpers import (
    REQUIRED_RATING_COLS,
    load_clean_parquet,
    load_unique_movies,
    rmse_mae,
    sample_parquet,
)
from scripts import neural_models as nm


@dataclass
class PipelineConfig:
    embed_dim: int = 64
    gmf_dim: int = 32
    mlp_dim: int = 32
    hidden: tuple[int, ...] = (128, 64)
    dropout: float = 0.2
    batch_size: int = 8192
    epochs: int = 5
    lr: float = 1e-3
    seed: int = 42


def _model_slug(name: str) -> str:
    slug = name.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug).strip("_")
    return slug


def _filter_eval_rows(df: pd.DataFrame, mappings: nm.IdMappings) -> pd.DataFrame:
    return df[
        df["userId"].isin(mappings.user_to_idx) & df["movieId"].isin(mappings.movie_to_idx)
    ].copy()


def clean_from_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    from scripts.model_helpers import clean_ratings_df

    return clean_ratings_df(df, required_cols=REQUIRED_RATING_COLS)


def _config_fingerprint(cfg: PipelineConfig, args: argparse.Namespace) -> dict:
    return {
        "embed_dim": cfg.embed_dim,
        "batch_size": cfg.batch_size,
        "epochs": cfg.epochs,
        "lr": cfg.lr,
        "seed": cfg.seed,
        "max_rows": args.max_rows,
    }


def _fingerprint_matches(saved: dict, current: dict) -> bool:
    return saved == current


def _baseline_metrics(val_df: pd.DataFrame, test_df: pd.DataFrame, global_mean: float) -> dict:
    val_pred = np.full(len(val_df), global_mean)
    test_pred = np.full(len(test_df), global_mean)
    val_rmse, val_mae = rmse_mae(val_df["rating"].to_numpy(), val_pred)
    test_rmse, test_mae = rmse_mae(test_df["rating"].to_numpy(), test_pred)
    return {
        "model": "Baseline (global mean)",
        "model_type": "baseline",
        "val_RMSE": val_rmse,
        "val_MAE": val_mae,
        "test_RMSE": test_rmse,
        "test_MAE": test_mae,
    }


def _build_model_specs(
    mappings: nm.IdMappings,
    content_dim: int,
    cfg: PipelineConfig,
) -> list[tuple[str, torch.nn.Module, nm.ModelType]]:
    return [
        (
            "GMF",
            nm.GMF(mappings.n_users, mappings.n_movies, embed_dim=cfg.embed_dim),
            "cf",
        ),
        (
            "NeuMF",
            nm.NeuMF(
                mappings.n_users,
                mappings.n_movies,
                gmf_dim=cfg.gmf_dim,
                mlp_dim=cfg.mlp_dim,
                hidden=cfg.hidden,
                dropout=cfg.dropout,
            ),
            "cf",
        ),
        (
            "Neural CF (MLP)",
            nm.NeuralCF(
                mappings.n_users,
                mappings.n_movies,
                embed_dim=cfg.embed_dim,
                hidden=cfg.hidden,
                dropout=cfg.dropout,
            ),
            "cf",
        ),
        (
            "ContentNet",
            nm.ContentNet(
                mappings.n_users,
                content_dim=content_dim,
                embed_dim=cfg.embed_dim,
                hidden=cfg.hidden,
                dropout=cfg.dropout,
            ),
            "content",
        ),
        (
            "HybridNet",
            nm.HybridNet(
                mappings.n_users,
                mappings.n_movies,
                content_dim=content_dim,
                embed_dim=cfg.embed_dim,
                hidden=cfg.hidden,
                dropout=cfg.dropout,
            ),
            "hybrid",
        ),
    ]


def _save_best_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    model_name: str,
    model_type: str,
    mappings: nm.IdMappings,
    content_lookup: np.ndarray,
    vocabulary: list[str],
    cfg: PipelineConfig,
    metrics: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": model_name,
            "model_type": model_type,
            "model_class": model.__class__.__name__,
            "n_users": mappings.n_users,
            "n_movies": mappings.n_movies,
            "content_dim": int(content_lookup.shape[1]),
            "user_ids": mappings.user_ids,
            "movie_ids": mappings.movie_ids,
            "content_lookup": content_lookup,
            "genre_vocabulary": vocabulary,
            "config": asdict(cfg),
            "metrics": metrics,
        },
        path,
    )


def _save_model_done(
    path: Path,
    *,
    model: torch.nn.Module,
    model_name: str,
    model_type: str,
    result: dict,
    history: list[dict],
    cfg: PipelineConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": model_name,
            "model_type": model_type,
            "model_class": model.__class__.__name__,
            "result": result,
            "history": history,
            "config": asdict(cfg),
        },
        path,
    )


def _load_model_done(
    path: Path,
    mappings: nm.IdMappings,
    content_dim: int,
    cfg: PipelineConfig,
) -> tuple[torch.nn.Module, dict, list[dict]]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    ckpt_cfg = {**asdict(cfg), **ckpt.get("config", {})}
    stub = {
        "config": ckpt_cfg,
        "model_class": ckpt["model_class"],
        "n_users": mappings.n_users,
        "n_movies": mappings.n_movies,
        "content_dim": content_dim,
        "model_state_dict": ckpt["model_state_dict"],
    }
    model = nm.build_model_from_checkpoint(stub)
    return model, ckpt["result"], ckpt.get("history", [])


def _load_pipeline_state(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_pipeline_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def _save_data_cache(
    cache_dir: Path,
    *,
    mappings: nm.IdMappings,
    content_lookup: np.ndarray,
    vocabulary: list[str],
    global_mean: float,
    train_stats: dict,
    val_stats: dict,
    test_stats: dict,
    fingerprint: dict,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "fingerprint": fingerprint,
            "user_ids": mappings.user_ids,
            "movie_ids": mappings.movie_ids,
            "content_lookup": content_lookup,
            "vocabulary": vocabulary,
            "global_mean": global_mean,
            "train_stats": train_stats,
            "val_stats": val_stats,
            "test_stats": test_stats,
        },
        cache_dir / "mappings_cache.pt",
    )


def _load_data_cache(cache_dir: Path, fingerprint: dict) -> dict | None:
    cache_path = cache_dir / "mappings_cache.pt"
    if not cache_path.exists():
        return None
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    if not _fingerprint_matches(cache.get("fingerprint", {}), fingerprint):
        return None
    return cache


def _mappings_from_cache(cache: dict) -> nm.IdMappings:
    user_ids = np.asarray(cache["user_ids"])
    movie_ids = np.asarray(cache["movie_ids"])
    return nm.IdMappings(
        user_ids=user_ids,
        movie_ids=movie_ids,
        user_to_idx={int(u): i for i, u in enumerate(user_ids)},
        movie_to_idx={int(m): i for i, m in enumerate(movie_ids)},
    )


def _clear_resume_files(output_dir: Path) -> None:
    for name in ("pipeline_state.json", "model_comparison.csv", "pipeline_report.json", "training_histories.json"):
        path = output_dir / name
        if path.exists():
            path.unlink()
    checkpoints = output_dir / "checkpoints"
    if checkpoints.exists():
        shutil.rmtree(checkpoints)
    cache = output_dir / "cache"
    if cache.exists():
        shutil.rmtree(cache)


def _finalize_run(
    *,
    output_dir: Path,
    results: list[dict],
    histories: dict[str, list[dict]],
    mappings: nm.IdMappings,
    content_lookup: np.ndarray,
    vocabulary: list[str],
    cfg: PipelineConfig,
    global_mean: float,
    train_df: pd.DataFrame,
    val_eval: pd.DataFrame,
    test_eval: pd.DataFrame,
    train_stats: dict,
    val_stats: dict,
    test_stats: dict,
    device: str,
    t0: float,
    model_specs: list[tuple[str, torch.nn.Module, nm.ModelType]],
) -> None:
    comparison = pd.DataFrame(results).sort_values("val_RMSE")
    comparison_path = output_dir / "model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    neural_rows = comparison[comparison["model_type"] != "baseline"]
    best_row = neural_rows.iloc[0]
    best_name = str(best_row["model"])
    done_path = output_dir / "checkpoints" / f"{_model_slug(best_name)}_done.pt"
    best_model, _, _ = _load_model_done(
        done_path, mappings, content_lookup.shape[1], cfg
    )

    best_metrics = {
        "selected_model": best_name,
        "selection_metric": "val_RMSE",
        "val_RMSE": float(best_row["val_RMSE"]),
        "val_MAE": float(best_row["val_MAE"]),
        "test_RMSE": float(best_row["test_RMSE"]),
        "test_MAE": float(best_row["test_MAE"]),
        "global_mean": global_mean,
        "train_rows": len(train_df),
        "val_rows_eval": len(val_eval),
        "test_rows_eval": len(test_eval),
        "device": device,
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    checkpoint_path = output_dir / "best_model.pt"
    best_type = next(t for n, _, t in model_specs if n == best_name)
    _save_best_checkpoint(
        checkpoint_path,
        model=best_model,
        model_name=best_name,
        model_type=best_type,
        mappings=mappings,
        content_lookup=content_lookup,
        vocabulary=vocabulary,
        cfg=cfg,
        metrics=best_metrics,
    )

    with open(output_dir / "pipeline_report.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "best": best_metrics,
                "all_models": results,
                "train_clean_stats": train_stats,
                "val_clean_stats": val_stats,
                "test_clean_stats": test_stats,
            },
            f,
            indent=2,
        )

    with open(output_dir / "training_histories.json", "w", encoding="utf-8") as f:
        json.dump(histories, f, indent=2)

    state = _load_pipeline_state(output_dir / "pipeline_state.json") or {}
    state["status"] = "completed"
    state["completed_models"] = [r["model"] for r in results if r["model_type"] != "baseline"]
    _save_pipeline_state(output_dir / "pipeline_state.json", state)

    print("\n" + "=" * 60)
    print("Model comparison (sorted by val RMSE):")
    print(comparison.to_string(index=False))
    print(f"\nBest model: {best_name}")
    print(f"  val  RMSE={best_row['val_RMSE']:.4f}  MAE={best_row['val_MAE']:.4f}")
    print(f"  test RMSE={best_row['test_RMSE']:.4f}  MAE={best_row['test_MAE']:.4f}")
    print("\nSaved:")
    print(f"  {checkpoint_path}")
    print(f"  {comparison_path}")
    print(f"  {output_dir / 'pipeline_report.json'}")
    print(f"Elapsed: {best_metrics['elapsed_seconds']:.0f}s")
    print("=" * 60)


def run(args: argparse.Namespace) -> None:
    root = project_root()
    processed = root / args.data_dir / "processed"
    output_dir = root / args.output_dir
    cache_dir = output_dir / "cache"
    checkpoints_dir = output_dir / "checkpoints"
    state_path = output_dir / "pipeline_state.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    cfg = PipelineConfig(
        embed_dim=args.embed_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
    )
    fingerprint = _config_fingerprint(cfg, args)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if args.fresh:
        print("Clearing previous checkpoints and cache (--fresh)...")
        _clear_resume_files(output_dir)

    train_path = processed / "train.parquet"
    val_path = processed / "val.parquet"
    test_path = processed / "test.parquet"

    print("=" * 60)
    print("CineMatch-AI training pipeline")
    print(f"Device: {device}")
    resume = not args.fresh and state_path.exists()
    if resume:
        print("Resume: will continue from last checkpoint if config matches")
    print("=" * 60)

    existing_state = _load_pipeline_state(state_path)
    if existing_state and not args.fresh:
        saved_fp = existing_state.get("config_fingerprint", {})
        if saved_fp and not _fingerprint_matches(saved_fp, fingerprint):
            print(
                "\nError: existing checkpoint was trained with different settings.",
                file=sys.stderr,
            )
            print("Use --fresh to discard old checkpoints and start over.", file=sys.stderr)
            sys.exit(1)
        if existing_state.get("status") == "completed":
            print("\nPipeline already completed. Outputs are in:", output_dir)
            print("Use --fresh to retrain from scratch.")
            return

    t0 = time.time()
    train_cache_path = cache_dir / "train_clean.parquet"
    mappings_cache = _load_data_cache(cache_dir, fingerprint)

    if mappings_cache is not None and train_cache_path.exists() and not args.fresh:
        print("\n[1/5] Loading cached train split...")
        train_df = pd.read_parquet(train_cache_path)
        train_stats = mappings_cache["train_stats"]
        mappings = _mappings_from_cache(mappings_cache)
        content_lookup = np.asarray(mappings_cache["content_lookup"], dtype=np.float32)
        vocabulary = list(mappings_cache["vocabulary"])
        global_mean = float(mappings_cache["global_mean"])
        val_stats = mappings_cache["val_stats"]
        test_stats = mappings_cache["test_stats"]
        print(f"  train rows: {len(train_df):,}  (from cache)")
    else:
        print("\n[1/5] Loading train split...")
        load_cols = list(REQUIRED_RATING_COLS) + ["timestamp"]
        if args.max_rows is not None:
            train_raw = sample_parquet(train_path, args.max_rows, load_cols, seed=cfg.seed)
            train_df, train_stats = clean_from_frame(train_raw)
        else:
            train_df, train_stats = load_clean_parquet(train_path)
        print(f"  train rows: {len(train_df):,}  (clean stats: {train_stats})")

        print("\n[2/5] Building ID mappings and content features...")
        movies_unique = load_unique_movies(train_path)
        from scripts.model_helpers import build_movie_content_features

        movie_features_arr, vocabulary = build_movie_content_features(movies_unique)
        movie_features = pd.DataFrame(
            movie_features_arr, index=movies_unique["movieId"].values
        )
        mappings = nm.build_id_mappings(train_df)
        content_lookup = nm.build_content_lookup(mappings, movie_features)
        global_mean = float(train_df["rating"].mean())

        print("\n[3/5] Loading val / test splits...")
        val_df, val_stats = load_clean_parquet(val_path)
        test_df, test_stats = load_clean_parquet(test_path)

        cache_dir.mkdir(parents=True, exist_ok=True)
        print("  Caching cleaned train data for faster resume...")
        train_df.to_parquet(train_cache_path, index=False)
        _save_data_cache(
            cache_dir,
            mappings=mappings,
            content_lookup=content_lookup,
            vocabulary=vocabulary,
            global_mean=global_mean,
            train_stats=train_stats,
            val_stats=val_stats,
            test_stats=test_stats,
            fingerprint=fingerprint,
        )

    if mappings_cache is not None and train_cache_path.exists() and not args.fresh:
        print("\n[2/5] Using cached mappings and content features")
        print(f"  users: {mappings.n_users:,}  movies: {mappings.n_movies:,}")
        print(f"  content dim: {content_lookup.shape[1]}  global mean: {global_mean:.4f}")
        print("\n[3/5] Loading val / test splits...")
        val_df, _ = load_clean_parquet(val_path)
        test_df, _ = load_clean_parquet(test_path)

    val_eval = _filter_eval_rows(val_df, mappings)
    test_eval = _filter_eval_rows(test_df, mappings)
    print(f"  val: {len(val_df):,} rows ({len(val_eval):,} in train universe)")
    print(f"  test: {len(test_df):,} rows ({len(test_eval):,} in train universe)")

    model_specs = _build_model_specs(mappings, content_lookup.shape[1], cfg)
    completed: set[str] = set()
    results: list[dict] = []
    histories: dict[str, list[dict]] = {}

    if existing_state and not args.fresh:
        completed = set(existing_state.get("completed_models", []))
        results = list(existing_state.get("results", []))
        histories = dict(existing_state.get("histories", {}))
        if completed:
            print(f"\nResuming — {len(completed)} model(s) already finished: {', '.join(sorted(completed))}")

    state = {
        "status": "training",
        "config_fingerprint": fingerprint,
        "completed_models": sorted(completed),
        "results": results,
        "histories": histories,
        "started_at": existing_state.get("started_at") if existing_state else time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _save_pipeline_state(state_path, state)

    print("\n[4/5] Training candidate models...")
    if not any(r.get("model_type") == "baseline" for r in results):
        baseline = _baseline_metrics(val_eval, test_eval, global_mean)
        results.append(baseline)
        state["results"] = results
        _save_pipeline_state(state_path, state)
        print(
            f"  Baseline  val RMSE={baseline['val_RMSE']:.4f}  "
            f"test RMSE={baseline['test_RMSE']:.4f}"
        )

    for name, model, model_type in model_specs:
        slug = _model_slug(name)
        done_path = checkpoints_dir / f"{slug}_done.pt"
        epoch_ckpt = checkpoints_dir / f"{slug}_epoch.pt"

        if name in completed and done_path.exists():
            print(f"\n  --- {name} --- (skipped, already done)")
            continue

        print(f"\n  --- {name} ---")
        row, history = nm.evaluate_model(
            name,
            model,
            train_df,
            val_eval,
            mappings,
            model_type=model_type,
            content_lookup=content_lookup if model_type != "cf" else None,
            batch_size=cfg.batch_size,
            epochs=cfg.epochs,
            lr=cfg.lr,
            device=device,
            epoch_checkpoint=epoch_ckpt,
            resume=not args.fresh,
        )
        preds = nm.predict_ratings(
            model,
            test_eval,
            mappings,
            content_lookup=content_lookup if model_type != "cf" else None,
            device=device,
            model_type=model_type,
            batch_size=cfg.batch_size,
        )
        test_rmse, test_mae = rmse_mae(test_eval["rating"].to_numpy(), preds)
        result = {
            "model": name,
            "model_type": model_type,
            "val_RMSE": row["RMSE"],
            "val_MAE": row["MAE"],
            "test_RMSE": test_rmse,
            "test_MAE": test_mae,
        }
        results.append(result)
        histories[name] = history

        _save_model_done(
            done_path,
            model=model,
            model_name=name,
            model_type=model_type,
            result=result,
            history=history,
            cfg=cfg,
        )
        if epoch_ckpt.exists():
            epoch_ckpt.unlink()

        completed.add(name)
        state["completed_models"] = sorted(completed)
        state["results"] = results
        state["histories"] = histories
        _save_pipeline_state(state_path, state)

        comparison = pd.DataFrame(results).sort_values("val_RMSE")
        comparison.to_csv(output_dir / "model_comparison.csv", index=False)

        print(
            f"  val RMSE={result['val_RMSE']:.4f}  MAE={result['val_MAE']:.4f}  "
            f"| test RMSE={result['test_RMSE']:.4f}  MAE={result['test_MAE']:.4f}"
        )
        print(f"  checkpoint saved: {done_path.name}")

    print("\n[5/5] Saving best model and reports...")
    _finalize_run(
        output_dir=output_dir,
        results=results,
        histories=histories,
        mappings=mappings,
        content_lookup=content_lookup,
        vocabulary=vocabulary,
        cfg=cfg,
        global_mean=global_mean,
        train_df=train_df,
        val_eval=val_eval,
        test_eval=test_eval,
        train_stats=train_stats,
        val_stats=val_stats,
        test_stats=test_stats,
        device=device,
        t0=t0,
        model_specs=model_specs,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train all CineMatch neural models and save the best checkpoint."
    )
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory for best_model.pt and metrics",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default=None,
        help="cpu or cuda (default: auto-detect)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row cap on train for smoke tests",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Discard checkpoints and retrain from scratch",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
