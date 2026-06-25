"""
Load a saved CineMatch checkpoint and predict ratings.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.data_helpers import project_root
from scripts import neural_models as nm


def load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def predict_dataframe(
    df: pd.DataFrame,
    ckpt: dict,
    *,
    device: str = "cpu",
    batch_size: int = 8192,
) -> pd.DataFrame:
    model = nm.build_model_from_checkpoint(ckpt)
    mappings = nm.id_mappings_from_checkpoint(ckpt)
    content_lookup = np.asarray(ckpt["content_lookup"], dtype=np.float32)
    model_type = ckpt["model_type"]

    known = df[
        df["userId"].isin(mappings.user_to_idx) & df["movieId"].isin(mappings.movie_to_idx)
    ].copy()
    if known.empty:
        raise ValueError("No rows with user/movie IDs seen during training.")

    preds = nm.predict_ratings(
        model,
        known,
        mappings,
        content_lookup=content_lookup if model_type != "cf" else None,
        device=device,
        model_type=model_type,
        batch_size=batch_size,
    )
    known = known.copy()
    known["predicted_rating"] = preds
    return known


def run(args: argparse.Namespace) -> None:
    root = project_root()
    ckpt_path = root / args.checkpoint
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    ckpt = load_checkpoint(ckpt_path)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loaded {ckpt['model_name']} ({ckpt['model_class']})")
    if "metrics" in ckpt:
        m = ckpt["metrics"]
        print(
            f"  val RMSE={m.get('val_RMSE', 'n/a')}  "
            f"test RMSE={m.get('test_RMSE', 'n/a')}"
        )

    if args.user_id is not None and args.movie_id is not None:
        df = pd.DataFrame({"userId": [args.user_id], "movieId": [args.movie_id]})
        out = predict_dataframe(df, ckpt, device=device, batch_size=args.batch_size)
        row = out.iloc[0]
        print(f"Predicted rating: {row['predicted_rating']:.3f}")
        return

    if args.input_csv:
        df = pd.read_csv(root / args.input_csv)
        if not {"userId", "movieId"}.issubset(df.columns):
            print("Input CSV must contain userId and movieId columns.", file=sys.stderr)
            sys.exit(1)
        out = predict_dataframe(df, ckpt, device=device, batch_size=args.batch_size)
        output_path = root / args.output_csv
        out.to_csv(output_path, index=False)
        print(f"Wrote {len(out):,} predictions to {output_path}")
        return

    print(
        "Provide --user-id and --movie-id for a single prediction, "
        "or --input-csv for batch predictions.",
        file=sys.stderr,
    )
    sys.exit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict ratings with a saved CineMatch model.")
    parser.add_argument(
        "--checkpoint",
        default="artifacts/best_model_full.pt",
        help="Path to best_model_full.pt from train_pipeline.py --retrain-best-full",
    )
    parser.add_argument("--user-id", type=int, default=None)
    parser.add_argument("--movie-id", type=int, default=None)
    parser.add_argument("--input-csv", default=None, help="CSV with userId, movieId columns")
    parser.add_argument(
        "--output-csv",
        default="artifacts/predictions.csv",
        help="Output path when using --input-csv",
    )
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default=None)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
