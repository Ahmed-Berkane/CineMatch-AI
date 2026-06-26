"""Write a Space-sized movies catalog (no overview/search_text) for HF deploy."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def slim_catalog(src: Path, dest: Path) -> None:
    df = pd.read_parquet(src)
    slim = df.drop(columns=["overview", "search_text"], errors="ignore")
    dest.parent.mkdir(parents=True, exist_ok=True)
    slim.to_parquet(dest, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=Path)
    parser.add_argument("dest", type=Path)
    args = parser.parse_args()
    slim_catalog(args.src, args.dest)
    size_mb = args.dest.stat().st_size / 1024 / 1024
    print(f"Slim catalog: {size_mb:.1f} MB -> {args.dest}")


if __name__ == "__main__":
    main()
