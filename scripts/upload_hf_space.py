"""Upload staged Space files via Hugging Face Hub API (no git credentials)."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload a folder to a Hugging Face Space.")
    parser.add_argument("--repo-id", required=True, help="e.g. username/CineMatch-AI")
    parser.add_argument("--folder", type=Path, required=True, help="Local folder to upload")
    parser.add_argument("--message", default="Deploy CineMatch-AI Streamlit app")
    parser.add_argument("--token", default=None, help="Optional HF token (defaults to cached login)")
    parser.add_argument(
        "--code-only",
        action="store_true",
        help="Upload app code + catalog only (skip model checkpoint)",
    )
    parser.add_argument(
        "--with-model",
        action="store_true",
        help="Upload app code + slim catalog + model checkpoint (no train/val/test splits)",
    )
    args = parser.parse_args()

    folder = args.folder.resolve()
    if not folder.is_dir():
        raise SystemExit(f"Folder not found: {folder}")

    api = HfApi(token=args.token)
    print(f"Uploading {folder} -> spaces/{args.repo_id} ...")
    upload_kwargs: dict = {
        "folder_path": str(folder),
        "repo_id": args.repo_id,
        "repo_type": "space",
        "commit_message": args.message,
        "ignore_patterns": [".git", ".git/**", "**/.git/**"],
    }
    if args.code_only and args.with_model:
        raise SystemExit("Use only one of --code-only or --with-model")

    if args.code_only:
        upload_kwargs["allow_patterns"] = [
            "app.py",
            "Dockerfile",
            ".dockerignore",
            "requirements.txt",
            "Logo.png",
            "README.md",
            ".gitattributes",
            ".streamlit/**",
            "scripts/**",
            "artifacts/movies_catalog.parquet",
        ]
        print("Code-only mode: app code + movies_catalog.parquet (no model).")
    elif args.with_model:
        upload_kwargs["allow_patterns"] = [
            "app.py",
            "Dockerfile",
            ".dockerignore",
            "requirements.txt",
            "Logo.png",
            "README.md",
            ".gitattributes",
            ".streamlit/**",
            "scripts/**",
            "artifacts/movies_catalog.parquet",
            "artifacts/best_model_full.pt",
            "artifacts/best_model.pt",
        ]
        print("Model deploy: app code + catalog + checkpoint (no train/val/test splits).")
    api.upload_folder(**upload_kwargs)
    print(f"Done: https://huggingface.co/spaces/{args.repo_id}")


if __name__ == "__main__":
    main()
