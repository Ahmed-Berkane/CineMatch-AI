"""Remove orphaned LFS blobs from a Hugging Face Space to stay under the 1 GB quota."""

from __future__ import annotations

import argparse

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Permanently delete LFS files from a Space to free the 1 GB repo quota."
    )
    parser.add_argument("--repo-id", required=True, help="e.g. username/CineMatch-AI")
    parser.add_argument(
        "--prefix",
        action="append",
        default=[],
        help="Delete LFS objects whose path starts with this prefix (repeatable)",
    )
    parser.add_argument(
        "--exact",
        action="append",
        default=[],
        help="Delete LFS objects with this exact path (repeatable)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List matching LFS files without deleting",
    )
    args = parser.parse_args()

    prefixes = args.prefix or ["data/processed/"]
    exact = set(args.exact)

    api = HfApi()
    to_delete = []
    for f in api.list_lfs_files(args.repo_id, repo_type="space"):
        if f.filename in exact or any(f.filename.startswith(p) for p in prefixes):
            to_delete.append(f)

    if not to_delete:
        print("No matching LFS files to delete.")
        return

    total_mb = sum(f.size for f in to_delete) / 1024 / 1024
    print(f"Matched {len(to_delete)} LFS object(s), {total_mb:.1f} MB total:")
    for f in sorted(to_delete, key=lambda x: x.size, reverse=True):
        print(f"  {f.size / 1024 / 1024:.1f} MB  {f.filename}")

    if args.dry_run:
        print("Dry run — nothing deleted.")
        return

    api.permanently_delete_lfs_files(args.repo_id, to_delete, repo_type="space")
    print("Deleted. Quota may take a minute to update.")


if __name__ == "__main__":
    main()
