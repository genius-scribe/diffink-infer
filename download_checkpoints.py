#!/usr/bin/env python3
"""
Download DiffInk pretrained checkpoints and data files from Google Drive.

Usage:
    python download_checkpoints.py [--ckpt-dir checkpoints] [--data-dir data]

The Google Drive folder is:
    https://drive.google.com/drive/folders/1h_uLmn-55WmbSBGh1ES8-rftAbDs8riB

Install gdown first:
    uv run pip install gdown   # or: pip install gdown
"""

import argparse
import os
import subprocess
import sys


# ---------------------------------------------------------------------------
# File IDs — extracted from the Google Drive folder listing.
# Update these if the upstream folder changes.
# ---------------------------------------------------------------------------

CHECKPOINTS = {
    # InkVAE checkpoint
    "vae_epoch_100.pt": "11fprScAKJnML2Dv_BFZ5JDSQ1cQm341t",
    # InkDiT fine-tuned checkpoint
    "dit_epoch_1.pt":   "13sApjo9rqFHdfNnWmiWaRjFVWewJqECY",
}

DATA_FILES = {
    # Character vocabulary
    "All_zi.json":            "1yQpL0oxC5dv8yXTdWuHsdkaZpAI4XYeQ",
    # Writer split metadata
    "selected_400_100.json":  "1V7g1tTXQzuuri28mVQfSY8o2lbZRF9a3",
    # Validation HDF5 dataset (400-writer split)
    "val.h5":                 "1ZOgUD6UBUWpb154VeXkGvPt9EZ7t1QRg",
}


def gdown_download(file_id: str, dest: str):
    """Download a single file from Google Drive using gdown."""
    url = f"https://drive.google.com/uc?id={file_id}"
    cmd = [sys.executable, "-m", "gdown", url, "-O", dest, "--fuzzy"]
    print(f"  gdown {url} → {dest}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"  [warning] gdown failed for {dest} — try downloading manually.")


def main():
    parser = argparse.ArgumentParser(description="Download DiffInk assets from Google Drive")
    parser.add_argument("--ckpt-dir", default="checkpoints", help="Checkpoint output directory")
    parser.add_argument("--data-dir", default="data", help="Data output directory")
    parser.add_argument("--skip-data", action="store_true", help="Skip dataset download")
    args = parser.parse_args()

    try:
        import gdown  # noqa: F401
    except ImportError:
        print("gdown is not installed. Run: pip install gdown")
        sys.exit(1)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    print(f"\nDownloading checkpoints → {args.ckpt_dir}/")
    for name, fid in CHECKPOINTS.items():
        dest = os.path.join(args.ckpt_dir, name)
        if os.path.exists(dest):
            print(f"  [skip] {name} already exists")
        else:
            gdown_download(fid, dest)

    if not args.skip_data:
        os.makedirs(args.data_dir, exist_ok=True)
        print(f"\nDownloading data files → {args.data_dir}/")
        for name, fid in DATA_FILES.items():
            dest = os.path.join(args.data_dir, name)
            if os.path.exists(dest):
                print(f"  [skip] {name} already exists")
            else:
                gdown_download(fid, dest)

    print("\nDone. Update configs/inference.yaml if you used non-default directories.")


if __name__ == "__main__":
    main()
