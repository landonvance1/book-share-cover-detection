#!/usr/bin/env python3
"""Sync local ONNX Florence-2 model with pinned revision in constants.py

This keeps local development in sync with Docker builds by ensuring the
same ONNX model revision is downloaded locally.

The ONNX model is the only pre-downloaded model in local dev, since it's
loaded from a fixed local directory path. PyTorch and GLiNER are downloaded
on-demand at runtime and don't need syncing.

Usage:
    python scripts/sync_onnx_model.py              # Download to default location
    python scripts/sync_onnx_model.py --cache-dir /path/to/cache
"""

import argparse
import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoProcessor

# Add app module to path to import constants
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import constants


def main():
    parser = argparse.ArgumentParser(
        description="Sync local ONNX Florence-2 model with pinned revision"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Cache directory for downloaded model (defaults to current directory or $HF_HOME)",
    )

    args = parser.parse_args()

    # Determine download directory
    if args.cache_dir:
        download_dir = args.cache_dir
    else:
        # Use current directory's parent as default
        download_dir = Path.cwd()

    print(f"Syncing ONNX Florence-2 model...")
    print(f"  Model: {constants.FLORENCE2_ONNX_MODEL}")
    print(f"  Revision: {constants.FLORENCE2_ONNX_REVISION}")
    print()

    onnx_dir = download_dir / "florence2-onnx"

    if onnx_dir.exists():
        print(f"Removing existing model directory: {onnx_dir}")
        shutil.rmtree(onnx_dir)

    print(f"Downloading to: {onnx_dir}")
    try:
        snapshot_download(
            constants.FLORENCE2_ONNX_MODEL,
            local_dir=str(onnx_dir),
            revision=constants.FLORENCE2_ONNX_REVISION,
        )
        print()

        # Also download the processor (needed at runtime for tokenization)
        print(f"Downloading processor: {constants.FLORENCE2_PROCESSOR_MODEL}")
        AutoProcessor.from_pretrained(
            constants.FLORENCE2_PROCESSOR_MODEL, trust_remote_code=True
        )
        print()
        print("✓ ONNX Florence-2 model synced successfully!")
        return 0

    except Exception as e:
        print(f"✗ Error syncing model: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
