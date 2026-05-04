"""
download_table_models.py – Pre-download Microsoft Table Transformer models
===========================================================================
Downloads the two DETR-based models used by TableTransformerPipeline
so they are cached locally and don't need internet on first inference.

Models:
    1. microsoft/table-transformer-detection       (~220 MB)
    2. microsoft/table-transformer-structure-recognition (~220 MB)

Usage:
    python -m app.scripts.download_table_models [--cache-dir PATH]

Models are saved to:  data/models/table_transformer/
"""

import argparse
import sys
from pathlib import Path

# Ensure parent is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

DEFAULT_CACHE_DIR = (
    Path(__file__).resolve().parent.parent.parent / "data" / "models" / "table_transformer"
)

MODELS = [
    "microsoft/table-transformer-detection",
    "microsoft/table-transformer-structure-recognition",
]


def download_models(cache_dir: str) -> None:
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  TABLE TRANSFORMER MODEL DOWNLOAD")
    print("=" * 60)
    print(f"  Cache directory: {cache_path}")
    print()

    try:
        from transformers import (
            DetrImageProcessor,
            TableTransformerForObjectDetection,
        )
    except ImportError:
        print("[ERROR] 'transformers' package not installed.")
        print("        Run: pip install transformers>=4.40.0 timm>=0.9.0")
        sys.exit(1)

    for idx, model_name in enumerate(MODELS, start=1):
        print(f"[{idx}/{len(MODELS)}] Downloading: {model_name}")
        try:
            DetrImageProcessor.from_pretrained(
                model_name, cache_dir=str(cache_path)
            )
            TableTransformerForObjectDetection.from_pretrained(
                model_name,
                cache_dir=str(cache_path),
                low_cpu_mem_usage=True,
            )
            print(f"        ✓ Cached successfully.")
        except Exception as exc:
            print(f"        ✗ Failed: {exc}")
            sys.exit(1)

    print()
    print("=" * 60)
    print("  ALL MODELS DOWNLOADED SUCCESSFULLY")
    print(f"  Location: {cache_path}")
    total_size = sum(
        f.stat().st_size for f in cache_path.rglob("*") if f.is_file()
    )
    print(f"  Total size: {total_size / 1024 / 1024:.0f} MB")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-download Table Transformer models."
    )
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help=f"Directory to cache models (default: {DEFAULT_CACHE_DIR})",
    )
    args = parser.parse_args()
    download_models(args.cache_dir)
