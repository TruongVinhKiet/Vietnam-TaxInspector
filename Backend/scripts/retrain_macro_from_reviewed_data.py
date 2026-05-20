"""CLI for retraining macro digital-twin artifacts from reviewed data only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml_engine.macro_retrain_pipeline import run_retrain


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain macro models from canonical + human-approved scenario data.")
    parser.add_argument("--min-samples", type=int, default=3000, help="Minimum augmented samples per model.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-write", action="store_true", help="Build/train but do not write artifacts.")
    args = parser.parse_args()

    report = run_retrain(
        min_samples=args.min_samples,
        seed=args.seed,
        write_artifacts=not args.no_write,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

