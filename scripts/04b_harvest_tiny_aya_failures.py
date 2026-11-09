"""Stage 2 Step 4B: Harvest Zero-Shot Trajectories (1 per Problem, 681 Total) via `CohereLabs/tiny-aya-global`.

Usage (with your `CohereLabs/tiny-aya-global` HF GPU Endpoint running, or on local/Kaggle GPU):
  .\\venv\\Scripts\\python.exe scripts/04b_harvest_tiny_aya_failures.py --backend hf_gpu

Optional flags:
  --endpoint-url URL          Explicit HF Endpoint URL for CohereLabs/tiny-aya-global (auto-discovered if omitted)
  --backend local_cpu         Run locally on CPU if preferred
  --limit N                   Smoke-test on first N problems
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adversarial_physics.data.generation.model_a_harvester import (  # noqa: E402
    DEFAULT_CANONICAL_TRAIN_PATH,
    DEFAULT_TINY_AYA_CHECKPOINT_PATH,
    DEFAULT_TINY_AYA_OUTPUT_PATH,
    harvest_tiny_aya_trajectories,
)
from adversarial_physics.utils.runtime import TINY_AYA_GLOBAL  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2 Step 4B: Harvest 1 zero-shot trajectory per canonical problem via CohereLabs/tiny-aya-global."
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "kaggle_gpu", "hf_gpu", "local_cpu"],
        default="hf_gpu",
        help="Execution backend (default: hf_gpu).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=TINY_AYA_GLOBAL,
        help=f"Base Solver model repo ID (default: {TINY_AYA_GLOBAL}).",
    )
    parser.add_argument(
        "--endpoint-url",
        type=str,
        default=None,
        help="Optional explicit HF GPU Inference Endpoint URL for CohereLabs/tiny-aya-global.",
    )
    parser.add_argument(
        "--canonical-train",
        type=str,
        default=DEFAULT_CANONICAL_TRAIN_PATH,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_TINY_AYA_CHECKPOINT_PATH,
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_TINY_AYA_OUTPUT_PATH,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of problems for quick testing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    t0 = time.perf_counter()
    results = harvest_tiny_aya_trajectories(
        canonical_train_path=args.canonical_train,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        model_name=args.model,
        backend=args.backend,  # type: ignore[arg-type]
        endpoint_url=args.endpoint_url,
        limit=args.limit,
    )
    wrong_count = sum(1 for r in results if not r.is_correct)
    report = {
        "model": args.model,
        "backend": args.backend,
        "total_harvested": len(results),
        "flawed_trajectories": wrong_count,
        "accidentally_correct_trajectories": len(results) - wrong_count,
        "output_path": args.output,
        "total_elapsed_sec": round(time.perf_counter() - t0, 3),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
