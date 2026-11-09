"""Stage 2 Step 4A: Generate Minority-Domain Solution 2 (82) & Controlled Error Trajectories (681) via `CohereForAI/aya-expanse-8b`.

Usage (with your `CohereForAI/aya-expanse-8b` HF GPU Endpoint running):
  .\\venv\\Scripts\\python.exe scripts/04a_generate_aya8b_trajectories.py --backend hf_gpu

Optional flags:
  --step both|sol2|flawed     Run both tasks (default), only 82 minority Solution 2, or only 681 flawed trajectories
  --endpoint-url URL          Explicit HF Endpoint URL (auto-discovered if omitted)
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

from adversarial_physics.data.generation.candidate_generator import (  # noqa: E402
    DEFAULT_CANONICAL_TRAIN_PATH,
    DEFAULT_FLAWED_CHECKPOINT_PATH,
    DEFAULT_FLAWED_OUTPUT_PATH,
    DEFAULT_SOLUTION2_CHECKPOINT_PATH,
    DEFAULT_SOLUTION2_OUTPUT_PATH,
    generate_controlled_error_trajectories,
    generate_minority_alternative_solutions,
)
from adversarial_physics.utils.runtime import AYA_EXPANSE_8B  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2 Step 4A: Generate 82 Minority-Domain Solution 2 & 681 Controlled Error Trajectories with aya-expanse-8b."
    )
    parser.add_argument(
        "--step",
        choices=["both", "sol2", "flawed"],
        default="both",
        help="Which generation task(s) to run (default: both).",
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
        default=AYA_EXPANSE_8B,
        help=f"Teacher model repo ID (default: {AYA_EXPANSE_8B}).",
    )
    parser.add_argument(
        "--endpoint-url",
        type=str,
        default=None,
        help="Optional explicit HF GPU Inference Endpoint URL (auto-discovered by repository if omitted).",
    )
    parser.add_argument(
        "--canonical-train",
        type=str,
        default=DEFAULT_CANONICAL_TRAIN_PATH,
        help="Path to canonical_train.jsonl (681 problems).",
    )
    parser.add_argument(
        "--sol2-checkpoint",
        type=str,
        default=DEFAULT_SOLUTION2_CHECKPOINT_PATH,
    )
    parser.add_argument(
        "--sol2-output",
        type=str,
        default=DEFAULT_SOLUTION2_OUTPUT_PATH,
    )
    parser.add_argument(
        "--flawed-checkpoint",
        type=str,
        default=DEFAULT_FLAWED_CHECKPOINT_PATH,
    )
    parser.add_argument(
        "--flawed-output",
        type=str,
        default=DEFAULT_FLAWED_OUTPUT_PATH,
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
    report: dict[str, object] = {"step": args.step, "model": args.model, "backend": args.backend}

    if args.step in ("both", "sol2"):
        t_sol2 = time.perf_counter()
        sol2 = generate_minority_alternative_solutions(
            canonical_train_path=args.canonical_train,
            checkpoint_path=args.sol2_checkpoint,
            output_path=args.sol2_output,
            model_name=args.model,
            backend=args.backend,  # type: ignore[arg-type]
            endpoint_url=args.endpoint_url,
            limit=args.limit,
        )
        report["minority_solution2_count"] = len(sol2)
        report["minority_solution2_elapsed_sec"] = round(time.perf_counter() - t_sol2, 3)

    if args.step in ("both", "flawed"):
        t_flawed = time.perf_counter()
        flawed = generate_controlled_error_trajectories(
            canonical_train_path=args.canonical_train,
            checkpoint_path=args.flawed_checkpoint,
            output_path=args.flawed_output,
            model_name=args.model,
            backend=args.backend,  # type: ignore[arg-type]
            endpoint_url=args.endpoint_url,
            limit=args.limit,
        )
        report["controlled_error_trajectories_count"] = len(flawed)
        report["controlled_error_elapsed_sec"] = round(time.perf_counter() - t_flawed, 3)

    report["total_elapsed_sec"] = round(time.perf_counter() - t0, 3)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
