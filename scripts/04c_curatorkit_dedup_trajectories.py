"""Stage 2 Step 4C: Deterministic CuratorKIT Trajectory Filtering & Deduplication (Zero LLM Calls, <30s).

Eliminates similar/duplicate trajectories (`ExactDeduplicator` + `MinHashDeduplicator >= 0.85` + token cosine)
and filters out any `tiny-aya-global` trajectory that accidentally solved the problem correctly BEFORE
generating hints in Step 4D.

Usage (no GPU endpoint needed):
  .\\venv\\Scripts\\python.exe scripts/04c_curatorkit_dedup_trajectories.py
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

from adversarial_physics.data.curation.curatorkit_runner import (  # noqa: E402
    DEFAULT_CANONICAL_TRAIN_PATH,
    DEFAULT_CURATED_SOL2_PATH,
    DEFAULT_DEDUP_SUMMARY_PATH,
    DEFAULT_FLAWED_8B_INPUT_PATH,
    DEFAULT_RAW_SYNTHETIC_PATH,
    DEFAULT_REJECTED_TRAJECTORIES_PATH,
    DEFAULT_SOLUTION2_INPUT_PATH,
    DEFAULT_SURVIVING_FLAWED_PATH,
    DEFAULT_TINY_AYA_CORRECT_PATH,
    DEFAULT_TINY_AYA_INPUT_PATH,
    curate_and_deduplicate_stage2_trajectories,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2 Step 4C: Run CuratorKIT trajectory deduplication & ANSWER: filtering before directive synthesis."
    )
    parser.add_argument("--canonical-train", type=str, default=DEFAULT_CANONICAL_TRAIN_PATH)
    parser.add_argument("--sol2-input", type=str, default=DEFAULT_SOLUTION2_INPUT_PATH)
    parser.add_argument("--flawed-8b-input", type=str, default=DEFAULT_FLAWED_8B_INPUT_PATH)
    parser.add_argument("--tiny-aya-input", type=str, default=DEFAULT_TINY_AYA_INPUT_PATH)
    parser.add_argument("--raw-synthetic-output", type=str, default=DEFAULT_RAW_SYNTHETIC_PATH)
    parser.add_argument("--curated-sol2-output", type=str, default=DEFAULT_CURATED_SOL2_PATH)
    parser.add_argument("--tiny-aya-correct-output", type=str, default=DEFAULT_TINY_AYA_CORRECT_PATH)
    parser.add_argument("--surviving-flawed-output", type=str, default=DEFAULT_SURVIVING_FLAWED_PATH)
    parser.add_argument("--rejected-output", type=str, default=DEFAULT_REJECTED_TRAJECTORIES_PATH)
    parser.add_argument("--summary-output", type=str, default=DEFAULT_DEDUP_SUMMARY_PATH)
    parser.add_argument("--similarity-threshold", type=float, default=0.85)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    t0 = time.perf_counter()
    _, _, summary = curate_and_deduplicate_stage2_trajectories(
        canonical_train_path=args.canonical_train,
        solution2_input_path=args.sol2_input,
        flawed_8b_input_path=args.flawed_8b_input,
        tiny_aya_input_path=args.tiny_aya_input,
        raw_synthetic_output_path=args.raw_synthetic_output,
        curated_sol2_output_path=args.curated_sol2_output,
        tiny_aya_correct_output_path=args.tiny_aya_correct_output,
        surviving_flawed_output_path=args.surviving_flawed_output,
        rejected_output_path=args.rejected_output,
        summary_output_path=args.summary_output,
        similarity_threshold=args.similarity_threshold,
    )
    payload = summary.to_dict()
    payload["elapsed_sec"] = round(time.perf_counter() - t0, 3)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
