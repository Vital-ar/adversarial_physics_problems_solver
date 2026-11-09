"""Stage 2 Step 4D: Generate Self-Contained Directives (`[[CHALLENGE]]`) & Domain-Balanced Approvals (`[[APPROVE]]`)
via `CohereForAI/aya-expanse-8b`, then Export All 5 Progressive SFT Datasets (`E1`, `E2`, `E3`, `E4 Solver`, `E4 Verifier`).

Usage (with your `CohereForAI/aya-expanse-8b` HF GPU Endpoint running):
  .\\venv\\Scripts\\python.exe scripts/04d_generate_aya8b_directives_and_export.py --backend hf_gpu

Optional flags:
  --export-only               Skip LLM generation and re-run deterministic SFT dataset export from existing checkpoints
  --endpoint-url URL          Explicit HF Endpoint URL for CohereForAI/aya-expanse-8b
  --approvals-per-domain 30   Number of [[APPROVE]] examples per domain across all 6 domains (default: 30 -> 180 total)
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

from adversarial_physics.data.curation.dataset_exporter import (  # noqa: E402
    DEFAULT_CURATED_SOL2_PATH,
    DEFAULT_DIRECTIVES_INPUT_PATH,
    DEFAULT_E1_MANUAL_DET_POOL_PATH,
    DEFAULT_E2_CURATORKIT_NOLLM_POOL_PATH,
    DEFAULT_E3_CANONICAL_TRAIN_PATH,
    DEFAULT_SFT_OUTPUT_ROOT,
    DEFAULT_TEST_PATH,
    DEFAULT_VALIDATION_PATH,
    export_all_stage2_sft_datasets,
)
from adversarial_physics.data.generation.directive_generator import (  # noqa: E402
    DEFAULT_DIRECTIVES_CHECKPOINT_PATH,
    DEFAULT_SURVIVING_FLAWED_PATH,
    generate_stage2_directives_and_approvals,
)
from adversarial_physics.utils.runtime import AYA_EXPANSE_8B  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2 Step 4D: Generate directives & approvals with aya-expanse-8b and export all 5 SFT datasets."
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
        help="Optional explicit HF GPU Inference Endpoint URL.",
    )
    parser.add_argument(
        "--approvals-per-domain",
        type=int,
        default=30,
        help="Approvals per domain across all 6 domains (default: 30 -> 180 total = 26.4%% of 681).",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Only run the deterministic SFT dataset exporter from existing directive outputs.",
    )
    parser.add_argument("--canonical-train", type=str, default=DEFAULT_E3_CANONICAL_TRAIN_PATH)
    parser.add_argument("--validation", type=str, default=DEFAULT_VALIDATION_PATH)
    parser.add_argument("--test", type=str, default=DEFAULT_TEST_PATH)
    parser.add_argument("--e1-pool", type=str, default=DEFAULT_E1_MANUAL_DET_POOL_PATH)
    parser.add_argument("--e2-pool", type=str, default=DEFAULT_E2_CURATORKIT_NOLLM_POOL_PATH)
    parser.add_argument("--curated-sol2", type=str, default=DEFAULT_CURATED_SOL2_PATH)
    parser.add_argument("--surviving-flawed", type=str, default=DEFAULT_SURVIVING_FLAWED_PATH)
    parser.add_argument("--directives-checkpoint", type=str, default=DEFAULT_DIRECTIVES_CHECKPOINT_PATH)
    parser.add_argument("--directives-output", type=str, default=DEFAULT_DIRECTIVES_INPUT_PATH)
    parser.add_argument("--sft-output-root", type=str, default=DEFAULT_SFT_OUTPUT_ROOT)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    t0 = time.perf_counter()

    if not args.export_only:
        generate_stage2_directives_and_approvals(
            canonical_train_path=args.canonical_train,
            curated_sol2_path=args.curated_sol2,
            surviving_flawed_path=args.surviving_flawed,
            checkpoint_path=args.directives_checkpoint,
            output_path=args.directives_output,
            approvals_per_domain=args.approvals_per_domain,
            model_name=args.model,
            backend=args.backend,  # type: ignore[arg-type]
            endpoint_url=args.endpoint_url,
            limit=args.limit,
        )

    export_summary = export_all_stage2_sft_datasets(
        e1_pool_path=args.e1_pool,
        e2_pool_path=args.e2_pool,
        e3_pool_path=args.canonical_train,
        validation_path=args.validation,
        test_path=args.test,
        curated_sol2_path=args.curated_sol2,
        surviving_flawed_path=args.surviving_flawed,
        directives_input_path=args.directives_output,
        sft_output_root=args.sft_output_root,
    )
    payload = export_summary.to_dict()
    payload["total_elapsed_sec"] = round(time.perf_counter() - t0, 3)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
