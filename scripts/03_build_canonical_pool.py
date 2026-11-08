"""Stage 1.3 & 1.4 CLI: Build the Canonical Training Pool (`data/processed/canonical_train.jsonl`).

Runs:
  1. Mandatory Anti-Leakage Gate (`cosine_similarity >= 0.85` + 5-gram containment) against
     all 76 frozen Validation and Test problems, and saves Intermediate Dataset 1:
     `data/intermediate/01a_leak_free_raw_pool.jsonl` (for Experiment E1 `solver_raw` ablation).
  2. Strict 1-10 logical step, single-target, diagram, and external-reference filtering
     (including independent subproblem splitting `(a)`, `(b)`), within-pool deduplication,
     and CuratorKIT (`TextCleaner`, `SchemaGate`, `DiversityGate`, `RewardGate`, Adaptive Recovery),
     saving Intermediate Dataset 2: `data/intermediate/01b_deterministic_curated_pool.jsonl`.
  3. Topic-diversified selection of ALL valid problems across the 6 canonical domains
     (no 250-entry cap unless `--target-total` is explicitly specified).
  4. Optional resumable `CohereForAI/aya-expanse-8b` solution normalization (`--use-llm`).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adversarial_physics.data.processing.canonical_pool_builder import (  # noqa: E402
    DEFAULT_CANONICAL_CANDIDATES_PATH,
    DEFAULT_CANONICAL_MANIFEST_PATH,
    DEFAULT_CANONICAL_TRAIN_PATH,
    DEFAULT_CHECKSUMS_PATH,
    DEFAULT_DETERMINISTIC_CURATED_POOL_PATH,
    DEFAULT_INITIAL_BOOK_DATASET_PATH,
    DEFAULT_LEAK_FREE_RAW_POOL_PATH,
    DEFAULT_REJECTED_STAGE1_PATH,
    DEFAULT_TEST_PATH,
    DEFAULT_TRAIN_SEEDS_PATH,
    DEFAULT_VALIDATION_PATH,
    build_stage1_canonical_pool,
)
from adversarial_physics.data.processing.deduplication import (  # noqa: E402
    DEFAULT_ANTI_LEAKAGE_COSINE_THRESHOLD,
)
from adversarial_physics.utils.runtime import AYA_EXPANSE_8B  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 1.3 & 1.4: Filter initial_book_dataset.jsonl, run 0.85 anti-leakage deduplication "
            "against 76 held-out problems, save E1/E2 intermediate pools, and build canonical_train.jsonl."
        )
    )
    parser.add_argument("--train-seeds", type=str, default=DEFAULT_TRAIN_SEEDS_PATH)
    parser.add_argument("--validation", type=str, default=DEFAULT_VALIDATION_PATH)
    parser.add_argument("--test", type=str, default=DEFAULT_TEST_PATH)
    parser.add_argument("--initial-book-dataset", type=str, default=DEFAULT_INITIAL_BOOK_DATASET_PATH)
    parser.add_argument("--raw-pool-out", type=str, default=DEFAULT_LEAK_FREE_RAW_POOL_PATH)
    parser.add_argument("--deterministic-curated-out", type=str, default=DEFAULT_DETERMINISTIC_CURATED_POOL_PATH)
    parser.add_argument("--candidates-out", type=str, default=DEFAULT_CANONICAL_CANDIDATES_PATH)
    parser.add_argument("--rejected-out", type=str, default=DEFAULT_REJECTED_STAGE1_PATH)
    parser.add_argument("--output", type=str, default=DEFAULT_CANONICAL_TRAIN_PATH)
    parser.add_argument("--manifest-out", type=str, default=DEFAULT_CANONICAL_MANIFEST_PATH)
    parser.add_argument("--checksums-out", type=str, default=DEFAULT_CHECKSUMS_PATH)
    parser.add_argument(
        "--target-total",
        type=int,
        default=None,
        help="Optional cap on total problems. Default is None (include ALL valid problems).",
    )
    parser.add_argument("--min-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument(
        "--anti-leakage-threshold",
        type=float,
        default=DEFAULT_ANTI_LEAKAGE_COSINE_THRESHOLD,
        help="Cosine similarity threshold for the anti-leakage gate against Val/Test (default: 0.85).",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Run resumable CohereForAI/aya-expanse-8b normalization over all valid canonical problems.",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "kaggle_gpu", "hf_gpu", "local_cpu"],
        default="auto",
        help="Execution backend when --use-llm is enabled (default: auto).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=AYA_EXPANSE_8B,
        help="Model repo ID when --use-llm is enabled (default: CohereForAI/aya-expanse-8b).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    _, summary = build_stage1_canonical_pool(
        train_seeds_path=args.train_seeds,
        validation_path=args.validation,
        test_path=args.test,
        initial_book_dataset_path=args.initial_book_dataset,
        leak_free_raw_pool_path=args.raw_pool_out,
        deterministic_curated_pool_path=args.deterministic_curated_out,
        canonical_candidates_path=args.candidates_out,
        rejected_output_path=args.rejected_out,
        canonical_train_250_path=args.output,
        manifest_output_path=args.manifest_out,
        checksums_output_path=args.checksums_out,
        target_total=args.target_total,
        min_logical_steps=args.min_steps,
        max_logical_steps=args.max_steps,
        anti_leakage_cosine_threshold=args.anti_leakage_threshold,
        use_llm_refinement=args.use_llm,
        llm_backend=args.backend,  # type: ignore[arg-type]
        llm_model_name=args.model,
    )
    print(json.dumps(summary.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
