"""Stage 2.0 CLI: Subtopic-Grounded & Seed-Guided Synthetic Problem Generation (`03b_generate_synthetic_e2.py`).

Generates formal LaTeX physics problems from Part-1 clean textbook kernels, matched subtopics
(`data/literature/subtopic_chunks.jsonl`), and PhysGym `train_seeds`, then builds:
  * Revised E2 pool (`data/intermediate/01b_e2_books_plus_synthetic.jsonl`)
  * Strict CuratorKIT Part-2 verified E3 pool (`data/processed/canonical_train.jsonl`)

Default model: `CohereForAI/aya-expanse-8b` (or pass `--model CohereLabs/tiny-aya-global` as fallback).
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

from adversarial_physics.data.generation.synthetic_problem_generator import (  # noqa: E402
    DEFAULT_E2_COMBINED_POOL_PATH,
    DEFAULT_E2_READY_PROMPTS_PATH,
    DEFAULT_E2_READY_SUMMARY_PATH,
    DEFAULT_E3_CURATED_POOL_PATH,
    DEFAULT_SYNTHETIC_CHECKPOINT_PATH,
    DEFAULT_SYNTHETIC_RAW_PATH,
    DEFAULT_SYNTHETIC_SUMMARY_PATH,
    generate_synthetic_e2_and_e3_pools,
    prepare_e2_llm_generation_requests,
)
from adversarial_physics.data.ingestion.subtopic_extractor import DEFAULT_SUBTOPIC_CHUNKS_PATH  # noqa: E402
from adversarial_physics.utils.runtime import AYA_EXPANSE_8B  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean-books",
        type=str,
        default="data/intermediate/01b_deterministic_curated_pool.jsonl",
    )
    parser.add_argument("--subtopics", type=str, default=DEFAULT_SUBTOPIC_CHUNKS_PATH)
    parser.add_argument("--train-seeds", type=str, default="data/splits/train_seeds.jsonl")
    parser.add_argument("--validation", type=str, default="data/splits/validation.jsonl")
    parser.add_argument("--test", type=str, default="data/splits/test.jsonl")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_SYNTHETIC_CHECKPOINT_PATH)
    parser.add_argument("--synthetic-out", type=str, default=DEFAULT_SYNTHETIC_RAW_PATH)
    parser.add_argument("--e2-out", type=str, default=DEFAULT_E2_COMBINED_POOL_PATH)
    parser.add_argument("--e3-out", type=str, default=DEFAULT_E3_CURATED_POOL_PATH)
    parser.add_argument("--summary-out", type=str, default=DEFAULT_SYNTHETIC_SUMMARY_PATH)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Run zero-LLM preparation only: match Part-1 clean books + subtopics + seeds and save ready prompts.",
    )
    parser.add_argument("--prompts-out", type=str, default=DEFAULT_E2_READY_PROMPTS_PATH)
    parser.add_argument("--prompts-summary-out", type=str, default=DEFAULT_E2_READY_SUMMARY_PATH)
    parser.add_argument("--model", type=str, default=AYA_EXPANSE_8B)
    parser.add_argument(
        "--backend",
        choices=["auto", "kaggle_gpu", "hf_gpu", "local_cpu"],
        default="auto",
    )
    parser.add_argument("--max-generations", type=int, default=None)
    parser.add_argument(
        "--skip-part2-llm-judges",
        action="store_true",
        help="Skip CuratorKIT Part 2 Hallucination/Reward LLM judges (SymPy & strict hygiene still run).",
    )
    args = parser.parse_args(argv)

    if args.prepare_only:
        _, summary = prepare_e2_llm_generation_requests(
            clean_books_path=args.clean_books,
            subtopic_chunks_path=args.subtopics,
            train_seeds_path=args.train_seeds,
            output_path=args.prompts_out,
            summary_out=args.prompts_summary_out,
            max_generations=args.max_generations,
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    _, _, summary = generate_synthetic_e2_and_e3_pools(
        clean_books_path=args.clean_books,
        subtopic_chunks_path=args.subtopics,
        train_seeds_path=args.train_seeds,
        validation_path=args.validation,
        test_path=args.test,
        checkpoint_path=args.checkpoint,
        synthetic_raw_out=args.synthetic_out,
        e2_pool_out=args.e2_out,
        e3_pool_out=args.e3_out,
        summary_out=args.summary_out,
        model_name=args.model,
        backend=args.backend,  # type: ignore[arg-type]
        max_generations=args.max_generations,
        enable_part2_llm_judges=not args.skip_part2_llm_judges,
    )
    print(json.dumps(summary.to_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
