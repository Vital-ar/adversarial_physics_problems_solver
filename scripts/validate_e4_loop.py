"""Kaggle/GPU Validation Runner for Multi-Turn E4A (Solver) + E4B (Verifier) Loop.

Automatically pulls ``data/splits/validation.jsonl`` (falling back to HF Hub repos
or public ``YimengChen/PhysGym`` + ``configs/data/split_21_38_38.yaml``) and pulls
LoRA weights from HF Hub whenever local ``adapter_model.safetensors`` files are absent.

Usage on Kaggle:
    python scripts/validate_e4_loop.py --max-iterations 3
    python scripts/validate_e4_loop.py --max-iterations 3 --limit 5  # quick smoke test
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

# Ensure `src/` is on sys.path when invoked directly as `python scripts/validate_e4_loop.py`
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from adversarial_physics.orchestration.loop import (  # noqa: E402
    DEFAULT_BASE_MODEL,
    DEFAULT_VALIDATION_PATH,
    DualLoRAEngine,
    ProblemLoopResult,
    ensure_validation_dataset,
    run_adversarial_validation_loop,
)
from adversarial_physics.utils.logging import get_logger  # noqa: E402

logger = get_logger("validate_e4_loop")


def _fmt_pct(value: float | None) -> str:
    return f"{value * 100:.2f}%" if value is not None else "N/A"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run E4A (Solver) + E4B (Verifier) multi-turn validation loop on held-out PhysGym validation split."
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum Solver->Verifier iterations per problem (default: 3 = Turn 0 + up to 2 revisions).",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help=f"Base model repo ID or local path (default: {DEFAULT_BASE_MODEL}).",
    )
    parser.add_argument(
        "--solver-adapter",
        type=str,
        default=None,
        help="Solver (E4A) LoRA path or HF Hub repo ID (auto-resolves local weights or vital-ar HF Hub).",
    )
    parser.add_argument(
        "--verifier-adapter",
        type=str,
        default=None,
        help="Verifier (E4B) LoRA path or HF Hub repo ID (auto-resolves local weights or vital-ar HF Hub).",
    )
    parser.add_argument(
        "--validation-path",
        type=str,
        default=DEFAULT_VALIDATION_PATH,
        help=f"Local JSONL path for held-out validation split (default: {DEFAULT_VALIDATION_PATH}).",
    )
    parser.add_argument(
        "--validation-repo",
        type=str,
        default=None,
        help="Optional HF Hub dataset repo ID to pull validation.jsonl from if absent locally.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/validation/e4_loop_iter3",
        help="Directory where summary.json and results_per_problem.jsonl are written.",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=["raw", "chat_template", "alpaca"],
        default="raw",
        help="Prompt formatting mode. Keep 'raw' to match Stage 4 AlignTune instruction_following training.",
    )
    parser.add_argument("--solver-max-new-tokens", type=int, default=1024)
    parser.add_argument("--verifier-max-new-tokens", type=int, default=192)
    parser.add_argument("--temperature-solver", type=float, default=0.0)
    parser.add_argument("--temperature-verifier", type=float, default=0.0)
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Transformers device_map ('auto', 'balanced', 'cuda:0', or 'cpu').",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "fp16", "bf16", "fp32"],
        default="auto",
        help="Model compute precision (default: auto -> bf16 if supported else fp16 on CUDA).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of validation problems for quick smoke runs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted run by skipping problem_ids already present in results_per_problem.jsonl.",
    )
    parser.add_argument(
        "--skip-final-verify",
        action="store_true",
        help="Skip running Verifier on the final iteration (faster, omits final-turn Verifier stats).",
    )
    parser.add_argument(
        "--allow-duplicate-hints",
        action="store_true",
        help="Continue revising up to --max-iterations even if Verifier repeats an identical hint.",
    )
    parser.add_argument("--rel-tol", type=float, default=0.015, help="Numerical equivalence tolerance (default: 0.015).")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face Hub token (or set HF_TOKEN).")
    return parser


def _print_summary(payload: dict[str, Any]) -> None:
    metrics = payload["metrics"]
    solver = metrics["solver"]
    verifier = metrics["verifier"]
    eff = metrics["loop_efficiency"]
    meta = payload.get("metadata", {})

    print("\n" + "=" * 78)
    print("E4A (Solver) + E4B (Verifier) Multi-Turn Validation Summary")
    print("=" * 78)
    print(f"Problems evaluated     : {metrics['total_problems']} (max_iterations={metrics['max_iterations']})")
    print(f"Validation source      : {meta.get('validation_source', 'N/A')}")
    print(f"Solver adapter (E4A)   : {meta.get('solver_adapter', 'N/A')}")
    print(f"Verifier adapter (E4B) : {meta.get('verifier_adapter', 'N/A')}")
    print(f"Prompt mode            : {meta.get('prompt_mode', 'raw')}")
    print("-" * 78)
    print(
        f"Turn-0 Accuracy        : {_fmt_pct(solver['turn0_accuracy'])} "
        f"({solver['turn0_correct']}/{metrics['total_problems']}) | "
        f"ANSWER: compliance {_fmt_pct(solver['turn0_sentinel_compliance_rate'])}"
    )
    print(
        f"Final Loop Accuracy    : {_fmt_pct(solver['final_accuracy'])} "
        f"({solver['final_correct']}/{metrics['total_problems']}) | "
        f"ANSWER: compliance {_fmt_pct(solver['final_sentinel_compliance_rate'])}"
    )
    gain = solver["net_revision_gain"]
    gain_str = f"{gain * 100:+.2f} pp" if gain is not None else "N/A"
    print(f"Net Revision Gain      : {gain_str}")
    print(
        f"Conversion Rate        : {_fmt_pct(solver['correction_conversion_rate'])} "
        f"({solver['converted_count']}/{solver['turn0_wrong_count']} wrong repaired)"
    )
    print(
        f"Regression Rate        : {_fmt_pct(solver['regression_rate'])} "
        f"({solver['regressed_count']}/{solver['turn0_correct']} right broken)"
    )
    iter_line = " | ".join(
        f"{k}: {_fmt_pct(v['accuracy'])} ({v['correct']}/{v['total']})"
        for k, v in solver["per_iteration_cumulative"].items()
    )
    print(f"Cumulative by Iter     : {iter_line}")
    print("-" * 78)
    print(
        f"Verifier Reviews       : {verifier['total_reviews']} "
        f"(TP={verifier['tp_true_challenge']}, FP={verifier['fp_false_challenge']}, "
        f"TN={verifier['tn_true_approve']}, FN={verifier['fn_false_approve']}, "
        f"UNK={verifier['unknown_no_sentinel']})"
    )
    print(
        f"Verifier Decision Acc  : {_fmt_pct(verifier['verifier_sentinel_accuracy'])} | "
        f"Challenge F1={_fmt_pct(verifier['verifier_challenge_f1'])}"
    )
    print(
        f"False Approval Rate    : {_fmt_pct(verifier['verifier_false_approval_rate'])} (wrong -> APPROVE) | "
        f"False Challenge Rate: {_fmt_pct(verifier['verifier_false_challenge_rate'])} (right -> CHALLENGE)"
    )
    print(
        f"Mean Iterations / Prob : {eff['mean_iterations_used']} | "
        f"Mean Latency: {eff['mean_latency_s']}s | Stop reasons: {eff['stop_reason_counts']}"
    )
    print("-" * 78)
    print("Per-Domain Breakdown:")
    for domain, d in metrics["per_domain"].items():
        dg = d["net_revision_gain"]
        dg_str = f"{dg * 100:+.1f} pp" if dg is not None else "N/A"
        print(
            f"  {domain:<16} n={d['count']:>2} | "
            f"Turn0={_fmt_pct(d['turn0_accuracy']):>7} -> Final={_fmt_pct(d['final_accuracy']):>7} ({dg_str})"
        )
    print("=" * 78)
    print(f"Saved summary : {payload['artifacts']['summary_json']}")
    print(f"Saved traces  : {payload['artifacts']['results_per_problem_jsonl']}")


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    hf_token = args.hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

    if hf_token:
        try:
            from huggingface_hub import login

            login(token=hf_token, add_to_git_credential=False)
        except Exception as exc:
            logger.warning(f"Optional HF Hub login warning: {exc}")

    data_res = ensure_validation_dataset(
        args.validation_path,
        hf_repo_id=args.validation_repo,
        hf_token=hf_token,
    )
    problems = data_res.problems[: args.limit] if args.limit else data_res.problems
    logger.info(f"Prepared {len(problems)} validation problems (source={data_res.source_used})")

    engine = DualLoRAEngine(
        base_model_id=args.base_model,
        solver_adapter=args.solver_adapter,
        verifier_adapter=args.verifier_adapter,
        prompt_mode=args.prompt_mode,
        solver_max_new_tokens=args.solver_max_new_tokens,
        verifier_max_new_tokens=args.verifier_max_new_tokens,
        temperature_solver=args.temperature_solver,
        temperature_verifier=args.temperature_verifier,
        device_map=args.device_map,
        dtype=args.dtype,
        hf_token=hf_token,
    )

    def _on_progress(idx: int, total: int, res: ProblemLoopResult) -> None:
        logger.info(
            f"[{idx}/{total}] {res.problem_id} ({res.domain}) | "
            f"iters={res.iterations_used}/{res.max_iterations} ({res.stop_reason}) | "
            f"turn0={res.turn0_correct} -> final={res.final_correct} ({res.final_match_method})"
        )

    payload = run_adversarial_validation_loop(
        problems,
        engine=engine,
        max_iterations=args.max_iterations,
        verify_final_iteration=not args.skip_final_verify,
        stop_on_duplicate_hint=not args.allow_duplicate_hints,
        rel_tol=args.rel_tol,
        output_dir=args.output_dir,
        resume=args.resume,
        metadata={
            "validation_source": data_res.source_used,
            "validation_local_path": str(data_res.local_path),
            "limit": args.limit,
        },
        progress_callback=_on_progress,
    )
    _print_summary(payload)


if __name__ == "__main__":
    main()
