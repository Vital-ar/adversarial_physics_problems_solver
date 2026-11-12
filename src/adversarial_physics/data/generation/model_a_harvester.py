"""Stage 2.2B Real Model A (`CohereLabs/tiny-aya-global`) Trajectory Harvester (`model_a_harvester.py`).

Runs zero-shot inference with `CohereLabs/tiny-aya-global` across all `681` canonical training
problems (`1` trajectory per problem) using the exact Turn-0 Solver prompt.
Supports `--backend hf_gpu` (Dedicated HF Endpoint for `CohereLabs/tiny-aya-global`),
`--backend kaggle_gpu`, or `--backend local_cpu`, with per-problem resumable checkpointing.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

from adversarial_physics.data.curation.answer_validator import validate_candidate_answer
from adversarial_physics.data.processing.canonical_pool_builder import clean_book_solution_text
from adversarial_physics.data.schemas.problem import load_problems_jsonl
from adversarial_physics.data.schemas.trajectory import (
    CandidateSolution,
    format_solver_prompt,
    load_candidate_checkpoint_map,
    normalize_solution_fences_and_answer,
    save_candidates_jsonl,
)
from adversarial_physics.utils.config import resolve_project_path
from adversarial_physics.utils.logging import get_logger
from adversarial_physics.utils.runtime import TINY_AYA_GLOBAL, BackendName, build_aya_chat_generator

logger = get_logger("stage_2_tiny_aya_harvester")

DEFAULT_CANONICAL_TRAIN_PATH = "data/processed/canonical_train.jsonl"
DEFAULT_TINY_AYA_CHECKPOINT_PATH = "data/synthetic/02c_tiny_aya_harvest_checkpoint.jsonl"
DEFAULT_TINY_AYA_OUTPUT_PATH = "data/synthetic/02c_tiny_aya_trajectories.jsonl"

TINY_AYA_SOLVER_SYSTEM_PROMPT = (
    "You are a university physics problem solver. Solve the problem step by step in at most 10 logical steps "
    "and end your derivation on a new final line with ANSWER: <final_answer>."
)


_load_checkpoint_map = load_candidate_checkpoint_map


def harvest_tiny_aya_trajectories(
    *,
    canonical_train_path: str | Path = DEFAULT_CANONICAL_TRAIN_PATH,
    checkpoint_path: str | Path = DEFAULT_TINY_AYA_CHECKPOINT_PATH,
    output_path: str | Path = DEFAULT_TINY_AYA_OUTPUT_PATH,
    model_name: str = TINY_AYA_GLOBAL,
    backend: BackendName = "auto",
    endpoint_url: str | None = None,
    max_new_tokens: int = 1536,
    limit: int | None = None,
    chat_fn: Callable[[str, str], str] | None = None,
) -> list[CandidateSolution]:
    """Run zero-shot `CohereLabs/tiny-aya-global` across all 681 canonical training problems (1 trajectory per problem)."""
    problems = load_problems_jsonl(resolve_project_path(canonical_train_path))
    if limit is not None and limit > 0:
        problems = problems[:limit]

    ckpt_file = resolve_project_path(checkpoint_path)
    out_file = resolve_project_path(output_path)
    ckpt_file.parent.mkdir(parents=True, exist_ok=True)

    cached = _load_checkpoint_map(ckpt_file)
    logger.info(
        "Stage 2.2B Tiny-Aya Harvester: %d canonical problems (%d already cached in %s).",
        len(problems),
        len(cached),
        ckpt_file,
    )

    generator: Callable[[str, str], str] | None = chat_fn
    results: list[CandidateSolution] = []
    live_calls = 0
    live_sec = 0.0

    for idx, prob in enumerate(problems, start=1):
        if prob.problem_id in cached:
            results.append(cached[prob.problem_id])
            continue

        if generator is None:
            generator = build_aya_chat_generator(
                task="literature_extraction",
                model_name=model_name,
                backend=backend,
                temperature=0.1,
                max_new_tokens=max_new_tokens,
                endpoint_url=endpoint_url,
            )

        user_prompt = format_solver_prompt(prob.problem_statement)
        t0 = time.perf_counter()
        raw_out = generator(TINY_AYA_SOLVER_SYSTEM_PROMPT, user_prompt)
        call_sec = round(time.perf_counter() - t0, 3)
        live_calls += 1
        live_sec += call_sec
        avg_sec = live_sec / live_calls
        remaining = max(0, len(problems) - idx) * avg_sec

        cleaned = normalize_solution_fences_and_answer(raw_out, strip_fences=False)
        cleaned = clean_book_solution_text(cleaned, ensure_answer_sentinel=True)

        val_res = validate_candidate_answer(cleaned, prob.reference_solution)
        is_right = val_res.is_equivalent

        cand = CandidateSolution(
            problem_id=prob.problem_id,
            candidate_id=f"{prob.problem_id}_tiny_aya",
            solution_text=cleaned,
            is_correct=is_right,
            domain=prob.domain,
            error_type=None if is_right else "real_model_a_failure",
            paired_directive=None,
            source="tiny_aya_global_zero_shot",
            original_id=prob.original_id,
            latency_seconds=call_sec,
        )
        with ckpt_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(cand.to_dict(), ensure_ascii=False) + "\n")
        cached[prob.problem_id] = cand
        results.append(cand)
        logger.info(
            "[TinyAya %d/%d | %.2fs | avg=%.2fs | ETA~%.1fm] Harvested %s (is_correct=%s)",
            idx,
            len(problems),
            call_sec,
            avg_sec,
            remaining / 60.0,
            cand.candidate_id,
            is_right,
        )

    save_candidates_jsonl(results, out_file)
    num_wrong = sum(1 for r in results if not r.is_correct)
    logger.info(
        "Saved %d Tiny-Aya trajectories to %s (%d wrong/failure modes, %d accidentally correct).",
        len(results),
        out_file,
        num_wrong,
        len(results) - num_wrong,
    )
    return results
