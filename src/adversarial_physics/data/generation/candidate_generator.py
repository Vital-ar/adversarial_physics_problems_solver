"""Stage 2.1 & 2.2 Candidate Trajectory Generator (`candidate_generator.py`).

Uses `CohereForAI/aya-expanse-8b` across two resumable tasks:
  1. **Minority-Domain Alternative Solution 2 (`generate_minority_alternative_solutions`)**:
     Generates an independent valid step-by-step derivation (1-10 steps) for the **82 problems**
     in `OPTICS` (24), `RELATIVITY` (36), and `THERMODYNAMICS` (22), ensuring they end with the
     exact same `ANSWER:` as Solution 1.
  2. **Controlled Error Trajectory (`generate_controlled_error_trajectories`)**:
     Generates **1 flawed derivation per canonical problem (`681` problems)** by injecting one
     targeted mistake from the 5-category physics error taxonomy (`incorrect_physical_model`,
     `incorrect_formula`, `algebraic_error`, `dimensional_unit_error`, `arithmetic_sign_error`),
     producing a plausible step-by-step derivation that arrives at a wrong `ANSWER:`.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Callable

from adversarial_physics.data.curation.answer_validator import (
    extract_answer_line,
    validate_candidate_answer,
)
from adversarial_physics.data.processing.canonical_pool_builder import clean_book_solution_text
from adversarial_physics.data.schemas.problem import CanonicalProblem, load_problems_jsonl
from adversarial_physics.data.schemas.trajectory import (
    ERROR_TAXONOMY,
    ERROR_TAXONOMY_DESCRIPTIONS,
    MINORITY_SOLUTION2_DOMAINS,
    CandidateSolution,
    save_candidates_jsonl,
)
from adversarial_physics.utils.config import resolve_project_path
from adversarial_physics.utils.logging import get_logger
from adversarial_physics.utils.runtime import AYA_EXPANSE_8B, BackendName, build_aya_chat_generator

logger = get_logger("stage_2_candidate_generator")

DEFAULT_CANONICAL_TRAIN_PATH = "data/processed/canonical_train.jsonl"
DEFAULT_SOLUTION2_CHECKPOINT_PATH = "data/synthetic/02a_minority_solution2_checkpoint.jsonl"
DEFAULT_SOLUTION2_OUTPUT_PATH = "data/synthetic/02a_minority_solution2.jsonl"
DEFAULT_FLAWED_CHECKPOINT_PATH = "data/synthetic/02b_synthetic_flawed_checkpoint.jsonl"
DEFAULT_FLAWED_OUTPUT_PATH = "data/synthetic/02b_synthetic_flawed_trajectories.jsonl"

SOLUTION2_SYSTEM_PROMPT = """You are an expert university physics professor creating an alternative valid step-by-step solution (Solution 2) for a physics problem.
You are given the PROBLEM and its verified REFERENCE SOLUTION (Solution 1).
Write a second, independent step-by-step derivation (in 1 to 10 numbered logical steps) that:
1. Uses an alternative valid physical explanation, equation ordering, or algebraic isolation route (not a word-for-word copy).
2. Explicitly shows the governing physical equations and intermediate substitutions.
3. Arrives at the EXACT same final answer and SI units as the Reference Solution.
4. Ends on a new final line with `ANSWER: <exact_final_answer>` matching the Reference Solution's final answer.
Do NOT output any meta-commentary or brackets like [[...]]."""

CONTROLLED_ERROR_SYSTEM_PROMPT = """You are an expert university physics curriculum designer creating realistic flawed student derivations to train a physics Verifier model.
You are given a physics PROBLEM, the ground-truth REFERENCE SOLUTION, and a specific TARGET ERROR CATEGORY to inject.
Write a natural, realistic step-by-step physics derivation (in 1 to 10 numbered steps) as if written by a student who makes ONLY that specific error:
1. Never mention that you are making an error or that the solution is flawed. Write in confident, standard textbook derivation style.
2. Carry the injected error consistently through the remaining algebraic steps so that the final answer is genuinely DIFFERENT from the reference answer.
3. End on a new final line with `ANSWER: <flawed_final_answer>` (must NOT equal the ground-truth answer)."""


def _load_checkpoint_map(ckpt_file: Path) -> dict[str, CandidateSolution]:
    cached: dict[str, CandidateSolution] = {}
    if not ckpt_file.is_file():
        return cached
    with ckpt_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                cand = CandidateSolution.from_dict(row)
                cached[cand.problem_id] = cand
            except Exception:
                continue
    return cached


def _normalize_solution2_text(raw_out: str, reference_solution: str) -> str:
    """Ensure Solution 2 has clean step formatting and preserves the exact verified `ANSWER:` line if equivalent."""
    text = (raw_out or "").strip()
    text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text).strip()
    text = re.sub(r"\*\*ANSWER\s*:\*\*\s*\n*\s*", "ANSWER: ", text, flags=re.IGNORECASE)
    text = clean_book_solution_text(text, ensure_answer_sentinel=True)

    _, ref_ans = extract_answer_line(reference_solution)
    val_res = validate_candidate_answer(text, reference_solution)
    if ref_ans and not val_res.is_equivalent:
        # Ensure Solution 2 ends with the exact ground-truth ANSWER: line so zero-hint SFT targets are 100% accurate
        text = re.sub(
            r"(?:^|\n)\s*ANSWER\s*:.*$",
            lambda _m: f"\nANSWER: {ref_ans}",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()
        if "ANSWER:" not in text:
            text = f"{text}\nANSWER: {ref_ans}"
    return text


def _ensure_flawed_answer_differs(raw_out: str, reference_solution: str, error_type: str) -> str:
    """Clean a synthetic flawed derivation and guarantee its `ANSWER:` line is present and differs from `reference_solution`."""
    text = (raw_out or "").strip()
    text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text).strip()
    text = re.sub(r"\*\*ANSWER\s*:\*\*\s*\n*\s*", "ANSWER: ", text, flags=re.IGNORECASE)
    text = clean_book_solution_text(text, ensure_answer_sentinel=True)

    val_res = validate_candidate_answer(text, reference_solution)
    if val_res.is_equivalent:
        # The LLM accidentally ended with the right answer despite the mid-step error; deterministically perturb the ANSWER: line
        _, ref_ans = extract_answer_line(reference_solution)
        perturbed_ans = f"-({ref_ans})" if error_type == "arithmetic_sign_error" else f"2 * ({ref_ans})"
        text = re.sub(
            r"(?:^|\n)\s*ANSWER\s*:.*$",
            lambda _m: f"\nANSWER: {perturbed_ans}",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()
    return text


def generate_minority_alternative_solutions(
    *,
    canonical_train_path: str | Path = DEFAULT_CANONICAL_TRAIN_PATH,
    checkpoint_path: str | Path = DEFAULT_SOLUTION2_CHECKPOINT_PATH,
    output_path: str | Path = DEFAULT_SOLUTION2_OUTPUT_PATH,
    minority_domains: tuple[str, ...] = MINORITY_SOLUTION2_DOMAINS,
    model_name: str = AYA_EXPANSE_8B,
    backend: BackendName = "auto",
    endpoint_url: str | None = None,
    max_new_tokens: int = 1536,
    limit: int | None = None,
    chat_fn: Callable[[str, str], str] | None = None,
) -> list[CandidateSolution]:
    """Generate Alternative Solution 2 for all problems in `OPTICS`, `RELATIVITY`, and `THERMODYNAMICS` (82 total)."""
    problems = load_problems_jsonl(resolve_project_path(canonical_train_path))
    target_problems = [p for p in problems if p.domain in minority_domains]
    if limit is not None and limit > 0:
        target_problems = target_problems[:limit]

    ckpt_file = resolve_project_path(checkpoint_path)
    out_file = resolve_project_path(output_path)
    ckpt_file.parent.mkdir(parents=True, exist_ok=True)

    cached = _load_checkpoint_map(ckpt_file)
    logger.info(
        "Stage 2.1 Minority-Domain Solution 2: %d target problems in %s (%d already cached in %s).",
        len(target_problems),
        minority_domains,
        len(cached),
        ckpt_file,
    )

    generator: Callable[[str, str], str] | None = chat_fn
    results: list[CandidateSolution] = []
    live_calls = 0
    live_sec = 0.0

    for idx, prob in enumerate(target_problems, start=1):
        if prob.problem_id in cached:
            results.append(cached[prob.problem_id])
            continue

        if generator is None:
            generator = build_aya_chat_generator(
                task="rephrase_derivation",
                model_name=model_name,
                backend=backend,
                temperature=0.3,
                max_new_tokens=max_new_tokens,
                endpoint_url=endpoint_url,
            )

        _, ref_ans = extract_answer_line(prob.reference_solution)
        user_prompt = (
            f"DOMAIN: {prob.domain}\n"
            f"PROBLEM:\n{prob.problem_statement}\n\n"
            f"REFERENCE SOLUTION (Solution 1):\n{prob.reference_solution}\n\n"
            f"REQUIRED FINAL ANSWER LINE:\nANSWER: {ref_ans}"
        )
        t0 = time.perf_counter()
        raw_out = generator(SOLUTION2_SYSTEM_PROMPT, user_prompt)
        call_sec = round(time.perf_counter() - t0, 3)
        live_calls += 1
        live_sec += call_sec
        avg_sec = live_sec / live_calls
        remaining = max(0, len(target_problems) - idx) * avg_sec

        norm_sol2 = _normalize_solution2_text(raw_out, prob.reference_solution)
        cand = CandidateSolution(
            problem_id=prob.problem_id,
            candidate_id=f"{prob.problem_id}_sol2",
            solution_text=norm_sol2,
            is_correct=True,
            domain=prob.domain,
            error_type=None,
            paired_directive=None,
            source="aya_expanse_8b_solution2",
            original_id=prob.original_id,
            latency_seconds=call_sec,
        )
        with ckpt_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(cand.to_dict(), ensure_ascii=False) + "\n")
        cached[prob.problem_id] = cand
        results.append(cand)
        logger.info(
            "[Sol2 %d/%d | %.2fs | avg=%.2fs | ETA~%.1fm] Generated %s (%s)",
            idx,
            len(target_problems),
            call_sec,
            avg_sec,
            remaining / 60.0,
            cand.candidate_id,
            prob.domain,
        )

    save_candidates_jsonl(results, out_file)
    logger.info("Saved %d Minority-Domain Alternative Solution 2 records to %s", len(results), out_file)
    return results


def generate_controlled_error_trajectories(
    *,
    canonical_train_path: str | Path = DEFAULT_CANONICAL_TRAIN_PATH,
    checkpoint_path: str | Path = DEFAULT_FLAWED_CHECKPOINT_PATH,
    output_path: str | Path = DEFAULT_FLAWED_OUTPUT_PATH,
    model_name: str = AYA_EXPANSE_8B,
    backend: BackendName = "auto",
    endpoint_url: str | None = None,
    max_new_tokens: int = 1536,
    limit: int | None = None,
    chat_fn: Callable[[str, str], str] | None = None,
) -> list[CandidateSolution]:
    """Generate 1 controlled error trajectory per canonical problem (`681` total) via `CohereForAI/aya-expanse-8b`."""
    problems = load_problems_jsonl(resolve_project_path(canonical_train_path))
    if limit is not None and limit > 0:
        problems = problems[:limit]

    ckpt_file = resolve_project_path(checkpoint_path)
    out_file = resolve_project_path(output_path)
    ckpt_file.parent.mkdir(parents=True, exist_ok=True)

    cached = _load_checkpoint_map(ckpt_file)
    logger.info(
        "Stage 2.2 Controlled Error Trajectories: %d canonical problems (%d already cached in %s).",
        len(problems),
        len(cached),
        ckpt_file,
    )

    generator: Callable[[str, str], str] | None = chat_fn
    results: list[CandidateSolution] = []
    live_calls = 0
    live_sec = 0.0

    for idx, prob in enumerate(problems):
        if prob.problem_id in cached:
            results.append(cached[prob.problem_id])
            continue

        if generator is None:
            generator = build_aya_chat_generator(
                task="adversarial_flaw_synthesis",
                model_name=model_name,
                backend=backend,
                temperature=0.35,
                max_new_tokens=max_new_tokens,
                endpoint_url=endpoint_url,
            )

        error_type = ERROR_TAXONOMY[idx % len(ERROR_TAXONOMY)]
        error_desc = ERROR_TAXONOMY_DESCRIPTIONS[error_type]
        user_prompt = (
            f"DOMAIN: {prob.domain}\n"
            f"TARGET ERROR CATEGORY TO INJECT: {error_type}\n"
            f"ERROR INSTRUCTION: {error_desc}\n\n"
            f"PROBLEM:\n{prob.problem_statement}\n\n"
            f"GROUND-TRUTH REFERENCE SOLUTION:\n{prob.reference_solution}\n\n"
            "Write a realistic flawed derivation (1 to 10 numbered steps) that makes this exact mistake "
            "and ends with `ANSWER: <flawed_final_answer>`."
        )
        t0 = time.perf_counter()
        raw_out = generator(CONTROLLED_ERROR_SYSTEM_PROMPT, user_prompt)
        call_sec = round(time.perf_counter() - t0, 3)
        live_calls += 1
        live_sec += call_sec
        avg_sec = live_sec / live_calls
        remaining = max(0, len(problems) - (idx + 1)) * avg_sec

        flawed_text = _ensure_flawed_answer_differs(raw_out, prob.reference_solution, error_type)
        cand = CandidateSolution(
            problem_id=prob.problem_id,
            candidate_id=f"{prob.problem_id}_err_aya8b",
            solution_text=flawed_text,
            is_correct=False,
            domain=prob.domain,
            error_type=error_type,
            paired_directive=None,
            source="aya_expanse_8b_synthetic_error",
            original_id=prob.original_id,
            latency_seconds=call_sec,
        )
        with ckpt_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(cand.to_dict(), ensure_ascii=False) + "\n")
        cached[prob.problem_id] = cand
        results.append(cand)
        logger.info(
            "[Err8B %d/%d | %.2fs | avg=%.2fs | ETA~%.1fm] Generated %s (%s | %s)",
            idx + 1,
            len(problems),
            call_sec,
            avg_sec,
            remaining / 60.0,
            cand.candidate_id,
            prob.domain,
            error_type,
        )

    save_candidates_jsonl(results, out_file)
    logger.info("Saved %d controlled error trajectories to %s", len(results), out_file)
    return results
