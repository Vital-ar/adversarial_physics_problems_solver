"""Stage 2.4 Self-Contained Directive (`[[CHALLENGE]]`) & Domain-Balanced Approval (`[[APPROVE]]`) Generator (`directive_generator.py`).

Uses `CohereForAI/aya-expanse-8b` with resumable per-item checkpointing (`02g_directives_checkpoint.jsonl`):
  1. **Surviving Flawed Trajectory Directives (`1` hint per surviving trajectory)**:
     Writes a concise (`1-2` sentences, `<= 192` tokens), strictly self-contained physical/mathematical
     directive ending with `[[CHALLENGE]]` for every trajectory in `02f_surviving_flawed_trajectories.jsonl`.
     Enforces zero step/line references and zero final-answer leakage via `directive_leak_check.py`.
  2. **Complementary 2nd Hint for the 50% Two-Hint Solver Split (`341` problems)**:
     Stratifies the `681` problems into `340` one-hint problems (50%) and `341` two-hint problems (50%),
     prioritizing problems that already have 2 surviving flawed trajectories for the 2-hint half.
     For any problem in the 2-hint half that only had 1 surviving trajectory after deduplication,
     synthesizes a second distinct self-contained physical constraint directive so the 50/50 split is exact.
  3. **Domain-Balanced Approvals (`180` examples = `26.4%` of `681`, `30 per domain` across all 6 domains)**:
     Selects `30` correct solutions per domain across all 6 physics domains (`MECHANICS`, `ELECTROMAGNETISM`,
     `MODERN_PHYSICS`, `RELATIVITY`, `OPTICS`, `THERMODYNAMICS`) from the `763` valid correct derivations
     (`681` Canonical Solution 1 + `82` Minority-Domain Alternative Solution 2) and generates a concise
     physical confirmation ending with `[[APPROVE]]`.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from adversarial_physics.data.curation.directive_leak_check import (
    sanitize_directive_text,
    validate_verifier_directive,
)
from adversarial_physics.data.schemas.problem import CanonicalProblem, load_problems_jsonl
from adversarial_physics.data.schemas.trajectory import (
    VERDICT_APPROVE,
    VERDICT_CHALLENGE,
    CandidateSolution,
    load_candidates_jsonl,
)
from adversarial_physics.utils.config import resolve_project_path
from adversarial_physics.utils.logging import get_logger
from adversarial_physics.utils.runtime import AYA_EXPANSE_8B, BackendName, build_aya_chat_generator

logger = get_logger("stage_2_directive_generator")

DEFAULT_CANONICAL_TRAIN_PATH = "data/processed/canonical_train.jsonl"
DEFAULT_CURATED_SOL2_PATH = "data/synthetic/02d_curated_solution2.jsonl"
DEFAULT_SURVIVING_FLAWED_PATH = "data/synthetic/02f_surviving_flawed_trajectories.jsonl"
DEFAULT_DIRECTIVES_CHECKPOINT_PATH = "data/synthetic/02g_directives_checkpoint.jsonl"
DEFAULT_DIRECTIVES_OUTPUT_PATH = "data/synthetic/02g_generated_directives.jsonl"

CHALLENGE_DIRECTIVE_SYSTEM_PROMPT = """You are an expert university physics Verifier model trainer.
You are given a physics PROBLEM, the ground-truth REFERENCE SOLUTION, and a FLAWED PROPOSED SOLUTION.
Write a concise (1 to 2 sentences, at most 80 words) corrective directive that tells the Solver what physical law, equation, algebraic step, or unit constraint must be used to solve the problem correctly, followed on a new final line by `[[CHALLENGE]]`.

STRICT RULES:
1. NEVER reference step numbers, line numbers, or previous text (words like "in step 2", "in line 3", "above", "your derivation", "the proposed solution" are strictly FORBIDDEN because the Solver re-solves from scratch without seeing its flawed attempt).
2. NEVER reveal or leak the final numerical answer in the directive. State the physical principle, governing formula, or algebraic relationship instead.
3. End on a new line with `[[CHALLENGE]]`."""

SECOND_CONSTRAINT_SYSTEM_PROMPT = """You are an expert university physics Verifier model trainer.
You are given a physics PROBLEM, the ground-truth REFERENCE SOLUTION, and a FIRST DIRECTIVE already given to the Solver.
Write a SECOND, complementary self-contained physical or algebraic constraint directive (1 to 2 sentences, at most 70 words) that highlights an additional key step, boundary condition, substitution, or unit constraint from the Reference Solution (distinct from the First Directive) without leaking the final answer or referencing step numbers.
End on a new line with `[[CHALLENGE]]`."""

APPROVAL_DIRECTIVE_SYSTEM_PROMPT = """You are an expert university physics Verifier model trainer.
You are given a physics PROBLEM and a VALID PROPOSED SOLUTION.
Write a concise (1 to 2 sentences, at most 70 words), self-contained confirmation explaining why the governing physical law and algebraic derivation are sound, followed on a new final line by `[[APPROVE]]`.
Do NOT reference step numbers or line numbers ("in step 2", "above" are forbidden)."""


def _load_directive_checkpoint(ckpt_file: Path) -> dict[str, dict[str, Any]]:
    cached: dict[str, dict[str, Any]] = {}
    if not ckpt_file.is_file():
        return cached
    with ckpt_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                key = str(row.get("item_id") or "")
                if key:
                    cached[key] = row
            except Exception:
                continue
    return cached


def select_domain_balanced_approval_candidates(
    problems: list[CanonicalProblem],
    curated_sol2: list[CandidateSolution],
    *,
    per_domain_target: int = 30,
) -> list[CandidateSolution]:
    """Select `per_domain_target` (default `30`, total `30 * 6 = 180` = `26.4%` of `681`) valid correct solutions
    equally across all 6 canonical physics domains (`MECHANICS`, `ELECTROMAGNETISM`, `MODERN_PHYSICS`,
    `RELATIVITY`, `OPTICS`, `THERMODYNAMICS`).
    """
    by_domain: dict[str, list[CandidateSolution]] = defaultdict(list)
    for p in problems:
        by_domain[p.domain].append(
            CandidateSolution(
                problem_id=p.problem_id,
                candidate_id=f"{p.problem_id}_sol1_approve",
                solution_text=p.reference_solution,
                is_correct=True,
                domain=p.domain,
                error_type=None,
                paired_directive=None,
                source="canonical_solution1",
                original_id=p.original_id,
            )
        )
    for s2 in curated_sol2:
        by_domain[s2.domain].append(
            CandidateSolution(
                problem_id=s2.problem_id,
                candidate_id=f"{s2.candidate_id}_approve",
                solution_text=s2.solution_text,
                is_correct=True,
                domain=s2.domain,
                error_type=None,
                paired_directive=None,
                source="minority_solution2",
                original_id=s2.original_id,
            )
        )

    selected: list[CandidateSolution] = []
    for domain in sorted(by_domain.keys()):
        pool = by_domain[domain]
        selected.extend(pool[:per_domain_target])
    return selected


def generate_stage2_directives_and_approvals(
    *,
    canonical_train_path: str | Path = DEFAULT_CANONICAL_TRAIN_PATH,
    curated_sol2_path: str | Path = DEFAULT_CURATED_SOL2_PATH,
    surviving_flawed_path: str | Path = DEFAULT_SURVIVING_FLAWED_PATH,
    checkpoint_path: str | Path = DEFAULT_DIRECTIVES_CHECKPOINT_PATH,
    output_path: str | Path = DEFAULT_DIRECTIVES_OUTPUT_PATH,
    approvals_per_domain: int = 30,
    model_name: str = AYA_EXPANSE_8B,
    backend: BackendName = "auto",
    endpoint_url: str | None = None,
    max_new_tokens: int = 220,
    limit: int | None = None,
    chat_fn: Callable[[str, str], str] | None = None,
) -> list[dict[str, Any]]:
    """Generate self-contained `[[CHALLENGE]]` hints for all surviving flawed trajectories,
    complementary 2nd hints where needed for the 50/50 1-hint/2-hint Solver split,
    and 180 (`30/domain`) domain-balanced `[[APPROVE]]` confirmations.
    """
    problems = load_problems_jsonl(resolve_project_path(canonical_train_path))
    if limit is not None and limit > 0:
        problems = problems[:limit]
    prob_map: dict[str, CanonicalProblem] = {p.problem_id: p for p in problems}

    curated_sol2 = [
        c for c in load_candidates_jsonl(resolve_project_path(curated_sol2_path)) if c.problem_id in prob_map
    ]
    surviving_flawed = [
        c for c in load_candidates_jsonl(resolve_project_path(surviving_flawed_path)) if c.problem_id in prob_map
    ]

    ckpt_file = resolve_project_path(checkpoint_path)
    out_file = resolve_project_path(output_path)
    ckpt_file.parent.mkdir(parents=True, exist_ok=True)
    cached = _load_directive_checkpoint(ckpt_file)

    # Identify which problems are assigned to the 2-hint half (50% of problems, stratified by domain,
    # prioritizing problems that already have 2 surviving flawed trajectories)
    flawed_by_prob: dict[str, list[CandidateSolution]] = defaultdict(list)
    for c in surviving_flawed:
        flawed_by_prob[c.problem_id].append(c)

    two_hint_target_count = (len(problems) + 1) // 2
    probs_sorted_for_2hints = sorted(
        problems,
        key=lambda p: (-len(flawed_by_prob.get(p.problem_id, [])), p.domain, p.problem_id),
    )
    two_hint_problem_ids: set[str] = {p.problem_id for p in probs_sorted_for_2hints[:two_hint_target_count]}

    # Select domain-balanced approval candidates (30 per domain = 180 total on full pool)
    eff_per_domain = approvals_per_domain if limit is None else max(1, min(approvals_per_domain, len(problems) // 6 or 1))
    approval_candidates = select_domain_balanced_approval_candidates(
        problems, curated_sol2, per_domain_target=eff_per_domain
    )

    # Count total work items for progress/ETA reporting
    extra_2nd_hint_probs = [
        p for p in problems if p.problem_id in two_hint_problem_ids and len(flawed_by_prob.get(p.problem_id, [])) < 2
    ]
    total_items = len(surviving_flawed) + len(extra_2nd_hint_probs) + len(approval_candidates)
    logger.info(
        "Stage 2.4 Directive & Approval Generation: %d surviving flawed trajectories + %d complementary 2nd hints "
        "+ %d domain-balanced approvals = %d total items (%d cached in %s).",
        len(surviving_flawed),
        len(extra_2nd_hint_probs),
        len(approval_candidates),
        total_items,
        len(cached),
        ckpt_file,
    )

    generator: Callable[[str, str], str] | None = chat_fn
    live_calls = 0
    live_sec = 0.0
    all_records: list[dict[str, Any]] = []
    processed_idx = 0

    def _get_gen() -> Callable[[str, str], str]:
        nonlocal generator
        if generator is None:
            generator = build_aya_chat_generator(
                task="directive_generation",
                model_name=model_name,
                backend=backend,
                temperature=0.2,
                max_new_tokens=max_new_tokens,
                endpoint_url=endpoint_url,
            )
        return generator

    # Part A: 1 Challenge Directive per Surviving Flawed Trajectory
    primary_hint_by_prob: dict[str, str] = {}
    for cand in surviving_flawed:
        processed_idx += 1
        item_id = f"challenge:{cand.candidate_id}"
        if item_id in cached:
            rec = cached[item_id]
            all_records.append(rec)
            primary_hint_by_prob.setdefault(cand.problem_id, str(rec["directive_body"]))
            continue

        prob = prob_map[cand.problem_id]
        user_prompt = (
            f"DOMAIN: {prob.domain}\n"
            f"PROBLEM:\n{prob.problem_statement}\n\n"
            f"GROUND-TRUTH REFERENCE SOLUTION:\n{prob.reference_solution}\n\n"
            f"FLAWED PROPOSED SOLUTION:\n{cand.solution_text}"
        )
        t0 = time.perf_counter()
        raw_out = _get_gen()(CHALLENGE_DIRECTIVE_SYSTEM_PROMPT, user_prompt)
        call_sec = round(time.perf_counter() - t0, 3)
        live_calls += 1
        live_sec += call_sec
        avg_sec = live_sec / live_calls
        rem_sec = max(0, total_items - processed_idx) * avg_sec

        formatted = sanitize_directive_text(
            raw_out,
            expected_verdict=VERDICT_CHALLENGE,
            problem_statement=prob.problem_statement,
            reference_solution=prob.reference_solution,
        )
        val = validate_verifier_directive(
            formatted,
            problem_statement=prob.problem_statement,
            reference_solution=prob.reference_solution,
            expected_verdict=VERDICT_CHALLENGE,
        )
        rec = {
            "item_id": item_id,
            "kind": "surviving_flawed_challenge",
            "problem_id": cand.problem_id,
            "candidate_id": cand.candidate_id,
            "domain": cand.domain,
            "source": cand.source,
            "error_type": cand.error_type,
            "proposed_solution": cand.solution_text,
            "directive_body": val.cleaned_directive_body,
            "verdict": VERDICT_CHALLENGE,
            "formatted_output": val.formatted_full_output,
            "is_valid": val.is_valid,
            "latency_seconds": call_sec,
        }
        with ckpt_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        cached[item_id] = rec
        all_records.append(rec)
        primary_hint_by_prob.setdefault(cand.problem_id, val.cleaned_directive_body)
        logger.info(
            "[Dir %d/%d | %.2fs | avg=%.2fs | ETA~%.1fm] Challenge hint for %s (%s)",
            processed_idx,
            total_items,
            call_sec,
            avg_sec,
            rem_sec / 60.0,
            cand.candidate_id,
            cand.domain,
        )

    # Part B: Complementary 2nd Hint for any 2-hint problem that only had 1 surviving trajectory
    for prob in extra_2nd_hint_probs:
        processed_idx += 1
        item_id = f"challenge2:{prob.problem_id}_hint2"
        if item_id in cached:
            all_records.append(cached[item_id])
            continue

        first_hint = primary_hint_by_prob.get(prob.problem_id, "")
        user_prompt = (
            f"DOMAIN: {prob.domain}\n"
            f"PROBLEM:\n{prob.problem_statement}\n\n"
            f"GROUND-TRUTH REFERENCE SOLUTION:\n{prob.reference_solution}\n\n"
            f"FIRST DIRECTIVE ALREADY GIVEN:\n{first_hint}"
        )
        t0 = time.perf_counter()
        raw_out = _get_gen()(SECOND_CONSTRAINT_SYSTEM_PROMPT, user_prompt)
        call_sec = round(time.perf_counter() - t0, 3)
        live_calls += 1
        live_sec += call_sec
        avg_sec = live_sec / live_calls
        rem_sec = max(0, total_items - processed_idx) * avg_sec

        formatted = sanitize_directive_text(
            raw_out,
            expected_verdict=VERDICT_CHALLENGE,
            problem_statement=prob.problem_statement,
            reference_solution=prob.reference_solution,
        )
        val = validate_verifier_directive(
            formatted,
            problem_statement=prob.problem_statement,
            reference_solution=prob.reference_solution,
            expected_verdict=VERDICT_CHALLENGE,
        )
        rec = {
            "item_id": item_id,
            "kind": "complementary_second_hint",
            "problem_id": prob.problem_id,
            "candidate_id": f"{prob.problem_id}_hint2_constraint",
            "domain": prob.domain,
            "source": "aya_expanse_8b_second_hint",
            "error_type": "complementary_constraint",
            "proposed_solution": "",
            "directive_body": val.cleaned_directive_body,
            "verdict": VERDICT_CHALLENGE,
            "formatted_output": val.formatted_full_output,
            "is_valid": val.is_valid,
            "latency_seconds": call_sec,
        }
        with ckpt_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        cached[item_id] = rec
        all_records.append(rec)
        logger.info(
            "[Dir %d/%d | %.2fs | avg=%.2fs | ETA~%.1fm] 2nd complementary hint for %s (%s)",
            processed_idx,
            total_items,
            call_sec,
            avg_sec,
            rem_sec / 60.0,
            prob.problem_id,
            prob.domain,
        )

    # Part C: 180 Domain-Balanced Approvals (`30 per domain` across all 6 domains)
    for app_cand in approval_candidates:
        processed_idx += 1
        item_id = f"approve:{app_cand.candidate_id}"
        if item_id in cached:
            all_records.append(cached[item_id])
            continue

        prob = prob_map[app_cand.problem_id]
        user_prompt = (
            f"DOMAIN: {prob.domain}\n"
            f"PROBLEM:\n{prob.problem_statement}\n\n"
            f"VALID PROPOSED SOLUTION:\n{app_cand.solution_text}"
        )
        t0 = time.perf_counter()
        raw_out = _get_gen()(APPROVAL_DIRECTIVE_SYSTEM_PROMPT, user_prompt)
        call_sec = round(time.perf_counter() - t0, 3)
        live_calls += 1
        live_sec += call_sec
        avg_sec = live_sec / live_calls
        rem_sec = max(0, total_items - processed_idx) * avg_sec

        formatted = sanitize_directive_text(
            raw_out,
            expected_verdict=VERDICT_APPROVE,
            problem_statement=prob.problem_statement,
            reference_solution=prob.reference_solution,
        )
        val = validate_verifier_directive(
            formatted,
            problem_statement=prob.problem_statement,
            reference_solution=prob.reference_solution,
            expected_verdict=VERDICT_APPROVE,
        )
        rec = {
            "item_id": item_id,
            "kind": "domain_balanced_approval",
            "problem_id": app_cand.problem_id,
            "candidate_id": app_cand.candidate_id,
            "domain": app_cand.domain,
            "source": app_cand.source,
            "error_type": None,
            "proposed_solution": app_cand.solution_text,
            "directive_body": val.cleaned_directive_body,
            "verdict": VERDICT_APPROVE,
            "formatted_output": val.formatted_full_output,
            "is_valid": val.is_valid,
            "latency_seconds": call_sec,
        }
        with ckpt_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        cached[item_id] = rec
        all_records.append(rec)
        logger.info(
            "[Dir %d/%d | %.2fs | avg=%.2fs | ETA~%.1fm] Approval rationale for %s (%s)",
            processed_idx,
            total_items,
            call_sec,
            avg_sec,
            rem_sec / 60.0,
            app_cand.candidate_id,
            app_cand.domain,
        )

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info("Saved %d generated directive & approval records to %s", len(all_records), out_file)
    return all_records
