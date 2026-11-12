"""Stage 2.5 SFT Dataset Builder & CuratorKIT Provenance Exporter (`dataset_exporter.py`).

Constructs, curates with `CuratorKIT`, and exports all 5 progressive SFT datasets for `AlignTune`
training and separate Hugging Face Hub dataset repositories:
  1. `data/sft/solver_e1_manual_det/` (`1,076` train examples — Manual deterministic Python rules only, BEFORE `CuratorKIT`)
  2. `data/sft/solver_e2_curatorkit_nollm/` (`728` train examples — `CuratorKIT` without LLM tools)
  3. `data/sft/solver_e3_canonical_llm/` (`681` train examples — `CuratorKIT` + `aya-expanse-8b` 1-10 step canonical pool)
  4. `data/sft/solver_e4_adversarial/` (`1,444` train examples — `763` 0-hint [`681` Sol1 + `82` Minority Sol2] + `340` 1-hint + `341` 2-hint)
  5. `data/sft/verifier_e4_curated/` (`~1,482` train examples — all surviving deduplicated flawed trajectories + hints + `180` [`30/domain`] approvals)

Each dataset directory receives `train.jsonl` (plus `validation.jsonl` and `test.jsonl` for Solver datasets),
`manifest.json`, `lexsi_provenance.json`, and `checksums.txt` via `curatorkit.manifest.ProvenanceManifest`.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from curatorkit.gates.schema import SchemaGate
from curatorkit.manifest import ProvenanceManifest, build_provenance
from curatorkit.normalizers.clean import TextCleaner
from curatorkit.normalizers.dedup import ExactDeduplicator
from curatorkit.pipeline import PipelineResult
from curatorkit.schema import DataSample

from adversarial_physics.data.curation.curatorkit_runner import _normalize_step_lines_and_answer
from adversarial_physics.data.curation.directive_leak_check import (
    sanitize_directive_text,
    validate_verifier_directive,
)
from adversarial_physics.data.schemas.problem import CanonicalProblem, load_problems_jsonl
from adversarial_physics.data.schemas.trajectory import (
    VERDICT_APPROVE,
    VERDICT_CHALLENGE,
    CandidateSolution,
    SolverTrajectory,
    VerifierTrajectory,
    format_solver_prompt,
    format_verifier_prompt,
    load_candidates_jsonl,
)
from adversarial_physics.utils.config import resolve_project_path
from adversarial_physics.utils.logging import get_logger
from adversarial_physics.utils.runtime import AYA_EXPANSE_8B

logger = get_logger("stage_2_dataset_exporter")

DEFAULT_E1_MANUAL_DET_POOL_PATH = "data/intermediate/01a_manual_deterministic_only_pool.jsonl"
DEFAULT_E2_CURATORKIT_NOLLM_POOL_PATH = "data/intermediate/01b_deterministic_curated_pool.jsonl"
DEFAULT_E3_CANONICAL_TRAIN_PATH = "data/processed/canonical_train.jsonl"
DEFAULT_VALIDATION_PATH = "data/splits/validation.jsonl"
DEFAULT_TEST_PATH = "data/splits/test.jsonl"
DEFAULT_CURATED_SOL2_PATH = "data/synthetic/02d_curated_solution2.jsonl"
DEFAULT_SURVIVING_FLAWED_PATH = "data/synthetic/02f_surviving_flawed_trajectories.jsonl"
DEFAULT_DIRECTIVES_INPUT_PATH = "data/synthetic/02g_generated_directives.jsonl"
DEFAULT_SFT_OUTPUT_ROOT = "data/sft"


@dataclass(frozen=True)
class Stage2ExportSummary:
    """Complete export summary across all 5 progressive SFT datasets."""

    solver_e1_manual_det_count: int
    solver_e2_curatorkit_nollm_count: int
    solver_e3_canonical_llm_count: int
    solver_e4_adversarial_total_count: int
    solver_e4_turn0_sol1_count: int
    solver_e4_turn0_minority_sol2_count: int
    solver_e4_turn1_one_hint_count: int
    solver_e4_turn2_two_hints_count: int
    verifier_e4_total_count: int
    verifier_e4_challenged_count: int
    verifier_e4_approved_count: int
    verifier_e4_approvals_by_domain: dict[str, int]
    verifier_e4_challenges_by_source: dict[str, int]
    directives_valid_count: int
    directives_invalid_count: int
    output_files: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _problems_to_turn0_solver_trajectories(
    problems: list[CanonicalProblem],
    *,
    dataset_tier: str,
) -> list[SolverTrajectory]:
    rows: list[SolverTrajectory] = []
    for p in problems:
        prompt = format_solver_prompt(p.problem_statement, [])
        rows.append(
            SolverTrajectory(
                trajectory_id=f"{dataset_tier}:{p.problem_id}:turn0",
                problem_id=p.problem_id,
                domain=p.domain,
                turn=0,
                accumulated_directives=[],
                prompt=prompt,
                target_output=p.reference_solution,
                source_candidate_ids=[f"{p.problem_id}_sol1"],
                dataset_tier=dataset_tier,
            )
        )
    return rows


def _write_jsonl_rows(rows: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def _curate_and_write_dataset_provenance(
    *,
    dataset_dir: Path,
    dataset_name: str,
    method: str,
    rows: list[dict[str, Any]],
    prompt_field: str = "prompt",
    output_field: str = "target_output",
    id_field: str = "trajectory_id",
    base_model: str | None = AYA_EXPANSE_8B,
    input_refs: list[Path] | None = None,
    extra_params: dict[str, Any] | None = None,
    val_rows: list[dict[str, Any]] | None = None,
    test_rows: list[dict[str, Any]] | None = None,
) -> tuple[Path, list[dict[str, Any]]]:
    """Run CuratorKIT `TextCleaner`, `SchemaGate`, and `ExactDeduplicator`
    over a dataset's rows, write `train.jsonl` (plus `validation.jsonl`/`test.jsonl` when provided),
    and emit `manifest.json`, `lexsi_provenance.json`, and `checksums.txt` via `ProvenanceManifest`.
    (Note: MinHashDeduplicator is intentionally applied earlier at the trajectory candidate stage in
    curatorkit_runner.py, not here, to prevent collapsing legitimate multi-turn prompt variants
    such as turn0, turn1, and turn2 of the same underlying physics problem).
    """
    t0 = time.monotonic()
    dataset_dir.mkdir(parents=True, exist_ok=True)

    samples = [
        DataSample(
            id=str(r.get(id_field) or f"row_{idx:04d}"),
            source_uri=f"{dataset_name}:{r.get('problem_id', idx)}",
            instruction=str(r.get(prompt_field) or ""),
            input=str(r.get("proposed_solution") or ""),
            output=str(r.get(output_field) or ""),
            task_type="instruction_following",
            metadata=dict(r),
        )
        for idx, r in enumerate(rows)
    ]

    cleaner = TextCleaner(fields=["instruction", "output"])
    schema_gate = SchemaGate(required_fields=["instruction", "output"], min_tokens=20, max_tokens=8192)
    exact_dedup = ExactDeduplicator()

    cleaned = cleaner.run(samples)
    passed, rejected = schema_gate.run(cleaned)
    deduped = exact_dedup.run(passed)

    # Sync cleaned instruction/output back into row metadata
    curated_rows: list[dict[str, Any]] = []
    for s in deduped:
        row_dict = dict(s.metadata or {})
        row_dict[prompt_field] = s.instruction.strip()
        row_dict[output_field] = s.output.strip()
        curated_rows.append(row_dict)

    train_path = _write_jsonl_rows(curated_rows, dataset_dir / "train.jsonl")
    tracked_files = [train_path]

    if val_rows is not None:
        val_path = _write_jsonl_rows(val_rows, dataset_dir / "validation.jsonl")
        tracked_files.append(val_path)
    if test_rows is not None:
        test_path = _write_jsonl_rows(test_rows, dataset_dir / "test.jsonl")
        tracked_files.append(test_path)

    stage_counts = {
        "TextCleaner": {"input_count": len(samples), "output_count": len(cleaned)},
        "SchemaGate": {
            "input_count": len(cleaned),
            "output_count": len(passed),
            "rejected_count": len(rejected),
        },
        "ExactDeduplicator": {
            "input_count": len(passed),
            "output_count": len(deduped),
            "duplicates_removed": len(passed) - len(deduped),
        },
    }

    pipeline_res = PipelineResult(
        passed=deduped,
        rejected=list(rejected),
        stage_counts=stage_counts,
        wall_clock_seconds=time.monotonic() - t0,
    )

    cfg_hash = hashlib.sha256(
        json.dumps({"dataset": dataset_name, "method": method, "rows": len(curated_rows)}, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]

    prov = build_provenance(
        method=method,
        inputs=[{"kind": "dataset", "ref": p.name} for p in (input_refs or []) if p.exists()],
        params={
            "dataset_name": dataset_name,
            "author": "vital-ar",
            "train_count": len(curated_rows),
            "validation_count": len(val_rows) if val_rows is not None else 0,
            "test_count": len(test_rows) if test_rows is not None else 0,
            **(extra_params or {}),
        },
        base_model=base_model,
    )

    manifest_writer = ProvenanceManifest(
        result=pipeline_res,
        pipeline_config_hash=cfg_hash,
        output_dir=dataset_dir,
        provenance=prov,
    )
    manifest_path = manifest_writer.write()
    tracked_files.extend([manifest_path, dataset_dir / "lexsi_provenance.json"])
    manifest_writer.write_checksums(tracked_files)

    return train_path, curated_rows


def export_all_stage2_sft_datasets(
    *,
    e1_pool_path: str | Path = DEFAULT_E1_MANUAL_DET_POOL_PATH,
    e2_pool_path: str | Path = DEFAULT_E2_CURATORKIT_NOLLM_POOL_PATH,
    e3_pool_path: str | Path = DEFAULT_E3_CANONICAL_TRAIN_PATH,
    validation_path: str | Path = DEFAULT_VALIDATION_PATH,
    test_path: str | Path = DEFAULT_TEST_PATH,
    curated_sol2_path: str | Path = DEFAULT_CURATED_SOL2_PATH,
    surviving_flawed_path: str | Path = DEFAULT_SURVIVING_FLAWED_PATH,
    directives_input_path: str | Path = DEFAULT_DIRECTIVES_INPUT_PATH,
    sft_output_root: str | Path = DEFAULT_SFT_OUTPUT_ROOT,
) -> Stage2ExportSummary:
    """Build, curate with CuratorKIT, and export all 5 progressive SFT datasets (`E1`, `E2`, `E3`,
    `E4 Solver`, `E4 Verifier`) with per-dataset `manifest.json`, `lexsi_provenance.json`, and `checksums.txt`.
    """
    out_root = resolve_project_path(sft_output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    e1_file = resolve_project_path(e1_pool_path)
    e2_file = resolve_project_path(e2_pool_path)
    synth_e2_file = resolve_project_path("data/intermediate/01b_e2_books_plus_synthetic.jsonl")
    if e2_pool_path == DEFAULT_E2_CURATORKIT_NOLLM_POOL_PATH and synth_e2_file.is_file():
        e2_file = synth_e2_file
    e3_file = resolve_project_path(e3_pool_path)
    val_file = resolve_project_path(validation_path)
    test_file = resolve_project_path(test_path)
    sol2_file = resolve_project_path(curated_sol2_path)
    surv_file = resolve_project_path(surviving_flawed_path)
    dir_file = resolve_project_path(directives_input_path)

    e1_problems = load_problems_jsonl(e1_file)
    e2_problems = load_problems_jsonl(e2_file)
    e3_problems = load_problems_jsonl(e3_file)
    val_problems = load_problems_jsonl(val_file) if val_file.is_file() else []
    test_problems = load_problems_jsonl(test_file) if test_file.is_file() else []

    prob_map: dict[str, CanonicalProblem] = {p.problem_id: p for p in e3_problems}
    curated_sol2: list[CandidateSolution] = [
        c for c in load_candidates_jsonl(sol2_file) if c.problem_id in prob_map
    ]
    sol2_map: dict[str, str] = {c.candidate_id: c.solution_text for c in curated_sol2}

    surviving_flawed = load_candidates_jsonl(surv_file) if surv_file.is_file() else []
    surviving_map: dict[str, CandidateSolution] = {c.candidate_id: c for c in surviving_flawed}

    val_sft_rows = [
        r.to_dict() for r in _problems_to_turn0_solver_trajectories(val_problems, dataset_tier="validation")
    ]
    test_sft_rows = [
        r.to_dict() for r in _problems_to_turn0_solver_trajectories(test_problems, dataset_tier="test")
    ]

    # 0. Sanitize and validate all directives in `02g_generated_directives.jsonl` and sync with curated trajectories
    raw_directive_records: list[dict[str, Any]] = []
    if dir_file.is_file():
        with dir_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_directive_records.append(json.loads(line))

    directive_records: list[dict[str, Any]] = []
    valid_dir_count = 0
    invalid_dir_count = 0

    for rec in raw_directive_records:
        pid = str(rec.get("problem_id") or "")
        prob = prob_map.get(pid)
        if prob is None:
            continue
        cid = str(rec.get("candidate_id") or "")
        verdict = str(rec.get("verdict") or VERDICT_CHALLENGE).strip()

        # If this is a challenge record, only keep it if its trajectory survived in `02f_surviving_flawed_trajectories.jsonl`
        if verdict == VERDICT_CHALLENGE and str(rec.get("kind") or "") != "complementary_second_hint":
            if surviving_map and cid not in surviving_map:
                continue
            if cid in surviving_map:
                rec["proposed_solution"] = surviving_map[cid].solution_text
        elif verdict == VERDICT_APPROVE:
            base_cid = cid.removesuffix("_approve")
            if base_cid in sol2_map:
                rec["proposed_solution"] = sol2_map[base_cid]
            else:
                rec["proposed_solution"] = _normalize_step_lines_and_answer(prob.reference_solution)

        sanitized_full = sanitize_directive_text(
            str(rec.get("formatted_output") or rec.get("directive_body") or ""),
            expected_verdict=verdict,
            problem_statement=prob.problem_statement,
            reference_solution=prob.reference_solution,
        )
        val_res = validate_verifier_directive(
            sanitized_full,
            problem_statement=prob.problem_statement,
            reference_solution=prob.reference_solution,
            expected_verdict=verdict,
        )
        rec["directive_body"] = val_res.cleaned_directive_body
        rec["formatted_output"] = val_res.formatted_full_output
        rec["is_valid"] = val_res.is_valid
        if val_res.is_valid:
            valid_dir_count += 1
            directive_records.append(rec)
        else:
            invalid_dir_count += 1

    # Save cleaned `02g_generated_directives.jsonl`
    if directive_records:
        _write_jsonl_rows(directive_records, dir_file)

    # 1. Build & Export E1, E2, and E3 Turn-0 Solver SFT Datasets
    e1_solver_rows = [
        r.to_dict() for r in _problems_to_turn0_solver_trajectories(e1_problems, dataset_tier="E1_solver_manual_det")
    ]
    e2_solver_rows = [
        r.to_dict() for r in _problems_to_turn0_solver_trajectories(e2_problems, dataset_tier="E2_solver_curatorkit_nollm")
    ]
    e3_solver_rows = [
        r.to_dict() for r in _problems_to_turn0_solver_trajectories(e3_problems, dataset_tier="E3_solver_canonical_llm")
    ]

    e1_out, e1_curated = _curate_and_write_dataset_provenance(
        dataset_dir=out_root / "solver_e1_manual_det",
        dataset_name="vital-ar/physics-solver-e1-manual-rules",
        method="stage1_manual_deterministic_rules_only",
        rows=e1_solver_rows,
        base_model=None,
        input_refs=[e1_file],
        val_rows=val_sft_rows,
        test_rows=test_sft_rows,
    )
    e2_out, e2_curated = _curate_and_write_dataset_provenance(
        dataset_dir=out_root / "solver_e2_curatorkit_nollm",
        dataset_name="vital-ar/physics-solver-e2-curatorkit-nollm",
        method="stage1_curatorkit_deterministic_gates",
        rows=e2_solver_rows,
        base_model=None,
        input_refs=[e2_file],
        val_rows=val_sft_rows,
        test_rows=test_sft_rows,
    )
    e3_out, e3_curated = _curate_and_write_dataset_provenance(
        dataset_dir=out_root / "solver_e3_canonical_llm",
        dataset_name="vital-ar/physics-solver-e3-canonical-aya8b",
        method="stage1_curatorkit_plus_aya8b_canonical",
        rows=e3_solver_rows,
        base_model=AYA_EXPANSE_8B,
        input_refs=[e3_file],
        val_rows=val_sft_rows,
        test_rows=test_sft_rows,
    )

    # 2. Build & Export E4 Solver SFT Dataset (`solver_e4_adversarial`: 0 hints + 50% 1-hint + 50% 2-hints)
    e4_solver_objs: list[SolverTrajectory] = []
    for p in e3_problems:
        e4_solver_objs.append(
            SolverTrajectory(
                trajectory_id=f"E4_solver:{p.problem_id}:turn0_sol1",
                problem_id=p.problem_id,
                domain=p.domain,
                turn=0,
                accumulated_directives=[],
                prompt=format_solver_prompt(p.problem_statement, []),
                target_output=p.reference_solution,
                source_candidate_ids=[f"{p.problem_id}_sol1"],
                dataset_tier="E4_solver_adversarial",
            )
        )
    for s2 in curated_sol2:
        p = prob_map[s2.problem_id]
        e4_solver_objs.append(
            SolverTrajectory(
                trajectory_id=f"E4_solver:{p.problem_id}:turn0_sol2",
                problem_id=p.problem_id,
                domain=p.domain,
                turn=0,
                accumulated_directives=[],
                prompt=format_solver_prompt(p.problem_statement, []),
                target_output=s2.solution_text,
                source_candidate_ids=[s2.candidate_id],
                dataset_tier="E4_solver_adversarial",
            )
        )

    hints_by_prob: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for rec in directive_records:
        if rec.get("verdict") == VERDICT_CHALLENGE and rec.get("directive_body"):
            pid = str(rec["problem_id"])
            cid = str(rec["candidate_id"])
            hbody = str(rec["directive_body"]).strip()
            existing_bodies = [b for _, b in hints_by_prob[pid]]
            if hbody not in existing_bodies:
                hints_by_prob[pid].append((cid, hbody))

    two_hint_target = (len(e3_problems) + 1) // 2
    sorted_probs = sorted(
        e3_problems,
        key=lambda p: (-len(hints_by_prob.get(p.problem_id, [])), p.domain, p.problem_id),
    )
    two_hint_probs = sorted_probs[:two_hint_target]
    one_hint_probs = sorted_probs[two_hint_target:]

    turn1_count = 0
    for p in one_hint_probs:
        prob_hints = hints_by_prob.get(p.problem_id, [])
        if not prob_hints:
            continue
        cid1, h1 = prob_hints[0]
        e4_solver_objs.append(
            SolverTrajectory(
                trajectory_id=f"E4_solver:{p.problem_id}:turn1_1hint",
                problem_id=p.problem_id,
                domain=p.domain,
                turn=1,
                accumulated_directives=[h1],
                prompt=format_solver_prompt(p.problem_statement, [h1]),
                target_output=p.reference_solution,
                source_candidate_ids=[cid1],
                dataset_tier="E4_solver_adversarial",
            )
        )
        turn1_count += 1

    turn2_count = 0
    for p in two_hint_probs:
        prob_hints = hints_by_prob.get(p.problem_id, [])
        if not prob_hints:
            continue
        if len(prob_hints) >= 2:
            (cid1, h1), (cid2, h2) = prob_hints[0], prob_hints[1]
            directives = [h1, h2]
            cids = [cid1, cid2]
            turn_num = 2
        else:
            cid1, h1 = prob_hints[0]
            first_step_lines = [
                ln.strip()
                for ln in p.reference_solution.splitlines()
                if ln.strip() and not ln.strip().upper().startswith("ANSWER:")
            ]
            first_step = first_step_lines[0] if first_step_lines else "Apply conservation and boundary equations"
            h2_clean = sanitize_directive_text(
                f"Additionally, verify dimensional consistency and boundary substitutions when setting up the governing relation ({first_step[:140].rstrip('.')}).\n[[CHALLENGE]]",
                expected_verdict=VERDICT_CHALLENGE,
                problem_statement=p.problem_statement,
                reference_solution=p.reference_solution,
            ).replace(VERDICT_CHALLENGE, "").strip()
            directives = [h1, h2_clean]
            cids = [cid1, f"{p.problem_id}_second_constraint"]
            turn_num = 2
        e4_solver_objs.append(
            SolverTrajectory(
                trajectory_id=f"E4_solver:{p.problem_id}:turn2_2hints",
                problem_id=p.problem_id,
                domain=p.domain,
                turn=turn_num,
                accumulated_directives=directives,
                prompt=format_solver_prompt(p.problem_statement, directives),
                target_output=p.reference_solution,
                source_candidate_ids=cids,
                dataset_tier="E4_solver_adversarial",
            )
        )
        turn2_count += 1

    e4_solver_out, e4_solver_curated = _curate_and_write_dataset_provenance(
        dataset_dir=out_root / "solver_e4_adversarial",
        dataset_name="vital-ar/physics-solver-e4-adversarial",
        method="stage2_curatorkit_adversarial_solver_sft",
        rows=[r.to_dict() for r in e4_solver_objs],
        base_model=AYA_EXPANSE_8B,
        input_refs=[e3_file, sol2_file, dir_file],
        extra_params={
            "turn0_sol1_count": len(e3_problems),
            "turn0_minority_sol2_count": len(curated_sol2),
            "turn1_one_hint_count": turn1_count,
            "turn2_two_hints_count": turn2_count,
        },
        val_rows=val_sft_rows,
        test_rows=test_sft_rows,
    )

    # 3. Build & Export E4 Verifier SFT Dataset (`verifier_e4_curated`)
    e4_verifier_objs: list[VerifierTrajectory] = []
    for rec in directive_records:
        kind = str(rec.get("kind") or "")
        if kind == "complementary_second_hint" or not str(rec.get("proposed_solution") or "").strip():
            continue
        pid = str(rec["problem_id"])
        prob = prob_map.get(pid)
        if prob is None:
            continue
        proposed_sol = str(rec["proposed_solution"]).strip()
        verdict = str(rec["verdict"]).strip()
        directive_body = str(rec["directive_body"]).strip()
        formatted_out = f"{directive_body}\n{verdict}"
        part_tag = (
            "part_1_domain_balanced_approved"
            if verdict == VERDICT_APPROVE
            else (
                "part_3_real_tiny_aya_challenged"
                if "tiny_aya" in str(rec.get("source") or "")
                else "part_2_synthetic_aya8b_challenged"
            )
        )
        e4_verifier_objs.append(
            VerifierTrajectory(
                trajectory_id=f"E4_verifier:{rec['candidate_id']}",
                problem_id=pid,
                domain=prob.domain,
                proposed_solution=proposed_sol,
                directive_text=directive_body,
                verdict=verdict,
                error_type=rec.get("error_type"),
                part=part_tag,
                prompt=format_verifier_prompt(prob.problem_statement, proposed_sol),
                target_output=formatted_out,
            )
        )

    e4_verifier_out, e4_verifier_curated = _curate_and_write_dataset_provenance(
        dataset_dir=out_root / "verifier_e4_curated",
        dataset_name="vital-ar/physics-verifier-e4-curated",
        method="stage2_curatorkit_adversarial_verifier_sft",
        rows=[v.to_dict() for v in e4_verifier_objs],
        base_model=AYA_EXPANSE_8B,
        input_refs=[e3_file, sol2_file, surv_file, dir_file],
        extra_params={
            "challenged_count": sum(1 for v in e4_verifier_objs if v.verdict == VERDICT_CHALLENGE),
            "approved_count": sum(1 for v in e4_verifier_objs if v.verdict == VERDICT_APPROVE),
        },
    )

    challenged_rows = [r for r in e4_verifier_curated if r.get("verdict") == VERDICT_CHALLENGE]
    approved_rows = [r for r in e4_verifier_curated if r.get("verdict") == VERDICT_APPROVE]

    output_files = {
        "solver_e1_manual_det": str(e1_out),
        "solver_e2_curatorkit_nollm": str(e2_out),
        "solver_e3_canonical_llm": str(e3_out),
        "solver_e4_adversarial": str(e4_solver_out),
        "verifier_e4_curated": str(e4_verifier_out),
    }

    summary = Stage2ExportSummary(
        solver_e1_manual_det_count=len(e1_curated),
        solver_e2_curatorkit_nollm_count=len(e2_curated),
        solver_e3_canonical_llm_count=len(e3_curated),
        solver_e4_adversarial_total_count=len(e4_solver_curated),
        solver_e4_turn0_sol1_count=len(e3_problems),
        solver_e4_turn0_minority_sol2_count=len(curated_sol2),
        solver_e4_turn1_one_hint_count=turn1_count,
        solver_e4_turn2_two_hints_count=turn2_count,
        verifier_e4_total_count=len(e4_verifier_curated),
        verifier_e4_challenged_count=len(challenged_rows),
        verifier_e4_approved_count=len(approved_rows),
        verifier_e4_approvals_by_domain=dict(sorted(Counter(r["domain"] for r in approved_rows).items())),
        verifier_e4_challenges_by_source=dict(sorted(Counter(r["part"] for r in challenged_rows).items())),
        directives_valid_count=valid_dir_count,
        directives_invalid_count=invalid_dir_count,
        output_files=output_files,
    )

    summary_path = out_root / "stage2_export_summary.json"
    summary_path.write_text(json.dumps(summary.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(
        "Stage 2.5 Export complete: E1=%d | E2=%d | E3=%d | E4_Solver=%d (%d Sol1 + %d Sol2 + %d 1-hint + %d 2-hint) "
        "| E4_Verifier=%d (%d challenged + %d approved) | directives_valid=%d (invalid=%d)",
        summary.solver_e1_manual_det_count,
        summary.solver_e2_curatorkit_nollm_count,
        summary.solver_e3_canonical_llm_count,
        summary.solver_e4_adversarial_total_count,
        summary.solver_e4_turn0_sol1_count,
        summary.solver_e4_turn0_minority_sol2_count,
        summary.solver_e4_turn1_one_hint_count,
        summary.solver_e4_turn2_two_hints_count,
        summary.verifier_e4_total_count,
        summary.verifier_e4_challenged_count,
        summary.verifier_e4_approved_count,
        summary.directives_valid_count,
        summary.directives_invalid_count,
    )
    return summary
