"""Unit tests for Stage 5 multi-turn E4A (Solver) + E4B (Verifier) validation loop."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from adversarial_physics.data.schemas.problem import CanonicalProblem, save_problems_jsonl
from adversarial_physics.orchestration.loop import (
    DEFAULT_SOLVER_ADAPTER_REPO,
    canonicalize_validation_record,
    ensure_validation_dataset,
    format_model_prompt,
    parse_verifier_response,
    prepare_reference_for_validation,
    resolve_adapter_source,
    run_adversarial_validation_loop,
    run_single_problem_loop,
    score_solver_output,
)


def _sample_problems() -> list[CanonicalProblem]:
    return [
        CanonicalProblem(
            problem_id="p_approve_iter1",
            domain="MECHANICS",
            problem_statement="Find the speed of an object dropped from height h.",
            reference_solution="Using energy conservation:\n$$\\boxed{v = \\sqrt{2 g h}}$$",
        ),
        CanonicalProblem(
            problem_id="p_convert_iter2",
            domain="ELECTROMAGNETISM",
            problem_statement="Compute the orbital period T.",
            reference_solution="$$\\displaystyle T = \\frac{2\\pi m}{q B_0}$$",
        ),
        CanonicalProblem(
            problem_id="p_convert_iter3",
            domain="THERMODYNAMICS",
            problem_statement="Compute angular velocity omega_B.",
            reference_solution="$$\\omega_B = 4 \\sqrt{\\frac{13 k q Q}{23 m L^3}} \\tag{6}$$",
        ),
        CanonicalProblem(
            problem_id="p_regress",
            domain="OPTICS",
            problem_statement="Find refractive index n.",
            reference_solution="\\therefore n = 1.5",
        ),
    ]


def test_prepare_reference_handles_raw_physgym_latex_tails() -> None:
    ref_boxed = (
        "Deriving:\n$$\\boxed{T(h) = \\frac{\\mu G M_m}{n R (R_m + h)}}$$\n"
        "This result indicates that temperature decreases inversely with height \\(h\\)."
    )
    norm_boxed, ans_boxed = prepare_reference_for_validation(ref_boxed)
    assert "ANSWER:" in norm_boxed
    assert ans_boxed == r"T(h) = \frac{\mu G M_m}{n R (R_m + h)}"

    ref_tag = r"$$\omega_B = 4 \sqrt{ \frac{13 k q Q}{23 m L^3} } \tag{6}$$"
    _, ans_tag = prepare_reference_for_validation(ref_tag)
    assert r"\tag" not in ans_tag
    assert r"\frac{13 k q Q}{23 m L^3}" in ans_tag

    ref_therefore_prose = (
        "\\[\n\\tan\\varphi = \\frac{P_{2}-P_{1}\\cos(2\\beta)}{P_{1}\\sin(2\\beta)}.\n\\]\n"
        "This is the relationship that \\(\\varphi\\) satisfies when the system is at equilibrium."
    )
    _, ans_prose = prepare_reference_for_validation(ref_therefore_prose)
    assert r"\tan\varphi =" in ans_prose

    ref_inline = (
        "$$a = b$$\n"
        "The terminal velocity is: $v=-\\sqrt{3 g l \\sin\\theta_0}$"
    )
    _, ans_inline = prepare_reference_for_validation(ref_inline)
    assert ans_inline == r"v=-\sqrt{3 g l \sin\theta_0}"


def test_score_solver_output_handles_duplicated_answer_prefix_and_numeric() -> None:
    ref = "Result:\n$$v = 12.0\\ \\mathrm{m/s}$$"
    cand = "Step 1: Compute speed.\nANSWER: 12.0 m/s\nANSWER: ANSWER: 12.0 m/s"
    res = score_solver_output(cand, ref)
    assert res.has_answer_sentinel is True
    assert res.is_equivalent is True


def test_parse_verifier_response_sanitizes_relative_phrases() -> None:
    raw = (
        "**Corrective Directive:** In step 2 of your previous equation, apply conservation of angular momentum.\n"
        "[[CHALLENGE]]"
    )
    parsed = parse_verifier_response(raw, problem_statement="A rotating rod problem.")
    assert parsed.verdict == "CHALLENGE"
    assert parsed.had_sentinel is True
    assert "step 2" not in parsed.directive.lower()
    assert "angular momentum" in parsed.directive.lower()

    missing = parse_verifier_response("Looks somewhat okay without a sentinel.")
    assert missing.verdict == "UNKNOWN"
    assert missing.had_sentinel is False


def test_format_model_prompt_matches_raw_training_boundary() -> None:
    prompt = "Solve this physics problem."
    assert format_model_prompt(prompt, prompt_mode="raw") == "Solve this physics problem.\n\n"
    assert format_model_prompt(prompt, prompt_mode="alpaca").startswith("### Instruction:\n")


def test_resolve_adapter_source_ignores_weightless_git_dir(tmp_path: Path) -> None:
    weightless_dir = tmp_path / "models" / "adapters" / "solver_e4_adversarial_lora"
    (weightless_dir / "final_adapter").mkdir(parents=True)
    (weightless_dir / "final_adapter" / "adapter_config.json").write_text("{}", encoding="utf-8")

    with patch(
        "adversarial_physics.orchestration.loop.resolve_project_path",
        side_effect=lambda p: tmp_path / p if not Path(p).is_absolute() else Path(p),
    ):
        resolved = resolve_adapter_source(
            None,
            default_local_dir="models/adapters/solver_e4_adversarial_lora",
            default_hf_repo=DEFAULT_SOLVER_ADAPTER_REPO,
        )
    assert resolved == DEFAULT_SOLVER_ADAPTER_REPO

    weights_file = weightless_dir / "final_adapter" / "adapter_model.safetensors"
    weights_file.write_bytes(b"stub")
    with patch(
        "adversarial_physics.orchestration.loop.resolve_project_path",
        side_effect=lambda p: tmp_path / p if not Path(p).is_absolute() else Path(p),
    ):
        resolved_local = resolve_adapter_source(
            None,
            default_local_dir="models/adapters/solver_e4_adversarial_lora",
            default_hf_repo=DEFAULT_SOLVER_ADAPTER_REPO,
        )
    assert resolved_local == str(weightless_dir / "final_adapter")


def test_canonicalize_sft_validation_row_and_auto_pull_fallbacks(tmp_path: Path) -> None:
    sft_row = {
        "problem_id": "physgym_0001",
        "domain": "mechanics",
        "prompt": (
            "Solve the following physics problem step by step.\n\n"
            "PROBLEM:\nA particle slides without friction.\n\n"
        ),
        "target_output": "Energy gives\n$$v = 5\\ \\mathrm{m/s}$$",
    }
    prob = canonicalize_validation_record(sft_row)
    assert prob.problem_id == "physgym_0001"
    assert prob.domain == "MECHANICS"
    assert prob.problem_statement == "A particle slides without friction."
    assert "v = 5" in prob.reference_solution

    # 1. Local hit
    local_val = tmp_path / "data" / "splits" / "validation.jsonl"
    save_problems_jsonl(_sample_problems()[:2], local_val)
    res_local = ensure_validation_dataset(local_val)
    assert len(res_local.problems) == 2
    assert res_local.source_used.startswith("local:")

    # 2. Remote Hub unavailable -> PhysGym + split manifest fallback
    missing_val = tmp_path / "remote_missing" / "validation.jsonl"
    manifest_yaml = tmp_path / "split.yaml"
    manifest_yaml.write_text("splits:\n  val_ids:\n    - physgym_0025\n", encoding="utf-8")
    reconstructed_pool = [
        CanonicalProblem(
            problem_id="physgym_0010",
            domain="MECHANICS",
            problem_statement="Train problem",
            reference_solution="ANSWER: 1",
        ),
        CanonicalProblem(
            problem_id="physgym_0025",
            domain="ELECTROMAGNETISM",
            problem_statement="Val problem",
            reference_solution="ANSWER: 2",
        ),
    ]
    with (
        patch("adversarial_physics.orchestration.loop._download_validation_from_hf_hub", return_value=None),
        patch(
            "adversarial_physics.orchestration.loop.load_physgym_dataset",
            return_value=(reconstructed_pool, {"kind": "hub"}),
        ),
    ):
        res_fallback = ensure_validation_dataset(
            missing_val,
            split_manifest_path=manifest_yaml,
            ingestion_config_path=tmp_path / "nonexistent_ing.yaml",
        )
    assert [p.problem_id for p in res_fallback.problems] == ["physgym_0025"]
    assert res_fallback.source_used.startswith("physgym_split:")
    assert missing_val.is_file()


def test_run_adversarial_validation_loop_iter3_metrics_and_resume(tmp_path: Path) -> None:
    problems = _sample_problems()

    def fake_solver(prompt: str) -> tuple[str, int]:
        if "height h" in prompt:
            return "Step 1: Equate mgh = 0.5 m v^2.\nANSWER: \\sqrt{2 g h}", 20
        if "orbital period" in prompt:
            if "CRITICAL PHYSICAL CONSTRAINTS" not in prompt:
                return "Step 1: Wrong radius factor.\nANSWER: \\frac{\\pi m}{q B_0}", 25
            return "Step 1: Full cyclotron period.\nANSWER: \\frac{2\\pi m}{q B_0}", 30
        if "omega_B" in prompt:
            if "CRITICAL PHYSICAL CONSTRAINTS" not in prompt:
                return "Step 1: Initial guess.\nANSWER: 1.0", 15
            if prompt.count("\n- ") == 1:
                return "Step 1: Better but wrong coefficient.\nANSWER: 2.0", 20
            return "Step 1: Exact torque balance.\n" + r"ANSWER: 4 \sqrt{\frac{13 k q Q}{23 m L^3}}", 35
        if "refractive index" in prompt:
            if "CRITICAL PHYSICAL CONSTRAINTS" not in prompt:
                return "Step 1: Snell law.\nANSWER: 1.5", 12
            return "Step 1: Over-adjusted.\nANSWER: 1.8", 12
        raise AssertionError(f"Unexpected prompt: {prompt}")

    def fake_verifier(prompt: str) -> tuple[str, int]:
        if r"\sqrt{2 g h}" in prompt:
            return "Conservation of mechanical energy is applied properly.\n[[APPROVE]]", 10
        if r"\frac{\pi m}{q B_0}" in prompt:
            return "Use full 2 pi radians per revolution in cyclotron period.\n[[CHALLENGE]]", 12
        if r"\frac{2\pi m}{q B_0}" in prompt:
            return "Correct cyclotron period formula.\n[[APPROVE]]", 10
        if "ANSWER: 1.0" in prompt:
            return "Apply rotational inertia about the pivot.\n[[CHALLENGE]]", 12
        if "ANSWER: 2.0" in prompt:
            return "Include electrostatic torque factor 13/23 carefully.\n[[CHALLENGE]]", 12
        if r"\frac{13 k q Q}{23 m L^3}" in prompt:
            return "Derivation matches torque balance.\n[[APPROVE]]", 10
        if "ANSWER: 1.5" in prompt:
            # False challenge against a correct Turn-0 solution causing regression
            return "Re-evaluate Snell ratio.\n[[CHALLENGE]]", 10
        if "ANSWER: 1.8" in prompt:
            # False approval of a now-wrong revised solution
            return "Looks consistent.\n[[APPROVE]]", 10
        raise AssertionError(f"Unexpected verifier prompt: {prompt}")

    out_dir = tmp_path / "val_out"
    # Verify single problem 3-iteration progression first
    res_iter3 = run_single_problem_loop(
        problems[2],
        solver_generate=fake_solver,
        verifier_generate=fake_verifier,
        max_iterations=3,
    )
    assert res_iter3.iterations_used == 3
    assert res_iter3.stop_reason == "approved"
    assert res_iter3.turn0_correct is False
    assert res_iter3.final_correct is True
    assert res_iter3.converted is True
    assert len(res_iter3.traces[2].accumulated_directives) == 2

    # Run full batch on first 3 problems, then resume with all 4
    partial = run_adversarial_validation_loop(
        problems[:3],
        solver_generate=fake_solver,
        verifier_generate=fake_verifier,
        max_iterations=3,
        output_dir=out_dir,
    )
    assert partial["metrics"]["total_problems"] == 3

    calls = {"solver": 0}

    def counting_solver(prompt: str) -> tuple[str, int]:
        calls["solver"] += 1
        return fake_solver(prompt)

    full = run_adversarial_validation_loop(
        problems,
        solver_generate=counting_solver,
        verifier_generate=fake_verifier,
        max_iterations=3,
        output_dir=out_dir,
        resume=True,
    )
    # Only the 4th problem should execute during resume (2 solver calls: iter1 + iter2)
    assert calls["solver"] == 2
    m = full["metrics"]
    assert m["total_problems"] == 4
    assert m["solver"]["turn0_correct"] == 2
    assert m["solver"]["final_correct"] == 3
    assert m["solver"]["turn0_accuracy"] == pytest.approx(0.5)
    assert m["solver"]["final_accuracy"] == pytest.approx(0.75)
    assert m["solver"]["net_revision_gain"] == pytest.approx(0.25)
    assert m["solver"]["converted_count"] == 2
    assert m["solver"]["regressed_count"] == 1
    assert m["verifier"]["tp_true_challenge"] == 3
    assert m["verifier"]["fp_false_challenge"] == 1
    assert m["verifier"]["tn_true_approve"] == 3
    assert m["verifier"]["fn_false_approve"] == 1
    assert m["verifier"]["directive_self_containment_no_leak_rate"] is not None
    assert m["verifier"]["auditkit_directive_actionability_score"] is not None
    assert "per_tier" in m
    assert Path(full["artifacts"]["summary_json"]).is_file()
    assert len(Path(full["artifacts"]["results_per_problem_jsonl"]).read_text(encoding="utf-8").splitlines()) == 4

    # Also verify batched lockstep execution (`run_batch_problems_loop`) produces identical results
    from adversarial_physics.orchestration.loop import run_batch_problems_loop

    batch_results = run_batch_problems_loop(
        problems,
        solver_generate_batch=lambda prompts: [fake_solver(p) for p in prompts],
        verifier_generate_batch=lambda prompts: [fake_verifier(p) for p in prompts],
        max_iterations=3,
    )
    assert len(batch_results) == 4
    assert [r.final_correct for r in batch_results] == [True, True, True, False]
    assert [r.iterations_used for r in batch_results] == [1, 2, 3, 2]


def test_comparison_tables_bootstrap_ci_and_suite4() -> None:
    from adversarial_physics.evaluation.auditkit_bridge import run_verifier_suite4_audit
    from adversarial_physics.evaluation.comparison_tables import (
        bootstrap_ci95_and_p_value,
        compute_run_fingerprint,
    )

    stats = bootstrap_ci95_and_p_value(
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 1, 1, 0, 1, 0, 0],
        n_resamples=500,
        seed=42,
    )
    assert stats.n_samples == 10
    assert stats.candidate_mean == pytest.approx(0.6)
    assert stats.baseline_mean == pytest.approx(0.1)
    assert stats.delta_mean == pytest.approx(0.5)
    assert stats.wins_candidate == 5
    assert stats.wins_baseline == 0
    assert stats.mcnemar_p_value < 0.10

    fp1 = compute_run_fingerprint(
        {"total": 2, "correct": 1, "per_sample": [{"problem_id": "p1", "is_equivalent": True, "predicted_answer": "42"}]}
    )
    fp2 = compute_run_fingerprint(
        {"total": 2, "correct": 1, "per_sample": [{"problem_id": "p1", "is_equivalent": True, "predicted_answer": "42"}]}
    )
    assert len(fp1) == 64 and fp1 == fp2

    s4 = run_verifier_suite4_audit(
        [
            {
                "verifier_verdict": "CHALLENGE",
                "solver_is_correct": False,
                "verifier_directive": "Apply conservation of mechanical energy mgh = 1/2 mv^2 to solve for v.",
                "problem_statement": "A block slides down height h.",
                "reference_answer": "\\sqrt{2gh}",
            },
            {
                "verifier_verdict": "APPROVE",
                "solver_is_correct": True,
                "verifier_directive": "Valid derivation.",
            },
        ]
    )
    assert s4["verifier_sentinel_accuracy"] == pytest.approx(1.0)
    assert s4["verifier_challenge_f1"] == pytest.approx(1.0)
    assert s4["directive_self_containment_no_leak_rate"] == pytest.approx(1.0)
    assert s4["auditkit_directive_actionability_score"] > 0.5


def test_semantic_duplicate_hint_and_turn_regression_guard() -> None:
    """Verify semantic near-duplicate directive stopping and Turn Regression Guard fallback."""
    from adversarial_physics.orchestration.loop import _is_near_duplicate_directive

    d0 = (
        "To find the polar coordinate trajectory equation r(theta), apply conservation "
        "of mechanical energy and magnetic force F = q v x B."
    )
    d1_near = (
        "To determine the trajectory equation r(theta) in polar coordinates, apply mechanical "
        "energy conservation and the magnetic force law F = q v x B."
    )
    d1_diff = (
        "Use Gauss's law for electrostatics to integrate the enclosed line charge density lambda "
        "and compute the radial polarization force."
    )
    assert _is_near_duplicate_directive(d1_near, [d0]) is True
    assert _is_near_duplicate_directive(d1_diff, [d0]) is False

    # Turn Regression Guard: Turn 0 is correct & has ANSWER:, Turn 1 challenges with distinct directive,
    # Turn 1 loses ANSWER: sentinel -> regression guard falls back to Turn 0!
    prob = CanonicalProblem(
        problem_id="p_reg",
        domain="MECHANICS",
        problem_statement="A block slides down a frictionless ramp of height 5 m.",
        reference_solution="Step 1: v = sqrt(2gh).\nANSWER: 10",
    )
    s_calls = [
        "Step 1: v = sqrt(2gh).\nANSWER: 10",
        "Step 1: Attempting integral \\int dt/t without final sentinel",
    ]
    v_calls = [
        "Check gravitational potential energy conversion mgh = 1/2 m v^2 carefully.\n[[CHALLENGE]]",
        "Still missing final sentinel.\n[[CHALLENGE]]",
    ]
    res = run_single_problem_loop(
        prob,
        solver_generate=lambda _: s_calls.pop(0),
        verifier_generate=lambda _: v_calls.pop(0),
        max_iterations=2,
    )
    assert res.turn0_correct is True
    assert res.final_correct is True
    assert res.final_has_answer is True
    assert res.regressed is False


def test_comparison_tables_discovers_loop_and_e3_e4b_runs(tmp_path: Path) -> None:
    """Verify comparison_tables discovers flat loop summaries (E4b & E3+E4b), per_problem_correct, and renders tables."""
    import json

    from adversarial_physics.evaluation.comparison_tables import (
        build_all_version_tables,
        discover_version_runs,
    )

    # Create E0_v3 single-turn validation baseline
    e0_dir = tmp_path / "results/eval_e0_v3"
    e0_dir.mkdir(parents=True)
    e0_payload = {
        "total": 2,
        "correct": 0,
        "accuracy": 0.0,
        "sentinel_rate": 1.0,
        "domain_breakdown": {"MECHANICS": {"total": 2, "correct": 0, "accuracy": 0.0}},
        "per_sample": [
            {"problem_id": "p1", "domain": "MECHANICS", "source": "curated", "is_equivalent": False, "match_method": "mismatch", "has_answer_sentinel": True, "predicted_answer": "1", "reference_answer": "2"},
            {"problem_id": "p2", "domain": "MECHANICS", "source": "physgym", "is_equivalent": False, "match_method": "mismatch", "has_answer_sentinel": True, "predicted_answer": "3", "reference_answer": "4"},
        ],
        "lexsi_audit": {
            "physics_accuracy_metrics": {"tolerant_5pct_accuracy": 0.0, "aligntune_math_grade_accuracy": 0.0, "unit_consistency_rate": 1.0},
            "derivation_similarity_metrics": {"rouge1_f1": 0.2, "rougeL_f1": 0.15, "bleu_score": 0.01, "physics_equation_token_f1": 0.25},
            "reasoning_structure_metrics": {"step_structure_rate": 1.0, "mean_equations_per_response": 3.0},
            "alignment_audit": {"degenerate_rate": 0.0, "repetition_ratio": 0.1, "avg_response_tokens": 150.0, "meta_leakage_rate": 0.0},
            "stratified_breakdown": {
                "by_tier": {
                    "curated_in_domain": {"total": 1, "correct": 0, "accuracy": 0.0, "tolerant_5pct_accuracy": 0.0},
                    "physgym_olympiad": {"total": 1, "correct": 0, "accuracy": 0.0},
                }
            },
        },
    }
    (e0_dir / "E0_v3_validation_eval.json").write_text(json.dumps(e0_payload), encoding="utf-8")

    # Create E4a+E4b and E3+E4b flat loop summaries matching validate_e4_loop.py schema
    for subdir, turn0_corr, final_corr, f1, fal_rate, p1_ok, p2_ok in [
        ("results/eval_e4_loop_v3_val", 0, 1, 0.75, 0.25, True, False),
        ("results/eval_e3_e4b_loop_v3_val", 1, 2, 0.85, 0.0, True, True),
    ]:
        d = tmp_path / subdir
        d.mkdir(parents=True)
        summary = {
            "total_problems": 2,
            "max_iterations": 3,
            "solver": {
                "turn0_correct": turn0_corr,
                "turn0_accuracy": turn0_corr / 2.0,
                "final_correct": final_corr,
                "final_accuracy": final_corr / 2.0,
                "net_revision_gain": (final_corr - turn0_corr) / 2.0,
                "final_sentinel_compliance_rate": 1.0,
                "converted_count": final_corr - turn0_corr,
                "turn0_wrong_count": 2 - turn0_corr,
                "correction_conversion_rate": (final_corr - turn0_corr) / max(1, 2 - turn0_corr),
                "regressed_count": 0,
                "regression_rate": 0.0 if turn0_corr > 0 else None,
                "final_match_method_counts": {"sympy_symbolic": final_corr, "mismatch": 2 - final_corr},
            },
            "verifier": {
                "total_reviews": 3,
                "verifier_sentinel_accuracy": 0.85,
                "verifier_challenge_f1": f1,
                "verifier_false_approval_rate": fal_rate,
                "verifier_false_challenge_rate": 0.0,
                "verifier_approve_rate": 0.5,
                "directive_self_containment_no_leak_rate": 1.0,
                "auditkit_directive_actionability_score": 0.9,
            },
            "loop_efficiency": {"mean_iterations_used": 1.5, "mean_tokens_per_problem": 300.0},
            "per_domain": {"MECHANICS": {"count": 2, "turn0_correct": turn0_corr, "final_correct": final_corr, "final_accuracy": final_corr / 2.0}},
            "lexsi_audit_final": {
                "physics_accuracy_metrics": {"tolerant_5pct_accuracy": final_corr / 2.0, "aligntune_math_grade_accuracy": final_corr / 2.0, "unit_consistency_rate": 1.0},
                "derivation_similarity_metrics": {"rouge1_f1": 0.4, "rougeL_f1": 0.3, "bleu_score": 0.05, "physics_equation_token_f1": 0.45},
                "reasoning_structure_metrics": {"step_structure_rate": 1.0, "mean_equations_per_response": 6.0},
                "alignment_audit": {"degenerate_rate": 0.0, "repetition_ratio": 0.05, "avg_response_tokens": 220.0, "meta_leakage_rate": 0.0},
                "stratified_breakdown": {
                    "by_tier": {
                        "curated_in_domain": {"total": 1, "correct": 1 if p1_ok else 0, "accuracy": 1.0 if p1_ok else 0.0, "tolerant_5pct_accuracy": 1.0},
                        "physgym_olympiad": {"total": 1, "correct": 1 if p2_ok else 0, "accuracy": 1.0 if p2_ok else 0.0},
                    }
                },
            },
        }
        (d / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (d / "results_per_problem.jsonl").write_text(
            json.dumps({"problem_id": "p1", "final_correct": p1_ok}) + "\n"
            + json.dumps({"problem_id": "p2", "final_correct": p2_ok}) + "\n",
            encoding="utf-8",
        )

    runs = discover_version_runs(tmp_path)
    assert "E4b" in runs["v3"]["validation"]
    assert "E3+E4b" in runs["v3"]["validation"]
    assert runs["v3"]["validation"]["E3+E4b"].per_problem_correct == {"p1": 1, "p2": 1}
    assert runs["v3"]["validation"]["E4b"].curated_correct == 1

    built = build_all_version_tables(tmp_path)
    md_text = Path(built["markdown_path"]).read_text(encoding="utf-8")
    assert "**`E4b_v3`**" in md_text
    assert "**`E3+E4b_v3`**" in md_text
    assert "Adversarial Loop & Verifier Diagnostics" in md_text
    assert "E3+E4b_vs_E0" in built["payload"]["versions"]["v3"]["significance_validation"]
    assert Path(built["csv_path"]).is_file()



