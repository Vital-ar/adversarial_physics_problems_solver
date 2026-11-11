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
    assert Path(full["artifacts"]["summary_json"]).is_file()
    assert len(Path(full["artifacts"]["results_per_problem_jsonl"]).read_text(encoding="utf-8").splitlines()) == 4
