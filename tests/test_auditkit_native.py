import json
from pathlib import Path

from adversarial_physics.evaluation.auditkit_bridge import (
    is_auditkit_installed,
    run_physics_eval_audit,
)
from adversarial_physics.evaluation.auditkit_native import (
    compare_version_runs_with_auditkit,
    evaluate_loop_traces_with_auditkit,
    evaluate_predictions_with_auditkit,
)


def _sample_eval_payload(is_correct: bool, pred_text: str) -> dict:
    return {
        "per_sample": [
            {
                "problem_id": "p_mech_1",
                "domain": "MECHANICS",
                "source": "literature_curated",
                "is_equivalent": is_correct,
                "match_method": "exact_normalized" if is_correct else "mismatch",
                "has_answer_sentinel": "ANSWER:" in pred_text,
                "predicted_answer": "19.8 m/s" if is_correct else "10 m/s",
                "reference_answer": "19.8 m/s",
                "prediction": pred_text,
                "reference_solution": "Step 1: Use energy conservation.\nStep 2: Substitute v = sqrt(2gh).\nANSWER: 19.8 m/s",
            }
        ]
    }


def test_auditkit_native_evaluate_and_compare_runs() -> None:
    assert is_auditkit_installed() is True

    base_payload = _sample_eval_payload(
        False,
        "wrong wrong wrong wrong wrong wrong wrong wrong\nANSWER: 10 m/s",
    )
    cand_payload = _sample_eval_payload(
        True,
        "Step 1: Apply conservation of mechanical energy.\nStep 2: Solve v = sqrt(2*g*h) = 19.8 m/s.\nANSWER: 19.8 m/s",
    )

    base_res, base_summ = evaluate_predictions_with_auditkit(base_payload, experiment_name="base_e0")
    cand_res, cand_summ = evaluate_predictions_with_auditkit(cand_payload, experiment_name="cand_e3")

    assert base_res is not None and cand_res is not None
    assert len(base_summ["fingerprint"]) >= 8
    assert base_summ["headline"]["physics_exact_equiv"] == 0.0
    assert cand_summ["headline"]["physics_exact_equiv"] == 1.0
    assert base_summ["headline"]["physics_degenerate_rate"] == 1.0
    assert cand_summ["headline"]["physics_degenerate_rate"] == 0.0
    assert cand_summ["by_domain_accuracy"]["MECHANICS"]["accuracy"] == 1.0

    comp = compare_version_runs_with_auditkit({"E0": base_res, "E3": cand_res}, baseline_key="E0")
    assert comp["leaderboard"][0]["experiment"] == "E3"
    delta_block = comp["vs_baseline"]["E3_vs_E0"]
    assert delta_block["sample_flips_exact_equiv"]["newly_correct"] == 1
    assert delta_block["sample_flips_exact_equiv"]["newly_wrong"] == 0
    assert delta_block["metric_deltas"]["physics_exact_equiv"]["delta"] == 1.0
    assert delta_block["metric_deltas"]["physics_degenerate_rate"]["delta"] == -1.0

    bridge = run_physics_eval_audit(cand_payload, experiment_name="cand_bridge")
    assert bridge["auditkit_available"] is True
    assert bridge["auditkit_native"]["fingerprint"] == cand_summ["fingerprint"]


def test_auditkit_loop_agent_eval_and_judge_calibration(tmp_path: Path) -> None:
    trace_file = tmp_path / "results_per_problem.jsonl"
    rows = [
        {
            "problem_id": "p_ok",
            "domain": "RELATIVITY",
            "problem_statement": "Compute Lorentz factor for v = 0.6c.",
            "reference_solution": "ANSWER: 5/4",
            "normalized_reference_answer": "5/4",
            "stop_reason": "approved",
            "final_correct": True,
            "total_tokens": 350,
            "total_latency_s": 12.5,
            "traces": [
                {
                    "iteration": 1,
                    "solver_output": "Step 1: gamma = 1/sqrt(1-0.36) = 5/4.\nANSWER: 5/4",
                    "solver_is_correct": True,
                    "solver_validation": {"extracted_candidate_answer": "5/4"},
                    "verifier_verdict": "APPROVE",
                    "verifier_directive": "Check gamma algebra cleanly.",
                }
            ],
        },
        {
            "problem_id": "p_bad",
            "domain": "MECHANICS",
            "problem_statement": "Find period of pendulum.",
            "reference_solution": "ANSWER: 2*pi*sqrt(L/g)",
            "normalized_reference_answer": "2*pi*sqrt(L/g)",
            "stop_reason": "max_iterations_reached",
            "final_correct": False,
            "total_tokens": 600,
            "total_latency_s": 24.0,
            "traces": [
                {
                    "iteration": 1,
                    "solver_output": "Step 1: Wrong guess T = L/g.\nANSWER: L/g",
                    "solver_is_correct": False,
                    "solver_validation": {"extracted_candidate_answer": "L/g"},
                    "verifier_verdict": "CHALLENGE",
                    "verifier_directive": "Apply small-angle simple harmonic restoring torque omega = sqrt(g/L).",
                }
            ],
        },
    ]
    trace_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    out = evaluate_loop_traces_with_auditkit(trace_file, name="test_loop_eval")
    assert out["auditkit_available"] is True
    assert out["n_episodes"] == 2
    assert out["verified_success_count"] == 1
    assert out["loop_free_rate"] == 1.0
    cal = out["verifier_judge_calibration"]
    assert cal["cohens_kappa"] == 1.0
    assert cal["agreement"] == 1.0
    assert "RELATIVITY" in cal["by_dimension"]
