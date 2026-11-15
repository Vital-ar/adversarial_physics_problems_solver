"""Tests for leak checks, Cohen's kappa calibration, agent re-exports, and E0 v3 shared fallback."""
from __future__ import annotations

import json
from pathlib import Path

from adversarial_physics.agents import (
    DualLoRAEngine,
    parse_verifier_response,
    run_adversarial_validation_loop,
)
from adversarial_physics.evaluation.calibration_audit import (
    check_train_eval_leakage,
    compute_confusion_kappa,
)
from adversarial_physics.evaluation.comparison_tables import discover_version_runs


def test_agents_package_reexports_orchestration_symbols() -> None:
    assert DualLoRAEngine is not None
    assert callable(parse_verifier_response)
    assert callable(run_adversarial_validation_loop)


def test_leak_checker_detects_exact_and_disjoint_splits(tmp_path: Path) -> None:
    train_file = tmp_path / "train.jsonl"
    eval_clean = tmp_path / "eval_clean.jsonl"
    eval_leaky = tmp_path / "eval_leaky.jsonl"

    train_file.write_text(
        json.dumps(
            {
                "problem_id": "prob_1",
                "problem_statement": "Find the kinetic energy of a mass sliding down a frictionless ramp of height h.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    eval_clean.write_text(
        json.dumps(
            {
                "problem_id": "prob_2",
                "problem_statement": "Calculate the magnetic field inside an ideal solenoid carrying steady current I.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    eval_leaky.write_text(
        json.dumps(
            {
                "problem_id": "prob_99",
                "problem_statement": "Find the kinetic energy of a mass sliding down a frictionless ramp of height h.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    clean_res = check_train_eval_leakage([train_file], [eval_clean], jaccard_threshold=0.75)
    assert clean_res["leak_free"] is True
    assert clean_res["exact_normalized_text_overlap_count"] == 0

    leaky_res = check_train_eval_leakage([train_file], [eval_leaky], jaccard_threshold=0.75)
    assert leaky_res["leak_free"] is False
    assert leaky_res["exact_normalized_text_overlap_count"] == 1


def test_compute_confusion_kappa_balanced_and_chance() -> None:
    stats = compute_confusion_kappa(tp=93, fp=3, tn=1, fn=27)
    assert stats["n"] == 124
    assert stats["observed_agreement"] == round(94 / 124, 4)
    assert stats["cohens_kappa"] < 0.1  # heavily skewed base rate lowers kappa vs raw accuracy

    perfect = compute_confusion_kappa(tp=50, fp=0, tn=50, fn=0)
    assert perfect["cohens_kappa"] == 1.0
    assert perfect["f1"] == 1.0


def test_discover_version_runs_uses_e0_v2_fallback_for_v3_only_when_e0_v3_absent(tmp_path: Path) -> None:
    e0_v2_dir = tmp_path / "results/eval_e0_v2"
    e0_v2_dir.mkdir(parents=True)
    sample = {
        "total": 2,
        "correct": 1,
        "accuracy": 0.5,
        "sentinel_rate": 1.0,
        "per_sample": [
            {"problem_id": "p1", "is_equivalent": True, "domain": "MECHANICS"},
            {"problem_id": "p2", "is_equivalent": False, "domain": "MECHANICS"},
        ],
        "lexsi_audit": {
            "physics_accuracy_metrics": {
                "exact_or_equiv_accuracy": 0.5,
                "tolerant_5pct_accuracy": 0.5,
                "aligntune_math_grade_accuracy": 0.5,
                "answer_sentinel_rate": 1.0,
                "unit_consistency_rate": 1.0,
            }
        },
    }
    (e0_v2_dir / "E0_v2_test_eval.json").write_text(json.dumps(sample), encoding="utf-8")

    discovered = discover_version_runs(tmp_path)
    assert "E0" in discovered["v2"]["test"]
    assert "E0" in discovered["v3"]["test"]
    assert discovered["v3"]["test"]["E0"].accuracy == 0.5
