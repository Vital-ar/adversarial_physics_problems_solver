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


def test_check_embedding_train_eval_leakage_threshold_080(tmp_path: Path) -> None:
    import numpy as np
    from adversarial_physics.evaluation.calibration_audit import check_embedding_train_eval_leakage

    train_p = tmp_path / "train.jsonl"
    eval_p = tmp_path / "eval.jsonl"
    train_p.write_text(
        json.dumps({"problem_id": "t1", "problem_statement": "train block energy"}) + "\n",
        encoding="utf-8",
    )
    eval_p.write_text(
        json.dumps({"problem_id": "p_leak", "problem_statement": "different wording same concept"})
        + "\n"
        + json.dumps({"problem_id": "p_clean", "problem_statement": "completely orthogonal topic"})
        + "\n",
        encoding="utf-8",
    )

    def _mock_embed(texts: list[str]) -> np.ndarray:
        vecs = []
        for t in texts:
            if "train block" in t:
                vecs.append([1.0, 0.0, 0.0])
            elif "same concept" in t:
                vecs.append([0.85, float(np.sqrt(1.0 - 0.85**2)), 0.0])
            else:
                vecs.append([0.5, float(np.sqrt(1.0 - 0.5**2)), 0.0])
        return np.asarray(vecs, dtype=np.float32)

    emb_res = check_embedding_train_eval_leakage(
        [train_p],
        [eval_p],
        model_name="mock-unit-test",
        cosine_threshold=0.80,
        embed_fn=_mock_embed,
    )
    assert emb_res["leak_free"] is False
    assert emb_res["leaked_eval_count"] == 1
    assert emb_res["leaked_by_split"] == {"eval.jsonl": ["p_leak"]}
    assert emb_res["clean_by_split"] == {"eval.jsonl": ["p_clean"]}
    assert emb_res["max_cosine_overall"] == 0.85
    assert emb_res["max_clean_cosine"] == 0.5

    combined = check_train_eval_leakage(
        [train_p],
        [eval_p],
        shingle_n=5,
        jaccard_threshold=0.75,
        embedding_cosine_threshold=0.80,
        embedding_model_name="mock-unit-test",
        embed_fn=_mock_embed,
    )
    assert combined["exact_normalized_text_overlap_count"] == 0
    assert combined["near_duplicate_pair_count"] == 0
    assert combined["leak_free"] is False


def test_recalculate_clean_080_and_dual_tables(tmp_path: Path) -> None:
    from adversarial_physics.evaluation.comparison_tables import (
        build_all_version_tables,
        recalculate_run_on_clean_ids,
    )

    splits_dir = tmp_path / "data/splits"
    splits_dir.mkdir(parents=True)
    (splits_dir / "validation_v3.jsonl").write_text(
        json.dumps(
            {
                "problem_id": "p_clean",
                "domain": "MECHANICS",
                "problem_statement": "Clean problem",
                "reference_solution": "Step 1: Use F=ma.\nANSWER: m a",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    e3_dir = tmp_path / "results/eval_e3_v3"
    e3_dir.mkdir(parents=True)
    eval_path = e3_dir / "E3_v3_validation_eval.json"
    eval_path.write_text(
        json.dumps(
            {
                "experiment": "E3_v3",
                "split": "validation",
                "total": 2,
                "correct": 1,
                "accuracy": 0.5,
                "sentinel_rate": 1.0,
                "per_sample": [
                    {
                        "problem_id": "p_leak",
                        "domain": "MECHANICS",
                        "prediction": "ANSWER: 5",
                        "reference_answer": "5",
                        "is_equivalent": True,
                        "match_method": "exact_normalized",
                        "has_answer_sentinel": True,
                    },
                    {
                        "problem_id": "p_clean",
                        "domain": "MECHANICS",
                        "prediction": "ANSWER: 2 m",
                        "reference_answer": "m a",
                        "is_equivalent": False,
                        "match_method": "mismatch",
                        "has_answer_sentinel": True,
                    },
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
        ),
        encoding="utf-8",
    )

    clean_run = recalculate_run_on_clean_ids(
        eval_path,
        {"p_clean"},
        version="v3",
        experiment_id="E3",
        split="validation",
        root_dir=tmp_path,
    )
    assert clean_run is not None
    assert clean_run.total == 1
    assert clean_run.correct == 0
    assert clean_run.accuracy == 0.0

    built = build_all_version_tables(tmp_path)
    orig_v3 = built["payload"]["versions"]["v3"]["validation"]["E3"]
    clean_v3 = built["payload"]["versions_clean_080"]["v3"]["validation"]["E3"]
    assert (orig_v3["correct"], orig_v3["total"]) == (1, 2)
    assert (clean_v3["correct"], clean_v3["total"]) == (0, 1)

    md_text = Path(built["markdown_path"]).read_text(encoding="utf-8")
    assert "### Table 1:" in md_text
    assert "### Table 2:" in md_text
    assert Path(built["csv_path"]).is_file()
    assert Path(built["csv_clean_path"]).is_file()
