"""Offline leak check, automatic grader/verifier Cohen's kappa calibration, and compute audit."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from adversarial_physics.evaluation.calibration_audit import (
    audit_grader_agreement_from_eval_jsons,
    audit_verifier_from_loop_summary,
    check_train_eval_leakage,
    summarize_training_and_inference_compute,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("09_leak_and_calibration_audit")


def run_audit(root_dir: str | Path = ".") -> dict:
    """Execute full offline leak check, Cohen's kappa audit, and compute telemetry aggregation."""
    root = Path(root_dir)
    sft_v3_trains = sorted(root.glob("data/sft_v3/*/train.jsonl"))
    eval_splits_v2 = [
        root / "data/splits/validation_v2.jsonl",
        root / "data/splits/test_v2.jsonl",
    ]
    eval_splits_v1 = [
        root / "data/splits/validation.jsonl",
        root / "data/splits/test.jsonl",
    ]

    leak_v3_vs_v2 = check_train_eval_leakage(sft_v3_trains, eval_splits_v2, shingle_n=5, jaccard_threshold=0.75)
    leak_v3_vs_v1 = check_train_eval_leakage(sft_v3_trains, eval_splits_v1, shingle_n=5, jaccard_threshold=0.75)

    verifier_audits = []
    for label, rel_path in [
        ("E4b_v3_prefix_leaked (38 physgym)", "results/ablations/eval_e4_loop_v3_prefix_leaked/summary.json"),
        ("E4b_v2_test (37 physgym)", "results/eval_e4_loop_v2_test/summary.json"),
        ("E4b_v3_validation (n=60)", "results/eval_e4_loop_v3_val/summary.json"),
        ("E4b_v3_test (n=60)", "results/eval_e4_loop_v3_test/summary.json"),
    ]:
        p = root / rel_path
        if p.is_file():
            verifier_audits.append(audit_verifier_from_loop_summary(p, label=label))

    # Pooled retrained v3 verifier across all 266 val+test reviews
    val_audit = next((x for x in verifier_audits if "v3_validation" in x["label"]), None)
    test_audit = next((x for x in verifier_audits if "v3_test" in x["label"]), None)
    pooled_retrained_v3 = None
    if val_audit and test_audit:
        from adversarial_physics.evaluation.calibration_audit import compute_confusion_kappa

        va = val_audit["valid_sentinel_stats"]
        ta = test_audit["valid_sentinel_stats"]
        pooled_retrained_v3 = {
            "total_reviews": val_audit["total_reviews"] + test_audit["total_reviews"],
            "unknown_no_sentinel": val_audit["unknown_no_sentinel"] + test_audit["unknown_no_sentinel"],
            "valid_sentinel_stats": compute_confusion_kappa(
                tp=va["tp"] + ta["tp"],
                fp=va["fp"] + ta["fp"],
                tn=va["tn"] + ta["tn"],
                fn=va["fn"] + ta["fn"],
            ),
        }

    eval_json_files = sorted(
        list(root.glob("results/eval_e0_v2/E0_v2_*.json"))
        + list(root.glob("results/eval_e*_v3/*.json"))
    )
    grader_agreement = audit_grader_agreement_from_eval_jsons(eval_json_files)
    compute_summary = summarize_training_and_inference_compute(root)

    report = {
        "leak_checks": {
            "sft_v3_vs_validation_and_test_v2": leak_v3_vs_v2,
            "sft_v3_vs_validation_and_test_v1": leak_v3_vs_v1,
        },
        "verifier_calibration": {
            "runs": verifier_audits,
            "pooled_retrained_v3_val_and_test": pooled_retrained_v3,
        },
        "physics_grader_agreement_sympy_vs_aligntune": grader_agreement,
        "compute_and_latency_telemetry": compute_summary,
    }

    out_dir = root / "results/audit_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "leak_and_calibration_summary.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Audit report written to %s", out_path)
    return report


def main() -> None:
    report = run_audit(".")
    lc = report["leak_checks"]["sft_v3_vs_validation_and_test_v2"]
    print(
        f"Leak check (sft_v3 vs val/test v2): leak_free={lc['leak_free']} "
        f"| id_overlaps={lc['id_overlap_count']} | exact_text={lc['exact_normalized_text_overlap_count']} "
        f"| near_dups={lc['near_duplicate_pair_count']} | max_jaccard={lc['max_shingle_jaccard']}"
    )
    for va in report["verifier_calibration"]["runs"]:
        vs = va["valid_sentinel_stats"]
        print(
            f"Verifier [{va['label']}]: reviews={va['total_reviews']} acc={va['overall_sentinel_accuracy']} "
            f"F1={vs['f1']} kappa={vs['cohens_kappa']}"
        )
    pooled = report["verifier_calibration"]["pooled_retrained_v3_val_and_test"]
    if pooled:
        ps = pooled["valid_sentinel_stats"]
        print(
            f"Verifier POOLED v3 (val+test): reviews={pooled['total_reviews']} "
            f"valid={ps['n']} acc={ps['observed_agreement']} F1={ps['f1']} kappa={ps['cohens_kappa']}"
        )
    ga = report["physics_grader_agreement_sympy_vs_aligntune"]["agreement_stats"]
    print(
        f"SymPy vs AlignTune Math Grader (n={ga['n']}): agreement={ga['observed_agreement']} "
        f"kappa={ga['cohens_kappa']}"
    )
    comp = report["compute_and_latency_telemetry"]
    print(f"Total tracked training GPU time: {comp['total_training_seconds']}s ({comp['total_training_hours']}h)")


if __name__ == "__main__":
    main()
