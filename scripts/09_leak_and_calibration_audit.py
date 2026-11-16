"""Offline leak check (< 0.80 MiniLM cosine), grader/verifier Cohen's kappa calibration, and compute audit."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

from adversarial_physics.evaluation.auditkit_native import run_full_auditkit_suite
from adversarial_physics.evaluation.calibration_audit import (
    audit_grader_agreement_from_eval_jsons,
    audit_verifier_from_loop_summary,
    check_embedding_train_eval_leakage,
    check_train_eval_leakage,
    compute_confusion_kappa,
    summarize_training_and_inference_compute,
)
from adversarial_physics.evaluation.comparison_tables import discover_clean_version_runs

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("09_leak_and_calibration_audit")


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        s = line.strip()
        if s:
            rows.append(json.loads(s))
    return rows


def _pool_verifier_stats(runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Pool v3 validation + test verifier confusion counts into combined kappa/F1 metrics."""
    val_audit = next((x for x in runs if "v3" in x["label"] and "val" in x["label"].lower()), None)
    test_audit = next((x for x in runs if "v3" in x["label"] and "test" in x["label"].lower()), None)
    if not (val_audit and test_audit):
        return None
    va = val_audit["valid_sentinel_stats"]
    ta = test_audit["valid_sentinel_stats"]
    return {
        "total_reviews": val_audit["total_reviews"] + test_audit["total_reviews"],
        "unknown_no_sentinel": val_audit["unknown_no_sentinel"] + test_audit["unknown_no_sentinel"],
        "valid_sentinel_stats": compute_confusion_kappa(
            tp=va["tp"] + ta["tp"],
            fp=va["fp"] + ta["fp"],
            tn=va["tn"] + ta["tn"],
            fn=va["fn"] + ta["fn"],
        ),
    }


def _verifier_audit_from_clean_run(run: Any, label: str) -> dict[str, Any] | None:
    """Extract calibration statistics from a clean (< 0.80) NormalizedExperimentRun."""
    ver = (run.loop_diagnostics or {}).get("verifier")
    if not ver:
        return None
    tp = int(ver.get("tp_true_challenge", 0))
    fp = int(ver.get("fp_false_challenge", 0))
    tn = int(ver.get("tn_true_approve", 0))
    fn = int(ver.get("fn_false_approve", 0))
    unknown = int(ver.get("unknown_no_sentinel", 0))
    total_reviews = tp + fp + tn + fn + unknown
    return {
        "label": label,
        "source": str(run.source_path),
        "problems_evaluated": int(run.total),
        "total_reviews": total_reviews,
        "unknown_no_sentinel": unknown,
        "overall_sentinel_accuracy": round((tp + tn) / total_reviews, 4) if total_reviews else None,
        "valid_sentinel_stats": compute_confusion_kappa(tp=tp, fp=fp, tn=tn, fn=fn),
        "false_approval_rate": ver.get("verifier_false_approval_rate"),
        "false_challenge_rate": ver.get("verifier_false_challenge_rate"),
        "directive_actionability": ver.get("auditkit_directive_actionability_score"),
        "directive_no_leak_rate": ver.get("directive_self_containment_no_leak_rate"),
    }


def run_audit(
    root_dir: str | Path = ".",
    *,
    cosine_threshold: float = 0.80,
    embed_fn: Callable[[list[str]], Any] | None = None,
) -> dict[str, Any]:
    """Execute full v1/v2/v3 leak check (< 0.80 cosine), Cohen's kappa audit, and compute telemetry."""
    root = Path(root_dir)
    version_configs = [
        ("v1", sorted(root.glob("data/sft/*/train.jsonl")), [root / "data/splits/validation.jsonl", root / "data/splits/test.jsonl"]),
        ("v2", sorted(root.glob("data/sft_v2/*/train.jsonl")), [root / "data/splits/validation_v2.jsonl", root / "data/splits/test_v2.jsonl"]),
        ("v3", sorted(root.glob("data/sft_v3/*/train.jsonl")), [root / "data/splits/validation_v3.jsonl", root / "data/splits/test_v3.jsonl"]),
    ]

    leak_checks: dict[str, Any] = {}
    for ver_tag, train_paths, eval_paths in version_configs:
        leak_checks[f"{ver_tag}_post_filter_splits"] = check_train_eval_leakage(
            train_paths,
            eval_paths,
            shingle_n=5,
            jaccard_threshold=0.75,
            embedding_cosine_threshold=cosine_threshold,
            embed_fn=embed_fn,
        )

    # Reconstruct pre-filter semantic leakage baseline from archived excluded items if available
    excluded_path = root / "results/audit_reports/excluded_leaked_eval_items_080.json"
    pre_filter_baseline: dict[str, Any] = {}
    if excluded_path.is_file():
        excluded_data = json.loads(excluded_path.read_text(encoding="utf-8-sig"))
        for ver_tag, train_paths, eval_paths in version_configs:
            ver_arch = excluded_data.get(ver_tag, {})
            raw_rows: list[tuple[str, dict[str, Any]]] = []
            for ep in eval_paths:
                for row in _load_jsonl_rows(ep):
                    raw_rows.append((ep.name, row))
            for row in ver_arch.get("excluded_validation", []):
                raw_rows.append((eval_paths[0].name, row))
            for row in ver_arch.get("excluded_test", []):
                raw_rows.append((eval_paths[1].name, row))
            if train_paths and raw_rows:
                pre_audit = check_embedding_train_eval_leakage(
                    train_paths,
                    eval_paths,
                    cosine_threshold=cosine_threshold,
                    embed_fn=embed_fn,
                    eval_rows_override=raw_rows,
                )
                pre_filter_baseline[ver_tag] = {
                    "raw_counts": ver_arch.get("raw_counts", {}),
                    "clean_counts": ver_arch.get("clean_counts", {}),
                    "leaked_eval_count": pre_audit["leaked_eval_count"],
                    "leaked_by_split": pre_audit["leaked_by_split"],
                    "max_cosine_overall": pre_audit["max_cosine_overall"],
                    "max_clean_cosine": pre_audit["max_clean_cosine"],
                }

    # Historical original verifier audits
    verifier_audits: list[dict[str, Any]] = []
    for label, rel_path in [
        ("E4b_v3_prefix_leaked (38 physgym)", "results/ablations/eval_e4_loop_v3_prefix_leaked/summary.json"),
        ("E4b_v2_test (37 physgym)", "results/eval_e4_loop_v2_test/summary.json"),
        ("E4b_v3_validation (n=60)", "results/eval_e4_loop_v3_val/summary.json"),
        ("E4b_v3_test (n=60)", "results/eval_e4_loop_v3_test/summary.json"),
    ]:
        p = root / rel_path
        if p.is_file():
            verifier_audits.append(audit_verifier_from_loop_summary(p, label=label))
    pooled_retrained_v3 = _pool_verifier_stats(verifier_audits)

    # Clean (< 0.80) recalculated verifier calibration audits
    clean_runs = discover_clean_version_runs(root)
    clean_verifier_audits: list[dict[str, Any]] = []
    for ver_tag, split_name, lbl in [
        ("v2", "test", "E4b_v2_test_clean (37 physgym)"),
        ("v3", "validation", "E4b_v3_val_clean (n=48)"),
        ("v3", "test", "E4b_v3_test_clean (n=55)"),
    ]:
        run = clean_runs.get(ver_tag, {}).get(split_name, {}).get("E4b")
        if run is not None:
            va = _verifier_audit_from_clean_run(run, label=lbl)
            if va is not None:
                clean_verifier_audits.append(va)
    pooled_clean_v3 = _pool_verifier_stats(clean_verifier_audits)

    eval_json_files = sorted(
        list(root.glob("results/eval_e0_v2/E0_v2_*.json"))
        + list(root.glob("results/eval_e*_v3/*.json"))
    )
    grader_agreement = audit_grader_agreement_from_eval_jsons(eval_json_files)
    compute_summary = summarize_training_and_inference_compute(root)

    clean_ids_map: dict[tuple[str, str], set[str]] = {}
    for ver_tag, _, eval_paths in version_configs:
        val_rows = _load_jsonl_rows(eval_paths[0])
        test_rows = _load_jsonl_rows(eval_paths[1])
        if val_rows:
            clean_ids_map[(ver_tag, "validation")] = {str(r["problem_id"]) for r in val_rows if "problem_id" in r}
        if test_rows:
            clean_ids_map[(ver_tag, "test")] = {str(r["problem_id"]) for r in test_rows if "problem_id" in r}

    auditkit_native = run_full_auditkit_suite(root, clean_ids_by_version_split=clean_ids_map)

    out_dir = root / "results/audit_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    native_out_path = out_dir / "auditkit_native_summary.json"
    native_out_path.write_text(json.dumps(auditkit_native, indent=2, ensure_ascii=False), encoding="utf-8")

    report = {
        "cosine_threshold": cosine_threshold,
        "embedding_model": "all-MiniLM-L6-v2",
        "leak_checks": leak_checks,
        "pre_filter_080_baseline": pre_filter_baseline,
        "verifier_calibration": {
            "runs": verifier_audits,
            "pooled_retrained_v3_val_and_test": pooled_retrained_v3,
            "clean_080_runs": clean_verifier_audits,
            "clean_080_pooled_v3_val_and_test": pooled_clean_v3,
        },
        "physics_grader_agreement_sympy_vs_aligntune": grader_agreement,
        "compute_and_latency_telemetry": compute_summary,
        "auditkit_native_report_path": str(native_out_path.relative_to(root)).replace("\\", "/"),
        "auditkit_native_highlights": {
            "auditkit_version": auditkit_native.get("auditkit_version"),
            "v3_clean_080_validation_comparison": (
                auditkit_native.get("views", {})
                .get("clean_080", {})
                .get("v3", {})
                .get("validation", {})
                .get("comparison")
            ),
            "v3_clean_080_test_comparison": (
                auditkit_native.get("views", {})
                .get("clean_080", {})
                .get("v3", {})
                .get("test", {})
                .get("comparison")
            ),
            "loop_agent_eval_and_judge_calibration": auditkit_native.get(
                "loop_agent_eval_and_judge_calibration"
            ),
        },
    }

    out_path = out_dir / "leak_and_calibration_summary.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Audit report written to %s", out_path)
    return report


def main() -> None:
    report = run_audit(".")
    for key, lc in report["leak_checks"].items():
        emb = lc.get("embedding_leak_audit") or {}
        print(
            f"Leak check [{key}]: leak_free={lc['leak_free']} | eval_rows={lc['eval_rows_checked']} "
            f"| id_overlaps={lc['id_overlap_count']} | exact_text={lc['exact_normalized_text_overlap_count']} "
            f"| near_dups={lc['near_duplicate_pair_count']} | max_cos={emb.get('max_cosine_overall')}"
        )
    for ver_tag, pre in report.get("pre_filter_080_baseline", {}).items():
        print(
            f"Pre-filter [{ver_tag}]: leaked={pre['leaked_eval_count']} "
            f"max_cos={pre['max_cosine_overall']} -> clean max_cos={pre['max_clean_cosine']}"
        )
    for va in report["verifier_calibration"]["runs"]:
        vs = va["valid_sentinel_stats"]
        print(
            f"Verifier Orig [{va['label']}]: reviews={va['total_reviews']} acc={va['overall_sentinel_accuracy']} "
            f"F1={vs['f1']} kappa={vs['cohens_kappa']}"
        )
    for va in report["verifier_calibration"]["clean_080_runs"]:
        vs = va["valid_sentinel_stats"]
        print(
            f"Verifier Clean [{va['label']}]: reviews={va['total_reviews']} acc={va['overall_sentinel_accuracy']} "
            f"F1={vs['f1']} kappa={vs['cohens_kappa']}"
        )
    pooled_clean = report["verifier_calibration"]["clean_080_pooled_v3_val_and_test"]
    if pooled_clean:
        ps = pooled_clean["valid_sentinel_stats"]
        print(
            f"Verifier Clean POOLED v3 (val+test): reviews={pooled_clean['total_reviews']} "
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
