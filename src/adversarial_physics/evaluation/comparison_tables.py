"""Multi-Version (v1, v2, v3) Experiment Comparison Tables & Statistical Significance Suite.

Implements Blueprint Section 8.2 requirements:
- Paired bootstrap 95% confidence intervals (`bootstrap_ci95_and_p_value`) and paired p-values
- Cryptographic SHA-256 `RunResult` provenance fingerprints (`compute_run_fingerprint`)
- On-the-fly 22-metric AuditKIT + AlignTune backfill for legacy v1 evaluation JSONs
- Unified comparison tables across Version 1 (`v1`), Version 2 (`v2`), and Version 3 (`v3`)
  for `[E0, E1, E2, E3, E4a, E4b]` on both Validation and Test splits.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from adversarial_physics.evaluation.auditkit_bridge import (
    run_physics_eval_audit,
    run_verifier_suite4_audit,
)


@dataclass
class PairedComparisonStats:
    """Paired bootstrap 95% confidence intervals and significance test between two models."""

    n_samples: int
    baseline_mean: float
    candidate_mean: float
    delta_mean: float
    baseline_ci95: tuple[float, float]
    candidate_ci95: tuple[float, float]
    delta_ci95: tuple[float, float]
    paired_bootstrap_p_value: float
    mcnemar_p_value: float
    wins_candidate: int
    wins_baseline: int
    ties: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "baseline_mean": round(self.baseline_mean, 4),
            "candidate_mean": round(self.candidate_mean, 4),
            "delta_mean": round(self.delta_mean, 4),
            "baseline_ci95": [round(self.baseline_ci95[0], 4), round(self.baseline_ci95[1], 4)],
            "candidate_ci95": [round(self.candidate_ci95[0], 4), round(self.candidate_ci95[1], 4)],
            "delta_ci95": [round(self.delta_ci95[0], 4), round(self.delta_ci95[1], 4)],
            "paired_bootstrap_p_value": round(self.paired_bootstrap_p_value, 4),
            "mcnemar_p_value": round(self.mcnemar_p_value, 4),
            "wins_candidate": self.wins_candidate,
            "wins_baseline": self.wins_baseline,
            "ties": self.ties,
        }


def _exact_binomial_two_sided_p(wins_a: int, wins_b: int) -> float:
    """Exact two-sided McNemar binomial p-value on discordant pairs."""
    n = wins_a + wins_b
    if n == 0:
        return 1.0
    k = min(wins_a, wins_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def bootstrap_ci95_and_p_value(
    baseline_scores: Sequence[float | int | bool],
    candidate_scores: Sequence[float | int | bool],
    *,
    n_resamples: int = 2000,
    seed: int = 42,
) -> PairedComparisonStats:
    """Compute paired bootstrap 95% CIs and two-sided paired p-values (Blueprint Section 8.2)."""
    if len(baseline_scores) != len(candidate_scores):
        raise ValueError(
            f"Paired score sequences must have equal length: {len(baseline_scores)} != {len(candidate_scores)}"
        )
    n = len(baseline_scores)
    if n == 0:
        return PairedComparisonStats(
            n_samples=0,
            baseline_mean=0.0,
            candidate_mean=0.0,
            delta_mean=0.0,
            baseline_ci95=(0.0, 0.0),
            candidate_ci95=(0.0, 0.0),
            delta_ci95=(0.0, 0.0),
            paired_bootstrap_p_value=1.0,
            mcnemar_p_value=1.0,
            wins_candidate=0,
            wins_baseline=0,
            ties=0,
        )

    base = [float(x) for x in baseline_scores]
    cand = [float(x) for x in candidate_scores]
    base_mean = sum(base) / n
    cand_mean = sum(cand) / n
    delta_mean = cand_mean - base_mean

    wins_cand = sum(1 for b, c in zip(base, cand) if c > b)
    wins_base = sum(1 for b, c in zip(base, cand) if b > c)
    ties = n - wins_cand - wins_base

    rng = random.Random(seed)
    boot_base: list[float] = []
    boot_cand: list[float] = []
    boot_delta: list[float] = []

    for _ in range(max(100, n_resamples)):
        idxs = [rng.randrange(n) for _ in range(n)]
        bm = sum(base[i] for i in idxs) / n
        cm = sum(cand[i] for i in idxs) / n
        boot_base.append(bm)
        boot_cand.append(cm)
        boot_delta.append(cm - bm)

    boot_base.sort()
    boot_cand.sort()
    boot_delta.sort()

    lo_idx = int(0.025 * len(boot_delta))
    hi_idx = min(len(boot_delta) - 1, int(0.975 * len(boot_delta)))

    # Two-sided bootstrap p-value for H0: delta == 0
    if delta_mean == 0.0 and wins_cand == 0 and wins_base == 0:
        boot_p = 1.0
    else:
        le_zero = sum(1 for d in boot_delta if d <= 0.0) / len(boot_delta)
        ge_zero = sum(1 for d in boot_delta if d >= 0.0) / len(boot_delta)
        boot_p = min(1.0, 2.0 * min(le_zero, ge_zero))

    mcnemar_p = _exact_binomial_two_sided_p(wins_cand, wins_base)

    return PairedComparisonStats(
        n_samples=n,
        baseline_mean=base_mean,
        candidate_mean=cand_mean,
        delta_mean=delta_mean,
        baseline_ci95=(boot_base[lo_idx], boot_base[hi_idx]),
        candidate_ci95=(boot_cand[lo_idx], boot_cand[hi_idx]),
        delta_ci95=(boot_delta[lo_idx], boot_delta[hi_idx]),
        paired_bootstrap_p_value=boot_p,
        mcnemar_p_value=mcnemar_p,
        wins_candidate=wins_cand,
        wins_baseline=wins_base,
        ties=ties,
    )


def compute_run_fingerprint(eval_payload: dict[str, Any]) -> str:
    """Compute a deterministic SHA-256 cryptographic fingerprint over a model run's predictions."""
    per_sample = eval_payload.get("per_sample", []) or []
    canonical_rows = [
        {
            "problem_id": str(r.get("problem_id", "")),
            "is_equivalent": bool(r.get("is_equivalent", False)),
            "match_method": str(r.get("match_method", "")),
            "predicted_answer": str(r.get("predicted_answer", "")),
        }
        for r in sorted(per_sample, key=lambda x: str(x.get("problem_id", "")))
    ]
    blob = json.dumps(
        {
            "total": eval_payload.get("total", len(canonical_rows)),
            "correct": eval_payload.get("correct", 0),
            "rows": canonical_rows,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass
class NormalizedExperimentRun:
    """Standardized representation of one experiment (`E0..E4b`) on one split (`validation` or `test`)."""

    version: str
    experiment_id: str
    split: str
    source_path: str
    total: int
    correct: int
    accuracy: float
    tolerant_5pct_accuracy: float
    aligntune_math_grade_accuracy: float
    sentinel_rate: float
    unit_consistency_rate: float
    curated_total: int
    curated_correct: int
    curated_accuracy: float
    curated_tol5_accuracy: float
    physgym_total: int
    physgym_correct: int
    physgym_accuracy: float
    rouge1_f1: float
    rougeL_f1: float
    bleu_score: float
    physics_equation_token_f1: float
    step_structure_rate: float
    mean_equations_per_response: float
    degenerate_rate: float
    repetition_ratio: float
    avg_response_tokens: float
    meta_leakage_rate: float
    ci95_low: float
    ci95_high: float
    sha256_fingerprint: str
    domain_breakdown: dict[str, dict[str, Any]] = field(default_factory=dict)
    match_method_breakdown: dict[str, int] = field(default_factory=dict)
    per_problem_correct: dict[str, int] = field(default_factory=dict)
    loop_diagnostics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "experiment_id": self.experiment_id,
            "split": self.split,
            "source_path": self.source_path,
            "total": self.total,
            "correct": self.correct,
            "accuracy": round(self.accuracy, 4),
            "ci95": [round(self.ci95_low, 4), round(self.ci95_high, 4)],
            "tolerant_5pct_accuracy": round(self.tolerant_5pct_accuracy, 4),
            "aligntune_math_grade_accuracy": round(self.aligntune_math_grade_accuracy, 4),
            "sentinel_rate": round(self.sentinel_rate, 4),
            "unit_consistency_rate": round(self.unit_consistency_rate, 4),
            "curated_total": self.curated_total,
            "curated_correct": self.curated_correct,
            "curated_accuracy": round(self.curated_accuracy, 4),
            "curated_tol5_accuracy": round(self.curated_tol5_accuracy, 4),
            "physgym_total": self.physgym_total,
            "physgym_correct": self.physgym_correct,
            "physgym_accuracy": round(self.physgym_accuracy, 4),
            "rouge1_f1": round(self.rouge1_f1, 4),
            "rougeL_f1": round(self.rougeL_f1, 4),
            "bleu_score": round(self.bleu_score, 4),
            "physics_equation_token_f1": round(self.physics_equation_token_f1, 4),
            "step_structure_rate": round(self.step_structure_rate, 4),
            "mean_equations_per_response": round(self.mean_equations_per_response, 4),
            "degenerate_rate": round(self.degenerate_rate, 4),
            "repetition_ratio": round(self.repetition_ratio, 4),
            "avg_response_tokens": round(self.avg_response_tokens, 2),
            "meta_leakage_rate": round(self.meta_leakage_rate, 4),
            "sha256_fingerprint": self.sha256_fingerprint,
            "domain_breakdown": self.domain_breakdown,
            "match_method_breakdown": self.match_method_breakdown,
            "loop_diagnostics": self.loop_diagnostics,
        }


def load_and_normalize_eval_json(
    path: str | Path,
    *,
    version: str,
    experiment_id: str,
    split: str,
) -> NormalizedExperimentRun:
    """Load either a single-turn eval JSON or an E4b loop summary JSON and normalize all 22 metrics."""
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))

    # Handle E4b multi-turn loop summary.json format if passed
    if "metrics" in raw and "solver" in raw.get("metrics", {}):
        metrics = raw["metrics"]
        solver = metrics["solver"]
        lexsi_final = (raw.get("lexsi_audit") or {}).get("final") or {}
        pa = lexsi_final.get("physics_accuracy_metrics", {})
        ds = lexsi_final.get("derivation_similarity_metrics", {})
        rs = lexsi_final.get("reasoning_structure_metrics", {})
        aa = lexsi_final.get("alignment_audit", {})
        tiers = (lexsi_final.get("stratified_breakdown") or {}).get("by_tier", {})
        cur = tiers.get("curated_in_domain", {}) or (metrics.get("per_tier") or {}).get("curated_in_domain", {})
        pg = tiers.get("physgym_olympiad", {}) or (metrics.get("per_tier") or {}).get("physgym_olympiad", {})

        total = int(metrics.get("total_problems", 0))
        correct = int(solver.get("final_correct", 0))
        acc = float(solver.get("final_accuracy") or 0.0)
        binary_vec = [1] * correct + [0] * max(0, total - correct)
        ci = bootstrap_ci95_and_p_value(binary_vec, binary_vec) if binary_vec else None

        return NormalizedExperimentRun(
            version=version,
            experiment_id=experiment_id,
            split=split,
            source_path=str(p).replace("\\", "/"),
            total=total,
            correct=correct,
            accuracy=acc,
            tolerant_5pct_accuracy=float(pa.get("tolerant_5pct_accuracy", acc)),
            aligntune_math_grade_accuracy=float(pa.get("aligntune_math_grade_accuracy", acc)),
            sentinel_rate=float(solver.get("final_sentinel_compliance_rate") or 0.0),
            unit_consistency_rate=float(pa.get("unit_consistency_rate", 0.0)),
            curated_total=int(cur.get("total", cur.get("count", 0))),
            curated_correct=int(cur.get("correct", cur.get("final_correct", 0))),
            curated_accuracy=float(cur.get("accuracy", cur.get("final_accuracy", 0.0)) or 0.0),
            curated_tol5_accuracy=float(cur.get("tolerant_5pct_accuracy", cur.get("final_accuracy", 0.0)) or 0.0),
            physgym_total=int(pg.get("total", pg.get("count", 0))),
            physgym_correct=int(pg.get("correct", pg.get("final_correct", 0))),
            physgym_accuracy=float(pg.get("accuracy", pg.get("final_accuracy", 0.0)) or 0.0),
            rouge1_f1=float(ds.get("rouge1_f1", 0.0)),
            rougeL_f1=float(ds.get("rougeL_f1", 0.0)),
            bleu_score=float(ds.get("bleu_score", 0.0)),
            physics_equation_token_f1=float(ds.get("physics_equation_token_f1", 0.0)),
            step_structure_rate=float(rs.get("step_structure_rate", 0.0)),
            mean_equations_per_response=float(rs.get("mean_equations_per_response", 0.0)),
            degenerate_rate=float(aa.get("degenerate_rate", 0.0)),
            repetition_ratio=float(aa.get("repetition_ratio", 0.0)),
            avg_response_tokens=float(
                aa.get("avg_response_tokens")
                or (metrics.get("loop_efficiency") or {}).get("mean_tokens_per_problem")
                or 0.0
            ),
            meta_leakage_rate=float(aa.get("meta_leakage_rate", 0.0)),
            ci95_low=ci.candidate_ci95[0] if ci else 0.0,
            ci95_high=ci.candidate_ci95[1] if ci else 0.0,
            sha256_fingerprint=hashlib.sha256(p.read_bytes()).hexdigest(),
            domain_breakdown=metrics.get("per_domain", {}),
            match_method_breakdown=solver.get("final_match_method_counts", {}),
            per_problem_correct={},
            loop_diagnostics={
                "solver": solver,
                "verifier": metrics.get("verifier", {}),
                "loop_efficiency": metrics.get("loop_efficiency", {}),
            },
        )

    # Single-turn evaluation JSON: backfill `lexsi_audit` on the fly if missing or legacy 3-metric (e.g., v1 files)
    audit = raw.get("lexsi_audit")
    if not audit or "physics_accuracy_metrics" not in audit:
        per_sample_raw = raw.get("per_sample", []) or []
        if per_sample_raw and not any(r.get("reference_solution") for r in per_sample_raw):
            ref_lookup: dict[str, str] = {}
            for sp_file in ("data/splits/validation_v2.jsonl", "data/splits/test_v2.jsonl", "data/splits/validation.jsonl", "data/splits/test.jsonl"):
                sp_p = Path(sp_file)
                if sp_p.is_file():
                    for line in sp_p.read_text(encoding="utf-8").splitlines():
                        if line.strip():
                            row_obj = json.loads(line)
                            pid_k = str(row_obj.get("problem_id", ""))
                            if pid_k and row_obj.get("reference_solution"):
                                ref_lookup[pid_k] = str(row_obj["reference_solution"])
            for r in per_sample_raw:
                pid_k = str(r.get("problem_id", ""))
                if pid_k in ref_lookup and not r.get("reference_solution"):
                    r["reference_solution"] = ref_lookup[pid_k]
        audit = run_physics_eval_audit(raw, experiment_name=f"{experiment_id}_{version}_{split}")

    pa = audit.get("physics_accuracy_metrics", {})
    ds = audit.get("derivation_similarity_metrics", {})
    rs = audit.get("reasoning_structure_metrics", {})
    aa = audit.get("alignment_audit", {})
    tiers = (audit.get("stratified_breakdown") or {}).get("by_tier", {})
    cur = tiers.get("curated_in_domain", {})
    pg = tiers.get("physgym_olympiad", {})

    per_sample = raw.get("per_sample", []) or []
    per_prob = {
        str(r.get("problem_id", f"idx_{i}")): (1 if r.get("is_equivalent") else 0)
        for i, r in enumerate(per_sample)
    }
    vec = [1 if r.get("is_equivalent") else 0 for r in per_sample]
    ci = bootstrap_ci95_and_p_value(vec, vec) if vec else None

    return NormalizedExperimentRun(
        version=version,
        experiment_id=experiment_id,
        split=split,
        source_path=str(p).replace("\\", "/"),
        total=int(raw.get("total", len(per_sample))),
        correct=int(raw.get("correct", sum(vec))),
        accuracy=float(raw.get("accuracy", pa.get("exact_or_equiv_accuracy", 0.0))),
        tolerant_5pct_accuracy=float(pa.get("tolerant_5pct_accuracy", 0.0)),
        aligntune_math_grade_accuracy=float(pa.get("aligntune_math_grade_accuracy", 0.0)),
        sentinel_rate=float(raw.get("sentinel_rate", pa.get("answer_sentinel_rate", 0.0))),
        unit_consistency_rate=float(pa.get("unit_consistency_rate", 0.0)),
        curated_total=int(cur.get("total", 0)),
        curated_correct=int(cur.get("correct", 0)),
        curated_accuracy=float(cur.get("accuracy", 0.0)),
        curated_tol5_accuracy=float(cur.get("tolerant_5pct_accuracy", 0.0)),
        physgym_total=int(pg.get("total", 0)),
        physgym_correct=int(pg.get("correct", 0)),
        physgym_accuracy=float(pg.get("accuracy", 0.0)),
        rouge1_f1=float(ds.get("rouge1_f1", 0.0)),
        rougeL_f1=float(ds.get("rougeL_f1", 0.0)),
        bleu_score=float(ds.get("bleu_score", 0.0)),
        physics_equation_token_f1=float(ds.get("physics_equation_token_f1", 0.0)),
        step_structure_rate=float(rs.get("step_structure_rate", 0.0)),
        mean_equations_per_response=float(rs.get("mean_equations_per_response", 0.0)),
        degenerate_rate=float(aa.get("degenerate_rate", 0.0)),
        repetition_ratio=float(aa.get("repetition_ratio", 0.0)),
        avg_response_tokens=float(aa.get("avg_response_tokens", 0.0)),
        meta_leakage_rate=float(aa.get("meta_leakage_rate", 0.0)),
        ci95_low=ci.candidate_ci95[0] if ci else 0.0,
        ci95_high=ci.candidate_ci95[1] if ci else 0.0,
        sha256_fingerprint=compute_run_fingerprint(raw),
        domain_breakdown=(audit.get("stratified_breakdown") or {}).get("by_domain", raw.get("domain_breakdown", {})),
        match_method_breakdown=audit.get("match_method_breakdown", {}),
        per_problem_correct=per_prob,
    )


def discover_version_runs(root_dir: str | Path = ".") -> dict[str, dict[str, dict[str, NormalizedExperimentRun]]]:
    """Discover all available evaluation runs for `v1`, `v2`, and `v3` across `validation` and `test`."""
    root = Path(root_dir)
    versions: dict[str, dict[str, dict[str, NormalizedExperimentRun]]] = {
        "v1": {"validation": {}, "test": {}},
        "v2": {"validation": {}, "test": {}},
        "v3": {"validation": {}, "test": {}},
    }

    # Version 1 discovery (`results/eval_v1/` + legacy 38-problem files)
    v1_candidates = {
        "E0": {
            "validation": [
                root / "results/eval_v1/E0_v1_validation_eval.json",
                root / "results/eval_e0_v2/E0_base_tiny_aya_v2_validation_eval.json",
            ],
            "test": [
                root / "results/eval_v1/E0_v1_test_eval.json",
                root / "results/eval_e0_v2/E0_base_tiny_aya_v2_test_eval.json",
            ],
        },
        "E1": {
            "validation": [
                root / "results/eval_v1/E1_v1_validation_eval.json",
                root / "results/eval_e1_v2/E1_solver_manual_det_v2_validation_eval.json",
            ],
            "test": [
                root / "results/eval_v1/E1_v1_test_eval.json",
                root / "results/eval_e1_v2/E1_solver_manual_det_v2_test_eval.json",
            ],
        },
        "E2": {
            "validation": [root / "results/eval_v1/E2_v1_validation_eval.json"],
            "test": [root / "results/eval_v1/E2_v1_test_eval.json"],
        },
        "E3": {
            "validation": [root / "results/eval_v1/E3_v1_validation_eval.json"],
            "test": [root / "results/eval_v1/E3_v1_test_eval.json"],
        },
        "E4a": {
            "validation": [root / "results/eval_v1/E4a_v1_validation_eval.json"],
            "test": [root / "results/eval_v1/E4a_v1_test_eval.json"],
        },
        "E4b": {
            "validation": [
                root / "results/eval_v1/E4b_v1_validation_eval.json",
                root / "outputs/validation/e4_loop_v1/summary.json",
            ],
            "test": [root / "results/eval_v1/E4b_v1_test_eval.json"],
        },
    }

    for ver_tag in ("v2", "v3"):
        ver_map = {
            "E0": {
                "validation": [root / f"results/eval_e0_{ver_tag}/E0_{ver_tag}_validation_eval.json"],
                "test": [root / f"results/eval_e0_{ver_tag}/E0_{ver_tag}_test_eval.json"],
            },
            "E1": {
                "validation": [root / f"results/eval_e1_{ver_tag}/E1_{ver_tag}_validation_eval.json"],
                "test": [root / f"results/eval_e1_{ver_tag}/E1_{ver_tag}_test_eval.json"],
            },
            "E2": {
                "validation": [root / f"results/eval_e2_{ver_tag}/E2_{ver_tag}_validation_eval.json"],
                "test": [root / f"results/eval_e2_{ver_tag}/E2_{ver_tag}_test_eval.json"],
            },
            "E3": {
                "validation": [root / f"results/eval_e3_{ver_tag}/E3_{ver_tag}_validation_eval.json"],
                "test": [root / f"results/eval_e3_{ver_tag}/E3_{ver_tag}_test_eval.json"],
            },
            "E4a": {
                "validation": [root / f"results/eval_e4a_{ver_tag}/E4a_{ver_tag}_validation_eval.json"],
                "test": [root / f"results/eval_e4a_{ver_tag}/E4a_{ver_tag}_test_eval.json"],
            },
            "E4b": {
                "validation": [
                    root / f"results/eval_e4b_{ver_tag}/E4b_{ver_tag}_validation_eval.json",
                    root / f"results/eval_e4b_{ver_tag}/summary.json",
                    root / f"outputs/validation/e4_loop_{ver_tag}/summary.json",
                    root / ("outputs/validation/e4_loop_iter3/summary.json" if ver_tag == "v2" else "nonexistent"),
                ],
                "test": [
                    root / f"results/eval_e4b_{ver_tag}/E4b_{ver_tag}_test_eval.json",
                    root / f"outputs/test/e4_loop_{ver_tag}/summary.json",
                ],
            },
        }
        cand_lookup = v1_candidates if ver_tag == "v1" else ver_map
        for exp_id, split_dict in cand_lookup.items():
            for split_name, paths in split_dict.items():
                for p in paths:
                    if p.is_file():
                        versions[ver_tag][split_name][exp_id] = load_and_normalize_eval_json(
                            p, version=ver_tag, experiment_id=exp_id, split=split_name
                        )
                        break

    for exp_id, split_dict in v1_candidates.items():
        for split_name, paths in split_dict.items():
            for p in paths:
                if p.is_file():
                    versions["v1"][split_name][exp_id] = load_and_normalize_eval_json(
                        p, version="v1", experiment_id=exp_id, split=split_name
                    )
                    break

    return versions


def discover_training_summaries(root_dir: str | Path = ".") -> list[dict[str, Any]]:
    """Collect training dynamics & checkpoint selection summaries across v1, v2, and v3 adapters."""
    root = Path(root_dir)
    rows: list[dict[str, Any]] = []
    adapters_dir = root / "models/adapters"
    if not adapters_dir.is_dir():
        return rows

    for sub in sorted(adapters_dir.iterdir()):
        if not sub.is_dir():
            continue
        summary_file = sub / "training_summary.json"
        if not summary_file.is_file():
            summary_file = sub / "final_adapter/training_summary.json"
        if not summary_file.is_file():
            continue
        try:
            d = json.loads(summary_file.read_text(encoding="utf-8"))
            name = d.get("adapter_name", sub.name)
            version = "v3" if "_v3_" in name else ("v2" if "_v2_" in name else "v1")
            tm = d.get("train_metrics") or {}
            inner_m = tm.get("metrics") or {}
            ckpt = d.get("checkpoint_selection") or {}
            rows.append(
                {
                    "version": version,
                    "adapter_name": name,
                    "dataset_key": d.get("dataset_key", ""),
                    "learning_rate": d.get("learning_rate"),
                    "lora_r": d.get("lora_r"),
                    "lora_alpha": d.get("lora_alpha"),
                    "max_epochs_configured": d.get("max_epochs"),
                    "epochs_completed": inner_m.get("epoch", tm.get("epochs")),
                    "total_steps": tm.get("steps"),
                    "final_train_loss": round(float(tm.get("final_loss") or 0.0), 4),
                    "selected_checkpoint": Path(str(ckpt.get("checkpoint_dir", "final"))).name,
                    "selected_eval_loss": (
                        round(float(ckpt["eval_loss"]), 4) if ckpt.get("eval_loss") is not None else None
                    ),
                    "selected_val_accuracy": (
                        round(float(ckpt["val_accuracy"]), 4) if ckpt.get("val_accuracy") is not None else None
                    ),
                    "selected_composite_score": (
                        round(float(ckpt["composite_score"]), 4) if ckpt.get("composite_score") is not None else None
                    ),
                    "train_runtime_s": round(float(tm.get("training_time") or 0.0), 1),
                }
            )
        except Exception:
            continue
    return rows


def compute_version_pairwise_significance(
    runs_for_split: dict[str, NormalizedExperimentRun],
) -> dict[str, dict[str, Any]]:
    """Compute paired bootstrap 95% CIs and p-values vs E0 (Zero-Shot) and E1 (Manual Det SFT)."""
    out: dict[str, dict[str, Any]] = {}
    for ref_key in ("E0", "E1"):
        if ref_key not in runs_for_split:
            continue
        ref_run = runs_for_split[ref_key]
        if not ref_run.per_problem_correct:
            continue
        for cand_key, cand_run in runs_for_split.items():
            if cand_key == ref_key or not cand_run.per_problem_correct:
                continue
            shared_ids = sorted(set(ref_run.per_problem_correct) & set(cand_run.per_problem_correct))
            if not shared_ids:
                continue
            b_vec = [ref_run.per_problem_correct[pid] for pid in shared_ids]
            c_vec = [cand_run.per_problem_correct[pid] for pid in shared_ids]
            stats = bootstrap_ci95_and_p_value(b_vec, c_vec)
            out[f"{cand_key}_vs_{ref_key}"] = stats.to_dict()
    return out


def _fmt_pct(val: float | None) -> str:
    return f"{val * 100:.2f}%" if val is not None else "N/A"


def build_all_version_tables(
    root_dir: str | Path = ".",
    output_dir: str | Path = "results/comparison_tables",
) -> dict[str, Any]:
    """Generate Markdown, JSON, and CSV comparison tables across Version 1 (`v1`), Version 2 (`v2`), and Version 3 (`v3`)."""
    root = Path(root_dir)
    out_dir = root / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    discovered = discover_version_runs(root)
    training_rows = discover_training_summaries(root)

    full_payload: dict[str, Any] = {
        "versions": {},
        "training_dynamics": training_rows,
        "cross_version_progression": [],
    }

    md_lines: list[str] = [
        "# Adversarial Physics Solver-Verifier — Multi-Version Experiment Tables (`v1`, `v2`, `v3`)",
        "",
        "Generated automatically with full 22-metric `AuditKIT` + `AlignTune` + `SymPy` evaluation,",
        "stratified by tier (`Curated In-Domain [22]` vs `PhysGym Olympiad [38]`), paired bootstrap 95% CIs,",
        "and SHA-256 `RunResult` provenance fingerprints.",
        "",
    ]

    csv_rows: list[dict[str, Any]] = []

    for ver in ("v1", "v2", "v3"):
        ver_data = discovered[ver]
        has_any = bool(ver_data["validation"] or ver_data["test"])
        full_payload["versions"][ver] = {
            "validation": {k: v.to_dict() for k, v in ver_data["validation"].items()},
            "test": {k: v.to_dict() for k, v in ver_data["test"].items()},
            "significance_validation": compute_version_pairwise_significance(ver_data["validation"]),
            "significance_test": compute_version_pairwise_significance(ver_data["test"]),
        }

        md_lines.append(f"## Version `{ver.upper()}` Experiment Tables")
        md_lines.append("")
        if not has_any:
            md_lines.append(
                f"> *No evaluation runs recorded yet for `{ver.upper()}`. Run `scripts/07_build_experiment_tables.py` "
                f"after placing `{ver}` evaluation JSONs in `results/eval_*_{ver}/`.*"
            )
            md_lines.append("")
            continue

        for split in ("validation", "test"):
            split_runs = ver_data[split]
            if not split_runs:
                continue
            sig_map = full_payload["versions"][ver][f"significance_{split}"]

            md_lines.append(f"### `{ver.upper()}` — `{split.capitalize()}` Split: Primary Physics & Tier Accuracy")
            md_lines.append("")
            md_lines.append(
                "| Exp | Overall Acc (n) | 95% Bootstrap CI | Curated In-Domain (n=22) | PhysGym Olympiad (n=38) | "
                "Tolerant 5% | AlignTune Math | Sentinel % | Unit % | p-val vs E0 |"
            )
            md_lines.append("|---|---|---|---|---|---|---|---|---|---|")

            for exp_id in ("E0", "E1", "E2", "E3", "E4a", "E4b"):
                if exp_id not in split_runs:
                    continue
                r = split_runs[exp_id]
                sig_e0 = sig_map.get(f"{exp_id}_vs_E0")
                p_str = f"{sig_e0['paired_bootstrap_p_value']:.4f}" if sig_e0 else "—"
                cur_str = (
                    f"**{_fmt_pct(r.curated_accuracy)}** ({r.curated_correct}/{r.curated_total})"
                    if r.curated_total > 0
                    else "N/A (0/0)"
                )
                pg_str = (
                    f"{_fmt_pct(r.physgym_accuracy)} ({r.physgym_correct}/{r.physgym_total})"
                    if r.physgym_total > 0
                    else "N/A"
                )
                md_lines.append(
                    f"| **`{exp_id}_{ver}`** | **{_fmt_pct(r.accuracy)}** ({r.correct}/{r.total}) | "
                    f"[{_fmt_pct(r.ci95_low)}, {_fmt_pct(r.ci95_high)}] | {cur_str} | {pg_str} | "
                    f"{_fmt_pct(r.tolerant_5pct_accuracy)} | {_fmt_pct(r.aligntune_math_grade_accuracy)} | "
                    f"{_fmt_pct(r.sentinel_rate)} | {_fmt_pct(r.unit_consistency_rate)} | {p_str} |"
                )
                csv_rows.append(r.to_dict())

            md_lines.append("")
            md_lines.append(
                f"### `{ver.upper()}` — `{split.capitalize()}` Split: Derivation Similarity, Reasoning Structure & Hygiene"
            )
            md_lines.append("")
            md_lines.append(
                "| Exp | Physics Eq Token F1 | ROUGE-1 | ROUGE-L | BLEU | Step Struct % | "
                "Eqs / Resp | Degenerate % | Repetition | Avg Tokens | Meta Leak % | SHA-256 Prefix |"
            )
            md_lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
            for exp_id in ("E0", "E1", "E2", "E3", "E4a", "E4b"):
                if exp_id not in split_runs:
                    continue
                r = split_runs[exp_id]
                md_lines.append(
                    f"| **`{exp_id}_{ver}`** | {r.physics_equation_token_f1:.4f} | {r.rouge1_f1:.4f} | "
                    f"{r.rougeL_f1:.4f} | {r.bleu_score:.4f} | {_fmt_pct(r.step_structure_rate)} | "
                    f"{r.mean_equations_per_response:.2f} | {_fmt_pct(r.degenerate_rate)} | "
                    f"{r.repetition_ratio:.4f} | {r.avg_response_tokens:.1f} | "
                    f"{_fmt_pct(r.meta_leakage_rate)} | `{r.sha256_fingerprint[:12]}` |"
                )
            md_lines.append("")

            # Per-Domain table
            md_lines.append(f"### `{ver.upper()}` — `{split.capitalize()}` Split: Per-Domain Accuracy Breakdown")
            md_lines.append("")
            domains = ["MECHANICS", "ELECTROMAGNETISM", "MODERN_PHYSICS", "OPTICS", "RELATIVITY", "THERMODYNAMICS"]
            md_lines.append("| Exp | " + " | ".join(domains) + " |")
            md_lines.append("|---|" + "|".join(["---"] * len(domains)) + "|")
            for exp_id in ("E0", "E1", "E2", "E3", "E4a", "E4b"):
                if exp_id not in split_runs:
                    continue
                r = split_runs[exp_id]
                cells: list[str] = []
                for dom in domains:
                    d_info = r.domain_breakdown.get(dom, {})
                    dtot = d_info.get("total", d_info.get("count", 0))
                    dcor = d_info.get("correct", d_info.get("final_correct", 0))
                    dacc = d_info.get("accuracy", d_info.get("final_accuracy", 0.0)) or 0.0
                    cells.append(f"{_fmt_pct(dacc)} ({dcor}/{dtot})" if dtot else "—")
                md_lines.append(f"| **`{exp_id}_{ver}`** | " + " | ".join(cells) + " |")
            md_lines.append("")

    # Cross-Version Progression Table (v1 -> v2 -> v3)
    md_lines.append("## Cross-Version Progression (`v1 -> v2 -> v3`)")
    md_lines.append("")
    md_lines.append(
        "| Experiment | Split | `v1` Overall (n=38) | `v2` Overall (n=60) | `v2` Curated (n=22) | "
        "`v2` Sentinel % | `v2` Degenerate % | `v3` Overall | `v3` Curated |"
    )
    md_lines.append("|---|---|---|---|---|---|---|---|---|")
    for split in ("validation", "test"):
        for exp_id in ("E0", "E1", "E2", "E3", "E4a", "E4b"):
            r1 = discovered["v1"][split].get(exp_id)
            r2 = discovered["v2"][split].get(exp_id)
            r3 = discovered["v3"][split].get(exp_id)
            if not (r1 or r2 or r3):
                continue
            v1_str = f"{_fmt_pct(r1.accuracy)} ({r1.correct}/{r1.total})" if r1 else "—"
            v2_all = f"{_fmt_pct(r2.accuracy)} ({r2.correct}/{r2.total})" if r2 else "—"
            v2_cur = f"**{_fmt_pct(r2.curated_accuracy)}** ({r2.curated_correct}/{r2.curated_total})" if r2 else "—"
            v2_sent = _fmt_pct(r2.sentinel_rate) if r2 else "—"
            v2_deg = _fmt_pct(r2.degenerate_rate) if r2 else "—"
            v3_all = f"{_fmt_pct(r3.accuracy)} ({r3.correct}/{r3.total})" if r3 else "*Pending v3*"
            v3_cur = f"**{_fmt_pct(r3.curated_accuracy)}** ({r3.curated_correct}/{r3.curated_total})" if r3 else "*Pending v3*"
            md_lines.append(
                f"| **`{exp_id}`** | `{split}` | {v1_str} | {v2_all} | {v2_cur} | {v2_sent} | {v2_deg} | {v3_all} | {v3_cur} |"
            )
            full_payload["cross_version_progression"].append(
                {
                    "experiment_id": exp_id,
                    "split": split,
                    "v1_accuracy": r1.accuracy if r1 else None,
                    "v2_accuracy": r2.accuracy if r2 else None,
                    "v2_curated_accuracy": r2.curated_accuracy if r2 else None,
                    "v3_accuracy": r3.accuracy if r3 else None,
                    "v3_curated_accuracy": r3.curated_accuracy if r3 else None,
                }
            )
    md_lines.append("")

    # Training Dynamics Table
    if training_rows:
        md_lines.append("## Training Dynamics & Checkpoint Selection (`v1`, `v2`, `v3`)")
        md_lines.append("")
        md_lines.append(
            "| Version | Adapter | LR | LoRA (r/α) | Max Epochs | Epochs Run | Steps | "
            "Final Train Loss | Winning Checkpoint | Eval Loss | Val Acc | Composite Score |"
        )
        md_lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for tr in training_rows:
            md_lines.append(
                f"| `{tr['version']}` | `{tr['adapter_name']}` | {tr['learning_rate']} | "
                f"{tr['lora_r']}/{tr['lora_alpha']} | {tr['max_epochs_configured']} | "
                f"{tr['epochs_completed']} | {tr['total_steps']} | {tr['final_train_loss']:.4f} | "
                f"`{tr['selected_checkpoint']}` | "
                f"{tr['selected_eval_loss'] if tr['selected_eval_loss'] is not None else '—'} | "
                f"{_fmt_pct(tr['selected_val_accuracy']) if tr['selected_val_accuracy'] is not None else '—'} | "
                f"{tr['selected_composite_score'] if tr['selected_composite_score'] is not None else '—'} |"
            )
        md_lines.append("")

    md_path = out_dir / "experiment_comparison_v1_v2_v3.md"
    json_path = out_dir / "experiment_comparison_v1_v2_v3.json"
    csv_path = out_dir / "experiment_comparison_v1_v2_v3.csv"

    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(full_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if csv_rows:
        scalar_keys = [
            k
            for k in csv_rows[0].keys()
            if k not in ("domain_breakdown", "match_method_breakdown", "loop_diagnostics", "ci95")
        ]
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=scalar_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(csv_rows)

    return {
        "markdown_path": str(md_path),
        "json_path": str(json_path),
        "csv_path": str(csv_path),
        "payload": full_payload,
    }
