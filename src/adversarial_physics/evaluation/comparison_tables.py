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

from adversarial_physics.data.curation.answer_validator import extract_answer_line
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


def _load_jsonl_list(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _backfill_reference_solutions(per_sample: list[dict[str, Any]], root_dir: Path | None = None) -> None:
    """Populate missing `reference_solution` strings on legacy (v1) eval rows from splits, E0_v2, and archive."""
    if not per_sample or all(r.get("reference_solution") for r in per_sample):
        return
    base = root_dir or Path(".")
    ref_lookup: dict[str, str] = {}
    candidate_split_files = [
        base / "data/splits/validation_v3.jsonl",
        base / "data/splits/test_v3.jsonl",
        base / "data/splits/validation_v2.jsonl",
        base / "data/splits/test_v2.jsonl",
        base / "data/splits/validation.jsonl",
        base / "data/splits/test.jsonl",
    ]
    for sp_p in candidate_split_files:
        for row_obj in _load_jsonl_list(sp_p):
            pid_k = str(row_obj.get("problem_id", ""))
            if pid_k and row_obj.get("reference_solution"):
                ref_lookup[pid_k] = str(row_obj["reference_solution"])

    # Fall back to E0_v2 raw JSONs and archived excluded leak items (preserves physgym_0204 ref after pruning)
    for e0_p in (
        base / "results/eval_e0_v2/E0_v2_validation_eval.json",
        base / "results/eval_e0_v2/E0_v2_test_eval.json",
    ):
        if e0_p.is_file():
            e0_data = json.loads(e0_p.read_text(encoding="utf-8"))
            for row_obj in e0_data.get("per_sample", []) or []:
                pid_k = str(row_obj.get("problem_id", ""))
                if pid_k and row_obj.get("reference_solution") and pid_k not in ref_lookup:
                    ref_lookup[pid_k] = str(row_obj["reference_solution"])

    excl_p = base / "results/audit_reports/excluded_leaked_eval_items_080.json"
    if excl_p.is_file():
        excl_data = json.loads(excl_p.read_text(encoding="utf-8"))
        for ver_key in ("v1", "v2", "v3"):
            ver_blk = excl_data.get(ver_key, {})
            for grp in ("excluded_validation", "excluded_test"):
                for row_obj in ver_blk.get(grp, []) or []:
                    pid_k = str(row_obj.get("problem_id", ""))
                    if pid_k and row_obj.get("reference_solution") and pid_k not in ref_lookup:
                        ref_lookup[pid_k] = str(row_obj["reference_solution"])

    for r in per_sample:
        pid_k = str(r.get("problem_id", ""))
        if pid_k in ref_lookup and not r.get("reference_solution"):
            r["reference_solution"] = ref_lookup[pid_k]


def load_clean_split_ids(root_dir: str | Path = ".") -> dict[str, dict[str, set[str]]]:
    """Load strictly leak-free (<0.80 cosine) problem_id sets for v1, v2, and v3 validation/test."""
    root = Path(root_dir)
    mapping = {
        "v1": ("data/splits/validation.jsonl", "data/splits/test.jsonl"),
        "v2": ("data/splits/validation_v2.jsonl", "data/splits/test_v2.jsonl"),
        "v3": ("data/splits/validation_v3.jsonl", "data/splits/test_v3.jsonl"),
    }
    out: dict[str, dict[str, set[str]]] = {}
    for ver, (val_rel, test_rel) in mapping.items():
        val_p = root / val_rel
        test_p = root / test_rel
        # Graceful fallback if v3 splits aren't written yet
        if ver == "v3" and not val_p.is_file():
            val_p = root / "data/splits/validation_v2.jsonl"
        if ver == "v3" and not test_p.is_file():
            test_p = root / "data/splits/test_v2.jsonl"
        out[ver] = {
            "validation": {str(r.get("problem_id", "")) for r in _load_jsonl_list(val_p) if r.get("problem_id")},
            "test": {str(r.get("problem_id", "")) for r in _load_jsonl_list(test_p) if r.get("problem_id")},
        }
    return out


def _normalize_from_raw_payload(
    raw: dict[str, Any],
    *,
    source_path: Path,
    version: str,
    experiment_id: str,
    split: str,
    root_dir: Path | None = None,
    force_recompute_audit: bool = False,
) -> NormalizedExperimentRun:
    """Normalize an in-memory single-turn eval or loop summary payload into NormalizedExperimentRun."""
    # Handle multi-turn loop summary.json format
    if ("metrics" in raw and "solver" in raw.get("metrics", {})) or (
        "solver" in raw and "total_problems" in raw
    ):
        metrics = raw["metrics"] if isinstance(raw.get("metrics"), dict) else raw
        solver = metrics["solver"]
        lexsi_final = (
            raw.get("lexsi_audit_final")
            or (raw.get("lexsi_audit") or {}).get("final")
            or {}
        )
        pa = lexsi_final.get("physics_accuracy_metrics", {})
        ds = lexsi_final.get("derivation_similarity_metrics", {})
        rs = lexsi_final.get("reasoning_structure_metrics", {})
        aa = lexsi_final.get("alignment_audit", {})
        tiers = (lexsi_final.get("stratified_breakdown") or {}).get("by_tier", {})
        cur = tiers.get("curated_in_domain", {}) or (metrics.get("per_tier") or {}).get("curated_in_domain", {})
        pg = tiers.get("physgym_olympiad", {}) or (metrics.get("per_tier") or {}).get("physgym_olympiad", {})

        per_problem_correct: dict[str, int] = dict(raw.get("_clean_per_problem_correct") or {})
        if not per_problem_correct:
            traces_path = source_path.parent / "results_per_problem.jsonl"
            for rec in _load_jsonl_list(traces_path):
                pid = str(rec.get("problem_id", ""))
                if pid:
                    per_problem_correct[pid] = 1 if rec.get("final_correct") else 0

        total = int(metrics.get("total_problems", 0))
        correct = int(solver.get("final_correct", 0))
        acc = float(solver.get("final_accuracy") or 0.0)
        binary_vec = (
            [per_problem_correct[pid] for pid in sorted(per_problem_correct)]
            if per_problem_correct
            else ([1] * correct + [0] * max(0, total - correct))
        )
        ci = bootstrap_ci95_and_p_value(binary_vec, binary_vec) if binary_vec else None
        fp_val = str(
            raw.get("_sha256_override")
            or hashlib.sha256(source_path.read_bytes()).hexdigest()
        )

        return NormalizedExperimentRun(
            version=version,
            experiment_id=experiment_id,
            split=split,
            source_path=str(source_path).replace("\\", "/"),
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
            sha256_fingerprint=fp_val,
            domain_breakdown=metrics.get("per_domain", {}),
            match_method_breakdown=solver.get("final_match_method_counts", {}),
            per_problem_correct=per_problem_correct,
            loop_diagnostics={
                "solver": solver,
                "verifier": metrics.get("verifier", {}),
                "loop_efficiency": metrics.get("loop_efficiency", {}),
            },
        )

    # Single-turn evaluation JSON: backfill references & AuditKIT metrics if needed or forced
    per_sample_raw = raw.get("per_sample", []) or []
    audit = None if force_recompute_audit else raw.get("lexsi_audit")
    if not audit or "physics_accuracy_metrics" not in audit:
        _backfill_reference_solutions(per_sample_raw, root_dir=root_dir)
        audit = run_physics_eval_audit(raw, experiment_name=f"{experiment_id}_{version}_{split}")

    pa = audit.get("physics_accuracy_metrics", {})
    ds = audit.get("derivation_similarity_metrics", {})
    rs = audit.get("reasoning_structure_metrics", {})
    aa = audit.get("alignment_audit", {})
    tiers = (audit.get("stratified_breakdown") or {}).get("by_tier", {})
    cur = tiers.get("curated_in_domain", {})
    pg = tiers.get("physgym_olympiad", {})

    per_prob = {
        str(r.get("problem_id", f"idx_{i}")): (1 if r.get("is_equivalent") else 0)
        for i, r in enumerate(per_sample_raw)
    }
    vec = [1 if r.get("is_equivalent") else 0 for r in per_sample_raw]
    ci = bootstrap_ci95_and_p_value(vec, vec) if vec else None

    total_n = len(per_sample_raw) if force_recompute_audit else int(raw.get("total", len(per_sample_raw)))
    correct_n = sum(vec) if force_recompute_audit else int(raw.get("correct", sum(vec)))
    acc_val = (correct_n / total_n) if (force_recompute_audit and total_n) else float(
        raw.get("accuracy", pa.get("exact_or_equiv_accuracy", 0.0))
    )

    return NormalizedExperimentRun(
        version=version,
        experiment_id=experiment_id,
        split=split,
        source_path=str(source_path).replace("\\", "/"),
        total=total_n,
        correct=correct_n,
        accuracy=acc_val,
        tolerant_5pct_accuracy=float(pa.get("tolerant_5pct_accuracy", 0.0)),
        aligntune_math_grade_accuracy=float(pa.get("aligntune_math_grade_accuracy", 0.0)),
        sentinel_rate=float(
            pa.get("answer_sentinel_rate", 0.0)
            if force_recompute_audit
            else raw.get("sentinel_rate", pa.get("answer_sentinel_rate", 0.0))
        ),
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
        sha256_fingerprint=str(raw.get("_sha256_override") or compute_run_fingerprint(raw)),
        domain_breakdown=(audit.get("stratified_breakdown") or {}).get("by_domain", raw.get("domain_breakdown", {})),
        match_method_breakdown=audit.get("match_method_breakdown", {}),
        per_problem_correct=per_prob,
    )


def load_and_normalize_eval_json(
    path: str | Path,
    *,
    version: str,
    experiment_id: str,
    split: str,
    root_dir: str | Path | None = None,
) -> NormalizedExperimentRun:
    """Load either a single-turn eval JSON or an E4b loop summary JSON and normalize all 22 metrics."""
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    resolved_root = Path(root_dir) if root_dir is not None else None
    return _normalize_from_raw_payload(
        raw,
        source_path=p,
        version=version,
        experiment_id=experiment_id,
        split=split,
        root_dir=resolved_root,
    )


def recalculate_run_on_clean_ids(
    path: str | Path,
    clean_ids: set[str],
    *,
    version: str,
    experiment_id: str,
    split: str,
    root_dir: str | Path = ".",
) -> NormalizedExperimentRun | None:
    """Recalculate all metrics strictly on the clean (<0.80 cosine) subset without mutating raw artifacts."""
    from adversarial_physics.orchestration.loop import ProblemLoopResult, TurnTrace, compute_loop_metrics

    p = Path(path)
    root = Path(root_dir)
    raw = json.loads(p.read_text(encoding="utf-8"))

    # Loop run: reconstruct ProblemLoopResult records for matching IDs and reuse compute_loop_metrics
    if ("metrics" in raw and "solver" in raw.get("metrics", {})) or (
        "solver" in raw and "total_problems" in raw
    ):
        traces_path = p.parent / "results_per_problem.jsonl"
        recs = [
            rec
            for rec in _load_jsonl_list(traces_path)
            if str(rec.get("problem_id", "")) in clean_ids
        ]
        if not recs:
            return None

        loop_objs: list[ProblemLoopResult] = []
        final_per_sample: list[dict[str, Any]] = []
        per_prob_corr: dict[str, int] = {}
        max_iters = int((raw.get("metrics") or raw).get("max_iterations") or 3)
        for rec in recs:
            traces = [
                TurnTrace(
                    iteration=int(t.get("iteration", 1)),
                    turn=int(t.get("turn", 0)),
                    accumulated_directives=list(t.get("accumulated_directives") or []),
                    solver_prompt=str(t.get("solver_prompt") or ""),
                    solver_output=str(t.get("solver_output") or ""),
                    solver_validation=dict(t.get("solver_validation") or {}),
                    solver_is_correct=bool(t.get("solver_is_correct", False)),
                    solver_tokens=int(t.get("solver_tokens", 0)),
                    solver_latency_s=float(t.get("solver_latency_s", 0.0)),
                    verifier_prompt=t.get("verifier_prompt"),
                    verifier_output=t.get("verifier_output"),
                    verifier_verdict=t.get("verifier_verdict"),
                    verifier_directive=t.get("verifier_directive"),
                    verifier_is_decision_correct=t.get("verifier_is_decision_correct"),
                    verifier_tokens=int(t.get("verifier_tokens", 0)),
                    verifier_latency_s=float(t.get("verifier_latency_s", 0.0)),
                )
                for t in rec.get("traces", [])
            ]
            pid = str(rec.get("problem_id", ""))
            loop_objs.append(
                ProblemLoopResult(
                    problem_id=pid,
                    domain=str(rec.get("domain", "UNKNOWN")),
                    problem_statement=str(rec.get("problem_statement", "")),
                    reference_solution=str(rec.get("reference_solution", "")),
                    normalized_reference_answer=str(rec.get("normalized_reference_answer", "")),
                    iterations_used=int(rec.get("iterations_used", len(traces))),
                    max_iterations=int(rec.get("max_iterations", max_iters)),
                    stop_reason=str(rec.get("stop_reason", "")),
                    turn0_correct=bool(rec.get("turn0_correct", False)),
                    final_correct=bool(rec.get("final_correct", False)),
                    turn0_has_answer=bool(rec.get("turn0_has_answer", False)),
                    final_has_answer=bool(rec.get("final_has_answer", False)),
                    turn0_match_method=str(rec.get("turn0_match_method", "unknown")),
                    final_match_method=str(rec.get("final_match_method", "unknown")),
                    converted=bool(rec.get("converted", False)),
                    regressed=bool(rec.get("regressed", False)),
                    total_latency_s=float(rec.get("total_latency_s", 0.0)),
                    total_tokens=int(rec.get("total_tokens", 0)),
                    traces=traces,
                )
            )
            per_prob_corr[pid] = 1 if rec.get("final_correct") else 0
            last_sol = traces[-1].solver_output if traces else ""
            has_s, pred_a = extract_answer_line(last_sol)
            final_per_sample.append(
                {
                    "problem_id": pid,
                    "domain": str(rec.get("domain", "UNKNOWN")),
                    "prediction": last_sol,
                    "reference_solution": str(rec.get("reference_solution", "")),
                    "reference_answer": str(rec.get("normalized_reference_answer", "")),
                    "predicted_answer": pred_a if has_s else "",
                    "is_equivalent": bool(rec.get("final_correct", False)),
                    "match_method": str(rec.get("final_match_method", "unknown")),
                    "has_answer_sentinel": bool(rec.get("final_has_answer", has_s)),
                }
            )

        clean_metrics = compute_loop_metrics(loop_objs, max_iterations=max_iters)
        clean_audit = run_physics_eval_audit(
            {"per_sample": final_per_sample},
            experiment_name=f"{experiment_id}_{version}_{split}_clean080",
        )
        clean_payload = {
            "metrics": clean_metrics,
            "lexsi_audit_final": clean_audit,
            "_clean_per_problem_correct": per_prob_corr,
            "_sha256_override": hashlib.sha256(
                json.dumps(clean_metrics, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest(),
        }
        return _normalize_from_raw_payload(
            clean_payload,
            source_path=p,
            version=version,
            experiment_id=experiment_id,
            split=split,
            root_dir=root,
        )

    # Single-turn eval JSON
    raw_samples = raw.get("per_sample", []) or []
    kept_samples = [dict(r) for r in raw_samples if str(r.get("problem_id", "")) in clean_ids]
    if not kept_samples:
        return None
    _backfill_reference_solutions(kept_samples, root_dir=root)
    clean_raw = {
        "experiment": raw.get("experiment", f"{experiment_id}_{version}_clean080"),
        "split": split,
        "per_sample": kept_samples,
    }
    clean_raw["_sha256_override"] = compute_run_fingerprint(clean_raw)
    return _normalize_from_raw_payload(
        clean_raw,
        source_path=p,
        version=version,
        experiment_id=experiment_id,
        split=split,
        root_dir=root,
        force_recompute_audit=True,
    )


def _get_version_candidate_map(root: Path) -> dict[str, dict[str, dict[str, list[Path]]]]:
    """Build ordered discovery candidate file paths for v1, v2, and v3."""
    candidate_map: dict[str, dict[str, dict[str, list[Path]]]] = {
        "v1": {
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
                    root / "results/eval_e4_loop_v1_val/summary.json",
                    root / "results/eval_v1/E4b_v1_validation_eval.json",
                    root / "outputs/validation/e4_loop_v1/summary.json",
                ],
                "test": [
                    root / "results/eval_e4_loop_v1_test/summary.json",
                    root / "results/eval_v1/E4b_v1_test_eval.json",
                ],
            },
            "E3+E4b": {
                "validation": [
                    root / "results/eval_e3_e4b_loop_v1_val/summary.json",
                    root / "results/eval_v1/E3_E4b_v1_validation_eval.json",
                ],
                "test": [
                    root / "results/eval_e3_e4b_loop_v1_test/summary.json",
                    root / "results/eval_v1/E3_E4b_v1_test_eval.json",
                ],
            },
        }
    }

    for ver_tag in ("v2", "v3"):
        e0_val_candidates = [root / f"results/eval_e0_{ver_tag}/E0_{ver_tag}_validation_eval.json"]
        e0_test_candidates = [root / f"results/eval_e0_{ver_tag}/E0_{ver_tag}_test_eval.json"]
        if ver_tag == "v3":
            # Validation/test v2 and v3 share identical untrained base E0 generations over overlapping IDs.
            e0_val_candidates.append(root / "results/eval_e0_v2/E0_v2_validation_eval.json")
            e0_test_candidates.append(root / "results/eval_e0_v2/E0_v2_test_eval.json")
        candidate_map[ver_tag] = {
            "E0": {"validation": e0_val_candidates, "test": e0_test_candidates},
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
                    root / f"results/eval_e4_loop_{ver_tag}_val/summary.json",
                    root / f"results/eval_e4_loop_{ver_tag}_validation/summary.json",
                    root / f"results/eval_e4b_{ver_tag}/E4b_{ver_tag}_validation_eval.json",
                    root / f"results/eval_e4b_{ver_tag}/summary.json",
                    root / f"outputs/validation/e4_loop_{ver_tag}/summary.json",
                    root / ("outputs/validation/e4_loop_iter3/summary.json" if ver_tag == "v2" else "nonexistent"),
                ],
                "test": [
                    root / f"results/eval_e4_loop_{ver_tag}_test/summary.json",
                    root / f"results/eval_e4b_{ver_tag}/E4b_{ver_tag}_test_eval.json",
                    root / f"results/eval_e4b_{ver_tag}/summary.json",
                    root / f"outputs/test/e4_loop_{ver_tag}/summary.json",
                ],
            },
            "E3+E4b": {
                "validation": [
                    root / f"results/eval_e3_e4b_loop_{ver_tag}_val/summary.json",
                    root / f"results/eval_e3_e4b_loop_{ver_tag}_validation/summary.json",
                    root / f"results/eval_e3_e4b_{ver_tag}/E3_E4b_{ver_tag}_validation_eval.json",
                    root / f"results/eval_e3_e4b_{ver_tag}/summary.json",
                    root / f"outputs/validation/e3_e4b_loop_{ver_tag}/summary.json",
                ],
                "test": [
                    root / f"results/eval_e3_e4b_loop_{ver_tag}_test/summary.json",
                    root / f"results/eval_e3_e4b_{ver_tag}/E3_E4b_{ver_tag}_test_eval.json",
                    root / f"results/eval_e3_e4b_{ver_tag}/summary.json",
                    root / f"outputs/test/e3_e4b_loop_{ver_tag}/summary.json",
                ],
            },
        }
    return candidate_map


def discover_version_runs(root_dir: str | Path = ".") -> dict[str, dict[str, dict[str, NormalizedExperimentRun]]]:
    """Discover all original (raw evaluated) runs for `v1`, `v2`, and `v3` across `validation` and `test`."""
    root = Path(root_dir)
    cand_map = _get_version_candidate_map(root)
    versions: dict[str, dict[str, dict[str, NormalizedExperimentRun]]] = {
        "v1": {"validation": {}, "test": {}},
        "v2": {"validation": {}, "test": {}},
        "v3": {"validation": {}, "test": {}},
    }
    for ver_tag, exp_dict in cand_map.items():
        for exp_id, split_dict in exp_dict.items():
            for split_name, paths in split_dict.items():
                for p in paths:
                    if p.is_file():
                        versions[ver_tag][split_name][exp_id] = load_and_normalize_eval_json(
                            p, version=ver_tag, experiment_id=exp_id, split=split_name, root_dir=root
                        )
                        break
    return versions


def discover_clean_version_runs(
    root_dir: str | Path = ".",
    clean_split_ids: dict[str, dict[str, set[str]]] | None = None,
) -> dict[str, dict[str, dict[str, NormalizedExperimentRun]]]:
    """Recalculate all `v1`, `v2`, `v3` evaluation runs strictly on clean (`cosine < 0.80`) problem IDs."""
    root = Path(root_dir)
    clean_ids_map = clean_split_ids if clean_split_ids is not None else load_clean_split_ids(root)
    cand_map = _get_version_candidate_map(root)
    versions: dict[str, dict[str, dict[str, NormalizedExperimentRun]]] = {
        "v1": {"validation": {}, "test": {}},
        "v2": {"validation": {}, "test": {}},
        "v3": {"validation": {}, "test": {}},
    }
    for ver_tag, exp_dict in cand_map.items():
        ver_clean = clean_ids_map.get(ver_tag, {})
        for exp_id, split_dict in exp_dict.items():
            for split_name, paths in split_dict.items():
                allowed_ids = ver_clean.get(split_name, set())
                # E4b_v2_test evaluated 37 PhysGym validation items saved under eval_e4_loop_v2_test/
                if ver_tag == "v2" and exp_id == "E4b" and split_name == "test":
                    allowed_ids = ver_clean.get("validation", set()) | ver_clean.get("test", set())
                if not allowed_ids:
                    continue
                for p in paths:
                    if p.is_file():
                        run = recalculate_run_on_clean_ids(
                            p,
                            allowed_ids,
                            version=ver_tag,
                            experiment_id=exp_id,
                            split=split_name,
                            root_dir=root,
                        )
                        if run is not None:
                            versions[ver_tag][split_name][exp_id] = run
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


def _fmt_acc_cell(run: NormalizedExperimentRun | None, fallback: str = "—") -> str:
    if run is None:
        return fallback
    return f"{_fmt_pct(run.accuracy)} ({run.correct}/{run.total})"


def _fmt_curated_cell(run: NormalizedExperimentRun | None, missing_text: str) -> str:
    if run is None:
        return missing_text
    if run.curated_total <= 0:
        return "N/A (0/0)"
    return f"**{_fmt_pct(run.curated_accuracy)}** ({run.curated_correct}/{run.curated_total})"


def _fmt_physgym_cell(run: NormalizedExperimentRun | None, missing_text: str = "—") -> str:
    if run is None:
        return missing_text
    if run.physgym_total <= 0:
        return "N/A (0/0)"
    return f"{_fmt_pct(run.physgym_accuracy)} ({run.physgym_correct}/{run.physgym_total})"


def _populate_version_payload(
    discovered_map: dict[str, dict[str, dict[str, NormalizedExperimentRun]]],
) -> dict[str, dict[str, Any]]:
    """Serialize discovered runs and pairwise bootstrap significance per version and split."""
    out: dict[str, dict[str, Any]] = {}
    for ver in ("v1", "v2", "v3"):
        ver_data = discovered_map[ver]
        out[ver] = {
            "validation": {k: v.to_dict() for k, v in ver_data["validation"].items()},
            "test": {k: v.to_dict() for k, v in ver_data["test"].items()},
            "significance_validation": compute_version_pairwise_significance(ver_data["validation"]),
            "significance_test": compute_version_pairwise_significance(ver_data["test"]),
        }
    return out


def _render_cross_version_table(
    discovered_map: dict[str, dict[str, dict[str, NormalizedExperimentRun]]],
    *,
    heading: str,
    subtitle: str,
    exp_order: tuple[str, ...],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Render a compact 9-column cross-version comparison table and JSON summary rows."""
    lines = [
        heading,
        "",
        subtitle,
        "",
        "| Exp | Split | `v1` Overall | `v2` Overall | `v2` Curated | `v2` PhysGym | `v3` Overall | `v3` Curated | `v3` PhysGym |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    prog_rows: list[dict[str, Any]] = []
    for split in ("validation", "test"):
        for exp_id in exp_order:
            r1 = discovered_map["v1"][split].get(exp_id)
            r2 = discovered_map["v2"][split].get(exp_id)
            r3 = discovered_map["v3"][split].get(exp_id)
            if not (r1 or r2 or r3):
                continue
            lines.append(
                f"| **`{exp_id}`** | `{split}` | {_fmt_acc_cell(r1)} | "
                f"{_fmt_acc_cell(r2)} | {_fmt_curated_cell(r2, '—')} | {_fmt_physgym_cell(r2)} | "
                f"{_fmt_acc_cell(r3)} | {_fmt_curated_cell(r3, '—')} | {_fmt_physgym_cell(r3)} |"
            )
            prog_rows.append(
                {
                    "experiment_id": exp_id,
                    "split": split,
                    "v1_accuracy": r1.accuracy if r1 else None,
                    "v1_total": r1.total if r1 else None,
                    "v2_accuracy": r2.accuracy if r2 else None,
                    "v2_total": r2.total if r2 else None,
                    "v2_curated_accuracy": r2.curated_accuracy if r2 else None,
                    "v2_physgym_accuracy": r2.physgym_accuracy if r2 else None,
                    "v3_accuracy": r3.accuracy if r3 else None,
                    "v3_total": r3.total if r3 else None,
                    "v3_curated_accuracy": r3.curated_accuracy if r3 else None,
                    "v3_physgym_accuracy": r3.physgym_accuracy if r3 else None,
                }
            )
    lines.append("")
    return lines, prog_rows


def _render_loop_summary_table(
    discovered_orig: dict[str, dict[str, dict[str, NormalizedExperimentRun]]],
    discovered_clean: dict[str, dict[str, dict[str, NormalizedExperimentRun]]],
    *,
    exp_order: tuple[str, ...],
) -> list[str]:
    """Render side-by-side Table 1 (Original) vs Table 2 (Clean < 0.80) adversarial loop comparison."""
    lines = [
        "### Multi-Turn Loop & Verifier Summary (Table 1 Original vs Table 2 Clean `< 0.80`)",
        "",
        "| View | Run | Split | Turn-0 | Final | Net Gain | Verifier Acc | Challenge F1 | False App / Chal |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    added = 0
    for view_label, src_map in (
        ("Table 1 (Orig)", discovered_orig),
        ("Table 2 (Clean)", discovered_clean),
    ):
        for ver in ("v2", "v3"):
            for split in ("validation", "test"):
                split_runs = src_map[ver][split]
                for exp_id in exp_order:
                    r = split_runs.get(exp_id)
                    if r is None or not r.loop_diagnostics:
                        continue
                    s_d = r.loop_diagnostics.get("solver") or {}
                    v_d = r.loop_diagnostics.get("verifier") or {}
                    gain = s_d.get("net_revision_gain")
                    gain_s = f"{float(gain) * 100:+.2f} pp" if gain is not None else "N/A"
                    fa = _fmt_pct(v_d.get("verifier_false_approval_rate"))
                    fc = _fmt_pct(v_d.get("verifier_false_challenge_rate"))
                    lines.append(
                        f"| {view_label} | **`{exp_id}_{ver}`** | `{split}` | "
                        f"{_fmt_pct(s_d.get('turn0_accuracy'))} ({s_d.get('turn0_correct', 0)}/{r.total}) | "
                        f"**{_fmt_pct(r.accuracy)}** ({r.correct}/{r.total}) | {gain_s} | "
                        f"{_fmt_pct(v_d.get('verifier_sentinel_accuracy'))} | "
                        f"{_fmt_pct(v_d.get('verifier_challenge_f1'))} | {fa} / {fc} |"
                    )
                    added += 1
    if not added:
        return []
    lines.append("")
    return lines


def _render_version_block(
    ver: str,
    ver_data: dict[str, dict[str, NormalizedExperimentRun]],
    ver_payload: dict[str, Any],
    *,
    exp_order: tuple[str, ...],
    view_label: str,
    csv_sink: list[dict[str, Any]],
) -> list[str]:
    """Render compact <=9-column subtables for a single version across validation and test."""
    lines: list[str] = [f"### Version `{ver.upper()}` — {view_label}", ""]
    if not (ver_data["validation"] or ver_data["test"]):
        lines.append(f"> *No `{ver.upper()}` runs available for `{view_label}`.*")
        lines.append("")
        return lines

    domain_keys = ["MECHANICS", "ELECTROMAGNETISM", "MODERN_PHYSICS", "OPTICS", "RELATIVITY", "THERMODYNAMICS"]
    domain_hdrs = ["MECH", "EM", "MOD", "OPT", "REL", "THERMO"]

    for split in ("validation", "test"):
        split_runs = ver_data[split]
        if not split_runs:
            continue
        sig_map = ver_payload[f"significance_{split}"]

        # Subtable 1: Primary Accuracy & Tiers (8 columns)
        lines.extend(
            [
                f"#### `{ver.upper()}` `{split.capitalize()}` — Primary Accuracy & Tier Stratification",
                "",
                "| Exp | Overall (n) | 95% CI | Curated | PhysGym | Tol 5% | Sentinel % | p vs E0 |",
                "|---|---|---|---|---|---|---|---|",
            ]
        )
        for exp_id in exp_order:
            r = split_runs.get(exp_id)
            if r is None:
                continue
            sig_e0 = sig_map.get(f"{exp_id}_vs_E0")
            p_str = f"{sig_e0['paired_bootstrap_p_value']:.4f}" if sig_e0 else "—"
            lines.append(
                f"| **`{exp_id}_{ver}`** | **{_fmt_pct(r.accuracy)}** ({r.correct}/{r.total}) | "
                f"[{_fmt_pct(r.ci95_low)}, {_fmt_pct(r.ci95_high)}] | "
                f"{_fmt_curated_cell(r, 'N/A (0/0)')} | {_fmt_physgym_cell(r, 'N/A')} | "
                f"{_fmt_pct(r.tolerant_5pct_accuracy)} | {_fmt_pct(r.sentinel_rate)} | {p_str} |"
            )
            csv_sink.append(r.to_dict())
        lines.append("")

        # Subtable 2: Derivation Similarity & Hygiene (9 columns)
        lines.extend(
            [
                f"#### `{ver.upper()}` `{split.capitalize()}` — Derivation Similarity & Reasoning Hygiene",
                "",
                "| Exp | Eq F1 | ROUGE-L | Step % | Eqs/Resp | Degenerate % | Rep Ratio | Avg Tokens | Meta Leak % |",
                "|---|---|---|---|---|---|---|---|---|",
            ]
        )
        for exp_id in exp_order:
            r = split_runs.get(exp_id)
            if r is None:
                continue
            lines.append(
                f"| **`{exp_id}_{ver}`** | {r.physics_equation_token_f1:.4f} | {r.rougeL_f1:.4f} | "
                f"{_fmt_pct(r.step_structure_rate)} | {r.mean_equations_per_response:.2f} | "
                f"{_fmt_pct(r.degenerate_rate)} | {r.repetition_ratio:.4f} | "
                f"{r.avg_response_tokens:.1f} | {_fmt_pct(r.meta_leakage_rate)} |"
            )
        lines.append("")

        # Subtable 3: Per-Domain Accuracy Breakdown (7 columns)
        lines.extend(
            [
                f"#### `{ver.upper()}` `{split.capitalize()}` — Per-Domain Accuracy",
                "",
                "| Exp | " + " | ".join(domain_hdrs) + " |",
                "|---|" + "|".join(["---"] * len(domain_hdrs)) + "|",
            ]
        )
        for exp_id in exp_order:
            r = split_runs.get(exp_id)
            if r is None:
                continue
            cells: list[str] = []
            for dom in domain_keys:
                d_info = r.domain_breakdown.get(dom, {})
                dtot = d_info.get("total", d_info.get("count", 0))
                dcor = d_info.get("correct", d_info.get("final_correct", 0))
                dacc = d_info.get("accuracy", d_info.get("final_accuracy", 0.0)) or 0.0
                cells.append(f"{_fmt_pct(dacc)} ({dcor}/{dtot})" if dtot else "—")
            lines.append(f"| **`{exp_id}_{ver}`** | " + " | ".join(cells) + " |")
        lines.append("")

        # Subtable 4: Adversarial Multi-Turn Loop & Verifier Diagnostics (9 columns)
        loop_exp_ids = [eid for eid in exp_order if eid in split_runs and split_runs[eid].loop_diagnostics]
        if loop_exp_ids:
            lines.extend(
                [
                    f"#### `{ver.upper()}` `{split.capitalize()}` — Adversarial Loop & Verifier Diagnostics",
                    "",
                    "| Exp | Turn-0 | Final | Net Gain | Conv / Reg | Mean Iters | Verifier Acc | Challenge F1 | False App / Chal |",
                    "|---|---|---|---|---|---|---|---|---|",
                ]
            )
            for exp_id in loop_exp_ids:
                r = split_runs[exp_id]
                diag = r.loop_diagnostics or {}
                s_diag = diag.get("solver") or {}
                v_diag = diag.get("verifier") or {}
                e_diag = diag.get("loop_efficiency") or {}
                gain = s_diag.get("net_revision_gain")
                gain_str = f"{float(gain) * 100:+.2f} pp" if gain is not None else "N/A"
                conv = int(s_diag.get("converted_count") or 0)
                reg = int(s_diag.get("regressed_count") or 0)
                mean_iters = e_diag.get("mean_iterations_used")
                iters_str = f"{float(mean_iters):.2f}" if mean_iters is not None else "—"
                fa = _fmt_pct(v_diag.get("verifier_false_approval_rate"))
                fc = _fmt_pct(v_diag.get("verifier_false_challenge_rate"))
                lines.append(
                    f"| **`{exp_id}_{ver}`** | {_fmt_pct(s_diag.get('turn0_accuracy'))} "
                    f"({s_diag.get('turn0_correct', 0)}/{r.total}) | "
                    f"**{_fmt_pct(r.accuracy)}** ({r.correct}/{r.total}) | "
                    f"{gain_str} | +{conv} / -{reg} | {iters_str} | "
                    f"{_fmt_pct(v_diag.get('verifier_sentinel_accuracy'))} | "
                    f"{_fmt_pct(v_diag.get('verifier_challenge_f1'))} | {fa} / {fc} |"
                )
            lines.append("")
    return lines


def _render_training_dynamics_table(training_rows: list[dict[str, Any]]) -> list[str]:
    """Render compact 9-column training dynamics & checkpoint selection summary."""
    if not training_rows:
        return []
    lines = [
        "## 4. Training Dynamics & Checkpoint Selection (`v1`, `v2`, `v3`)",
        "",
        "| Version | Adapter | LR | LoRA | Epochs (Run/Cfg) | Steps | Final Loss | Win Ckpt | Val Acc |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for tr in training_rows:
        val_acc_s = _fmt_pct(tr["selected_val_accuracy"]) if tr["selected_val_accuracy"] is not None else "—"
        lines.append(
            f"| `{tr['version']}` | `{tr['adapter_name']}` | {tr['learning_rate']} | "
            f"{tr['lora_r']}/{tr['lora_alpha']} | {tr['epochs_completed']}/{tr['max_epochs_configured']} | "
            f"{tr['total_steps']} | {tr['final_train_loss']:.4f} | "
            f"`{tr['selected_checkpoint']}` | {val_acc_s} |"
        )
    lines.append("")
    return lines


def _write_csv(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    scalar_keys = [
        k
        for k in rows[0].keys()
        if k not in ("domain_breakdown", "match_method_breakdown", "loop_diagnostics", "ci95")
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=scalar_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_all_version_tables(
    root_dir: str | Path = ".",
    output_dir: str | Path = "results/comparison_tables",
) -> dict[str, Any]:
    """Generate dual Table 1 (Original) & Table 2 (Clean < 0.80) Markdown, JSON, and CSV comparison reports."""
    root = Path(root_dir)
    out_dir = root / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    discovered_orig = discover_version_runs(root)
    discovered_clean = discover_clean_version_runs(root)
    training_rows = discover_training_summaries(root)
    exp_order = ("E0", "E1", "E2", "E3", "E4a", "E4b", "E3+E4b")

    full_payload: dict[str, Any] = {
        "versions": _populate_version_payload(discovered_orig),
        "versions_clean_080": _populate_version_payload(discovered_clean),
        "training_dynamics": training_rows,
        "cross_version_progression": [],
        "cross_version_progression_clean_080": [],
    }

    md_lines: list[str] = [
        "# Adversarial Physics Solver-Verifier — Experiment Comparison (`v1`, `v2`, `v3`)",
        "",
        "Two views are reported identically across `AuditKIT` + `AlignTune` + `SymPy` metrics:",
        "1. **Table 1 — Original Evaluated Splits:** Exact historical metrics measured on the raw evaluation splits (`v1`: 38, `v2`/`v3`: 60, `E4b_v2` test: 37 PhysGym).",
        "2. **Table 2 — Clean (`all-MiniLM-L6-v2` Cosine < 0.80) Recalculated Splits:** Deterministic subset recalculation from saved per-sample JSONs & traces after removing every problem with cosine similarity `>= 0.80` against any training split (`v1`: 38 val / 37 test, `v2`: 44 val / 49 test, `v3`: 48 val / 55 test).",
        "",
        "## 1. Executive Cross-Version Comparison (`v1 -> v2 -> v3`)",
        "",
    ]

    t1_lines, t1_prog = _render_cross_version_table(
        discovered_orig,
        heading="### Table 1: Original Evaluated Splits (Real Pre-Filter Splits)",
        subtitle="Raw historical splits prior to `< 0.80` MiniLM semantic deduplication.",
        exp_order=exp_order,
    )
    t2_lines, t2_prog = _render_cross_version_table(
        discovered_clean,
        heading="### Table 2: Leak-Free Splits (`all-MiniLM-L6-v2` Cosine `< 0.80` Recalculated)",
        subtitle="Strict zero-leakage subset (`cosine < 0.80` vs all training splits); `E4b_v2` evaluates 37 clean PhysGym validation items.",
        exp_order=exp_order,
    )
    full_payload["cross_version_progression"] = t1_prog
    full_payload["cross_version_progression_clean_080"] = t2_prog
    md_lines.extend(t1_lines)
    md_lines.extend(t2_lines)
    md_lines.extend(_render_loop_summary_table(discovered_orig, discovered_clean, exp_order=exp_order))

    csv_rows: list[dict[str, Any]] = []
    csv_clean_rows: list[dict[str, Any]] = []

    md_lines.extend(
        [
            "## 2. Detailed Per-Version Breakdown — Table 1 (Original Evaluated Splits)",
            "",
            "*Domain codes: `MECH` = Mechanics, `EM` = Electromagnetism, `MOD` = Modern Physics, `OPT` = Optics, `REL` = Relativity, `THERMO` = Thermodynamics.*",
            "",
        ]
    )
    for ver in ("v1", "v2", "v3"):
        md_lines.extend(
            _render_version_block(
                ver,
                discovered_orig[ver],
                full_payload["versions"][ver],
                exp_order=exp_order,
                view_label="Table 1 (Original Evaluated Splits)",
                csv_sink=csv_rows,
            )
        )

    md_lines.extend(
        [
            "## 3. Detailed Per-Version Breakdown — Table 2 (Clean `< 0.80` Recalculated Splits)",
            "",
            "Recalculated from saved `per_sample` predictions and `results_per_problem.jsonl` traces keeping only items with `all-MiniLM-L6-v2` cosine `< 0.80`.",
            "",
        ]
    )
    for ver in ("v1", "v2", "v3"):
        md_lines.extend(
            _render_version_block(
                ver,
                discovered_clean[ver],
                full_payload["versions_clean_080"][ver],
                exp_order=exp_order,
                view_label="Table 2 (Clean < 0.80)",
                csv_sink=csv_clean_rows,
            )
        )

    md_lines.extend(_render_training_dynamics_table(training_rows))

    md_path = out_dir / "experiment_comparison_v1_v2_v3.md"
    json_path = out_dir / "experiment_comparison_v1_v2_v3.json"
    csv_path = out_dir / "experiment_comparison_v1_v2_v3.csv"
    csv_clean_path = out_dir / "experiment_comparison_v1_v2_v3_clean_080.csv"

    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(full_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(csv_path, csv_rows)
    _write_csv(csv_clean_path, csv_clean_rows)

    return {
        "markdown_path": str(md_path),
        "json_path": str(json_path),
        "csv_path": str(csv_path),
        "csv_clean_path": str(csv_clean_path),
        "payload": full_payload,
    }
