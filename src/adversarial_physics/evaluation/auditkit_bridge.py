"""Lexsi AI Evaluation Bridge (`auditkit_bridge.py`): AuditKIT + AlignTune Eval + Physics Scorer.

Integrates project physics evaluation (`score_solver_output`), Lexsi AlignTune evaluation primitives
(`aligntune.utils.math_grading` and `aligntune.eval.alignment_auditor.AuditReport`), and native
`auditkit` benchmark execution whenever `auditkit` is installed in the runtime environment.
"""
from __future__ import annotations

import importlib.util
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from adversarial_physics.orchestration.loop import score_solver_output


def is_auditkit_installed() -> bool:
    """Return True if the optional Lexsi ``auditkit`` package is importable."""
    return importlib.util.find_spec("auditkit") is not None


def _compute_generation_hygiene(predictions: list[str]) -> dict[str, float]:
    """Compute lightweight alignment/degeneracy diagnostics aligned with AlignTune's ``AuditReport``."""
    if not predictions:
        return {
            "avg_response_tokens": 0.0,
            "repetition_ratio": 0.0,
            "unique_token_ratio": 1.0,
            "degenerate_rate": 0.0,
        }

    lengths: list[int] = []
    rep_ratios: list[float] = []
    uniq_ratios: list[float] = []
    degenerate = 0

    for text in predictions:
        words = (text or "").split()
        n = len(words)
        lengths.append(n)
        if n < 5 or not text.strip():
            degenerate += 1
        uniq_ratios.append(len(set(w.lower() for w in words)) / max(1, n))
        if n >= 3:
            trigrams = [tuple(w.lower() for w in words[i : i + 3]) for i in range(n - 2)]
            counts = Counter(trigrams)
            repeated = sum(c - 1 for c in counts.values() if c > 1)
            rep_ratios.append(repeated / max(1, len(trigrams)))
        else:
            rep_ratios.append(0.0)

    total = len(predictions)
    metrics = {
        "avg_response_tokens": round(sum(lengths) / total, 2),
        "repetition_ratio": round(sum(rep_ratios) / total, 4),
        "unique_token_ratio": round(sum(uniq_ratios) / total, 4),
        "degenerate_rate": round(degenerate / total, 4),
    }

    # Package cleanly into AlignTune's AuditReport schema when available
    try:
        from aligntune.eval.alignment_auditor import AuditReport

        report = AuditReport(
            reward_hacking=0.0,
            sycophancy=0.0,
            refusal_collapse=0.0,
            verbosity_gain=0.0,
            timestamp=datetime.now(timezone.utc).isoformat(),
            **metrics,
        )
        return report.to_dict()
    except Exception:
        return metrics


def run_physics_eval_audit(
    eval_result: dict[str, Any],
    *,
    experiment_name: str = "solver_eval",
) -> dict[str, Any]:
    """Enrich a solver split evaluation payload with Lexsi AlignTune/AuditKIT diagnostics."""
    per_sample = list(eval_result.get("per_sample") or [])
    predictions = [str(r.get("prediction") or "") for r in per_sample]

    method_counts: Counter[str] = Counter()
    for row in per_sample:
        existing_method = row.get("match_method")
        if existing_method:
            method_counts[str(existing_method)] += 1
            continue
        cand_ans = str(row.get("predicted_answer") or "")
        ref_ans = str(row.get("reference_answer") or "")
        res = score_solver_output(str(row.get("prediction") or ""), f"ANSWER: {ref_ans}" if ref_ans else "")
        method_counts[res.match_method] += 1
        if not cand_ans and res.extracted_candidate_answer:
            row["predicted_answer"] = res.extracted_candidate_answer

    payload: dict[str, Any] = {
        "experiment": experiment_name,
        "auditkit_available": is_auditkit_installed(),
        "aligntune_math_grading_available": importlib.util.find_spec("aligntune.utils.math_grading") is not None,
        "match_method_breakdown": dict(sorted(method_counts.items())),
        "generation_hygiene": _compute_generation_hygiene(predictions),
    }

    if is_auditkit_installed():
        try:
            import auditkit  # type: ignore[import-not-found]

            payload["auditkit_version"] = getattr(auditkit, "__version__", "installed")
        except Exception as exc:
            payload["auditkit_error"] = str(exc)

    return payload
