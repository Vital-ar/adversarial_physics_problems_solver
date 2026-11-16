"""Native AuditKIT (auditkit>=1.1.12) evaluation, comparison, and agent-trace suite.

Executes AuditKIT's core evaluation primitives (`evaluate`, `compare`,
`AgentEvalRunner`, `judge_calibration`) deterministically over saved Solver
predictions and Solver-Verifier multi-turn traces across both Original and
Clean (`all-MiniLM-L6-v2` cosine < 0.80) evaluation splits.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from adversarial_physics.data.curation.answer_validator import (
    extract_answer_line,
    validate_candidate_answer,
)
from adversarial_physics.evaluation.auditkit_bridge import (
    STEP_PATTERN,
    has_degenerate_repetition_loop,
)
from adversarial_physics.orchestration.loop import score_solver_output

try:
    import auditkit as ak
    from auditkit.agent_eval.types import AgentCase, AgentEpisode, AgentEvent
    from auditkit.metric import Metric
    from auditkit.types import Direction, ScoreKind

    _HAS_AUDITKIT = True
except Exception:  # pragma: no cover - optional runtime guard
    ak = None  # type: ignore[assignment]
    AgentCase = None  # type: ignore[assignment]
    AgentEpisode = None  # type: ignore[assignment]
    AgentEvent = None  # type: ignore[assignment]
    Metric = object  # type: ignore[assignment,misc]
    Direction = None  # type: ignore[assignment]
    ScoreKind = None  # type: ignore[assignment]
    _HAS_AUDITKIT = False


def _ref_from_sample(sample: Any) -> str:
    """Extract gold physics reference text from an AuditKIT Sample."""
    meta = getattr(sample, "metadata", None) or {}
    ref_sol = str(meta.get("reference_solution") or "").strip()
    if ref_sol:
        return ref_sol
    ref_ans = str(getattr(sample, "target", None) or "").strip()
    return f"ANSWER: {ref_ans}" if ref_ans else ""


class PhysicsExactEquivMetric(Metric):
    """AuditKIT metric checking SymPy symbolic + <=1.5% numeric physics equivalence."""

    name = "physics_exact_equiv"
    kind = ScoreKind.BENCHMARK if ScoreKind else "benchmark"  # type: ignore[assignment]
    direction = Direction.MAXIMIZE if Direction else "maximize"  # type: ignore[assignment]

    def identity(self) -> dict[str, Any]:
        return {"name": self.name, "version": "1", "rel_tol": 0.015}

    def score(self, sample: Any, output: str, context: Any = None) -> Any:
        del context
        meta = getattr(sample, "metadata", None) or {}
        if "precomputed_is_equivalent" in meta:
            ok = bool(meta["precomputed_is_equivalent"])
            reason = str(meta.get("match_method") or ("precomputed_true" if ok else "mismatch"))
        else:
            ref = _ref_from_sample(sample)
            if not ref:
                return ak.Score(
                    name=self.name,
                    value=0.0,
                    kind=self.kind,
                    direction=self.direction,
                    reason="missing_reference",
                )
            scored = score_solver_output(output, ref)
            ok = bool(scored.is_equivalent)
            reason = scored.match_method
        return ak.Score(
            name=self.name,
            value=1.0 if ok else 0.0,
            kind=self.kind,
            direction=self.direction,
            reason=reason,
        )


class PhysicsTolerant5Metric(Metric):
    """AuditKIT metric checking SymPy or <=5% tolerant physics answer equivalence."""

    name = "physics_tolerant_5pct"
    kind = ScoreKind.BENCHMARK if ScoreKind else "benchmark"  # type: ignore[assignment]
    direction = Direction.MAXIMIZE if Direction else "maximize"  # type: ignore[assignment]

    def identity(self) -> dict[str, Any]:
        return {"name": self.name, "version": "1", "rel_tol": 0.05}

    def score(self, sample: Any, output: str, context: Any = None) -> Any:
        del context
        meta = getattr(sample, "metadata", None) or {}
        if meta.get("precomputed_is_equivalent"):
            ok = True
        else:
            ref = _ref_from_sample(sample)
            if not ref:
                ok = False
            else:
                has_sent, _ = extract_answer_line(output)
                if has_sent:
                    try:
                        ok = bool(validate_candidate_answer(output, ref, rel_tol=0.05).is_equivalent)
                    except Exception:
                        ok = False
                else:
                    ok = False
        return ak.Score(
            name=self.name,
            value=1.0 if ok else 0.0,
            kind=self.kind,
            direction=self.direction,
        )


class PhysicsStepStructureMetric(Metric):
    """AuditKIT metric verifying structured >=2-step physical derivation format."""

    name = "physics_step_structured"
    kind = ScoreKind.CODE if ScoreKind else "code"  # type: ignore[assignment]
    direction = Direction.MAXIMIZE if Direction else "maximize"  # type: ignore[assignment]

    def identity(self) -> dict[str, Any]:
        return {"name": self.name, "version": "1", "min_steps": 2}

    def score(self, sample: Any, output: str, context: Any = None) -> Any:
        del sample, context
        ok = len(STEP_PATTERN.findall(output or "")) >= 2
        return ak.Score(
            name=self.name,
            value=1.0 if ok else 0.0,
            kind=self.kind,
            direction=self.direction,
        )


class PhysicsDegenerateCollapseMetric(Metric):
    """AuditKIT metric detecting token-repetition collapse (0.0 ideal)."""

    name = "physics_degenerate_rate"
    kind = ScoreKind.CODE if ScoreKind else "code"  # type: ignore[assignment]
    direction = Direction.MINIMIZE if Direction else "minimize"  # type: ignore[assignment]

    def identity(self) -> dict[str, Any]:
        return {"name": self.name, "version": "1"}

    def score(self, sample: Any, output: str, context: Any = None) -> Any:
        del sample, context
        tokens = [t for t in (output or "").lower().split() if t]
        collapsed = has_degenerate_repetition_loop(tokens)
        return ak.Score(
            name=self.name,
            value=1.0 if collapsed else 0.0,
            kind=self.kind,
            direction=self.direction,
        )


def _default_physics_scorers() -> list[Any]:
    """Return the native AuditKIT metric suite used across all evaluation runs."""
    return [
        PhysicsExactEquivMetric(),
        PhysicsTolerant5Metric(),
        PhysicsStepStructureMetric(),
        PhysicsDegenerateCollapseMetric(),
        ak.Regex(r"ANSWER\s*:"),
        ak.Bleu(),
        ak.RogueL(),
        ak.WordCount(min_words=15),
    ]


def _serialize_stat(stat: Any) -> dict[str, Any]:
    return {
        "mean": round(float(getattr(stat, "mean", 0.0)), 6),
        "std": round(float(getattr(stat, "std", 0.0)), 6),
        "se": round(float(getattr(stat, "se", 0.0)), 6),
        "n": int(getattr(stat, "n", 0)),
    }


def summarize_auditkit_result(result: Any) -> dict[str, Any]:
    """Convert an `auditkit.Result` into a compact JSON-serializable summary."""
    headline = {k: round(float(v), 6) for k, v in (getattr(result, "headline", None) or {}).items()}
    stats = {k: _serialize_stat(v) for k, v in (getattr(result, "stats", None) or {}).items()}
    per_domain: dict[str, dict[str, float]] = {}
    for pred in getattr(result, "predictions", None) or []:
        dom = str(getattr(pred, "task", "") or "physics")
        score_rows = (getattr(pred, "metadata", None) or {}).get("scores") or []
        exact_row = next(
            (s for s in score_rows if isinstance(s, dict) and s.get("name") == "physics_exact_equiv"),
            None,
        )
        if exact_row is not None:
            bucket = per_domain.setdefault(dom, {"total": 0, "correct": 0})
            bucket["total"] += 1
            if float(exact_row.get("value", 0.0)) >= 0.5:
                bucket["correct"] += 1
    for dom, b in per_domain.items():
        b["accuracy"] = round(b["correct"] / max(1, b["total"]), 6)

    return {
        "auditkit_version": getattr(ak, "__version__", "unknown") if ak is not None else None,
        "run_id": str(getattr(result, "run_id", "")),
        "fingerprint": str(getattr(result, "fingerprint", "")),
        "n_samples": len(getattr(result, "predictions", None) or []),
        "headline": headline,
        "stats": stats,
        "by_domain_accuracy": per_domain,
    }


def evaluate_predictions_with_auditkit(
    eval_result: dict[str, Any],
    *,
    experiment_name: str,
    allowed_problem_ids: set[str] | Iterable[str] | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> tuple[Any | None, dict[str, Any]]:
    """Run real `auditkit.evaluate()` over saved `per_sample` evaluation rows."""
    if not _HAS_AUDITKIT or ak is None:
        return None, {"auditkit_available": False, "error": "auditkit_not_installed"}

    per_sample: list[dict[str, Any]] = list(eval_result.get("per_sample", []) or [])
    allow_set = {str(x) for x in allowed_problem_ids} if allowed_problem_ids is not None else None

    samples: list[Any] = []
    out_by_prompt: dict[str, str] = {}
    seen_ids: dict[str, int] = {}

    for idx, row in enumerate(per_sample):
        pid = str(row.get("problem_id") or f"row_{idx}")
        if allow_set is not None and pid not in allow_set:
            continue
        count = seen_ids.get(pid, 0)
        seen_ids[pid] = count + 1
        prompt_key = pid if count == 0 else f"{pid}#{count}"
        pred_text = str(row.get("prediction") or "")
        out_by_prompt[prompt_key] = pred_text

        ref_sol = str(row.get("reference_solution") or "")
        ref_ans = str(row.get("reference_answer") or "")
        target_text = ref_sol or (f"ANSWER: {ref_ans}" if ref_ans else "")

        meta: dict[str, Any] = {
            "source": str(row.get("source") or ""),
            "reference_solution": ref_sol,
            "reference_answer": ref_ans,
        }
        if "is_equivalent" in row and "match_method" in row:
            meta["precomputed_is_equivalent"] = bool(row["is_equivalent"])
            meta["match_method"] = str(row.get("match_method") or "")

        samples.append(
            ak.Sample(
                input=prompt_key,
                target=target_text,
                id=prompt_key,
                task=str(row.get("domain") or "physics"),
                metadata=meta,
            )
        )

    if not samples:
        return None, {
            "auditkit_available": True,
            "auditkit_version": getattr(ak, "__version__", "unknown"),
            "n_samples": 0,
            "headline": {},
            "stats": {},
        }

    def _saved_prediction_model(prompts: list[str]) -> list[str]:
        return [out_by_prompt.get(p, "") for p in prompts]

    cfg = ak.RunConfig(seed=42, temperature=temperature, max_tokens=max_tokens)
    result = ak.evaluate(
        samples,
        _saved_prediction_model,
        scorers=_default_physics_scorers(),
        config=cfg,
        experiment_name=experiment_name,
        cache=False,
    )
    summary = summarize_auditkit_result(result)
    summary["auditkit_available"] = True
    return result, summary


def compare_version_runs_with_auditkit(
    runs_by_exp: dict[str, Any],
    *,
    baseline_key: str = "E0",
) -> dict[str, Any]:
    """Use `auditkit.compare()` for ranked leaderboard + pairwise `RunComparison` deltas vs baseline."""
    if not _HAS_AUDITKIT or ak is None:
        return {"auditkit_available": False}

    valid_runs = {k: v for k, v in runs_by_exp.items() if v is not None}
    if not valid_runs:
        return {"auditkit_available": True, "leaderboard": [], "vs_baseline": {}}

    leaderboard = ak.compare(list(valid_runs.values()), metric="physics_exact_equiv")
    id_to_exp = {r.run_id: exp for exp, r in valid_runs.items()}
    for row in leaderboard:
        row["experiment"] = id_to_exp.get(row.get("run_id", ""), "")

    vs_baseline: dict[str, Any] = {}
    base_run = valid_runs.get(baseline_key)
    if base_run is not None:
        for exp_key, cand_run in valid_runs.items():
            if exp_key == baseline_key:
                continue
            cmp = ak.compare(base_run, cand_run)
            deltas = {}
            for d in cmp.metric_deltas():
                deltas[d.metric] = {
                    "baseline": round(float(d.baseline), 6) if d.baseline is not None else None,
                    "candidate": round(float(d.candidate), 6) if d.candidate is not None else None,
                    "delta": round(float(d.delta), 6) if d.delta is not None else None,
                    "grade": d.grade.value,
                    "direction": d.direction,
                    "pct_change": round(float(d.pct_change), 2) if d.pct_change is not None else None,
                }
            sig = cmp.significance(metric="physics_exact_equiv")
            diff_counts = cmp.sample_summary("physics_exact_equiv")
            vs_baseline[f"{exp_key}_vs_{baseline_key}"] = {
                "overall_grade": cmp.grade().value,
                "significance_exact_equiv": sig,
                "sample_flips_exact_equiv": diff_counts,
                "metric_deltas": deltas,
            }

    return {
        "auditkit_available": True,
        "auditkit_version": getattr(ak, "__version__", "unknown"),
        "leaderboard": leaderboard,
        "vs_baseline": vs_baseline,
    }


def _build_loop_cases_episodes_and_reviews(
    rows: Sequence[dict[str, Any]],
) -> tuple[list[Any], list[Any], list[dict[str, Any]]]:
    """Convert recorded loop trace dicts into AuditKIT AgentCases, AgentEpisodes, and Verifier rows."""
    cases: list[Any] = []
    episodes: list[Any] = []
    verifier_rows: list[dict[str, Any]] = []

    for row in rows:
        pid = str(row.get("problem_id") or "")
        domain = str(row.get("domain") or "physics")
        traces = list(row.get("traces", []) or [])
        events: list[Any] = []
        ev_idx = 0
        last_ans = ""

        for tr in traces:
            s_out = str(tr.get("solver_output") or "")
            events.append(
                AgentEvent(
                    index=ev_idx,
                    role="assistant",
                    type="message",
                    payload={"text": s_out},
                    turn_id=tr.get("iteration"),
                )
            )
            ev_idx += 1
            last_ans = str((tr.get("solver_validation") or {}).get("extracted_candidate_answer") or "")

            verdict = str(tr.get("verifier_verdict") or "").upper()
            directive = str(tr.get("verifier_directive") or "")
            if verdict:
                events.append(
                    AgentEvent(
                        index=ev_idx,
                        role="assistant",
                        type="tool_call",
                        payload={
                            "name": "verifier_review",
                            "arguments": {
                                "verdict": verdict,
                                "directive": directive[:240],
                            },
                        },
                        turn_id=tr.get("iteration"),
                    )
                )
                ev_idx += 1
                events.append(
                    AgentEvent(
                        index=ev_idx,
                        role="tool",
                        type="tool_result",
                        payload={"verdict": verdict, "stop_reason": row.get("stop_reason")},
                        turn_id=tr.get("iteration"),
                    )
                )
                ev_idx += 1

            if verdict in {"APPROVE", "CHALLENGE"} and "solver_is_correct" in tr:
                verifier_rows.append(
                    {
                        "human": bool(tr["solver_is_correct"]),
                        "judge": verdict == "APPROVE",
                        "dimension": domain,
                    }
                )

        cases.append(
            AgentCase(
                id=pid,
                task=str(row.get("problem_statement") or "")[:200],
                category=domain,
                allowed_tools=["verifier_review"],
                outcome=ak.FinalStateAssertion("final_correct", equals=True),
            )
        )
        episodes.append(
            AgentEpisode(
                case_id=pid,
                events=events,
                final_answer=last_ans,
                final_state={"final_correct": bool(row.get("final_correct", False))},
                counters={
                    "tokens": int(row.get("total_tokens") or 0),
                    "latency_ms": round(float(row.get("total_latency_s") or 0.0) * 1000.0, 2),
                },
                stop_reason=str(row.get("stop_reason") or "completed"),
                coverage={
                    "final_answer": "observed",
                    "tool_call_names": "observed",
                    "tool_call_arguments": "observed",
                    "tool_results": "observed",
                    "final_state": "observed",
                },
            )
        )

    return cases, episodes, verifier_rows


def evaluate_loop_traces_with_auditkit(
    trace_path: str | Path,
    *,
    name: str,
    allowed_problem_ids: set[str] | Iterable[str] | None = None,
) -> dict[str, Any]:
    """Audit saved E4b multi-turn traces via `AgentEvalRunner` + `AgentLoopDetection` + `judge_calibration`."""
    if not _HAS_AUDITKIT or ak is None:
        return {"auditkit_available": False, "error": "auditkit_not_installed"}

    p = Path(trace_path)
    if not p.is_file():
        return {"auditkit_available": True, "error": f"missing_trace:{p}"}

    allow_set = {str(x) for x in allowed_problem_ids} if allowed_problem_ids is not None else None
    rows: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8-sig").splitlines():
        s = line.strip()
        if not s:
            continue
        row = json.loads(s)
        if allow_set is not None and str(row.get("problem_id")) not in allow_set:
            continue
        rows.append(row)

    if not rows:
        return {"auditkit_available": True, "name": name, "n_episodes": 0}

    cases, episodes, verifier_rows = _build_loop_cases_episodes_and_reviews(rows)
    spec = ak.AgentEvalSpec(
        cases=cases,
        episodes=episodes,
        mode="recorded",
        scorers=["agent_loop_detection"],
        name=name,
    )
    agent_res = ak.AgentEvalRunner().run(spec)

    verified_success = sum(1 for r in agent_res.rows if (r.outcome or {}).get("verdict") == "success")
    loop_scores = [
        float(r.diagnostics["agent_loop_detection"]["value"])
        for r in agent_res.rows
        if "agent_loop_detection" in (r.diagnostics or {})
        and r.diagnostics["agent_loop_detection"].get("value") is not None
    ]
    flagged_loop_ids = [
        r.case_id
        for r in agent_res.rows
        if float((r.diagnostics or {}).get("agent_loop_detection", {}).get("value", 1.0)) < 0.5
    ]

    cal = ak.judge_calibration(verifier_rows) if verifier_rows else {"status": "unknown"}

    return {
        "auditkit_available": True,
        "auditkit_version": getattr(ak, "__version__", "unknown"),
        "name": name,
        "n_episodes": len(episodes),
        "verified_success_count": verified_success,
        "verified_success_rate": round(verified_success / max(1, len(episodes)), 6),
        "agent_loop_detection_evaluated": len(loop_scores),
        "loop_free_rate": round(sum(loop_scores) / len(loop_scores), 6) if loop_scores else None,
        "flagged_loop_case_ids": flagged_loop_ids,
        "verifier_judge_calibration": cal,
    }


def _loop_trace_to_eval_payload(trace_path: Path) -> dict[str, Any]:
    """Convert E4b `results_per_problem.jsonl` final iterations into `per_sample` eval payload."""
    rows: list[dict[str, Any]] = []
    for line in trace_path.read_text(encoding="utf-8-sig").splitlines():
        s = line.strip()
        if not s:
            continue
        rec = json.loads(s)
        traces = list(rec.get("traces", []) or [])
        last_trace = traces[-1] if traces else {}
        s_out = str(last_trace.get("solver_output") or "")
        sv = last_trace.get("solver_validation") or {}
        rows.append(
            {
                "problem_id": str(rec.get("problem_id") or ""),
                "domain": str(rec.get("domain") or "physics"),
                "source": "loop_eval",
                "is_equivalent": bool(rec.get("final_correct", False)),
                "match_method": str(
                    rec.get("final_match_method") or ("exact" if rec.get("final_correct") else "mismatch")
                ),
                "has_answer_sentinel": bool(rec.get("final_has_answer", False)),
                "predicted_answer": str(sv.get("extracted_candidate_answer") or ""),
                "reference_answer": str(rec.get("normalized_reference_answer") or ""),
                "prediction": s_out,
                "reference_solution": str(rec.get("reference_solution") or ""),
            }
        )
    return {"per_sample": rows}


def run_full_auditkit_suite(
    root_dir: str | Path = ".",
    *,
    clean_ids_by_version_split: dict[tuple[str, str], set[str]] | None = None,
) -> dict[str, Any]:
    """Run native AuditKIT evaluation, baseline comparisons, and loop audits over all saved runs."""
    if not _HAS_AUDITKIT or ak is None:
        return {"auditkit_available": False, "error": "auditkit_not_installed"}

    root = Path(root_dir)
    eval_specs: dict[tuple[str, str], dict[str, Path]] = {
        ("v1", "validation"): {
            "E0": root / "results/eval_v1/E0_v1_validation_eval.json",
            "E1": root / "results/eval_v1/E1_v1_validation_eval.json",
            "E2": root / "results/eval_v1/E2_v1_validation_eval.json",
            "E4a": root / "results/eval_v1/E4a_v1_validation_eval.json",
        },
        ("v1", "test"): {
            "E0": root / "results/eval_v1/E0_v1_test_eval.json",
            "E1": root / "results/eval_v1/E1_v1_test_eval.json",
            "E2": root / "results/eval_v1/E2_v1_test_eval.json",
            "E4a": root / "results/eval_v1/E4a_v1_test_eval.json",
        },
        ("v2", "validation"): {
            "E0": root / "results/eval_e0_v2/E0_v2_validation_eval.json",
            "E1": root / "results/eval_e1_v2/E1_v2_validation_eval.json",
            "E2": root / "results/eval_e2_v2/E2_v2_validation_eval.json",
            "E3": root / "results/eval_e3_v2/E3_v2_validation_eval.json",
            "E4a": root / "results/eval_e4a_v2/E4a_v2_validation_eval.json",
        },
        ("v2", "test"): {
            "E0": root / "results/eval_e0_v2/E0_v2_test_eval.json",
            "E1": root / "results/eval_e1_v2/E1_v2_test_eval.json",
            "E2": root / "results/eval_e2_v2/E2_v2_test_eval.json",
            "E3": root / "results/eval_e3_v2/E3_v2_test_eval.json",
            "E4a": root / "results/eval_e4a_v2/E4a_v2_test_eval.json",
        },
        ("v3", "validation"): {
            "E0": root / "results/eval_e0_v2/E0_v2_validation_eval.json",
            "E1": root / "results/eval_e1_v3/E1_v3_validation_eval.json",
            "E2": root / "results/eval_e2_v3/E2_v3_validation_eval.json",
            "E3": root / "results/eval_e3_v3/E3_v3_validation_eval.json",
            "E4a": root / "results/eval_e4a_v3/E4a_v3_validation_eval.json",
            "E4b": root / "results/eval_e4_loop_v3_val/results_per_problem.jsonl",
        },
        ("v3", "test"): {
            "E0": root / "results/eval_e0_v2/E0_v2_test_eval.json",
            "E1": root / "results/eval_e1_v3/E1_v3_test_eval.json",
            "E2": root / "results/eval_e2_v3/E2_v3_test_eval.json",
            "E3": root / "results/eval_e3_v3/E3_v3_test_eval.json",
            "E4a": root / "results/eval_e4a_v3/E4a_v3_test_eval.json",
            "E4b": root / "results/eval_e4_loop_v3_test/results_per_problem.jsonl",
        },
    }

    views: dict[str, Any] = {"original": {}, "clean_080": {}}
    for view_name in ("original", "clean_080"):
        for (ver, split), exp_paths in eval_specs.items():
            allow_ids = None
            if view_name == "clean_080" and clean_ids_by_version_split:
                allow_ids = clean_ids_by_version_split.get((ver, split))
                if allow_ids is None:
                    continue

            run_objs: dict[str, Any] = {}
            run_summaries: dict[str, Any] = {}
            for exp, path in exp_paths.items():
                if not path.is_file():
                    continue
                if path.suffix == ".jsonl":
                    payload = _loop_trace_to_eval_payload(path)
                else:
                    payload = json.loads(path.read_text(encoding="utf-8-sig"))
                res_obj, summ = evaluate_predictions_with_auditkit(
                    payload,
                    experiment_name=f"{exp}_{ver}_{split}_{view_name}",
                    allowed_problem_ids=allow_ids,
                )
                if res_obj is not None:
                    run_objs[exp] = res_obj
                    run_summaries[exp] = summ

            if run_objs:
                comp = compare_version_runs_with_auditkit(run_objs, baseline_key="E0")
                views[view_name].setdefault(ver, {})[split] = {
                    "runs": run_summaries,
                    "comparison": comp,
                }

    loop_audits: dict[str, Any] = {}
    loop_sources = [
        # `eval_e4_loop_v2_test` evaluated 37 clean PhysGym validation IDs (`validation_v2.jsonl`)
        ("v2", "validation", "E4b_v2_test", root / "results/eval_e4_loop_v2_test/results_per_problem.jsonl"),
        ("v3", "validation", "E4b_v3_val", root / "results/eval_e4_loop_v3_val/results_per_problem.jsonl"),
        ("v3", "test", "E4b_v3_test", root / "results/eval_e4_loop_v3_test/results_per_problem.jsonl"),
    ]
    for ver, split, label, lp in loop_sources:
        if not lp.is_file():
            continue
        loop_audits[f"{label}_original"] = evaluate_loop_traces_with_auditkit(lp, name=f"{label}_original")
        if clean_ids_by_version_split and (ver, split) in clean_ids_by_version_split:
            loop_audits[f"{label}_clean_080"] = evaluate_loop_traces_with_auditkit(
                lp,
                name=f"{label}_clean_080",
                allowed_problem_ids=clean_ids_by_version_split[(ver, split)],
            )

    return {
        "auditkit_available": True,
        "auditkit_version": getattr(ak, "__version__", "unknown"),
        "views": views,
        "loop_agent_eval_and_judge_calibration": loop_audits,
    }
