"""Stage 2 Step 4E: Deterministically Rescue All 5 v2 SFT Datasets (Zero New LLM Calls),
Verify Strict Pre-Upload Quality & Anti-Leakage Assertions, and Upload to HuggingFace `-v2` Repos.

Usage:
  .\\venv\\Scripts\\python.exe scripts/04e_rescue_and_upload_v2.py --no-upload   # Build & verify locally only
  .\\venv\\Scripts\\python.exe scripts/04e_rescue_and_upload_v2.py               # Build, verify, and upload to HF
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adversarial_physics.data.curation.answer_validator import extract_answer_line  # noqa: E402
from adversarial_physics.data.curation.dataset_exporter import (  # noqa: E402
    export_all_stage2_sft_datasets,
    is_clean_symbolic_or_numeric_answer,
)
from adversarial_physics.data.schemas.problem import (  # noqa: E402
    CanonicalProblem,
    load_problems_jsonl,
    save_problems_jsonl,
)
from adversarial_physics.data.schemas.trajectory import (  # noqa: E402
    CandidateSolution,
    load_candidates_jsonl,
    save_candidates_jsonl,
)
from adversarial_physics.orchestration.loop import (  # noqa: E402
    prepare_reference_for_validation,
    score_solver_output,
)

DOMAIN_VAL_TEST_QUOTA: dict[str, int] = {
    "MECHANICS": 8,
    "ELECTROMAGNETISM": 5,
    "MODERN_PHYSICS": 3,
    "OPTICS": 2,
    "RELATIVITY": 2,
    "THERMODYNAMICS": 2,
}

HF_V2_REPOS: dict[str, str] = {
    "solver_e1_manual_det": "vital-ar/physics-solver-e1-manual-rules-v2",
    "solver_e2_curatorkit_nollm": "vital-ar/physics-solver-e2-curatorkit-nollm-v2",
    "solver_e3_canonical_llm": "vital-ar/physics-solver-e3-canonical-llm-v2",
    "solver_e4_adversarial": "vital-ar/physics-solver-e4-adversarial-v2",
    "verifier_e4_curated": "vital-ar/physics-verifier-e4-curated-v2",
}


def _load_jsonl_dicts(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def _save_jsonl_dicts(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _norm_text_for_leakage(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _word_trigrams(text: str) -> set[tuple[str, str, str]]:
    toks = re.findall(r"[a-z0-9]+", (text or "").lower())
    if len(toks) < 3:
        return set()
    return {tuple(toks[i : i + 3]) for i in range(len(toks) - 2)}  # type: ignore[misc]


def _is_near_heldout_leak(
    problem_statement: str,
    heldout_norm_statements: set[str],
    heldout_trigrams: list[set[tuple[str, str, str]]],
    *,
    jaccard_threshold: float = 0.70,
) -> bool:
    norm = _norm_text_for_leakage(problem_statement)
    if norm in heldout_norm_statements:
        return True
    tri = _word_trigrams(norm)
    if not tri:
        return False
    for h_tri in heldout_trigrams:
        if not h_tri:
            continue
        inter = len(tri & h_tri)
        if inter == 0:
            continue
        jacc = inter / len(tri | h_tri)
        if jacc >= jaccard_threshold:
            return True
    return False


def _resolve_hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        return token.strip()
    env_path = PROJECT_ROOT / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            if k.strip() in {"HF_TOKEN", "HUGGINGFACE_TOKEN"} and v.strip():
                return v.strip().strip('"').strip("'")
    return None


def _is_ultra_clean_problem(problem_statement: str, reference_solution: str) -> bool:
    ps = (problem_statement or "").strip()
    rs = (reference_solution or "").strip()
    if len(ps) < 20 or len(rs) < 400:
        return False
    ans = extract_answer_line(rs)
    if not ans or not is_clean_symbolic_or_numeric_answer(ans):
        return False
    return bool(score_solver_output(rs, rs).is_equivalent)


def _format_problem_with_answer(p: CanonicalProblem) -> CanonicalProblem:
    formatted_ref, _ = prepare_reference_for_validation(p.reference_solution)
    return dataclasses.replace(p, reference_solution=formatted_ref)


def _select_balanced_challenges(
    challenge_rows: list[dict[str, Any]],
    sol2_by_pid: dict[str, CandidateSolution],
    rng: random.Random,
    *,
    max_challenges: int = 720,
    max_pairs: int = 160,
) -> list[dict[str, Any]]:
    """Deterministically cap CHALLENGE directives to `max_challenges` while preserving Sol2, 2-hint pairs, and balance."""
    by_pid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in challenge_rows:
        by_pid[str(row["problem_id"])].append(row)

    # Sort each problem's challenges preferring real tiny_aya student failures first, then aya8b
    for pid in by_pid:
        by_pid[pid].sort(
            key=lambda r: (
                0 if "tiny_aya" in str(r.get("source") or "") else 1,
                str(r.get("source") or ""),
                str(r.get("item_id") or ""),
            )
        )

    selected: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()
    used_pids_count: Counter[str] = Counter()

    def _add_row(row: dict[str, Any]) -> bool:
        key = (str(row["problem_id"]), str(row.get("source") or ""))
        if key in seen_keys or len(selected) >= max_challenges:
            return False
        seen_keys.add(key)
        selected.append(row)
        used_pids_count[key[0]] += 1
        return True

    # (a) Prioritize problems in sol2_by_pid first for full Sol2 hinted-turn coverage
    for pid in sorted(sol2_by_pid):
        for row in by_pid.get(pid, []):
            _add_row(row)

    # (b) Add problems with >= 2 distinct challenge directives up to `max_pairs` paired problems
    paired_pids = [pid for pid, rows in by_pid.items() if len(rows) >= 2 and pid not in sol2_by_pid]
    rng.shuffle(paired_pids)
    pairs_added = sum(1 for pid in used_pids_count if used_pids_count[pid] >= 2)
    for pid in paired_pids:
        if pairs_added >= max_pairs or len(selected) + 2 > max_challenges:
            break
        added_here = 0
        for row in by_pid[pid][:2]:
            if _add_row(row):
                added_here += 1
        if added_here >= 2:
            pairs_added += 1

    # (c) Add 1 challenge directive per remaining problem balancing source & domain up to max_challenges
    single_candidates: list[dict[str, Any]] = []
    for pid, rows in by_pid.items():
        if used_pids_count[pid] > 0:
            continue
        single_candidates.append(rows[0])

    rng.shuffle(single_candidates)
    # Interleave tiny_aya_global_zero_shot and aya_expanse_8b_synthetic_error round-robin by domain
    tiny_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    aya_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in single_candidates:
        dom = str(r.get("domain") or "MECHANICS")
        if "tiny_aya" in str(r.get("source") or ""):
            tiny_bucket[dom].append(r)
        else:
            aya_bucket[dom].append(r)

    domains = sorted(set(tiny_bucket) | set(aya_bucket))
    ordered_singles: list[dict[str, Any]] = []
    toggle = 0
    while any(tiny_bucket[d] or aya_bucket[d] for d in domains):
        for d in domains:
            primary = tiny_bucket[d] if toggle % 2 == 0 else aya_bucket[d]
            secondary = aya_bucket[d] if toggle % 2 == 0 else tiny_bucket[d]
            if primary:
                ordered_singles.append(primary.pop())
            elif secondary:
                ordered_singles.append(secondary.pop())
        toggle += 1

    for row in ordered_singles:
        if len(selected) >= max_challenges:
            break
        _add_row(row)

    return selected


def _normalize_exported_prompts_and_checksums(sft_root: Path) -> None:
    """Ensure 100% of prompt & instruction strings in exported JSONLs end with \\n\\n and update checksums.txt."""
    for folder in HF_V2_REPOS:
        ds_dir = sft_root / folder
        if not ds_dir.is_dir():
            continue
        for split_name in ("train", "validation", "test"):
            sp_path = ds_dir / f"{split_name}.jsonl"
            if not sp_path.is_file():
                continue
            rows = _load_jsonl_dicts(sp_path)
            modified = False
            for r in rows:
                for key in ("prompt", "instruction"):
                    val = r.get(key)
                    if isinstance(val, str) and val and not val.endswith("\n\n"):
                        r[key] = val.rstrip() + "\n\n"
                        modified = True
            if modified:
                _save_jsonl_dicts(rows, sp_path)

        # Recompute SHA-256 checksums for all files in this dataset folder
        checksum_lines: list[str] = []
        for fpath in sorted(ds_dir.iterdir(), key=lambda p: p.name):
            if not fpath.is_file() or fpath.name == "checksums.txt":
                continue
            digest = hashlib.sha256(fpath.read_bytes()).hexdigest()
            checksum_lines.append(f"{digest}  {fpath.name}")
        (ds_dir / "checksums.txt").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")


def build_and_verify_rescued_v2() -> dict[str, Any]:
    """Run all deterministic rescue stages, run strict assertions, and return summary stats."""
    rng = random.Random(42)

    # 1. Load fallback E1 pool and locked PhysGym validation/test splits
    p01a_list = load_problems_jsonl(PROJECT_ROOT / "data/intermediate/01a_manual_deterministic_only_pool.jsonl")
    p01a_map: dict[str, CanonicalProblem] = {p.problem_id: p for p in p01a_list}

    val_physgym_raw = load_problems_jsonl(PROJECT_ROOT / "data/splits/validation.jsonl")
    test_physgym_raw = load_problems_jsonl(PROJECT_ROOT / "data/splits/test.jsonl")
    val_physgym = [_format_problem_with_answer(p) for p in val_physgym_raw]
    test_physgym = [_format_problem_with_answer(p) for p in test_physgym_raw]
    physgym_holdout_pids = {p.problem_id for p in val_physgym + test_physgym}

    # Read untouched directive checkpoints (is_valid=True) for idempotency & non-directive preference
    v2_dir_raw = [
        r
        for r in _load_jsonl_dicts(PROJECT_ROOT / "data/synthetic/02e_aya8b_directives_checkpoint_v2.jsonl")
        if r.get("is_valid", True)
    ]
    v1_dir_raw = [
        r
        for r in _load_jsonl_dicts(PROJECT_ROOT / "data/synthetic/02g_directives_checkpoint.jsonl")
        if r.get("is_valid", True)
    ]
    v2_dir_pids = {str(r["problem_id"]) for r in v2_dir_raw if r.get("problem_id")}
    v1_dir_pids = {str(r["problem_id"]) for r in v1_dir_raw if r.get("problem_id")}

    # Merge ultra-clean candidates in priority order
    ultra_sources: list[tuple[str, list[dict[str, Any]]]] = [
        ("canon_v2", _load_jsonl_dicts(PROJECT_ROOT / "data/processed/canonical_train_v2.jsonl")),
        ("syn_e2", _load_jsonl_dicts(PROJECT_ROOT / "data/synthetic/03b_e2_synthetic_raw_v2.jsonl")),
        ("canon_v1", _load_jsonl_dicts(PROJECT_ROOT / "data/processed/canonical_train.jsonl")),
        ("ckpt_llm", _load_jsonl_dicts(PROJECT_ROOT / "data/processed/canonical_train_llm_checkpoint.jsonl")),
    ]

    ultra_clean_map: dict[str, CanonicalProblem] = {}
    for _, raw_rows in ultra_sources:
        for r in raw_rows:
            pid = str(r.get("problem_id") or "").strip()
            if not pid or pid in physgym_holdout_pids or pid in ultra_clean_map:
                continue
            fallback = p01a_map.get(pid)
            ps = str(r.get("problem_statement") or (fallback.problem_statement if fallback else "")).strip()
            dom = str(r.get("domain") or (fallback.domain if fallback else "MECHANICS")).strip()
            src = str(r.get("source") or (fallback.source if fallback else "literature_curated")).strip()
            oid = str(r.get("original_id") or (fallback.original_id if fallback else pid)).strip()
            rs = str(r.get("reference_solution") or "").strip()
            if not _is_ultra_clean_problem(ps, rs):
                continue
            ultra_clean_map[pid] = CanonicalProblem(
                problem_id=pid,
                domain=dom,
                problem_statement=ps,
                reference_solution=rs,
                source=src,
                original_id=oid,
            )

    print(f"[Step 1] Built ultra_clean_map with {len(ultra_clean_map)} unique >=400-char self-scoring problems.")

    # 2. Deterministically select 22 validation + 22 test in-domain problems stratified by domain
    by_domain: dict[str, list[CanonicalProblem]] = defaultdict(list)
    for prob in ultra_clean_map.values():
        by_domain[prob.domain].append(prob)

    curated_val: list[CanonicalProblem] = []
    curated_test: list[CanonicalProblem] = []

    for dom, quota in DOMAIN_VAL_TEST_QUOTA.items():
        pool = list(by_domain.get(dom, []))
        rng.shuffle(pool)
        # Prioritize problems not covered by directives so E4 directive training pool stays maximal
        pool.sort(
            key=lambda p: (
                0 if (p.problem_id not in v2_dir_pids and p.problem_id not in v1_dir_pids) else (
                    1 if p.problem_id not in v2_dir_pids else 2
                ),
            )
        )
        needed = 2 * quota
        if len(pool) < needed:
            raise RuntimeError(f"Insufficient ultra-clean problems in domain {dom}: need {needed}, got {len(pool)}")
        picked = pool[:needed]
        curated_val.extend(picked[:quota])
        curated_test.extend(picked[quota : 2 * quota])

    validation_v2 = val_physgym + [_format_problem_with_answer(p) for p in curated_val]
    test_v2 = test_physgym + [_format_problem_with_answer(p) for p in curated_test]
    save_problems_jsonl(validation_v2, PROJECT_ROOT / "data/splits/validation_v2.jsonl")
    save_problems_jsonl(test_v2, PROJECT_ROOT / "data/splits/test_v2.jsonl")

    heldout_all = validation_v2 + test_v2
    heldout_pids = {p.problem_id for p in heldout_all}
    heldout_norm_statements = {_norm_text_for_leakage(p.problem_statement) for p in heldout_all}
    heldout_trigrams = [_word_trigrams(p.problem_statement) for p in heldout_all]
    assert len(validation_v2) == 60 and len(test_v2) == 60 and len(heldout_pids) == 120
    print(f"[Step 2] Saved validation_v2 ({len(validation_v2)}) and test_v2 ({len(test_v2)}), heldout_pids={len(heldout_pids)}.")

    # 3. Save rescued E3, E2, and E1 pools with zero heldout leakage (exact + near-dup guard)
    def _is_leak_free(p: CanonicalProblem) -> bool:
        if p.problem_id in heldout_pids:
            return False
        return not _is_near_heldout_leak(p.problem_statement, heldout_norm_statements, heldout_trigrams)

    e3_pool = [p for pid, p in ultra_clean_map.items() if pid not in heldout_pids and _is_leak_free(p)]
    save_problems_jsonl(e3_pool, PROJECT_ROOT / "data/processed/canonical_train_v2.jsonl")
    e3_map: dict[str, CanonicalProblem] = {p.problem_id: p for p in e3_pool}
    e3_train_pids = set(e3_map.keys())
    print(f"[Step 3a] Saved rescued E3 pool: {len(e3_pool)} problems.")

    # Build rescued E2 pool
    orig_01b = load_problems_jsonl(PROJECT_ROOT / "data/intermediate/01b_e2_books_plus_synthetic_v2.jsonl")
    orig_03b = load_problems_jsonl(PROJECT_ROOT / "data/synthetic/03b_e2_synthetic_raw_v2.jsonl")
    prefer_01b_03b_ids = {p.problem_id for p in orig_01b + orig_03b}
    e2_candidates: dict[str, CanonicalProblem] = {}
    for p in orig_01b + orig_03b + p01a_list:
        if p.problem_id in e2_candidates or not _is_leak_free(p):
            continue
        formatted = _format_problem_with_answer(p)
        rs = formatted.reference_solution.strip()
        ans = extract_answer_line(rs)
        if p.problem_id in prefer_01b_03b_ids and (
            len(rs) < 250 or not ans or not is_clean_symbolic_or_numeric_answer(ans)
        ):
            if p.problem_id in ultra_clean_map and _is_leak_free(ultra_clean_map[p.problem_id]):
                formatted = _format_problem_with_answer(ultra_clean_map[p.problem_id])
                rs = formatted.reference_solution.strip()
                ans = extract_answer_line(rs)
        if not ans or not is_clean_symbolic_or_numeric_answer(ans):
            continue
        if not score_solver_output(rs, rs).is_equivalent:
            continue
        e2_candidates[p.problem_id] = formatted

    e2_pool = list(e2_candidates.values())
    save_problems_jsonl(e2_pool, PROJECT_ROOT / "data/intermediate/01b_e2_books_plus_synthetic_v2.jsonl")
    print(f"[Step 3b] Saved rescued E2 pool: {len(e2_pool)} problems.")

    # Build leak-free E1 pool
    e1_pool = [_format_problem_with_answer(p) for p in p01a_list if _is_leak_free(p)]
    save_problems_jsonl(e1_pool, PROJECT_ROOT / "data/intermediate/01a_manual_deterministic_only_pool_v2.jsonl")
    print(f"[Step 3c] Saved rescued E1 pool: {len(e1_pool)} problems.")

    # 4. Merge & filter Stage 2 trajectories from untouched raw/checkpoint sources
    def _merge_clean_correct_candidates(paths: list[Path]) -> list[CandidateSolution]:
        kept: dict[str, CandidateSolution] = {}
        for pth in paths:
            for cand in load_candidates_jsonl(pth):
                pid = cand.problem_id
                if pid not in e3_train_pids or pid in kept:
                    continue
                sol = (cand.solution_text or "").strip()
                if len(sol) < 400:
                    continue
                ans = extract_answer_line(sol)
                if not ans or not is_clean_symbolic_or_numeric_answer(ans):
                    continue
                if not score_solver_output(sol, e3_map[pid].reference_solution).is_equivalent:
                    continue
                kept[pid] = dataclasses.replace(cand, is_correct=True)
        return list(kept.values())

    rescued_sol2 = _merge_clean_correct_candidates(
        [
            PROJECT_ROOT / "data/synthetic/02a_aya8b_solution2_raw_v2.jsonl",
            PROJECT_ROOT / "data/synthetic/02d_curated_solution2.jsonl",
        ]
    )
    save_candidates_jsonl(rescued_sol2, PROJECT_ROOT / "data/synthetic/02d_curated_solution2_v2.jsonl")
    sol2_by_pid = {c.problem_id: c for c in rescued_sol2}

    rescued_tiny_ok = _merge_clean_correct_candidates(
        [
            PROJECT_ROOT / "data/synthetic/02d_tiny_aya_correct_curated_v2.jsonl",
            PROJECT_ROOT / "data/synthetic/02e_tiny_aya_correct_trajectories.jsonl",
        ]
    )
    save_candidates_jsonl(rescued_tiny_ok, PROJECT_ROOT / "data/synthetic/02d_tiny_aya_correct_curated_v2.jsonl")

    flawed_dedup: dict[tuple[str, str], CandidateSolution] = {}
    for pth in [
        PROJECT_ROOT / "data/synthetic/02b_aya8b_flawed_raw_v2.jsonl",
        PROJECT_ROOT / "data/synthetic/02c_tiny_aya_failures_raw_v2.jsonl",
        PROJECT_ROOT / "data/synthetic/02f_surviving_flawed_trajectories.jsonl",
    ]:
        for cand in load_candidates_jsonl(pth):
            if cand.problem_id not in e3_train_pids:
                continue
            if cand.is_correct or len((cand.solution_text or "").strip()) < 80:
                continue
            key = (cand.problem_id, cand.source)
            if key not in flawed_dedup:
                flawed_dedup[key] = cand
    rescued_flawed = list(flawed_dedup.values())
    save_candidates_jsonl(rescued_flawed, PROJECT_ROOT / "data/synthetic/02d_surviving_flawed_v2.jsonl")

    # Separate [[APPROVE]] vs [[CHALLENGE]] directives cleanly
    approve_dedup: dict[tuple[str, str, str], dict[str, Any]] = {}
    approve_sol1_by_pid: dict[str, dict[str, Any]] = {}
    challenge_dedup: dict[tuple[str, str], dict[str, Any]] = {}

    for raw_rows in [v2_dir_raw, v1_dir_raw]:
        for r in raw_rows:
            pid = str(r.get("problem_id") or "")
            if pid not in e3_train_pids:
                continue
            kind = str(r.get("kind") or "")
            src = str(r.get("source") or "")
            verdict = str(r.get("verdict") or "")

            if verdict == "[[APPROVE]]" or kind == "domain_balanced_approval":
                rec = dict(r)
                if src == "minority_solution2":
                    if pid not in sol2_by_pid:
                        continue
                    rec["proposed_solution"] = sol2_by_pid[pid].solution_text
                else:
                    rec["proposed_solution"] = e3_map[pid].reference_solution
                    approve_sol1_by_pid.setdefault(pid, rec)
                body_prefix = str(rec.get("directive_body") or "")[:80]
                akey = (pid, src, body_prefix)
                if akey not in approve_dedup:
                    approve_dedup[akey] = rec
            else:
                ckey = (pid, src)
                if ckey not in challenge_dedup:
                    challenge_dedup[ckey] = dict(r)

    # Clone Sol1 [[APPROVE]] for any Sol2 problem having a Sol1 approval but no minority_solution2 approval
    existing_sol2_approve_pids = {k[0] for k in approve_dedup if k[1] == "minority_solution2"}
    for pid, sol2_cand in sorted(sol2_by_pid.items()):
        if pid in existing_sol2_approve_pids or pid not in approve_sol1_by_pid:
            continue
        cloned = dict(approve_sol1_by_pid[pid])
        cloned["source"] = "minority_solution2"
        cloned["candidate_id"] = sol2_cand.candidate_id
        cloned["item_id"] = f"approve:{sol2_cand.candidate_id}"
        cloned["proposed_solution"] = sol2_cand.solution_text
        body_prefix = str(cloned.get("directive_body") or "")[:80]
        approve_dedup[(pid, "minority_solution2", body_prefix)] = cloned

    approvals = list(approve_dedup.values())
    challenges = _select_balanced_challenges(
        list(challenge_dedup.values()),
        sol2_by_pid=sol2_by_pid,
        rng=rng,
        max_challenges=720,
        max_pairs=160,
    )
    rescued_directives = approvals + challenges
    _save_jsonl_dicts(rescued_directives, PROJECT_ROOT / "data/synthetic/02e_aya8b_directives_curated_v2.jsonl")
    print(
        f"[Step 4] Merged Stage 2 files: sol2={len(rescued_sol2)}, tiny_correct={len(rescued_tiny_ok)}, "
        f"flawed={len(rescued_flawed)}, directives={len(rescued_directives)} "
        f"(approvals={len(approvals)}, challenges={len(challenges)})"
    )

    # 5. Run Stage 2 Exporter into data/sft_v2 and normalize \n\n + checksums
    export_summary = export_all_stage2_sft_datasets(
        e1_pool_path=str(PROJECT_ROOT / "data/intermediate/01a_manual_deterministic_only_pool_v2.jsonl"),
        e2_pool_path=str(PROJECT_ROOT / "data/intermediate/01b_e2_books_plus_synthetic_v2.jsonl"),
        e3_pool_path=str(PROJECT_ROOT / "data/processed/canonical_train_v2.jsonl"),
        validation_path=str(PROJECT_ROOT / "data/splits/validation_v2.jsonl"),
        test_path=str(PROJECT_ROOT / "data/splits/test_v2.jsonl"),
        curated_sol2_path=str(PROJECT_ROOT / "data/synthetic/02d_curated_solution2_v2.jsonl"),
        tiny_aya_correct_path=str(PROJECT_ROOT / "data/synthetic/02d_tiny_aya_correct_curated_v2.jsonl"),
        surviving_flawed_path=str(PROJECT_ROOT / "data/synthetic/02d_surviving_flawed_v2.jsonl"),
        directives_input_path=str(PROJECT_ROOT / "data/synthetic/02e_aya8b_directives_curated_v2.jsonl"),
        sft_output_root=str(PROJECT_ROOT / "data/sft_v2"),
        max_repeated_ref_hints=150,
        require_clean_e2_e3_e4=True,
        min_e3_e4_chars=400,
    )

    sft_root = PROJECT_ROOT / "data/sft_v2"
    _normalize_exported_prompts_and_checksums(sft_root)

    # 6. Strict Pre-Upload Assertions
    audit_report: dict[str, Any] = {}
    for folder in HF_V2_REPOS:
        ds_dir = sft_root / folder
        train_rows = _load_jsonl_dicts(ds_dir / "train.jsonl")
        assert len(train_rows) > 0, f"{folder}/train.jsonl is empty!"

        # Check 0 leakage by problem_id across all train rows
        train_pids = {str(r.get("problem_id") or "") for r in train_rows}
        overlap = train_pids & heldout_pids
        assert not overlap, f"Leakage in {folder}/train.jsonl: {len(overlap)} heldout problem_ids found!"

        split_stats: dict[str, Any] = {"train_count": len(train_rows)}
        for split_name in ("train", "validation", "test"):
            sp_path = ds_dir / f"{split_name}.jsonl"
            if not sp_path.is_file():
                continue
            sp_rows = _load_jsonl_dicts(sp_path)
            for idx, r in enumerate(sp_rows):
                pr = str(r.get("prompt") or "")
                assert pr.endswith("\n\n"), f"{folder}/{split_name}.jsonl row {idx} prompt missing trailing \\n\\n!"

            if split_name in ("validation", "test"):
                assert len(sp_rows) == 60, f"{folder}/{split_name}.jsonl expected 60 rows, got {len(sp_rows)}"
                for idx, r in enumerate(sp_rows):
                    ref = str(
                        r.get("reference_solution")
                        or r.get("target_output")
                        or r.get("output")
                        or r.get("completion")
                        or ""
                    )
                    ans = extract_answer_line(ref)
                    assert ans, f"{folder}/{split_name}.jsonl row {idx} missing ANSWER: sentinel!"
                    assert score_solver_output(ref, ref).is_equivalent, (
                        f"{folder}/{split_name}.jsonl row {idx} ({r.get('problem_id')}) failed self-score!"
                    )
                split_stats[f"{split_name}_count"] = len(sp_rows)

        if folder in ("solver_e3_canonical_llm", "solver_e4_adversarial"):
            for idx, r in enumerate(train_rows):
                comp = str(r.get("target_output") or r.get("output") or r.get("completion") or "")
                assert len(comp) >= 400, f"{folder}/train.jsonl row {idx} len(target_output)={len(comp)} < 400!"
                ans = extract_answer_line(comp)
                assert ans, f"{folder}/train.jsonl row {idx} missing ANSWER: sentinel!"
                assert is_clean_symbolic_or_numeric_answer(ans), (
                    f"{folder}/train.jsonl row {idx} unclean ANSWER: {ans!r}"
                )

        if folder == "solver_e4_adversarial":
            comp_counts = Counter(
                str(r.get("target_output") or r.get("output") or r.get("completion") or "") for r in train_rows
            )
            max_dup = max(comp_counts.values())
            assert max_dup <= 3, f"solver_e4_adversarial max target multiplicity is {max_dup} (> 3)!"
            turn_dist = dict(sorted(Counter(r.get("turn") for r in train_rows).items()))
            sol2_rows = sum(
                1
                for r in train_rows
                if "sol2" in str(r.get("source_candidate_ids") or "") or "sol2" in str(r.get("trajectory_id") or "")
            )
            split_stats["turn_dist"] = turn_dist
            split_stats["sol2_trajectory_rows"] = sol2_rows
            split_stats["max_completion_multiplicity"] = max_dup

        if folder == "verifier_e4_curated":
            verdict_ok = 0
            app_n = 0
            for r in train_rows:
                c = str(r.get("target_output") or r.get("output") or r.get("completion") or "").strip()
                first_line = c.splitlines()[0].strip() if c else ""
                if first_line.startswith("[[APPROVE]]") or c.endswith("[[APPROVE]]"):
                    app_n += 1
                    verdict_ok += 1
                elif first_line.startswith("[[CHALLENGE]]") or c.endswith("[[CHALLENGE]]"):
                    verdict_ok += 1
            assert verdict_ok == len(train_rows), (
                f"verifier_e4_curated: only {verdict_ok}/{len(train_rows)} have clean [[APPROVE]]/[[CHALLENGE]] verdict!"
            )
            app_rate = app_n / len(train_rows)
            assert 0.15 <= app_rate <= 0.35, f"verifier_e4_curated approval rate {app_rate:.4f} outside [0.15, 0.35]!"
            split_stats["approve_count"] = app_n
            split_stats["challenge_count"] = len(train_rows) - app_n
            split_stats["approve_rate"] = round(app_rate, 4)

        split_stats["domain_dist"] = dict(sorted(Counter(str(r.get("domain") or "") for r in train_rows).items()))
        audit_report[folder] = split_stats

    return {
        "ultra_clean_pool_count": len(ultra_clean_map),
        "e1_pool_count": len(e1_pool),
        "e2_pool_count": len(e2_pool),
        "e3_pool_count": len(e3_pool),
        "export_summary": export_summary.to_dict(),
        "assertions": audit_report,
    }


def upload_v2_to_hf(token: str) -> dict[str, str]:
    """Upload all 5 verified rescued datasets under `data/sft_v2/` to HuggingFace."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    sft_root = PROJECT_ROOT / "data/sft_v2"
    uploaded: dict[str, str] = {}
    for folder, repo_id in HF_V2_REPOS.items():
        local_dir = sft_root / folder
        print(f"[HF Upload] Uploading {local_dir} -> {repo_id} ...")
        info = api.upload_folder(
            folder_path=str(local_dir),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Rescue v2 clean export: {folder} (0 leakage, BPE \\n\\n boundary, clean >=400ch derivations)",
        )
        uploaded[repo_id] = str(getattr(info, "commit_url", info))
    return uploaded


def main() -> int:
    parser = argparse.ArgumentParser(description="Rescue and verify all 5 v2 SFT datasets and upload to HuggingFace.")
    parser.add_argument("--no-upload", action="store_true", help="Run local rescue and assertions only without uploading to HF.")
    args = parser.parse_args()

    result = build_and_verify_rescued_v2()
    if not args.no_upload:
        token = _resolve_hf_token()
        if not token:
            raise SystemExit("HF_TOKEN missing in environment and .env — aborting upload.")
        result["hf_uploads"] = upload_v2_to_hf(token)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
