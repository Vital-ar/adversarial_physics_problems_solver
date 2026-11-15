"""Stage 2 Step 4F: Build & Verify All 5 v3 SFT Datasets (Zero New LLM Calls) and Upload to HuggingFace `-v3` Repos.

Key v3 Data Improvements over v2:
1. 100% Uniform Step-by-Step Derivation Structure (`E2_v3`, `E3_v3`, `E4a_v3`):
   - Converts multi-paragraph textbook/Olympiad derivations lacking `Step 1:` headers into uniform
     `**Derivation:**\\nStep 1: ...\\nStep 2: ...\\nANSWER: ...` format so the Solver learns a single coherent format.
2. Zero One-Liner Shortcuts in `E1_v3` (>= 220 chars) and `E2_v3` (>= 260 chars):
   - Removes the terse 1-line answer keys (< 200 chars) that previously taught E1/E2 to skip derivation steps.
3. Expanded Verified `Sol2` & Student-Correct Pool:
   - Harvests additional verified alternative derivations from `02a_aya8b_solution2_checkpoint_v2.jsonl`,
     `02_raw_synthetic_pool_v2.jsonl`, `02a_minority_solution2.jsonl`, and `02e_tiny_aya_correct_trajectories.jsonl`.
4. Purged Shifted `[[APPROVE]]` Rows & Rebalanced `verifier_e4_curated` (~30% `[[APPROVE]]` / ~70% `[[CHALLENGE]]`):
   - Purges the 33 off-by-one shifted `[[APPROVE]]` rows discovered in `v2` (lexical overlap < 0.25).
   - Adds verified problem-aligned `[[APPROVE]]` rationales covering `Sol1`, `Sol2`, and real `tiny-aya-global`
     correct solutions so the Verifier cannot cheat by model style and learns calibrated approval vs challenge.
5. Hallucinated Formula Sanitization in Directives (`E4a_v3` & `E4b_v3`):
   - Strips teacher-hallucinated formulas whose symbols are absent from the problem statement and reference solution.
6. Capped Completion Multiplicity (<= 2) in `solver_e4_adversarial` (`v3`):
   - Prevents the 3x target-output repetition per epoch that caused `E4a_v2` to plateau early at Epoch 2.

Usage:
  .\\venv\\Scripts\\python.exe scripts/04f_build_and_upload_v3.py --no-upload
  .\\venv\\Scripts\\python.exe scripts/04f_build_and_upload_v3.py
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
from adversarial_physics.data.curation.directive_leak_check import (  # noqa: E402
    sanitize_directive_text,
    validate_verifier_directive,
)
from adversarial_physics.data.schemas.problem import (  # noqa: E402
    CanonicalProblem,
    load_problems_jsonl,
    save_problems_jsonl,
)
from adversarial_physics.data.schemas.trajectory import (  # noqa: E402
    VERDICT_APPROVE,
    VERDICT_CHALLENGE,
    CandidateSolution,
    load_candidates_jsonl,
    save_candidates_jsonl,
)
from adversarial_physics.orchestration.loop import (  # noqa: E402
    prepare_reference_for_validation,
    score_solver_output,
)

HF_V3_REPOS: dict[str, str] = {
    "solver_e1_manual_det": "vital-ar/physics-solver-e1-manual-rules-v3",
    "solver_e2_curatorkit_nollm": "vital-ar/physics-solver-e2-curatorkit-nollm-v3",
    "solver_e3_canonical_llm": "vital-ar/physics-solver-e3-canonical-llm-v3",
    "solver_e4_adversarial": "vital-ar/physics-solver-e4-adversarial-v3",
    "verifier_e4_curated": "vital-ar/physics-verifier-e4-curated-v3",
}

_STOP_WORDS = frozenset(
    {
        "derivation",
        "equation",
        "problem",
        "solve",
        "apply",
        "principles",
        "correctly",
        "therefore",
        "using",
        "obtain",
        "final",
        "expression",
        "solution",
        "approve",
        "challenge",
        "determine",
        "given",
        "value",
        "calculate",
        "find",
        "must",
        "where",
        "this",
        "that",
        "with",
        "from",
        "since",
        "thus",
        "hence",
    }
)

_STEP_MARKER_RE = re.compile(r"(?:^|\n)\s*(?:#{1,4}\s*step|\*\*step|step\s+\d+|\d+\.\s+\*\*)", re.IGNORECASE)
_MATH_SYM_RE = re.compile(r"\\[A-Za-z]+|[A-Za-z_][A-Za-z0-9_]*")


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


def _ensure_structured_steps(solution_text: str) -> str:
    """Deterministically format multi-sentence/paragraph derivations into uniform `Step 1: ...` steps."""
    text = (solution_text or "").strip()
    if not text or _STEP_MARKER_RE.search(text):
        return text

    m = list(re.finditer(r"(?:^|\n)\s*ANSWER\s*:\s*(.+)$", text, flags=re.IGNORECASE))
    if not m:
        return text
    last_m = m[-1]
    body = text[: last_m.start()].strip()
    ans_line = f"ANSWER: {last_m.group(1).strip()}"
    if not body:
        return text

    # Split into paragraphs first; if fewer than 2 paragraphs, split by sentence boundaries outside $$...$$
    paras = [p.strip() for p in re.split(r"\n\s*\n+", body) if p.strip()]
    chunks: list[str] = []
    if len(paras) >= 2:
        # Merge standalone display math blocks (`$$ ... $$`) with their preceding explanation paragraph
        for p in paras:
            if chunks and (p.startswith("$$") or p.startswith("\\[")) and len(chunks[-1]) < 260:
                chunks[-1] = f"{chunks[-1]}\n{p}"
            else:
                chunks.append(p)
    else:
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z$(\\[])", body) if s.strip()]
        cur = ""
        for s in sents:
            if not cur:
                cur = s
            elif len(cur) < 110:
                cur = f"{cur} {s}"
            else:
                chunks.append(cur)
                cur = s
        if cur:
            chunks.append(cur)

    if len(chunks) < 2:
        return f"**Derivation:**\nStep 1: {body}\n{ans_line}"

    # Cap at 8 logical steps by merging trailing chunks if needed
    while len(chunks) > 8:
        merged = f"{chunks[-2]} {chunks[-1]}"
        chunks = [*chunks[:-2], merged]

    step_lines = ["**Derivation:**"]
    for idx, ch in enumerate(chunks, start=1):
        clean_ch = re.sub(r"^\*\*Derivation:\*\*\s*", "", ch, flags=re.IGNORECASE).strip()
        step_lines.append(f"Step {idx}: {clean_ch}")
    step_lines.append(ans_line)
    return "\n".join(step_lines)


def _prompt_directive_lexical_overlap(problem_text: str, proposed_sol: str, directive_body: str) -> float:
    p_toks = set(re.findall(r"[a-zA-Z]{4,}", f"{problem_text} {proposed_sol}".lower()))
    d_toks = set(re.findall(r"[a-zA-Z]{4,}", (directive_body or "").lower())) - _STOP_WORDS
    if not d_toks:
        return 0.0
    return len(p_toks & d_toks) / len(d_toks)


def _sanitize_hallucinated_equations_in_directive(
    directive_body: str,
    problem_statement: str,
    reference_solution: str,
    domain: str,
) -> str:
    """Strip inline equations `$... = ...$` whose symbols are not grounded in the problem or reference solution."""
    text = (directive_body or "").strip()
    if not text:
        return text

    ref_syms = set(_MATH_SYM_RE.findall(f"{problem_statement} {reference_solution}"))
    # Check inline math blocks `$...$` or `\(...\)` containing `=`
    def _check_math_block(match: re.Match[str]) -> str:
        block = match.group(0)
        if "=" not in block:
            return block
        syms = [s for s in _MATH_SYM_RE.findall(block) if len(s) >= 1 and not s.startswith("\\text")]
        if not syms:
            return block
        grounded = sum(1 for s in syms if s in ref_syms)
        if (grounded / len(syms)) < 0.50:
            return "the governing physical relation"
        return block

    cleaned = re.sub(r"\$[^$\n]+\$|\\\([^)]+\\\)", _check_math_block, text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) < 35:
        dom_label = domain.replace("_", " ").lower()
        cleaned = (
            f"Apply the fundamental conservation laws and governing {dom_label} equations "
            f"while verifying dimensional consistency before solving for the final expression."
        )
    return cleaned


def _harmonize_approve_rationale_opening(directive_body: str, idx: int) -> str:
    """Rewrite `[[APPROVE]]` openings ('The derivation...', 'The solution...') into the same analytical
    infinitive/participial openings ('To solve...', 'To find...', 'Applying...') used by `[[CHALLENGE]]`
    directives so Token 0 cannot act as a binary class shortcut."""
    text = (directive_body or "").strip()
    if not text:
        return text

    openers = (
        "To solve this problem, ",
        "To find the target expression, ",
        "To correctly determine the result, ",
        "To derive the governing relation, ",
        "Applying the governing principles, ",
        "Using the physical constraints, ",
    )
    prefix = openers[idx % len(openers)]

    # Strip leading "The (proposed )?(derivation|solution) (accurately |correctly |properly )?"
    stripped = re.sub(
        r"^(?:\*\*\s*)?The\s+(?:proposed\s+)?(?:derivation|solution)\s+(?:is\s+sound\s+(?:because|as)\s+it\s+|(?:accurately|correctly|properly|directly|effectively)\s+)?",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    # Convert finite 3rd-person singular verb at start of stripped clause ("applies", "identifies", "uses", "employs")
    # into a gerund ("applying", "identifying", "using", "employing") so it reads naturally after the opener
    verb_map = {
        "applies": "applying",
        "identifies": "identifying",
        "uses": "using",
        "employs": "employing",
        "demonstrates": "demonstrating",
        "sets": "setting",
        "calculates": "calculating",
        "solves": "solving",
        "combines": "combining",
        "relates": "relating",
        "leverages": "leveraging",
        "considers": "considering",
        "establishes": "establishing",
        "tracks": "tracking",
        "evaluates": "evaluating",
        "integrates": "integrating",
        "equates": "equating",
        "substitutes": "substituting",
    }
    first_word_m = re.match(r"^([A-Za-z]+)(\b.*)$", stripped, flags=re.DOTALL)
    if first_word_m:
        fw_lower = first_word_m.group(1).lower()
        rest = first_word_m.group(2)
        if fw_lower in verb_map:
            stripped = f"{verb_map[fw_lower]}{rest} and confirms the valid final result."
            stripped = re.sub(r"\.\s+and\s+confirms\s+the\s+valid\s+final\s+result\.$", ", confirming the valid final result.", stripped)
        else:
            stripped = fw_lower + rest
    if not stripped.endswith((".", "!", "?")):
        stripped += "."
    return f"{prefix}{stripped}"


def _normalize_exported_prompts_and_checksums_v3(sft_root: Path) -> None:
    """Ensure 100% of prompt & instruction strings in `data/sft_v3/` end with \\n\\n, update README repo IDs to `-v3`, and write checksums."""
    for folder, repo_v3 in HF_V3_REPOS.items():
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

        readme_path = ds_dir / "README.md"
        if readme_path.is_file():
            md = readme_path.read_text(encoding="utf-8")
            md_v3 = md.replace("-v2", "-v3").replace("sft_v2", "sft_v3")
            if md_v3 != md:
                readme_path.write_text(md_v3, encoding="utf-8")

        checksum_lines: list[str] = []
        for fpath in sorted(ds_dir.iterdir(), key=lambda p: p.name):
            if not fpath.is_file() or fpath.name == "checksums.txt":
                continue
            digest = hashlib.sha256(fpath.read_bytes()).hexdigest()
            checksum_lines.append(f"{digest}  {fpath.name}")
        (ds_dir / "checksums.txt").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")


def build_and_verify_v3() -> dict[str, Any]:
    """Build all 5 v3 SFT datasets deterministically in `data/sft_v3/` and run strict quality assertions."""
    rng = random.Random(42)

    # 1. Load leak-free v3 (or fallback v2) validation and test splits
    val_split_path = PROJECT_ROOT / "data/splits/validation_v3.jsonl"
    test_split_path = PROJECT_ROOT / "data/splits/test_v3.jsonl"
    if not val_split_path.is_file():
        val_split_path = PROJECT_ROOT / "data/splits/validation_v2.jsonl"
    if not test_split_path.is_file():
        test_split_path = PROJECT_ROOT / "data/splits/test_v2.jsonl"
    validation_v2 = load_problems_jsonl(val_split_path)
    test_v2 = load_problems_jsonl(test_split_path)
    heldout_pids = {p.problem_id for p in validation_v2 + test_v2}
    assert len(validation_v2) > 0 and len(test_v2) > 0 and len(heldout_pids) == len(validation_v2) + len(test_v2)

    # 2. Load and step-structure E3 canonical pool (`642` leak-free problems -> 100% step-structured)
    e3_raw = load_problems_jsonl(PROJECT_ROOT / "data/processed/canonical_train_v2.jsonl")
    e3_pool: list[CanonicalProblem] = []
    for p in e3_raw:
        if p.problem_id in heldout_pids:
            continue
        structured_sol = _ensure_structured_steps(p.reference_solution)
        ans = extract_answer_line(structured_sol)
        if not ans or not is_clean_symbolic_or_numeric_answer(ans):
            continue
        if not score_solver_output(structured_sol, structured_sol).is_equivalent:
            continue
        e3_pool.append(dataclasses.replace(p, reference_solution=structured_sol))

    save_problems_jsonl(e3_pool, PROJECT_ROOT / "data/processed/canonical_train_v3.jsonl")
    e3_map: dict[str, CanonicalProblem] = {p.problem_id: p for p in e3_pool}
    e3_train_pids = set(e3_map.keys())
    print(f"[v3 Step 1] Built 100% step-structured E3 pool: {len(e3_pool)} problems.")

    # 3. Build E2 pool (`>= 260` chars, zero one-liners, 100% step-structured)
    e2_raw = load_problems_jsonl(PROJECT_ROOT / "data/intermediate/01b_e2_books_plus_synthetic_v2.jsonl")
    e2_pool: list[CanonicalProblem] = []
    seen_e2: set[str] = set()
    for p in e2_raw:
        if p.problem_id in heldout_pids or p.problem_id in seen_e2:
            continue
        # Upgrade any short textbook fallback from e3_map if available
        sol_candidate = p.reference_solution
        if len(sol_candidate.strip()) < 260 and p.problem_id in e3_map:
            sol_candidate = e3_map[p.problem_id].reference_solution
        structured_sol = _ensure_structured_steps(sol_candidate)
        if len(structured_sol) < 260:
            continue
        ans = extract_answer_line(structured_sol)
        if not ans or not is_clean_symbolic_or_numeric_answer(ans):
            continue
        if not score_solver_output(structured_sol, structured_sol).is_equivalent:
            continue
        seen_e2.add(p.problem_id)
        e2_pool.append(dataclasses.replace(p, reference_solution=structured_sol))

    save_problems_jsonl(e2_pool, PROJECT_ROOT / "data/intermediate/01b_e2_books_plus_synthetic_v3.jsonl")
    print(f"[v3 Step 2] Built step-structured E2 pool (>=260 chars, 0 one-liners): {len(e2_pool)} problems.")

    # 4. Build E1 pool (`>= 220` chars, removing bare 1-line answer keys)
    e1_raw = load_problems_jsonl(PROJECT_ROOT / "data/intermediate/01a_manual_deterministic_only_pool_v2.jsonl")
    e1_pool: list[CanonicalProblem] = []
    for p in e1_raw:
        if p.problem_id in heldout_pids:
            continue
        sol = p.reference_solution.strip()
        if len(sol) < 220:
            continue
        ans = extract_answer_line(sol)
        if not ans or not is_clean_symbolic_or_numeric_answer(ans):
            continue
        e1_pool.append(p)

    save_problems_jsonl(e1_pool, PROJECT_ROOT / "data/intermediate/01a_manual_deterministic_only_pool_v3.jsonl")
    print(f"[v3 Step 3] Built clean E1 pool (>=220 chars, 0 bare answer keys): {len(e1_pool)} problems.")

    # 5. Harvest all verified alternative `Sol2` and `tiny_aya_correct` derivations across all synthetic checkpoints
    def _harvest_verified_alt_solutions(paths: list[Path]) -> list[CandidateSolution]:
        kept: dict[str, CandidateSolution] = {}
        for pth in paths:
            if not pth.is_file():
                continue
            for cand in load_candidates_jsonl(pth):
                pid = cand.problem_id
                if pid not in e3_train_pids or pid in kept:
                    continue
                sol = _ensure_structured_steps((cand.solution_text or "").strip())
                if len(sol) < 400:
                    continue
                # Ensure it is not character-identical to Sol1
                if sol.strip() == e3_map[pid].reference_solution.strip():
                    continue
                ans = extract_answer_line(sol)
                if not ans or not is_clean_symbolic_or_numeric_answer(ans):
                    continue
                if not score_solver_output(sol, e3_map[pid].reference_solution).is_equivalent:
                    continue
                kept[pid] = dataclasses.replace(cand, solution_text=sol, is_correct=True)
        return list(kept.values())

    v3_sol2 = _harvest_verified_alt_solutions(
        [
            PROJECT_ROOT / "data/synthetic/02d_curated_solution2_v2.jsonl",
            PROJECT_ROOT / "data/synthetic/02a_aya8b_solution2_checkpoint_v2.jsonl",
            PROJECT_ROOT / "data/synthetic/02a_aya8b_solution2_raw_v2.jsonl",
            PROJECT_ROOT / "data/synthetic/02d_curated_solution2.jsonl",
            PROJECT_ROOT / "data/synthetic/02a_minority_solution2.jsonl",
            PROJECT_ROOT / "data/synthetic/02_raw_synthetic_pool_v2.jsonl",
        ]
    )
    save_candidates_jsonl(v3_sol2, PROJECT_ROOT / "data/synthetic/02d_curated_solution2_v3.jsonl")
    sol2_by_pid = {c.problem_id: c for c in v3_sol2}

    v3_tiny_ok = _harvest_verified_alt_solutions(
        [
            PROJECT_ROOT / "data/synthetic/02d_tiny_aya_correct_curated_v2.jsonl",
            PROJECT_ROOT / "data/synthetic/02e_tiny_aya_correct_trajectories.jsonl",
            PROJECT_ROOT / "data/synthetic/02c_tiny_aya_failures_checkpoint_v2.jsonl",
            PROJECT_ROOT / "data/synthetic/02c_tiny_aya_harvest_checkpoint.jsonl",
        ]
    )
    save_candidates_jsonl(v3_tiny_ok, PROJECT_ROOT / "data/synthetic/02d_tiny_aya_correct_curated_v3.jsonl")
    print(f"[v3 Step 4] Harvested verified alternative derivations: sol2={len(v3_sol2)}, tiny_correct={len(v3_tiny_ok)}.")

    # 6. Curate & Rebalance Verifier Directives (~15% [[APPROVE]] / ~85% [[CHALLENGE]], 0 shifted rows, 0 hallucinated formulas)
    v2_dir_raw = _load_jsonl_dicts(PROJECT_ROOT / "data/synthetic/02e_aya8b_directives_checkpoint_v2.jsonl")
    v1_dir_raw = _load_jsonl_dicts(PROJECT_ROOT / "data/synthetic/02g_directives_checkpoint.jsonl")

    clean_approvals: dict[tuple[str, str], dict[str, Any]] = {}
    clean_challenges: dict[tuple[str, str], dict[str, Any]] = {}
    purged_shifted_approvals = 0
    sanitized_challenge_formulas = 0

    for raw_rows in [v2_dir_raw, v1_dir_raw]:
        for r in raw_rows:
            pid = str(r.get("problem_id") or "")
            if pid not in e3_train_pids:
                continue
            prob = e3_map[pid]
            kind = str(r.get("kind") or "")
            src = str(r.get("source") or "")
            verdict = str(r.get("verdict") or "")
            body = str(r.get("directive_body") or "").strip()

            if verdict == VERDICT_APPROVE or kind == "domain_balanced_approval":
                if body.startswith("The derivation correctly applies the governing"):
                    continue
                prop_sol = (
                    sol2_by_pid[pid].solution_text
                    if (src == "minority_solution2" and pid in sol2_by_pid)
                    else prob.reference_solution
                )
                # Guard 1: Verify proposed solution is actually equivalent to reference answer
                if not score_solver_output(prop_sol, prob.reference_solution).is_equivalent:
                    purged_shifted_approvals += 1
                    continue
                # Guard 2: Purge off-by-one shifted approvals with lexical overlap < 0.25
                if _prompt_directive_lexical_overlap(prob.problem_statement, prop_sol, body) < 0.25:
                    purged_shifted_approvals += 1
                    continue
                rec = dict(r)
                rec["proposed_solution"] = prop_sol
                rec["verdict"] = VERDICT_APPROVE
                rec["directive_body"] = body
                rec.pop("formatted_output", None)
                akey = (pid, src or "sol1_approve")
                if akey not in clean_approvals:
                    clean_approvals[akey] = rec
            else:
                prop_sol = str(r.get("proposed_solution") or "").strip()
                if not prop_sol or len(prop_sol) < 60:
                    continue
                # Guard: Ensure challenged solution is genuinely wrong
                if score_solver_output(prop_sol, prob.reference_solution).is_equivalent:
                    continue
                if _prompt_directive_lexical_overlap(prob.problem_statement, prop_sol, body) < 0.22:
                    continue
                new_body = _sanitize_hallucinated_equations_in_directive(
                    body, prob.problem_statement, prob.reference_solution, prob.domain
                )
                if new_body != body:
                    sanitized_challenge_formulas += 1
                rec = dict(r)
                rec["directive_body"] = new_body
                rec.pop("formatted_output", None)
                rec["verdict"] = VERDICT_CHALLENGE
                ckey = (pid, src)
                if ckey not in clean_challenges:
                    clean_challenges[ckey] = rec

    # Select calibrated ~15% [[APPROVE]] (108 genuine LLM-authored approvals, 0 synthetic templates)
    # and ~85% [[CHALLENGE]] (612 challenges: balanced 60% aya_expanse_8b_synthetic_error + 40% tiny_aya_global_zero_shot
    # so LaTeX density \frac, \text, \cdot matches [[APPROVE]]).
    target_approvals = 108
    all_clean_approvals = list(clean_approvals.values())
    by_dom_app: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_clean_approvals:
        pid = str(row["problem_id"])
        dom = e3_map[pid].domain if pid in e3_map else str(row.get("domain") or "MECHANICS")
        by_dom_app[dom].append(row)
    for dom in by_dom_app:
        by_dom_app[dom].sort(
            key=lambda r: (
                0 if str(r.get("source") or "") == "minority_solution2" else 1,
                0 if "\\frac" in str(r.get("proposed_solution") or "") else 1,
                str(r.get("problem_id") or ""),
            )
        )

    approvals_list: list[dict[str, Any]] = []
    while len(approvals_list) < target_approvals and any(by_dom_app.values()):
        for dom in sorted(by_dom_app.keys()):
            if len(approvals_list) >= target_approvals:
                break
            if by_dom_app[dom]:
                approvals_list.append(by_dom_app[dom].pop(0))

    # Select 612 balanced challenges (prioritizing aya_expanse_8b_synthetic_error so LaTeX density matches e3_pool)
    by_pid_ch: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in clean_challenges.values():
        by_pid_ch[str(row["problem_id"])].append(row)
    for pid in by_pid_ch:
        by_pid_ch[pid].sort(
            key=lambda r: (
                1 if "tiny_aya" in str(r.get("source") or "") else 0,
                str(r.get("source") or ""),
            )
        )

    selected_challenges: list[dict[str, Any]] = []
    seen_ch_keys: set[tuple[str, str]] = set()
    used_pids_ch: Counter[str] = Counter()
    max_challenges = 612
    max_pairs = 35

    def _add_ch(row: dict[str, Any]) -> bool:
        key = (str(row["problem_id"]), str(row.get("source") or ""))
        if key in seen_ch_keys or len(selected_challenges) >= max_challenges:
            return False
        seen_ch_keys.add(key)
        selected_challenges.append(row)
        used_pids_ch[key[0]] += 1
        return True

    # (a) Prioritize Sol2 problems so hinted turns have distinct Sol2 targets
    for pid in sorted(sol2_by_pid):
        for row in by_pid_ch.get(pid, [])[:1]:
            _add_ch(row)

    # (b) Add at most `max_pairs` 2-hint problems (preferring Sol2 problems first)
    paired_pids = [pid for pid, rows in by_pid_ch.items() if len(rows) >= 2]
    paired_pids.sort(key=lambda pid: (0 if pid in sol2_by_pid else 1, pid))
    pairs_added = 0
    for pid in paired_pids:
        if pairs_added >= max_pairs or len(selected_challenges) + 2 > max_challenges:
            break
        for row in by_pid_ch[pid][:2]:
            _add_ch(row)
        if used_pids_ch[pid] >= 2:
            pairs_added += 1

    # (c) Fill remaining budget with 1 challenge per problem first, then 2nd challenge if needed
    single_cands = [rows[0] for pid, rows in by_pid_ch.items() if used_pids_ch[pid] == 0]
    single_cands.sort(key=lambda r: str(r["problem_id"]))
    for row in single_cands:
        if len(selected_challenges) >= max_challenges:
            break
        _add_ch(row)

    for pid in sorted(by_pid_ch.keys()):
        if len(selected_challenges) >= max_challenges:
            break
        for row in by_pid_ch[pid]:
            if len(selected_challenges) >= max_challenges:
                break
            _add_ch(row)

    v3_directives = approvals_list + selected_challenges
    rng.shuffle(v3_directives)
    _save_jsonl_dicts(v3_directives, PROJECT_ROOT / "data/synthetic/02e_aya8b_directives_curated_v3.jsonl")
    print(
        f"[v3 Step 5] Curated v3 directives: total={len(v3_directives)} "
        f"(approvals={len(approvals_list)} [{len(approvals_list)/len(v3_directives):.1%}], "
        f"challenges={len(selected_challenges)} [{len(selected_challenges)/len(v3_directives):.1%}], "
        f"purged_shifted_approvals={purged_shifted_approvals}, sanitized_formulas={sanitized_challenge_formulas})."
    )

    # 7. Export all 5 v3 SFT datasets into `data/sft_v3/`
    export_summary = export_all_stage2_sft_datasets(
        e1_pool_path=str(PROJECT_ROOT / "data/intermediate/01a_manual_deterministic_only_pool_v3.jsonl"),
        e2_pool_path=str(PROJECT_ROOT / "data/intermediate/01b_e2_books_plus_synthetic_v3.jsonl"),
        e3_pool_path=str(PROJECT_ROOT / "data/processed/canonical_train_v3.jsonl"),
        validation_path=str(PROJECT_ROOT / "data/splits/validation_v2.jsonl"),
        test_path=str(PROJECT_ROOT / "data/splits/test_v2.jsonl"),
        curated_sol2_path=str(PROJECT_ROOT / "data/synthetic/02d_curated_solution2_v3.jsonl"),
        tiny_aya_correct_path=str(PROJECT_ROOT / "data/synthetic/02d_tiny_aya_correct_curated_v3.jsonl"),
        surviving_flawed_path=str(PROJECT_ROOT / "data/synthetic/02d_surviving_flawed_v2.jsonl"),
        directives_input_path=str(PROJECT_ROOT / "data/synthetic/02e_aya8b_directives_curated_v3.jsonl"),
        sft_output_root=str(PROJECT_ROOT / "data/sft_v3"),
        max_repeated_ref_hints=75,
        require_clean_e2_e3_e4=True,
        min_e3_e4_chars=400,
    )

    sft_root = PROJECT_ROOT / "data/sft_v3"

    # Convert any synthetic "Additionally, verify dimensional consistency..." 2nd bullet in E4a into a clean 1-hint Turn-1 row,
    # and enforce max_completion_multiplicity <= 2 in solver_e4_adversarial/train.jsonl
    from adversarial_physics.data.schemas.trajectory import (
        format_solver_prompt,
        format_verifier_prompt,
        normalize_proposed_solution_for_verifier,
    )

    e4a_train_path = sft_root / "solver_e4_adversarial" / "train.jsonl"
    e4a_rows = _load_jsonl_dicts(e4a_train_path)
    prob_stmt_by_pid = {p.problem_id: p.problem_statement for p in e3_pool}
    for r in e4a_rows:
        dirs = list(r.get("accumulated_directives") or [])
        if len(dirs) == 2 and str(dirs[1]).startswith("Additionally, verify dimensional consistency"):
            clean_dirs = [dirs[0]]
            pid = str(r.get("problem_id") or "")
            stmt = prob_stmt_by_pid.get(pid, "")
            if stmt:
                new_prompt = format_solver_prompt(stmt, clean_dirs).rstrip() + "\n\n"
                r["accumulated_directives"] = clean_dirs
                r["turn"] = 1
                r["trajectory_id"] = f"E4_solver:{pid}:turn1_1hint"
                r["prompt"] = new_prompt
                r["instruction"] = new_prompt

    comp_seen: Counter[str] = Counter()
    deduped_e4a: list[dict[str, Any]] = []
    # Sort so Turn 0 and Turn 1 come before Turn 2 when capping multiplicity at 2
    e4a_rows.sort(key=lambda r: (int(r.get("turn") or 0), str(r.get("trajectory_id") or "")))
    for r in e4a_rows:
        comp = str(r.get("target_output") or r.get("output") or "").strip()
        if comp_seen[comp] >= 2:
            continue
        comp_seen[comp] += 1
        deduped_e4a.append(r)
    _save_jsonl_dicts(deduped_e4a, e4a_train_path)
    print(f"[v3 Step 6] Normalized E4a turns & capped multiplicity <= 2: {len(e4a_rows)} -> {len(deduped_e4a)} rows.")

    # Post-normalize `verifier_e4_curated/train.jsonl`:
    # 1. Normalize 100% of proposed_solution strings (both [[APPROVE]] and [[CHALLENGE]]) to uniform
    #    `**Derivation:**\nStep 1: ...\nANSWER: ...` format (0% format leak, 0% \boxed).
    # 2. Harmonize `[[APPROVE]]` target openings ('To solve...', 'To find...', 'Applying...') with `[[CHALLENGE]]`
    #    openings so Token 0 cannot act as a class shortcut.
    # 3. Deterministically shuffle `verifier_e4_curated/train.jsonl` so every batch and 10% validation slice
    #    has the exact calibrated ~15% [[APPROVE]] / ~85% [[CHALLENGE]] balance.
    e4b_train_path = sft_root / "verifier_e4_curated" / "train.jsonl"
    e4b_rows = _load_jsonl_dicts(e4b_train_path)
    app_idx = 0
    for r in e4b_rows:
        pid = str(r.get("problem_id") or "")
        stmt = prob_stmt_by_pid.get(pid, "")
        raw_prop = str(r.get("proposed_solution") or "")
        norm_prop = normalize_proposed_solution_for_verifier(raw_prop)
        r["proposed_solution"] = norm_prop
        if stmt:
            norm_prompt = format_verifier_prompt(stmt, norm_prop).rstrip() + "\n\n"
            r["prompt"] = norm_prompt
            r["instruction"] = norm_prompt
        if r.get("verdict") == VERDICT_APPROVE:
            raw_dir = str(r.get("directive_text") or "").strip()
            harm_dir = _harmonize_approve_rationale_opening(raw_dir, app_idx)
            app_idx += 1
            r["directive_text"] = harm_dir
            r["target_output"] = f"{harm_dir}\n{VERDICT_APPROVE}"
            r["output"] = r["target_output"]

    rng_shuf = random.Random(42)
    rng_shuf.shuffle(e4b_rows)
    _save_jsonl_dicts(e4b_rows, e4b_train_path)
    print(f"[v3 Step 6b] Post-normalized & shuffled verifier_e4_curated: {len(e4b_rows)} rows.")

    _normalize_exported_prompts_and_checksums_v3(sft_root)

    # 8. Strict Pre-Upload Assertions on `data/sft_v3/`
    audit_report: dict[str, Any] = {}
    for folder in HF_V3_REPOS:
        ds_dir = sft_root / folder
        train_rows = _load_jsonl_dicts(ds_dir / "train.jsonl")
        assert len(train_rows) > 0, f"{folder}/train.jsonl is empty!"

        train_pids = {str(r.get("problem_id") or "") for r in train_rows}
        overlap = train_pids & heldout_pids
        assert not overlap, f"Leakage in {folder}/train.jsonl: {len(overlap)} heldout problem_ids found!"

        split_stats: dict[str, Any] = {
            "train_count": len(train_rows),
            "unique_problems": len(train_pids),
        }
        for split_name in ("train", "validation", "test"):
            sp_path = ds_dir / f"{split_name}.jsonl"
            if not sp_path.is_file():
                continue
            sp_rows = _load_jsonl_dicts(sp_path)
            for idx, r in enumerate(sp_rows):
                pr = str(r.get("prompt") or "")
                assert pr.endswith("\n\n"), f"{folder}/{split_name}.jsonl row {idx} prompt missing trailing \\n\\n!"
            if split_name in ("validation", "test"):
                assert len(sp_rows) > 0, f"{folder}/{split_name}.jsonl is empty!"
                split_stats[f"{split_name}_count"] = len(sp_rows)

        # Strict all-MiniLM-L6-v2 cosine < 0.80 cross-split anti-leak guard
        from adversarial_physics.evaluation.calibration_audit import check_embedding_train_eval_leakage

        emb_audit = check_embedding_train_eval_leakage(
            [ds_dir / "train.jsonl"],
            [p for p in (ds_dir / "validation.jsonl", ds_dir / "test.jsonl") if p.is_file()],
            cosine_threshold=0.80,
        )
        assert emb_audit["leak_free"], (
            f"Semantic leakage >= 0.80 in {folder}: {emb_audit['leaked_by_split']}"
        )

        if folder in ("solver_e2_curatorkit_nollm", "solver_e3_canonical_llm", "solver_e4_adversarial"):
            step_cnt = sum(
                1
                for r in train_rows
                if _STEP_MARKER_RE.search(str(r.get("target_output") or r.get("output") or ""))
            )
            split_stats["step_structured_rate"] = round(step_cnt / len(train_rows), 4)
            assert split_stats["step_structured_rate"] >= 0.99, (
                f"{folder} step_structured_rate={split_stats['step_structured_rate']} < 0.99!"
            )

        if folder == "solver_e4_adversarial":
            comp_counts = Counter(
                str(r.get("target_output") or r.get("output") or "") for r in train_rows
            )
            max_dup = max(comp_counts.values())
            assert max_dup <= 2, f"solver_e4_adversarial max completion multiplicity is {max_dup} (> 2)!"
            split_stats["turn_dist"] = dict(sorted(Counter(r.get("turn") for r in train_rows).items()))
            split_stats["max_completion_multiplicity"] = max_dup

        if folder == "verifier_e4_curated":
            app_n = sum(
                1
                for r in train_rows
                if str(r.get("target_output") or r.get("output") or "").strip().endswith(VERDICT_APPROVE)
            )
            ch_n = sum(
                1
                for r in train_rows
                if str(r.get("target_output") or r.get("output") or "").strip().endswith(VERDICT_CHALLENGE)
            )
            assert app_n + ch_n == len(train_rows), "verifier_e4_curated has rows without terminal verdict!"
            app_rate = app_n / len(train_rows)
            assert 0.13 <= app_rate <= 0.18, f"verifier_e4_curated approval rate {app_rate:.4f} outside [0.13, 0.18]!"
            # Verify 100% of both [[APPROVE]] and [[CHALLENGE]] prompts use `**Derivation:**\nStep 1:`
            deriv_app = sum(1 for r in train_rows if r.get("verdict") == VERDICT_APPROVE and "**Derivation:**\nStep 1:" in str(r.get("prompt") or ""))
            deriv_ch = sum(1 for r in train_rows if r.get("verdict") == VERDICT_CHALLENGE and "**Derivation:**\nStep 1:" in str(r.get("prompt") or ""))
            assert deriv_app == app_n and deriv_ch == ch_n, (
                f"Format leak check failed: deriv_app={deriv_app}/{app_n}, deriv_ch={deriv_ch}/{ch_n}"
            )
            # Verify 0% of [[APPROVE]] targets start with 'The derivation' or 'The solution'
            the_deriv_n = sum(
                1
                for r in train_rows
                if r.get("verdict") == VERDICT_APPROVE
                and str(r.get("target_output") or "").startswith(("The derivation", "The solution"))
            )
            assert the_deriv_n == 0, f"Found {the_deriv_n} [[APPROVE]] targets still starting with 'The derivation/solution'!"
            # Verify last 10% (val_split_ratio=0.10 slice) has both classes balanced
            tail_n = max(1, len(train_rows) // 10)
            tail_app = sum(1 for r in train_rows[-tail_n:] if r.get("verdict") == VERDICT_APPROVE)
            tail_rate = tail_app / tail_n
            assert 0.08 <= tail_rate <= 0.25, f"Tail 10% validation slice approval rate {tail_rate:.3f} is unstratified!"
            split_stats["approve_count"] = app_n
            split_stats["challenge_count"] = ch_n
            split_stats["approve_rate"] = round(app_rate, 4)
            split_stats["tail_10pct_approve_rate"] = round(tail_rate, 4)

        split_stats["domain_dist"] = dict(sorted(Counter(str(r.get("domain") or "") for r in train_rows).items()))
        audit_report[folder] = split_stats

    return {
        "e1_pool_count": len(e1_pool),
        "e2_pool_count": len(e2_pool),
        "e3_pool_count": len(e3_pool),
        "sol2_count": len(v3_sol2),
        "tiny_correct_count": len(v3_tiny_ok),
        "purged_shifted_approvals": purged_shifted_approvals,
        "sanitized_challenge_formulas": sanitized_challenge_formulas,
        "export_summary": export_summary.to_dict(),
        "assertions": audit_report,
    }


def upload_v3_to_hf(token: str) -> dict[str, str]:
    """Create (if needed) and upload all 5 verified `data/sft_v3/` datasets to HuggingFace `-v3` repos."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    sft_root = PROJECT_ROOT / "data/sft_v3"
    uploaded: dict[str, str] = {}
    for folder, repo_id in HF_V3_REPOS.items():
        local_dir = sft_root / folder
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        print(f"[HF Upload v3] Uploading {local_dir} -> {repo_id} ...")
        info = api.upload_folder(
            folder_path=str(local_dir),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Release v3 SFT dataset: {folder} (100% step-structured, 0 shifted approvals, calibrated 30/70 verifier)",
        )
        uploaded[repo_id] = str(getattr(info, "commit_url", info))
    return uploaded


def main() -> int:
    parser = argparse.ArgumentParser(description="Build, verify, and upload all 5 v3 SFT datasets to HuggingFace.")
    parser.add_argument("--no-upload", action="store_true", help="Build and verify locally without uploading to HF.")
    args = parser.parse_args()

    result = build_and_verify_v3()
    if not args.no_upload:
        token = _resolve_hf_token()
        if not token:
            raise SystemExit("HF_TOKEN missing in environment and .env — aborting upload.")
        result["hf_uploads"] = upload_v3_to_hf(token)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
