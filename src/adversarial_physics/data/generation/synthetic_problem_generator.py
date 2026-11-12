"""Seed-Guided & Subtopic-Grounded Synthetic Problem Generator (`synthetic_problem_generator.py`).

Generates formal symbolic/LaTeX physics problems for Experiment E2 & E3 by combining:
  1. A strict Part-1 clean textbook problem (``CanonicalProblem`` physical kernel),
  2. Its chapter-matched subtopic theory snippet retrieved via zero-LLM ``SubtopicMapper``, and
  3. A domain-matched PhysGym ``train_seed`` exemplar setting the formal LaTeX & variable style.

Avoids brittle nested JSON outputs by using lightweight plain delimiters
(``===PROBLEM===``, ``===SOLUTION===``, ``ANSWER:``, optional ``CHECK_VALUES:``) and performs
deterministic SymPy back-substitution verification against the Crowell anchor answer.
"""
from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from adversarial_physics.data.curation.answer_validator import (
    _extract_primary_number,
    _is_purely_numeric_answer,
    evaluate_symbolic_answer_with_params,
    extract_answer_line,
)
from adversarial_physics.data.curation.curatorkit_stage1 import (
    PART1_STRICT_CONFIG,
    PART2_STRICT_CONFIG,
    CuratorKitStrictConfig,
    check_strict_physics_hygiene,
    count_text_tokens,
    run_curatorkit_embedding_dedup,
    run_curatorkit_stage1_gates_and_recovery,
    run_curatorkit_within_pool_dedup,
)
from adversarial_physics.data.ingestion.subtopic_extractor import (
    DEFAULT_SUBTOPIC_CHUNKS_PATH,
    SubtopicChunk,
    SubtopicMapper,
)
from adversarial_physics.data.schemas.problem import (
    CanonicalProblem,
    load_problems_jsonl,
    save_problems_jsonl,
)
from adversarial_physics.utils.config import resolve_project_path
from adversarial_physics.utils.logging import get_logger
from adversarial_physics.utils.runtime import AYA_EXPANSE_8B, BackendName, build_aya_chat_generator

logger = get_logger("synthetic_problem_generator")

DEFAULT_E2_READY_PROMPTS_PATH = "data/intermediate/01c_e2_ready_for_llm_generation.jsonl"
DEFAULT_E2_READY_SUMMARY_PATH = "data/intermediate/01c_e2_ready_for_llm_generation_summary.json"
DEFAULT_SYNTHETIC_CHECKPOINT_PATH = "data/intermediate/02a_seed_grounded_synthetic_checkpoint.jsonl"
DEFAULT_SYNTHETIC_RAW_PATH = "data/intermediate/02a_seed_grounded_synthetic_raw.jsonl"
DEFAULT_E2_COMBINED_POOL_PATH = "data/intermediate/01b_e2_books_plus_synthetic.jsonl"
DEFAULT_E3_CURATED_POOL_PATH = "data/processed/canonical_train.jsonl"
DEFAULT_SYNTHETIC_SUMMARY_PATH = "data/intermediate/02_synthetic_generation_summary.json"

SYNTHETIC_SYSTEM_PROMPT = (
    "You are a university physics problem author creating rigorous, self-contained, LaTeX-formatted "
    "physics problems and step-by-step solutions.\n"
    "Follow the exact output tags requested. Output ONLY the tagged sections — no JSON, no markdown code fences."
)


@dataclass(frozen=True)
class ParsedSyntheticGeneration:
    """Lightweight parsed synthetic problem + optional numeric substitution check params."""

    problem_statement: str
    reference_solution: str
    answer_value: str
    check_values: dict[str, float]


@dataclass(frozen=True)
class SyntheticGenerationSummary:
    """Summary of Stage 2 seed-guided synthetic generation and CuratorKIT Part 2 curation."""

    source_clean_book_count: int
    synthetic_attempted: int
    synthetic_parsed_ok: int
    synthetic_sympy_verified: int
    e2_total_count: int
    e3_total_count: int
    e3_by_domain: dict[str, int]
    model_used: str
    wall_clock_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_seed_style_prompt(
    crowell_prob: CanonicalProblem,
    subtopic_chunks: list[SubtopicChunk],
    physgym_seed: CanonicalProblem | None,
    *,
    max_subtopic_words: int = 280,
    max_seed_sol_chars: int = 900,
) -> str:
    """Build a compact (<~1.5k tokens) generation prompt combining kernel, subtopic, and seed style."""
    if subtopic_chunks:
        ch = subtopic_chunks[0]
        words = ch.text.split()
        snippet_text = " ".join(words[:max_subtopic_words])
        subtopic_block = f"[{ch.section_id} {ch.section_title}]\n{snippet_text}"
    else:
        subtopic_block = f"[{crowell_prob.domain} reference theory — derive from first principles]"

    if physgym_seed is not None:
        seed_stmt = physgym_seed.problem_statement.strip()[:700]
        seed_sol = physgym_seed.reference_solution.strip()[:max_seed_sol_chars]
        style_block = (
            f"Style Seed Problem ({physgym_seed.domain}):\n{seed_stmt}\n\n"
            f"Style Seed Solution Excerpt:\n{seed_sol}"
        )
    else:
        style_block = "Use formal academic physics prose with explicit symbol definitions and standard LaTeX equations."

    return (
        "TASK: Transform the Textbook Problem below into a clean, self-contained, formal LaTeX physics problem "
        "in the rigorous style of the Style Seed Example, grounded in the Subtopic Theory.\n"
        "- Generalize concrete numbers into clean symbolic parameters where natural (or keep clear SI parameters), "
        "stating all assumptions explicitly.\n"
        "- Write a concise 3-8 step derivation in clean LaTeX (`$...$` / `$$...$$`).\n"
        "- End the solution with a single final line: `ANSWER: <final symbolic or numeric expression>`.\n"
        "- If your final ANSWER is symbolic AND corresponds to numeric parameters from the Textbook Problem, "
        "add one optional final line: `CHECK_VALUES: var1=num1, var2=num2, ...`.\n\n"
        f"--- STYLE EXEMPLAR ---\n{style_block}\n\n"
        f"--- SUBTOPIC THEORY ---\n{subtopic_block}\n\n"
        f"--- SOURCE TEXTBOOK KERNEL ({crowell_prob.domain}) ---\n"
        f"Problem: {crowell_prob.problem_statement.strip()}\n"
        f"Reference Answer/Solution: {crowell_prob.reference_solution.strip()[:900]}\n\n"
        "--- OUTPUT FORMAT (EXACT TAGS, NO JSON) ---\n"
        "===PROBLEM===\n"
        "<rewritten problem statement>\n"
        "===SOLUTION===\n"
        "<step-by-step LaTeX derivation>\n"
        "ANSWER: <final expression>\n"
        "CHECK_VALUES: <optional comma-separated numeric assignments, or NONE>"
    )


def _parse_check_values_line(line: str) -> dict[str, float]:
    """Parse ``var1=0.5, var2=55`` into ``dict[str, float]``; ignores units safely."""
    text = (line or "").strip()
    if not text or text.upper() == "NONE":
        return {}
    out: dict[str, float] = {}
    for pair in text.split(","):
        if "=" not in pair:
            continue
        k, v = pair.split("=", 1)
        var = re.sub(r"[^a-zA-Z0-9_]", "", k.strip().lstrip("\\"))
        m = re.search(r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", v.strip())
        if var and m:
            try:
                out[var] = float(m.group(1))
            except ValueError:
                continue
    return out


def parse_tagged_generation(raw_text: str) -> ParsedSyntheticGeneration | None:
    """Parse plain delimiter generation output without fragile JSON decoding."""
    text = (raw_text or "").strip()
    if "===PROBLEM===" not in text or "===SOLUTION===" not in text:
        return None

    _, after_prob = text.split("===PROBLEM===", 1)
    prob_part, sol_part = after_prob.split("===SOLUTION===", 1)
    problem_statement = prob_part.strip()
    sol_raw = sol_part.strip()

    check_values: dict[str, float] = {}
    m_chk = re.search(r"(?:^|\n)\s*CHECK_VALUES\s*:\s*(.+)$", sol_raw, flags=re.IGNORECASE | re.MULTILINE)
    if m_chk:
        check_values = _parse_check_values_line(m_chk.group(1))
        sol_raw = (sol_raw[: m_chk.start()] + sol_raw[m_chk.end() :]).strip()

    has_ans, ans_val = extract_answer_line(sol_raw)
    if not has_ans or not ans_val or ans_val in ("$$", "$"):
        return None

    # Canonicalize trailing ANSWER: line without disturbing any earlier body text mentioning "answer:"
    ans_clean = re.sub(r"^\$+|\$+$", "", ans_val).strip()
    ans_matches = list(
        re.finditer(r"(?:^|\n)\s*(?:\*\*?)?ANSWER(?:\*\*?)?\s*:", sol_raw, flags=re.IGNORECASE)
    )
    sol_wo_tail_ans = sol_raw[: ans_matches[-1].start()].strip() if ans_matches else sol_raw.strip()
    if not sol_wo_tail_ans or len(problem_statement.split()) < 10:
        return None
    normalized_solution = f"{sol_wo_tail_ans}\nANSWER: {ans_clean}"

    return ParsedSyntheticGeneration(
        problem_statement=problem_statement,
        reference_solution=normalized_solution,
        answer_value=ans_clean,
        check_values=check_values,
    )


def verify_sympy_consistency(
    parsed: ParsedSyntheticGeneration,
    anchor_problem: CanonicalProblem | None = None,
    *,
    rel_tol: float = 0.03,
) -> tuple[bool, str]:
    """Deterministically verify SymPy parseability and optional substitution against Crowell anchor."""
    ans = parsed.answer_value
    if not ans:
        return False, "empty_answer"

    # Reject dangling placeholder answers
    if re.search(r"<[^>]+>|TODO|unknown", ans, flags=re.IGNORECASE):
        return False, "placeholder_in_answer"

    # If purely numeric, ensure primary number parses cleanly
    if _is_purely_numeric_answer(ans):
        num = _extract_primary_number(ans)
        if num is None:
            return False, "unparseable_numeric_answer"
        if anchor_problem is not None:
            has_a, a_ans = extract_answer_line(anchor_problem.reference_solution)
            if has_a and _is_purely_numeric_answer(a_ans):
                a_num = _extract_primary_number(a_ans)
                if a_num is not None and abs(a_num) > 1e-12:
                    err = abs(num - a_num) / abs(a_num)
                    if err > rel_tol:
                        return False, f"numeric_mismatch_vs_anchor ({num} vs {a_num})"
        return True, "numeric_ok"

    # Symbolic expression check: verify SymPy parses the expression cleanly
    sym_val = evaluate_symbolic_answer_with_params(ans, parsed.check_values)
    if parsed.check_values:
        if sym_val is None:
            return False, "sympy_eval_failed_with_check_values"
        if anchor_problem is not None:
            has_a, a_ans = extract_answer_line(anchor_problem.reference_solution)
            if has_a and _is_purely_numeric_answer(a_ans):
                a_num = _extract_primary_number(a_ans)
                if a_num is not None and abs(a_num) > 1e-12:
                    err = abs(sym_val - a_num) / abs(a_num)
                    if err > rel_tol:
                        return False, f"sympy_subs_mismatch_vs_anchor ({sym_val:.4g} vs {a_num:.4g})"
        return True, "sympy_subs_ok"

    # Symbolic without CHECK_VALUES: confirm parseable with dummy unit parameters
    vars_found = set(re.findall(r"\b([a-zA-Z])\b", ans)) - {"e", "i"}
    dummy = {v: 2.0 for v in vars_found}
    if evaluate_symbolic_answer_with_params(ans, dummy) is None:
        return False, "unparseable_symbolic_expression"
    return True, "sympy_parse_ok"


def _pick_domain_seed(
    domain: str,
    seeds_by_domain: dict[str, list[CanonicalProblem]],
    all_seeds: list[CanonicalProblem],
    step_idx: int,
) -> CanonicalProblem | None:
    pool = seeds_by_domain.get(domain) or all_seeds
    if not pool:
        return None
    return pool[step_idx % len(pool)]


def _prepare_e2_prompt_records(
    *,
    clean_books_path: str | Path,
    subtopic_chunks_path: str | Path,
    train_seeds_path: str | Path,
    max_generations: int | None = None,
) -> tuple[list[CanonicalProblem], list[CanonicalProblem], list[CanonicalProblem], list[dict[str, Any]]]:
    """Assemble deterministic Part-1 clean books + subtopic snippets + domain seeds into LLM prompts."""
    source_pool = load_problems_jsonl(resolve_project_path(clean_books_path))
    book_sources = [p for p in source_pool if p.source not in ("physgym", "physgym_seed")]
    train_seeds = load_problems_jsonl(resolve_project_path(train_seeds_path))

    part1_clean_books, _, _ = run_curatorkit_stage1_gates_and_recovery(
        book_sources,
        strict_config=PART1_STRICT_CONFIG,
    )
    targets = part1_clean_books[:max_generations] if max_generations is not None else part1_clean_books

    subtopic_path = resolve_project_path(subtopic_chunks_path)
    mapper = SubtopicMapper.from_jsonl(subtopic_path) if subtopic_path.is_file() else SubtopicMapper([])

    clean_seeds = [
        sd
        for sd in train_seeds
        if count_text_tokens(sd.problem_statement, use_tiktoken=True) <= PART1_STRICT_CONFIG.max_problem_tokens
        and count_text_tokens(sd.reference_solution, use_tiktoken=True) <= PART1_STRICT_CONFIG.max_solution_tokens
    ]
    preferred_seeds = clean_seeds or train_seeds

    seeds_by_domain: dict[str, list[CanonicalProblem]] = defaultdict(list)
    for sd in clean_seeds:
        seeds_by_domain[sd.domain].append(sd)
    # Fallback per domain only if a domain (e.g. RELATIVITY) had zero seeds within strict length caps
    for sd in train_seeds:
        if sd.domain not in seeds_by_domain and count_text_tokens(sd.problem_statement, use_tiktoken=True) <= PART1_STRICT_CONFIG.max_problem_tokens:
            seeds_by_domain[sd.domain].append(sd)

    prompt_records: list[dict[str, Any]] = []
    for idx, anchor in enumerate(targets):
        match_tier: int | None = None
        if hasattr(mapper, "find_subtopics_with_tiers"):
            matched_pairs = mapper.find_subtopics_with_tiers(anchor, top_k=1)
            subtopics = [c for c, _ in matched_pairs]
            match_tier = matched_pairs[0][1] if matched_pairs else None
        else:
            subtopics = mapper.find_subtopics(anchor, top_k=1)
        seed = _pick_domain_seed(anchor.domain, seeds_by_domain, preferred_seeds, idx)
        user_prompt = build_seed_style_prompt(anchor, subtopics, seed)
        sub_meta = (
            {
                "chunk_id": subtopics[0].chunk_id,
                "book_id": subtopics[0].book_id,
                "chapter": subtopics[0].chapter,
                "chapter_slug": subtopics[0].chapter_slug,
                "section_id": subtopics[0].section_id,
                "section_title": subtopics[0].section_title,
                "match_tier": match_tier,
                "text": subtopics[0].text[:1500],
            }
            if subtopics
            else None
        )
        prompt_records.append(
            {
                "anchor_problem_id": anchor.problem_id,
                "domain": anchor.domain,
                "original_id": anchor.original_id,
                "anchor_problem": anchor.to_dict(),
                "matched_subtopic": sub_meta,
                "matched_seed_id": seed.problem_id if seed is not None else None,
                "system_prompt": SYNTHETIC_SYSTEM_PROMPT,
                "user_prompt": user_prompt,
                "estimated_prompt_tokens": max(
                    1, (len(SYNTHETIC_SYSTEM_PROMPT) + len(user_prompt)) // 4
                ),
            }
        )
    return book_sources, part1_clean_books, train_seeds, prompt_records


def prepare_e2_llm_generation_requests(
    *,
    clean_books_path: str | Path = "data/intermediate/01b_deterministic_curated_pool.jsonl",
    subtopic_chunks_path: str | Path = DEFAULT_SUBTOPIC_CHUNKS_PATH,
    train_seeds_path: str | Path = "data/splits/train_seeds.jsonl",
    output_path: str | Path = DEFAULT_E2_READY_PROMPTS_PATH,
    summary_out: str | Path = DEFAULT_E2_READY_SUMMARY_PATH,
    max_generations: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Export zero-LLM E2 ready-for-LLM generation prompts + coverage statistics."""
    book_sources, part1_clean_books, _, prompt_records = _prepare_e2_prompt_records(
        clean_books_path=clean_books_path,
        subtopic_chunks_path=subtopic_chunks_path,
        train_seeds_path=train_seeds_path,
        max_generations=max_generations,
    )

    out_file = resolve_project_path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        for rec in prompt_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    by_domain: dict[str, int] = defaultdict(int)
    by_tier: dict[str, int] = defaultdict(int)
    subtopic_hits = 0
    token_counts: list[int] = []
    for rec in prompt_records:
        by_domain[str(rec["domain"])] += 1
        ms = rec.get("matched_subtopic")
        if ms is not None:
            subtopic_hits += 1
            tier_val = ms.get("match_tier")
            by_tier[f"tier_{tier_val}" if tier_val is not None else "tier_unknown"] += 1
        token_counts.append(int(rec.get("estimated_prompt_tokens", 0)))

    summary: dict[str, Any] = {
        "input_book_sources": len(book_sources),
        "part1_strict_clean_count": len(part1_clean_books),
        "ready_prompt_count": len(prompt_records),
        "subtopic_matched_count": subtopic_hits,
        "subtopic_hit_rate": round(subtopic_hits / len(prompt_records), 4) if prompt_records else 0.0,
        "subtopic_match_tiers": dict(sorted(by_tier.items())),
        "by_domain": dict(sorted(by_domain.items())),
        "max_estimated_prompt_tokens": max(token_counts) if token_counts else 0,
        "mean_estimated_prompt_tokens": (
            round(sum(token_counts) / len(token_counts), 1) if token_counts else 0.0
        ),
        "output_path": str(output_path),
    }
    sum_file = resolve_project_path(summary_out)
    sum_file.parent.mkdir(parents=True, exist_ok=True)
    sum_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return prompt_records, summary


def generate_synthetic_e2_and_e3_pools(
    *,
    clean_books_path: str | Path = "data/intermediate/01b_deterministic_curated_pool.jsonl",
    subtopic_chunks_path: str | Path = DEFAULT_SUBTOPIC_CHUNKS_PATH,
    train_seeds_path: str | Path = "data/splits/train_seeds.jsonl",
    validation_path: str | Path = "data/splits/validation.jsonl",
    test_path: str | Path = "data/splits/test.jsonl",
    checkpoint_path: str | Path = DEFAULT_SYNTHETIC_CHECKPOINT_PATH,
    synthetic_raw_out: str | Path = DEFAULT_SYNTHETIC_RAW_PATH,
    e2_pool_out: str | Path = DEFAULT_E2_COMBINED_POOL_PATH,
    e3_pool_out: str | Path = DEFAULT_E3_CURATED_POOL_PATH,
    summary_out: str | Path = DEFAULT_SYNTHETIC_SUMMARY_PATH,
    model_name: str = AYA_EXPANSE_8B,
    backend: BackendName = "auto",
    endpoint_url: str | None = None,
    max_generations: int | None = None,
    enable_part2_llm_judges: bool = True,
    chat_fn: Callable[[str, str], str] | None = None,
    judge_chat_fn: Callable[[str, str], str] | None = None,
) -> tuple[list[CanonicalProblem], list[CanonicalProblem], SyntheticGenerationSummary]:
    """Generate subtopic-grounded, seed-styled synthetic problems and curate E2 & E3 pools."""
    t_start = time.perf_counter()

    _, part1_clean_books, train_seeds, prompt_records = _prepare_e2_prompt_records(
        clean_books_path=clean_books_path,
        subtopic_chunks_path=subtopic_chunks_path,
        train_seeds_path=train_seeds_path,
        max_generations=max_generations,
    )
    val_problems = load_problems_jsonl(resolve_project_path(validation_path))
    test_problems = load_problems_jsonl(resolve_project_path(test_path))

    source_pool_all = load_problems_jsonl(resolve_project_path(clean_books_path))
    part1_physgym_seeds = [
        p
        for p in source_pool_all
        if p.source in ("physgym", "physgym_seed")
        and check_strict_physics_hygiene(p, PART1_STRICT_CONFIG)[0]
    ]
    if not part1_physgym_seeds:
        part1_physgym_seeds = [
            p
            for p in train_seeds
            if check_strict_physics_hygiene(p, PART1_STRICT_CONFIG)[0]
        ]

    # Load resumable checkpoint
    ckpt_file = resolve_project_path(checkpoint_path)
    ckpt_file.parent.mkdir(parents=True, exist_ok=True)
    completed_by_source: dict[str, dict[str, Any]] = {}
    if ckpt_file.is_file():
        with ckpt_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    completed_by_source[str(rec["anchor_problem_id"])] = rec

    gen_fn = chat_fn
    if gen_fn is None:
        gen_fn = build_aya_chat_generator(
            task="synthetic_physics_generation",
            model_name=model_name,
            backend=backend,
            temperature=0.25,
            max_new_tokens=1024,
            endpoint_url=endpoint_url,
        )

    anchor_by_id: dict[str, CanonicalProblem] = {p.problem_id: p for p in part1_clean_books}
    check_params_by_id: dict[str, dict[str, float]] = {}
    source_contexts: dict[str, str] = {}

    attempted = 0
    parsed_ok = 0
    sympy_ok = 0
    raw_synthetic: list[CanonicalProblem] = []
    verified_synthetic_for_e2: list[CanonicalProblem] = []

    def _build_grounding_context(req_obj: dict[str, Any], anchor_obj: CanonicalProblem) -> str:
        ms = req_obj.get("matched_subtopic") or {}
        sec_hdr = f"[Subtopic {ms.get('section_id', '')} {ms.get('section_title', '')}]".strip()
        sub_txt = str(ms.get("text") or "").strip()
        parts: list[str] = []
        if sub_txt:
            parts.append(f"{sec_hdr}\n{sub_txt[:1500]}")
        parts.append(
            f"[Source Textbook Kernel]\nProblem: {anchor_obj.problem_statement}\n"
            f"Reference Solution: {anchor_obj.reference_solution}"
        )
        return "\n\n".join(parts)

    with ckpt_file.open("a", encoding="utf-8") as ckpt_out:
        for req in prompt_records:
            anchor = CanonicalProblem.from_dict(req["anchor_problem"])
            if anchor.problem_id in completed_by_source:
                cached = completed_by_source[anchor.problem_id]
                attempted += 1
                if cached.get("parsed_ok"):
                    parsed_ok += 1
                    prob = CanonicalProblem.from_dict(cached["problem"])
                    raw_synthetic.append(prob)
                    check_params_by_id[prob.problem_id] = cached.get("check_values") or {}
                    anchor_by_id[prob.problem_id] = anchor
                    source_contexts[prob.problem_id] = _build_grounding_context(req, anchor)
                    ok_sym_cached = bool(cached.get("sympy_ok"))
                    if ok_sym_cached:
                        sympy_ok += 1
                    if ok_sym_cached and check_strict_physics_hygiene(prob, PART1_STRICT_CONFIG)[0]:
                        verified_synthetic_for_e2.append(prob)
                continue

            raw_resp = gen_fn(str(req["system_prompt"]), str(req["user_prompt"]))
            parsed = parse_tagged_generation(raw_resp)
            attempted += 1

            if parsed is None:
                rec = {"anchor_problem_id": anchor.problem_id, "parsed_ok": False, "sympy_ok": False}
                ckpt_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                ckpt_out.flush()
                continue

            parsed_ok += 1
            syn_id = f"syn_seed_{anchor.problem_id}"
            syn_prob = CanonicalProblem(
                problem_id=syn_id,
                domain=anchor.domain,
                problem_statement=parsed.problem_statement,
                reference_solution=parsed.reference_solution,
                source="synthetic_seed_subtopic",
                original_id=anchor.original_id,
            )
            ok_sym, sym_reason = verify_sympy_consistency(parsed, anchor)
            if ok_sym:
                sympy_ok += 1
            raw_synthetic.append(syn_prob)
            check_params_by_id[syn_id] = parsed.check_values
            anchor_by_id[syn_id] = anchor
            source_contexts[syn_id] = _build_grounding_context(req, anchor)
            if ok_sym and check_strict_physics_hygiene(syn_prob, PART1_STRICT_CONFIG)[0]:
                verified_synthetic_for_e2.append(syn_prob)

            rec = {
                "anchor_problem_id": anchor.problem_id,
                "parsed_ok": True,
                "sympy_ok": ok_sym,
                "sympy_reason": sym_reason,
                "check_values": parsed.check_values,
                "problem": syn_prob.to_dict(),
            }
            ckpt_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            ckpt_out.flush()

    save_problems_jsonl(raw_synthetic, resolve_project_path(synthetic_raw_out))

    # Build E2: clean physgym_seeds + Part-1 clean textbook pool + deterministic-verified synthetic outputs,
    # deduplicated against held-out Validation/Test splits and uncleaned excess train_seeds.
    heldout = [(p, "validation") for p in val_problems] + [(p, "test") for p in test_problems]
    clean_seed_ids = {p.problem_id for p in part1_physgym_seeds}
    extra_unretained_seeds = [s for s in train_seeds if s.problem_id not in clean_seed_ids]

    e2_candidates_raw = list(part1_physgym_seeds) + list(part1_clean_books) + list(verified_synthetic_for_e2)
    e2_no_leak, _ = run_curatorkit_embedding_dedup(
        e2_candidates_raw,
        heldout,
        cosine_threshold=PART1_STRICT_CONFIG.anti_leakage_cosine_threshold,
    )
    e2_pool, _ = run_curatorkit_within_pool_dedup(
        e2_no_leak,
        seed_problems=extra_unretained_seeds,
        minhash_threshold=PART1_STRICT_CONFIG.minhash_threshold,
        diversity_threshold=PART1_STRICT_CONFIG.diversity_threshold,
    )
    save_problems_jsonl(e2_pool, resolve_project_path(e2_pool_out))

    # Build E3: Strict CuratorKIT Part 2 (SymPy consistency + HallucinationGate>=0.85 + RewardGate>=0.82 + DiversityGate)
    def _sympy_gate_fn(prob: CanonicalProblem) -> tuple[bool, str]:
        if prob.source != "synthetic_seed_subtopic":
            return True, "book_or_seed_verified"
        has_ans, ans_val = extract_answer_line(prob.reference_solution)
        if not has_ans:
            return False, "missing_answer"
        parsed_obj = ParsedSyntheticGeneration(
            problem_statement=prob.problem_statement,
            reference_solution=prob.reference_solution,
            answer_value=ans_val,
            check_values=check_params_by_id.get(prob.problem_id, {}),
        )
        return verify_sympy_consistency(parsed_obj, anchor_by_id.get(prob.problem_id))

    resolved_judge_fn = judge_chat_fn
    if enable_part2_llm_judges and resolved_judge_fn is None:
        resolved_judge_fn = build_aya_chat_generator(
            task="curatorkit_judge",
            model_name=model_name,
            backend=backend,
            temperature=0.0,
            max_new_tokens=512,
            endpoint_url=endpoint_url,
        )

    part2_cfg = CuratorKitStrictConfig(
        name=PART2_STRICT_CONFIG.name,
        enable_llm_gates=bool(enable_part2_llm_judges and resolved_judge_fn is not None),
        hallucination_threshold=PART2_STRICT_CONFIG.hallucination_threshold,
        reward_threshold=PART2_STRICT_CONFIG.reward_threshold,
        on_judge_error=PART2_STRICT_CONFIG.on_judge_error,
        diversity_threshold=PART2_STRICT_CONFIG.diversity_threshold,
    )
    e3_pool, _, _ = run_curatorkit_stage1_gates_and_recovery(
        e2_pool,
        strict_config=part2_cfg,
        sympy_verify_fn=_sympy_gate_fn,
        source_contexts=source_contexts,
        chat_fn=resolved_judge_fn,
    )
    save_problems_jsonl(e3_pool, resolve_project_path(e3_pool_out))

    by_domain: dict[str, int] = defaultdict(int)
    for p in e3_pool:
        by_domain[p.domain] += 1

    summary = SyntheticGenerationSummary(
        source_clean_book_count=len(part1_clean_books),
        synthetic_attempted=attempted,
        synthetic_parsed_ok=parsed_ok,
        synthetic_sympy_verified=sympy_ok,
        e2_total_count=len(e2_pool),
        e3_total_count=len(e3_pool),
        e3_by_domain=dict(sorted(by_domain.items())),
        model_used=model_name,
        wall_clock_seconds=round(time.perf_counter() - t_start, 3),
    )
    sum_path = resolve_project_path(summary_out)
    sum_path.parent.mkdir(parents=True, exist_ok=True)
    sum_path.write_text(json.dumps(summary.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return e2_pool, e3_pool, summary
