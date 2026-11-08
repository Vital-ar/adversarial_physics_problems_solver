"""Substage 1.3 & 1.4: Canonical Training Pool Builder (`canonical_train_250.jsonl`).

Transforms:
  * ``data/splits/train_seeds.jsonl`` (21 PhysGym training seeds)
  * ``data/literature/initial_book_dataset.jsonl`` (986 solution-matched Crowell book problems)
while enforcing strict isolation against:
  * ``data/splits/validation.jsonl`` (38 held-out Validation problems)
  * ``data/splits/test.jsonl`` (38 held-out Test problems)

Pipeline Stages:
  1. **PDF Artifact & LaTeX Hygiene (`clean_book_problem_text`, `clean_book_solution_text`)**:
     Removes running page headers/footers, margin symbols (`√`, `⋆`), and ensures an explicit
     ``ANSWER: <final_value_or_expression>`` sentinel at the end of ``reference_solution``.
  2. **Independent Multi-Part Subproblem Splitting (`split_independent_subproblems`)**:
     Splits multi-part textbook problems ``(a)``, ``(b)`` into standalone single-target records
     ONLY when subproblems are independently solvable, preserving ``<original_id>:part_a`` provenance.
  3. **Strict Self-Containment & External-Reference Gate (`evaluate_self_containment`)**:
     Rejects problems that depend on visual diagrams/figures/tables, external problems or pages
     (e.g., ``book_modern_0986`` referencing ``problem 2 on p. 1028``), or corrupted unrecovered
     PDF fractions (e.g., ``= 2825``).
  4. **Logical Step Complexity Gate (`estimate_logical_steps`, `1–10` Logical Steps)**:
     Counts logical physical/mathematical reasoning operations (not displayed equation lines)
     and enforces ``1 <= logical_steps <= 10`` and ``len(reference_solution) <= 4500``.
  5. **Mandatory Anti-Leakage Gate (`run_anti_leakage_gate`, `cosine_similarity >= 0.85`)**:
     Purges any candidate that matches any of the 76 held-out Validation/Test problems.
  6. **Domain- & Topic-Balanced Selection (`select_balanced_canonical_pool`)**:
     Combines the accepted PhysGym training seeds + curated book problems to reach exactly
     ``target_total = 250`` problems balanced across the 6 canonical domains and round-robin
     diversified across textbook chapters/topics.
  7. **Optional Resumable `CohereForAI/aya-expanse-8b` Refinement (`refine_pool_with_aya`)**:
     Normalizes LaTeX and ``ANSWER:`` lines with per-problem checkpointing and wall-clock timing
     tracking so it can run safely on ``hf_gpu``, ``kaggle_gpu``, or ``local_cpu``.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from adversarial_physics.data.curation.curatorkit_stage1 import (
    run_curatorkit_stage1_gates_and_recovery,
    write_curatorkit_provenance_bundle,
)
from adversarial_physics.data.ingestion.chapter_domain_extractor import build_initial_book_dataset
from adversarial_physics.data.ingestion.literature_extractor import is_diagram_dependent
from adversarial_physics.data.processing.deduplication import (
    DEFAULT_ANTI_LEAKAGE_COSINE_THRESHOLD,
    DEFAULT_ANTI_LEAKAGE_NGRAM_CONTAINMENT_THRESHOLD,
    DEFAULT_WITHIN_POOL_COSINE_THRESHOLD,
    LeakageMatch,
    deduplicate_candidate_pool,
    run_anti_leakage_gate,
)
from adversarial_physics.data.schemas.problem import (
    CanonicalProblem,
    load_problems_jsonl,
    save_problems_jsonl,
)
from adversarial_physics.utils.config import resolve_project_path
from adversarial_physics.utils.logging import get_logger
from adversarial_physics.utils.runtime import AYA_EXPANSE_8B, BackendName, build_aya_chat_generator

logger = get_logger("stage_1_3_canonical_pool_builder")

DEFAULT_TRAIN_SEEDS_PATH = "data/splits/train_seeds.jsonl"
DEFAULT_VALIDATION_PATH = "data/splits/validation.jsonl"
DEFAULT_TEST_PATH = "data/splits/test.jsonl"
DEFAULT_INITIAL_BOOK_DATASET_PATH = "data/literature/initial_book_dataset.jsonl"

# Intermediate datasets for Experiment E1 (manual deterministic rules before CuratorKIT) and E2 (CuratorKIT without LLM)
DEFAULT_LEAK_FREE_RAW_POOL_PATH = "data/intermediate/01a_leak_free_raw_pool.jsonl"
DEFAULT_MANUAL_DETERMINISTIC_ONLY_POOL_PATH = "data/intermediate/01a_manual_deterministic_only_pool.jsonl"
DEFAULT_DETERMINISTIC_CURATED_POOL_PATH = "data/intermediate/01b_deterministic_curated_pool.jsonl"

DEFAULT_CANONICAL_CANDIDATES_PATH = "data/literature/canonical_candidates.jsonl"
DEFAULT_REJECTED_STAGE1_PATH = "data/literature/rejected_stage1_candidates.jsonl"
DEFAULT_CANONICAL_TRAIN_PATH = "data/processed/canonical_train.jsonl"
DEFAULT_CANONICAL_TRAIN_250_PATH = DEFAULT_CANONICAL_TRAIN_PATH
DEFAULT_CANONICAL_MANIFEST_PATH = "data/processed/canonical_train_manifest.json"
DEFAULT_CHECKSUMS_PATH = "data/processed/checksums.txt"
DEFAULT_LLM_CHECKPOINT_PATH = "data/processed/canonical_train_llm_checkpoint.jsonl"

# Target domain weights for the 250-problem Canonical Training Pool
DEFAULT_DOMAIN_WEIGHTS: dict[str, float] = {
    "MECHANICS": 0.34,
    "ELECTROMAGNETISM": 0.24,
    "OPTICS": 0.13,
    "MODERN_PHYSICS": 0.13,
    "RELATIVITY": 0.08,
    "THERMODYNAMICS": 0.08,
}

# Patterns that indicate a problem depends on another problem, page, or external section
EXTERNAL_DEPENDENCY_PATTERNS: tuple[str, ...] = (
    r"\bin\s+(?:homework\s+)?problem\s+\d+\b",
    r"\bfrom\s+(?:homework\s+)?problem\s+\d+\b",
    r"\bof\s+(?:homework\s+)?problem\s+\d+\b",
    r"\bredo\s+problem\s+\d+\b",
    r"\bsee\s+problem\s+\d+\b",
    r"\bproblem\s+\d+\s+in\s+ch(?:apter|\.)?\s*\d+\b",
    r"\bproblem\s+\d+\s+on\s+p(?:age|\.)?\s*\d+\b",
    r"\btransitions\s+labeled\s+[A-Z]\s+and\s+[A-Z]\b",
    r"\bread\s+optional\s+section\s+\d+(?:\.\d+)*\b",
    r"\bas\s+described\s+in\s+example\s+\d+\s+on\s+p(?:age|\.)?\s*\d+\b",
    r"\breproduce\s+figure\s+[a-z0-9]+\b",
    r"\bin\s+ a\s+rough\s+sketch\b",
)

# Patterns that indicate a subproblem (b), (c), ... depends on an earlier subproblem (a)
SUBPROBLEM_DEPENDENCY_PATTERNS: tuple[str, ...] = (
    r"\bpart\s+[a-d]\b",
    r"\bparts\s+[a-d]\s+and\s+[a-d]\b",
    r"\bfrom\s+\(?[a-d]\)\b",
    r"\bin\s+\(?[a-d]\)\b",
    r"\byour\s+(?:answer|result|equation)s?\s+(?:from|in|to)\b",
    r"\busing\s+your\s+(?:answer|result)\b",
    r"\bbased\s+on\s+your\s+answers?\b",
    r"\bcomparing\s+your\s+answers?\b",
)

# Running headers/footers & margin symbols in Crowell PDFs
_CROWELL_HEADER_LINE_RE = re.compile(
    r"^(?:Problems\s+\d+|Problem\s+\d+\.?|\d+\s+CHAPTER\s+\d+\.[A-Z0-9\s,\-]+|CHAPTER\s+\d+\.[A-Z0-9\s,\-]+\s+\d+|[√⋆∫]+)$",
    flags=re.IGNORECASE,
)
_TRAILING_SYMBOLS_RE = re.compile(r"[\s√⋆∫]+$")

# Corrupted unrecovered PDF fraction pattern (e.g. "2^-2 - 5^-2 / 2^-2 - 4^-2 = 2825")
_CORRUPTED_FRACTION_RE = re.compile(
    r"(?:fraction|ratio\s+[λa-zA-Z0-9_/]+).*?=\s*\d{4,}\s*$",
    flags=re.IGNORECASE | re.DOTALL,
)

_SUBPART_SPLIT_RE = re.compile(r"(?:^|\n|(?<=[.?!:;]))\s*\(([a-f])\)\s+", flags=re.IGNORECASE)

AYA_REFINEMENT_SYSTEM_PROMPT = """You are an expert physics dataset curator.
Given a university physics problem and its official reference solution:
1. Normalize the solution into a clean, step-by-step physical derivation (1 to 10 logical steps) that ends on a new final line with:
ANSWER: <final numerical value with SI units or simplified symbolic expression>
2. If the provided reference solution is already complete, accurate, and well-structured, output the complete reference solution text directly ending with `ANSWER: <final value>`. Do NOT output a commentary note saying no changes are needed—always output the full derivation text itself.
3. ONLY if the problem statement is genuinely impossible to solve because it is missing a required visual diagram/figure/table or references another unprovided problem, output ONLY:
[[UNSOLVABLE_MISSING_CONTEXT: <short reason>]]

Strict Rules:
- Preserve the exact physics and ground-truth final answer from the official reference solution. Never invent a different answer.
- Never reject a problem just because its reference solution is already complete or requires no simplification.
- End with `ANSWER: <value>` on the very last line.
"""

_FALSE_REJECTION_PHRASES: tuple[str, ...] = (
    "solution is complete",
    "reference solution is complete",
    "no further simplification",
    "no normalization",
    "requiring no normalization",
    "not required",
    "already presented",
    "correctly stated",
    "well-structured",
    "well-explained",
    "valid, concise derivation",
    "no further steps are necessary",
)


@dataclass(frozen=True)
class Stage1RejectedRecord:
    """Detailed record for any candidate problem excluded during Stage 1.3 filtering."""

    problem_id: str
    domain: str
    source: str
    original_id: Any
    reason: str
    logical_steps: int
    problem_statement_preview: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CanonicalPoolSummary:
    """Summary statistics and stage wall-clock timings for the Stage 1 Canonical Training Pool."""

    physgym_seeds_total: int
    physgym_seeds_accepted: int
    physgym_seeds_rejected: int
    initial_book_problems_total: int
    leak_free_raw_pool_count: int
    candidates_after_subproblem_split: int
    candidates_passing_acceptance_gates: int
    candidates_purged_by_anti_leakage: int
    candidates_purged_by_within_pool_dedup: int
    canonical_candidates_count: int
    deterministic_curated_pool_count: int
    canonical_train_count: int
    canonical_train_250_count: int
    counts_by_domain: dict[str, int]
    counts_by_source: dict[str, int]
    rejection_reasons: dict[str, int]
    llm_refinement_applied: bool
    stage_timings_sec: dict[str, float] = field(default_factory=dict)
    total_elapsed_sec: float = 0.0
    output_paths: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def clean_book_problem_text(text: str) -> str:
    """Strip Crowell PDF running headers, margin symbols (√, ⋆), and normalize whitespace."""
    lines: list[str] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if _CROWELL_HEADER_LINE_RE.match(stripped):
            continue
        # Remove inline trailing margin checkmarks/stars
        cleaned_line = _TRAILING_SYMBOLS_RE.sub("", stripped).strip()
        if cleaned_line and not _CROWELL_HEADER_LINE_RE.match(cleaned_line):
            lines.append(cleaned_line)

    joined = "\n".join(lines)
    # Normalize spaced minus signs and powers from PDF extraction while keeping paragraph breaks
    joined = re.sub(r"[ \t]+", " ", joined)
    return _TRAILING_SYMBOLS_RE.sub("", joined).strip()


def extract_final_answer_from_solution(solution_text: str) -> str:
    """Extract the final numerical or symbolic answer expression from a reference solution."""
    # Check if ANSWER: sentinel already exists
    ans_match = re.search(r"(?:^|\n)\s*ANSWER\s*:\s*(.+)$", solution_text.strip(), flags=re.IGNORECASE)
    if ans_match:
        return ans_match.group(1).strip()

    # Check LaTeX display math block at the end ($$ ... $$ or \[ ... \])
    display_blocks = re.findall(r"(?:\$\$(.*?)\$\$|\\\[(.*?)\\\])", solution_text, flags=re.DOTALL)
    if display_blocks:
        last_math = (display_blocks[-1][0] or display_blocks[-1][1]).strip()
        if last_math:
            # If the equation has '=' take the RHS, or keep full equation if short
            last_line = [ln.strip() for ln in last_math.splitlines() if ln.strip()][-1]
            return re.sub(r"\s+", " ", last_line).strip(" .,;")

    # Check inline math or final sentence with '=' or '≈' or '∼'
    lines = [ln.strip() for ln in solution_text.splitlines() if ln.strip() and not _CROWELL_HEADER_LINE_RE.match(ln.strip())]
    if not lines:
        return ""

    tail = " ".join(lines[-2:]) if len(lines) >= 2 else lines[-1]
    eq_matches = list(re.finditer(r"(?:=|≈|∼|\\approx)\s*([^=;]+?)(?=\.(?:\s|$)|;|$)", tail))
    if eq_matches:
        candidate_rhs = eq_matches[-1].group(1).strip(" .,;")
        if candidate_rhs:
            return re.sub(r"\s+", " ", candidate_rhs)

    return re.sub(r"\s+", " ", lines[-1]).strip()


def clean_book_solution_text(solution_text: str, *, ensure_answer_sentinel: bool = True) -> str:
    """Clean PDF running headers from a reference solution and ensure an `ANSWER:` sentinel line."""
    lines: list[str] = []
    for raw_line in solution_text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            lines.append("")
            continue
        if _CROWELL_HEADER_LINE_RE.match(stripped):
            continue
        lines.append(stripped)

    cleaned = "\n".join(lines).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    if ensure_answer_sentinel and cleaned and not re.search(r"(?:^|\n)\s*ANSWER\s*:", cleaned, flags=re.IGNORECASE):
        final_ans = extract_final_answer_from_solution(cleaned)
        if final_ans:
            cleaned = f"{cleaned}\nANSWER: {final_ans}"
    return cleaned


def split_independent_subproblems(problem: CanonicalProblem) -> list[CanonicalProblem]:
    """Split a multi-part textbook problem `(a)`, `(b)` into standalone single-target records
    ONLY for subparts that are independently solvable.

    If a problem has no `(a)`/`(b)` markers, returns `[problem]` with cleaned text.
    If a problem has `(a)`/`(b)` markers in both statement and solution:
      * Extracts shared preamble text before `(a)`.
      * Emits `(a)` (and any subsequent subpart `(b)`, `(c)` that does NOT reference earlier parts)
        as a standalone single-target `CanonicalProblem` with `original_id = "<orig>:part_a"`.
    """
    stmt_cleaned = clean_book_problem_text(problem.problem_statement)
    sol_cleaned = clean_book_solution_text(problem.reference_solution, ensure_answer_sentinel=False)

    stmt_matches = list(_SUBPART_SPLIT_RE.finditer(stmt_cleaned))
    if len(stmt_matches) < 2:
        return [
            CanonicalProblem(
                problem_id=problem.problem_id,
                domain=problem.domain,
                problem_statement=stmt_cleaned,
                reference_solution=clean_book_solution_text(sol_cleaned, ensure_answer_sentinel=True),
                source=problem.source,
                original_id=problem.original_id,
            )
        ]

    sol_matches = list(_SUBPART_SPLIT_RE.finditer(sol_cleaned))
    if not sol_matches:
        # Cannot reliably align solution subparts; return cleaned original so the multipart gate handles it
        return [
            CanonicalProblem(
                problem_id=problem.problem_id,
                domain=problem.domain,
                problem_statement=stmt_cleaned,
                reference_solution=clean_book_solution_text(sol_cleaned, ensure_answer_sentinel=True),
                source=problem.source,
                original_id=problem.original_id,
            )
        ]

    preamble = stmt_cleaned[: stmt_matches[0].start()].strip()
    stmt_parts: dict[str, str] = {}
    for idx, match in enumerate(stmt_matches):
        letter = match.group(1).lower()
        start = match.end()
        end = stmt_matches[idx + 1].start() if idx + 1 < len(stmt_matches) else len(stmt_cleaned)
        stmt_parts[letter] = stmt_cleaned[start:end].strip()

    sol_parts: dict[str, str] = {}
    for idx, match in enumerate(sol_matches):
        letter = match.group(1).lower()
        start = match.end()
        end = sol_matches[idx + 1].start() if idx + 1 < len(sol_matches) else len(sol_cleaned)
        sol_parts[letter] = sol_cleaned[start:end].strip()

    extracted_subproblems: list[CanonicalProblem] = []
    for letter, part_stmt in stmt_parts.items():
        part_sol = sol_parts.get(letter, "").strip()
        if not part_stmt or not part_sol:
            continue

        # Part (b), (c), ... is only admissible if it does NOT depend on an earlier part
        if letter != "a":
            combined_check = f"{part_stmt}\n{part_sol}"
            if any(re.search(pat, combined_check, flags=re.IGNORECASE) for pat in SUBPROBLEM_DEPENDENCY_PATTERNS):
                continue

        full_stmt = f"{preamble}\n\n{part_stmt}".strip() if preamble else part_stmt
        full_sol = clean_book_solution_text(part_sol, ensure_answer_sentinel=True)
        extracted_subproblems.append(
            CanonicalProblem(
                problem_id=f"{problem.problem_id}_{letter}",
                domain=problem.domain,
                problem_statement=full_stmt,
                reference_solution=full_sol,
                source=problem.source,
                original_id=f"{problem.original_id}:part_{letter}",
            )
        )

    return extracted_subproblems


def estimate_logical_steps(solution_text: str) -> int:
    """Count logical physical/mathematical reasoning operations (not raw displayed equation lines).

    As specified in the blueprint:
      * A single algebraic rearrangement spread across multiple lines or a single LaTeX
        ``aligned`` / ``array`` block counts as 1 logical reasoning step.
      * Distinct physical equations, substitutions, or paragraph steps each count as 1 logical step.
    """
    # Strip final ANSWER: sentinel before counting steps
    body = re.sub(r"(?:^|\n)\s*ANSWER\s*:.*$", "", solution_text.strip(), flags=re.IGNORECASE).strip()
    if not body:
        return 0

    # Collapse multi-line LaTeX display math blocks ($$...$$ or \[...\]) into single logical blocks
    collapsed = re.sub(r"\$\$.*?\$\$", " [DISPLAY_EQ_BLOCK] ", body, flags=re.DOTALL)
    collapsed = re.sub(r"\\\[.*?\\\]", " [DISPLAY_EQ_BLOCK] ", collapsed, flags=re.DOTALL)

    n_display_blocks = collapsed.count("[DISPLAY_EQ_BLOCK]")
    if n_display_blocks > 0:
        # In PhysGym / LaTeX solutions, each distinct display block or prose transition represents a step;
        # consecutive display blocks that are pure intermediate rearrangements represent ~0.75 logical steps each.
        logical_count = max(1, round(n_display_blocks * 0.72))
        return logical_count

    # For textbook prose + inline equation solutions:
    # Count distinct prose sentences or primary equations (ignoring consecutive '= ...' continuation lines)
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    logical_ops = 0
    prev_was_eq_continuation = False
    for ln in lines:
        is_continuation = ln.startswith("=") or ln.startswith("≈") or ln.startswith("∼")
        if is_continuation and prev_was_eq_continuation:
            continue
        if "=" in ln or "≈" in ln or "∼" in ln:
            logical_ops += 1
            prev_was_eq_continuation = True
        else:
            prev_was_eq_continuation = False

    if logical_ops > 0:
        return max(1, min(logical_ops, max(1, round(logical_ops * 0.8))))

    sentences = [s.strip() for s in re.split(r"[.!?]+", body) if len(s.strip()) > 12]
    return max(1, len(sentences))


def evaluate_self_containment_and_acceptance(
    problem: CanonicalProblem,
    *,
    min_logical_steps: int = 1,
    max_logical_steps: int = 10,
    min_statement_chars: int = 35,
    min_solution_chars: int = 20,
    max_solution_chars: int = 4500,
) -> tuple[bool, str | None, int]:
    """Evaluate a `CanonicalProblem` against all Stage 1.3 strict acceptance criteria.

    Returns:
        ``(is_accepted, rejection_reason_or_None, estimated_logical_steps)``
    """
    stmt = problem.problem_statement.strip()
    sol = problem.reference_solution.strip()
    steps = estimate_logical_steps(sol)

    if len(stmt) < min_statement_chars:
        return False, "statement_too_short", steps
    if len(sol) < min_solution_chars:
        return False, "solution_too_short", steps
    if len(sol) > max_solution_chars:
        return False, f"solution_exceeds_max_chars ({len(sol)} > {max_solution_chars})", steps

    # 1. Visual diagram / figure / table dependency check on problem statement
    if is_diagram_dependent(stmt):
        return False, "diagram_or_figure_dependent", steps

    # Also check solution for explicit missing figure dependence (for textbook problems)
    if problem.source != "physgym" and is_diagram_dependent(sol):
        return False, "solution_diagram_dependent", steps

    # 2. External problem / page / section dependency check (e.g., book_modern_0986)
    combined = f"{stmt}\n{sol}"
    for pattern in EXTERNAL_DEPENDENCY_PATTERNS:
        if re.search(pattern, combined, flags=re.IGNORECASE):
            return False, "external_problem_or_page_dependency", steps

    # 3. Corrupted unrecovered PDF fraction check (e.g., "= 2825" for "28/25")
    if _CORRUPTED_FRACTION_RE.search(combined):
        return False, "corrupted_unrecovered_math_fraction", steps

    # 4. Unsplit multi-part check
    subpart_matches = _SUBPART_SPLIT_RE.findall(stmt)
    if len(set(m.lower() for m in subpart_matches)) >= 2:
        return False, "unsplit_multipart_question", steps

    # 5. Logical step complexity gate (1 to 10 logical steps)
    if steps < min_logical_steps or steps > max_logical_steps:
        return False, f"step_count_out_of_range ({steps} not in [{min_logical_steps}, {max_logical_steps}])", steps

    return True, None, steps


def _chapter_topic_key(problem: CanonicalProblem) -> str:
    """Extract `<book_id>:<chapter_stem>` topic key from `original_id` for topic-balanced selection."""
    orig = str(problem.original_id or "")
    parts = orig.split(":")
    if len(parts) >= 2:
        return f"{parts[0]}:{parts[1]}"
    return orig or problem.problem_id


def select_balanced_canonical_pool(
    accepted_physgym_seeds: list[CanonicalProblem],
    accepted_book_candidates: list[CanonicalProblem],
    *,
    target_total: int | None = None,
    domain_weights: dict[str, float] | None = None,
    candidate_validator: Callable[[CanonicalProblem], tuple[CanonicalProblem | None, str | None]] | None = None,
) -> tuple[list[CanonicalProblem], list[Stage1RejectedRecord]]:
    """Select `accepted_physgym_seeds` + curated book problems across the 6 canonical domains
    and round-robin diversified across textbook chapters/topics.

    When `target_total is None` (default), includes ALL accepted PhysGym seeds and ALL valid
    book candidates (not capped at 250). When `target_total` is a positive integer, selects
    up to `target_total` problems balanced by `domain_weights` and automatically backfills
    from remaining valid candidates if `candidate_validator` rejects an entry.
    """
    weights = domain_weights or DEFAULT_DOMAIN_WEIGHTS
    domains = sorted(
        set(weights.keys())
        | {p.domain for p in accepted_physgym_seeds}
        | {c.domain for c in accepted_book_candidates}
    )

    seeds_by_domain: dict[str, list[CanonicalProblem]] = defaultdict(list)
    for p in accepted_physgym_seeds:
        seeds_by_domain[p.domain].append(p)

    cands_by_domain: dict[str, list[CanonicalProblem]] = defaultdict(list)
    for c in accepted_book_candidates:
        cands_by_domain[c.domain].append(c)

    # Order ALL valid book candidates within each domain using chapter/topic round-robin
    diversified_by_domain: dict[str, list[CanonicalProblem]] = {}
    for d in domains:
        by_topic: dict[str, list[CanonicalProblem]] = defaultdict(list)
        for item in cands_by_domain[d]:
            by_topic[_chapter_topic_key(item)].append(item)
        topic_queues = [by_topic[k] for k in sorted(by_topic.keys())]
        rr_list: list[CanonicalProblem] = []
        idx = 0
        while True:
            added_any = False
            for q in topic_queues:
                if idx < len(q):
                    rr_list.append(q[idx])
                    added_any = True
            if not added_any:
                break
            idx += 1
        diversified_by_domain[d] = rr_list

    # Compute target total per domain, bounded by available (seeds + book candidates)
    available_by_domain = {
        d: len(seeds_by_domain[d]) + len(diversified_by_domain[d]) for d in domains
    }
    if target_total is None or target_total <= 0:
        # Uncapped mode: include ALL valid problems across all domains
        target_by_domain = dict(available_by_domain)
        effective_target = sum(available_by_domain.values())
    else:
        target_by_domain = {}
        for d in domains:
            desired = round(weights.get(d, 0.0) * target_total)
            target_by_domain[d] = min(available_by_domain[d], max(len(seeds_by_domain[d]), desired))

        # Reconcile sum(target_by_domain) to reach min(target_total, sum(available_by_domain))
        effective_target = min(target_total, sum(available_by_domain.values()))
        while sum(target_by_domain.values()) < effective_target:
            spare_domains = [d for d in domains if target_by_domain[d] < available_by_domain[d]]
            if not spare_domains:
                break
            best_d = max(
                spare_domains,
                key=lambda d: (weights.get(d, 0.0) * effective_target - target_by_domain[d], available_by_domain[d] - target_by_domain[d]),
            )
            target_by_domain[best_d] += 1

        while sum(target_by_domain.values()) > effective_target:
            reducible = [d for d in domains if target_by_domain[d] > len(seeds_by_domain[d])]
            if not reducible:
                break
            worst_d = max(reducible, key=lambda d: target_by_domain[d] - weights.get(d, 0.0) * effective_target)
            target_by_domain[worst_d] -= 1

    selected: list[CanonicalProblem] = []
    llm_rejected: list[Stage1RejectedRecord] = []
    domain_slug_map = {
        "MECHANICS": "mech",
        "ELECTROMAGNETISM": "em",
        "THERMODYNAMICS": "thermo",
        "OPTICS": "optics",
        "MODERN_PHYSICS": "modern",
        "RELATIVITY": "rel",
    }

    # Include all accepted PhysGym training seeds first
    for d in domains:
        for seed_prob in seeds_by_domain[d]:
            base_seed = CanonicalProblem(
                problem_id=seed_prob.problem_id,
                domain=seed_prob.domain,
                problem_statement=seed_prob.problem_statement,
                reference_solution=clean_book_solution_text(seed_prob.reference_solution, ensure_answer_sentinel=True),
                source="physgym_seed",
                original_id=seed_prob.original_id,
            )
            if candidate_validator is not None:
                validated, rej_reason = candidate_validator(base_seed)
                if validated is not None:
                    selected.append(validated)
                else:
                    llm_rejected.append(
                        Stage1RejectedRecord(
                            problem_id=base_seed.problem_id,
                            domain=base_seed.domain,
                            source=base_seed.source,
                            original_id=base_seed.original_id,
                            reason=rej_reason or "llm_rejected_physgym_seed",
                            logical_steps=estimate_logical_steps(base_seed.reference_solution),
                            problem_statement_preview=base_seed.problem_statement[:160],
                        )
                    )
            else:
                selected.append(base_seed)

    # Pull from the full candidate queue per domain until each domain's target quota is filled
    lit_counter = 1
    unused_by_domain: dict[str, list[CanonicalProblem]] = defaultdict(list)

    for d in domains:
        current_in_domain = sum(1 for p in selected if p.domain == d)
        needed_from_books = max(0, target_by_domain[d] - current_in_domain)
        slug = domain_slug_map.get(d, "phys")
        admitted_for_domain = 0

        for idx, book_prob in enumerate(diversified_by_domain[d]):
            if admitted_for_domain >= needed_from_books:
                unused_by_domain[d].extend(diversified_by_domain[d][idx:])
                break

            candidate_entry = CanonicalProblem(
                problem_id=f"lit_{slug}_{lit_counter:04d}",
                domain=d,
                problem_statement=book_prob.problem_statement,
                reference_solution=clean_book_solution_text(book_prob.reference_solution, ensure_answer_sentinel=True),
                source="literature_curated",
                original_id=book_prob.original_id,
            )
            if candidate_validator is not None:
                validated, rej_reason = candidate_validator(candidate_entry)
                if validated is None:
                    llm_rejected.append(
                        Stage1RejectedRecord(
                            problem_id=book_prob.problem_id,
                            domain=d,
                            source=book_prob.source,
                            original_id=book_prob.original_id,
                            reason=rej_reason or "llm_rejected",
                            logical_steps=estimate_logical_steps(book_prob.reference_solution),
                            problem_statement_preview=book_prob.problem_statement[:160],
                        )
                    )
                    continue
                candidate_entry = validated

            selected.append(candidate_entry)
            admitted_for_domain += 1
            lit_counter += 1

    # If any domain ran short due to LLM rejections, backfill from remaining unused valid candidates across other domains
    while len(selected) < effective_target:
        spare_domains = [d for d in domains if unused_by_domain[d]]
        if not spare_domains:
            break
        for d in spare_domains:
            if len(selected) >= effective_target or not unused_by_domain[d]:
                break
            book_prob = unused_by_domain[d].pop(0)
            slug = domain_slug_map.get(d, "phys")
            candidate_entry = CanonicalProblem(
                problem_id=f"lit_{slug}_{lit_counter:04d}",
                domain=d,
                problem_statement=book_prob.problem_statement,
                reference_solution=clean_book_solution_text(book_prob.reference_solution, ensure_answer_sentinel=True),
                source="literature_curated",
                original_id=book_prob.original_id,
            )
            if candidate_validator is not None:
                validated, rej_reason = candidate_validator(candidate_entry)
                if validated is None:
                    llm_rejected.append(
                        Stage1RejectedRecord(
                            problem_id=book_prob.problem_id,
                            domain=d,
                            source=book_prob.source,
                            original_id=book_prob.original_id,
                            reason=rej_reason or "llm_rejected",
                            logical_steps=estimate_logical_steps(book_prob.reference_solution),
                            problem_statement_preview=book_prob.problem_statement[:160],
                        )
                    )
                    continue
                candidate_entry = validated

            selected.append(candidate_entry)
            lit_counter += 1

    return selected, llm_rejected


def _is_false_llm_rejection_reason(reason_text: str) -> bool:
    """Return True if an LLM `[[REJECT: ...]]` note actually states that the solution is complete/valid."""
    lower = reason_text.strip().lower()
    return any(phrase in lower for phrase in _FALSE_REJECTION_PHRASES)


def _strip_llm_meta_tags(solution_text: str, fallback_solution: str) -> str:
    """Strip any `[[REJECT: ...]`, `[[NO DERIVATION NEEDED: ...]`, or `[[RENORMALIZED SOLUTION:` meta-headers from LLM output."""
    cleaned = re.sub(
        r"\[\[(?:REJECT|UNSOLVABLE_MISSING_CONTEXT|NO\s+DERIVATION\s+NEEDED)\s*:[^\]]*\]\]?",
        "",
        solution_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    cleaned = re.sub(
        r"\[\[RENORMALIZED\s+SOLUTION\s*:?",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = cleaned.replace("]]", "").strip()
    # Normalize bold `**ANSWER:**` followed by newline/value into standard `ANSWER: <val>`
    cleaned = re.sub(r"\*\*ANSWER\s*:\*\*\s*\n*\s*", "ANSWER: ", cleaned, flags=re.IGNORECASE)
    # Remove any trailing empty `ANSWER:` lines
    cleaned = re.sub(r"\n\s*ANSWER\s*:\s*$", "", cleaned, flags=re.IGNORECASE).strip()

    # If stripping the meta-commentary left only an `ANSWER:` line with no derivation body,
    # preserve the original reference derivation body + final answer.
    body_without_ans = re.sub(r"(?:^|\n)\s*ANSWER\s*:.*$", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
    if not body_without_ans:
        return clean_book_solution_text(fallback_solution, ensure_answer_sentinel=True)

    result = clean_book_solution_text(cleaned, ensure_answer_sentinel=True)
    # If the extracted `ANSWER:` line accidentally captured a numbered prose step header (e.g. `8. **Displacement...`),
    # replace the `ANSWER:` value with the clean answer from `fallback_solution`.
    ans_match = re.search(r"(?:^|\n)\s*ANSWER\s*:\s*(.+)$", result, flags=re.IGNORECASE)
    if ans_match:
        ans_val = ans_match.group(1).strip()
        if re.match(r"^\d+\.\s*\*\*", ans_val) or not ans_val:
            fb_ans = extract_final_answer_from_solution(fallback_solution)
            if fb_ans:
                result = re.sub(
                    r"(?:^|\n)\s*ANSWER\s*:.*$",
                    f"\nANSWER: {fb_ans}",
                    result,
                    flags=re.IGNORECASE,
                ).strip()
    return result


def build_aya_candidate_validator(
    *,
    checkpoint_path: str | Path = DEFAULT_LLM_CHECKPOINT_PATH,
    model_name: str = AYA_EXPANSE_8B,
    backend: BackendName = "auto",
    max_new_tokens: int = 1536,
    max_logical_steps: int = 10,
    total_candidates: int | None = None,
    chat_fn: Callable[[str, str], str] | None = None,
) -> Callable[[CanonicalProblem], tuple[CanonicalProblem | None, str | None]]:
    """Create a checkpointed `CohereForAI/aya-expanse-8b` validator/normalizer callable.

    Operates over the full `unique_book_candidates` pool during `select_balanced_canonical_pool`:
      * If a candidate is genuinely rejected by the LLM (`[[UNSOLVABLE_MISSING_CONTEXT: ...]]` or
        `[[REJECT: ...]]` where the reason is NOT a false-rejection "solution is already complete" note)
        or exceeds `max_logical_steps` after normalization, returns `(None, reason)`.
      * Automatically detects and rescues false-rejection notes where the LLM wrote
        `[[REJECT: The provided solution is complete...]]`.
      * Persists decisions in `checkpoint_path` (keyed by `original_id`) so interrupted overnight
        runs resume seamlessly without re-running the LLM.
    """
    ckpt_file = resolve_project_path(checkpoint_path)
    ckpt_file.parent.mkdir(parents=True, exist_ok=True)

    cached_decisions: dict[str, dict[str, Any]] = {}
    if ckpt_file.is_file():
        with ckpt_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    key = str(row.get("original_id") or row.get("problem_id") or "")
                    if key:
                        cached_decisions[key] = row
                except Exception:
                    continue
        logger.info("Loaded %d cached LLM candidate decisions from %s.", len(cached_decisions), ckpt_file)

    generator: Callable[[str, str], str] | None = chat_fn
    live_calls = 0
    live_total_sec = 0.0

    def _validate_and_normalize(prob: CanonicalProblem) -> tuple[CanonicalProblem | None, str | None]:
        nonlocal generator, live_calls, live_total_sec
        cache_key = str(prob.original_id or prob.problem_id)
        if cache_key in cached_decisions:
            cached = cached_decisions[cache_key]
            if cached.get("status") == "rejected":
                cached_reason = str(cached.get("reason") or "llm_rejected_cached")
                if not _is_false_llm_rejection_reason(cached_reason):
                    return None, cached_reason
                logger.info(
                    "Rescuing falsely rejected cached entry %s (%s)",
                    cache_key,
                    cached_reason,
                )
            sol_text = _strip_llm_meta_tags(
                str(cached.get("reference_solution") or prob.reference_solution),
                prob.reference_solution,
            )
            return (
                CanonicalProblem(
                    problem_id=prob.problem_id,
                    domain=prob.domain,
                    problem_statement=prob.problem_statement,
                    reference_solution=sol_text,
                    source=prob.source,
                    original_id=prob.original_id,
                ),
                None,
            )

        if generator is None:
            generator = build_aya_chat_generator(
                task="literature_extraction",
                model_name=model_name,
                backend=backend,
                temperature=0.1,
                max_new_tokens=max_new_tokens,
            )

        user_prompt = (
            f"DOMAIN: {prob.domain}\n"
            f"PROBLEM:\n{prob.problem_statement}\n\n"
            f"OFFICIAL REFERENCE SOLUTION:\n{prob.reference_solution}"
        )
        t_call = time.perf_counter()
        raw_out = generator(AYA_REFINEMENT_SYSTEM_PROMPT, user_prompt).strip()
        call_sec = round(time.perf_counter() - t_call, 3)
        live_calls += 1
        live_total_sec += call_sec
        avg_sec = live_total_sec / live_calls
        expected_total = total_candidates if (total_candidates is not None and total_candidates > 0) else 750
        remaining_est = max(0, expected_total - len(cached_decisions) - 1) * avg_sec

        rej_match = re.search(
            r"\[\[(?:REJECT|UNSOLVABLE_MISSING_CONTEXT)\s*:\s*(.*?)\]\]?",
            raw_out,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if rej_match:
            rej_note = rej_match.group(1).strip()
            if not _is_false_llm_rejection_reason(rej_note):
                reason = f"llm_rejected:{rej_note or 'unsuitable'}"
                row_data = {
                    "original_id": cache_key,
                    "status": "rejected",
                    "reason": reason,
                    "latency_seconds": call_sec,
                }
                with ckpt_file.open("a", encoding="utf-8") as ckpt_out:
                    ckpt_out.write(json.dumps(row_data, ensure_ascii=False) + "\n")
                cached_decisions[cache_key] = row_data
                logger.info(
                    "[LLM #%d | %.2fs | avg=%.2fs | ETA~%.1fm] Rejected %s (%s)",
                    live_calls,
                    call_sec,
                    avg_sec,
                    remaining_est / 60.0,
                    cache_key,
                    reason,
                )
                return None, reason

        norm_sol = _strip_llm_meta_tags(raw_out or prob.reference_solution, prob.reference_solution)
        steps = estimate_logical_steps(norm_sol)
        if steps > max_logical_steps:
            reason = f"llm_normalized_step_count_exceeds_{max_logical_steps} ({steps} steps)"
            row_data = {
                "original_id": cache_key,
                "status": "rejected",
                "reason": reason,
                "latency_seconds": call_sec,
            }
            with ckpt_file.open("a", encoding="utf-8") as ckpt_out:
                ckpt_out.write(json.dumps(row_data, ensure_ascii=False) + "\n")
            cached_decisions[cache_key] = row_data
            logger.info(
                "[LLM #%d | %.2fs | avg=%.2fs | ETA~%.1fm] Rejected %s (%s)",
                live_calls,
                call_sec,
                avg_sec,
                remaining_est / 60.0,
                cache_key,
                reason,
            )
            return None, reason

        updated = CanonicalProblem(
            problem_id=prob.problem_id,
            domain=prob.domain,
            problem_statement=prob.problem_statement,
            reference_solution=norm_sol,
            source=prob.source,
            original_id=prob.original_id,
        )
        accepted_row = {**updated.to_dict(), "status": "accepted", "latency_seconds": call_sec}
        with ckpt_file.open("a", encoding="utf-8") as ckpt_out:
            ckpt_out.write(json.dumps(accepted_row, ensure_ascii=False) + "\n")
        cached_decisions[cache_key] = accepted_row
        logger.info(
            "[LLM #%d | %.2fs | avg=%.2fs | ETA~%.1fm] Accepted %s (%s)",
            live_calls,
            call_sec,
            avg_sec,
            remaining_est / 60.0,
            updated.problem_id,
            cache_key,
        )
        return updated, None

    return _validate_and_normalize


def refine_pool_with_aya(
    problems: list[CanonicalProblem],
    *,
    checkpoint_path: str | Path = DEFAULT_LLM_CHECKPOINT_PATH,
    model_name: str = AYA_EXPANSE_8B,
    backend: BackendName = "auto",
    max_new_tokens: int = 1536,
    chat_fn: Callable[[str, str], str] | None = None,
) -> list[CanonicalProblem]:
    """Run resumable `CohereForAI/aya-expanse-8b` solution normalization over a list of problems."""
    validator = build_aya_candidate_validator(
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        backend=backend,
        max_new_tokens=max_new_tokens,
        total_candidates=len(problems),
        chat_fn=chat_fn,
    )
    refined: list[CanonicalProblem] = []
    for prob in problems:
        updated, _ = validator(prob)
        if updated is not None:
            refined.append(updated)
    return refined


def _compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_stage1_canonical_pool(
    *,
    train_seeds_path: str | Path = DEFAULT_TRAIN_SEEDS_PATH,
    validation_path: str | Path = DEFAULT_VALIDATION_PATH,
    test_path: str | Path = DEFAULT_TEST_PATH,
    initial_book_dataset_path: str | Path = DEFAULT_INITIAL_BOOK_DATASET_PATH,
    leak_free_raw_pool_path: str | Path = DEFAULT_LEAK_FREE_RAW_POOL_PATH,
    manual_deterministic_only_pool_path: str | Path = DEFAULT_MANUAL_DETERMINISTIC_ONLY_POOL_PATH,
    deterministic_curated_pool_path: str | Path = DEFAULT_DETERMINISTIC_CURATED_POOL_PATH,
    canonical_candidates_path: str | Path = DEFAULT_CANONICAL_CANDIDATES_PATH,
    rejected_output_path: str | Path = DEFAULT_REJECTED_STAGE1_PATH,
    canonical_train_250_path: str | Path = DEFAULT_CANONICAL_TRAIN_PATH,
    manifest_output_path: str | Path = DEFAULT_CANONICAL_MANIFEST_PATH,
    checksums_output_path: str | Path = DEFAULT_CHECKSUMS_PATH,
    target_total: int | None = None,
    min_logical_steps: int = 1,
    max_logical_steps: int = 10,
    anti_leakage_cosine_threshold: float = DEFAULT_ANTI_LEAKAGE_COSINE_THRESHOLD,
    use_llm_refinement: bool = False,
    llm_backend: BackendName = "auto",
    llm_model_name: str = AYA_EXPANSE_8B,
) -> tuple[list[CanonicalProblem], CanonicalPoolSummary]:
    """Execute the complete Stage 1.3 & 1.4 pipeline.

    By default (`target_total=None`), includes ALL valid, anti-leakage-verified problems
    across all 6 canonical domains without capping at 250. Also saves intermediate datasets:
      * `data/intermediate/01a_leak_free_raw_pool.jsonl`: Raw leak-free pool (for Experiment E1 `solver_raw`).
      * `data/intermediate/01b_deterministic_curated_pool.jsonl`: Full deterministic + CuratorKIT curated pool
        saved before optional LLM normalization (for Experiment E2/E3 and safe checkpointing).
      * `data/processed/canonical_train.jsonl`: Final canonical training dataset.
    """
    t_pipeline_start = time.perf_counter()
    stage_timings: dict[str, float] = {}

    t0 = time.perf_counter()
    train_seeds = load_problems_jsonl(resolve_project_path(train_seeds_path))
    val_problems = load_problems_jsonl(resolve_project_path(validation_path))
    test_problems = load_problems_jsonl(resolve_project_path(test_path))

    resolved_initial_book = resolve_project_path(initial_book_dataset_path)
    if not resolved_initial_book.is_file():
        logger.info(
            "Initial book dataset not found at %s; building it now from data/chapters/...",
            resolved_initial_book,
        )
        book_initial, _ = build_initial_book_dataset(output_path=resolved_initial_book)
    else:
        book_initial = load_problems_jsonl(resolved_initial_book)
    stage_timings["load_and_clean_sec"] = round(time.perf_counter() - t0, 3)

    rejection_reasons: Counter[str] = Counter()
    rejected_records: list[Stage1RejectedRecord] = []

    # 1. Filter PhysGym training seeds (keep raw archive untouched, exclude >10 logical-step outliers)
    t0 = time.perf_counter()
    accepted_seeds: list[CanonicalProblem] = []
    for seed_prob in train_seeds:
        steps = estimate_logical_steps(seed_prob.reference_solution)
        if steps > max_logical_steps:
            reason = f"physgym_seed_step_count_exceeds_{max_logical_steps} ({steps} steps)"
            rejection_reasons[reason.split(" ")[0]] += 1
            rejected_records.append(
                Stage1RejectedRecord(
                    problem_id=seed_prob.problem_id,
                    domain=seed_prob.domain,
                    source=seed_prob.source,
                    original_id=seed_prob.original_id,
                    reason=reason,
                    logical_steps=steps,
                    problem_statement_preview=seed_prob.problem_statement[:160],
                )
            )
        else:
            accepted_seeds.append(seed_prob)
    stage_timings["seed_filtering_sec"] = round(time.perf_counter() - t0, 3)

    # 2. Split independent subproblems on the 986 book problems
    t0 = time.perf_counter()
    split_candidates: list[CanonicalProblem] = []
    for book_prob in book_initial:
        split_candidates.extend(split_independent_subproblems(book_prob))
    stage_timings["book_subproblem_split_sec"] = round(time.perf_counter() - t0, 3)

    # 3. Run Mandatory Anti-Leakage Gate (cosine_similarity >= 0.85) against all 76 Val + Test problems
    #    Run this BEFORE step/diagram filtering so that Experiment E1's raw pool (01a_leak_free_raw_pool.jsonl)
    #    is also guaranteed 100% leak-free against the 76 held-out Val/Test problems!
    t0 = time.perf_counter()
    leak_free_split_candidates, leaked_matches = run_anti_leakage_gate(
        split_candidates,
        val_problems=val_problems,
        test_problems=test_problems,
        cosine_threshold=anti_leakage_cosine_threshold,
        ngram_containment_threshold=DEFAULT_ANTI_LEAKAGE_NGRAM_CONTAINMENT_THRESHOLD,
    )
    leaked_base_original_ids: set[str] = set()
    for lm in leaked_matches:
        rejection_reasons["anti_leakage_gate"] += 1
        orig_str = str(lm.candidate_original_id or "")
        base_orig = orig_str.split(":part_")[0]
        if base_orig:
            leaked_base_original_ids.add(base_orig)
        rejected_records.append(
            Stage1RejectedRecord(
                problem_id=lm.candidate_id,
                domain="UNKNOWN",
                source="book_initial",
                original_id=lm.candidate_original_id,
                reason=f"{lm.reason} -> {lm.matched_split}:{lm.matched_problem_id}",
                logical_steps=0,
                problem_statement_preview="",
            )
        )

    # Save Intermediate Dataset 1: 01a_leak_free_raw_pool.jsonl (for Experiment E1 solver_raw ablation)
    # Contains all 21 raw PhysGym train seeds + all raw book problems that did not leak against Val/Test,
    # BEFORE CuratorKIT gates, 1-10 step filtering, diagram filtering, and LLM normalization.
    leak_free_raw_pool: list[CanonicalProblem] = list(train_seeds) + [
        bp for bp in book_initial if str(bp.original_id or "") not in leaked_base_original_ids
    ]
    raw_pool_file = resolve_project_path(leak_free_raw_pool_path)
    save_problems_jsonl(leak_free_raw_pool, raw_pool_file)
    stage_timings["anti_leakage_and_raw_pool_sec"] = round(time.perf_counter() - t0, 3)

    # 4. Evaluate strict acceptance gates (1-10 steps, no diagrams, no external refs, single target)
    t0 = time.perf_counter()
    gate_passed_candidates: list[CanonicalProblem] = []
    for cand in leak_free_split_candidates:
        ok, reason, steps = evaluate_self_containment_and_acceptance(
            cand,
            min_logical_steps=min_logical_steps,
            max_logical_steps=max_logical_steps,
        )
        if ok:
            gate_passed_candidates.append(cand)
        else:
            reason_str = reason or "unknown"
            rejection_reasons[reason_str.split(" ")[0]] += 1
            rejected_records.append(
                Stage1RejectedRecord(
                    problem_id=cand.problem_id,
                    domain=cand.domain,
                    source=cand.source,
                    original_id=cand.original_id,
                    reason=reason_str,
                    logical_steps=steps,
                    problem_statement_preview=cand.problem_statement[:160],
                )
            )
    # Save Intermediate Dataset 1B: 01a_manual_deterministic_only_pool.jsonl (for Experiment E1 Model 1)
    # Contains all accepted PhysGym seeds (<=10 steps) + all split book candidates that passed manual
    # deterministic Python rules (no diagrams, no external refs, single target, 1-10 steps),
    # BEFORE any CuratorKIT within-pool deduplication or CuratorKIT quality/diversity gates.
    manual_det_only_pool: list[CanonicalProblem] = [
        CanonicalProblem(
            problem_id=sp.problem_id,
            domain=sp.domain,
            problem_statement=clean_book_problem_text(sp.problem_statement),
            reference_solution=clean_book_solution_text(sp.reference_solution, ensure_answer_sentinel=True),
            source=sp.source,
            original_id=sp.original_id,
        )
        for sp in accepted_seeds
    ] + [
        CanonicalProblem(
            problem_id=cand.problem_id,
            domain=cand.domain,
            problem_statement=cand.problem_statement,
            reference_solution=clean_book_solution_text(cand.reference_solution, ensure_answer_sentinel=True),
            source=cand.source,
            original_id=cand.original_id,
        )
        for cand in gate_passed_candidates
    ]
    manual_det_file = resolve_project_path(manual_deterministic_only_pool_path)
    save_problems_jsonl(manual_det_only_pool, manual_det_file)
    stage_timings["acceptance_gates_sec"] = round(time.perf_counter() - t0, 3)

    # 5. Run Within-Pool Deduplication against accepted PhysGym seeds and each other
    t0 = time.perf_counter()
    unique_book_candidates, pool_dups = deduplicate_candidate_pool(
        gate_passed_candidates,
        seed_problems=accepted_seeds,
        cosine_threshold=DEFAULT_WITHIN_POOL_COSINE_THRESHOLD,
    )
    for dm in pool_dups:
        rejection_reasons["within_pool_duplicate"] += 1
        rejected_records.append(
            Stage1RejectedRecord(
                problem_id=dm.candidate_id,
                domain="UNKNOWN",
                source="book_initial",
                original_id=dm.candidate_original_id,
                reason=f"{dm.reason} -> {dm.matched_problem_id}",
                logical_steps=0,
                problem_statement_preview="",
            )
        )
    stage_timings["within_pool_dedup_sec"] = round(time.perf_counter() - t0, 3)

    # 6. Build and save Intermediate Dataset 2: 01b_deterministic_curated_pool.jsonl
    #    (All valid problems after deterministic + CuratorKIT gates & adaptive recovery, BEFORE optional LLM normalization)
    t0 = time.perf_counter()
    def _deterministic_repair(prob: CanonicalProblem, _diag: str) -> CanonicalProblem:
        return CanonicalProblem(
            problem_id=prob.problem_id,
            domain=prob.domain,
            problem_statement=clean_book_problem_text(prob.problem_statement),
            reference_solution=clean_book_solution_text(prob.reference_solution, ensure_answer_sentinel=True)[:4500],
            source=prob.source,
            original_id=prob.original_id,
        )

    pre_llm_selected, _ = select_balanced_canonical_pool(
        accepted_seeds,
        unique_book_candidates,
        target_total=target_total,
        candidate_validator=None,
    )
    deterministic_curated_pool, ck_rejected, ck_gate_report = run_curatorkit_stage1_gates_and_recovery(
        pre_llm_selected,
        max_solution_chars=4500,
        require_answer_sentinel=True,
        enable_llm_gates=False,
        repair_fn=_deterministic_repair,
    )
    det_curated_file = resolve_project_path(deterministic_curated_pool_path)
    save_problems_jsonl(deterministic_curated_pool, det_curated_file)
    stage_timings["deterministic_curated_pool_sec"] = round(time.perf_counter() - t0, 3)

    # 7. Optional Resumable CohereForAI/aya-expanse-8b Refinement (--use-llm)
    t0 = time.perf_counter()
    if use_llm_refinement:
        llm_validator = build_aya_candidate_validator(
            model_name=llm_model_name,
            backend=llm_backend,
            max_logical_steps=max_logical_steps,
            total_candidates=len(accepted_seeds) + len(unique_book_candidates),
        )
        canonical_train, llm_rejected_records = select_balanced_canonical_pool(
            accepted_seeds,
            unique_book_candidates,
            target_total=target_total,
            candidate_validator=llm_validator,
        )
        for rec in llm_rejected_records:
            rejection_reasons[rec.reason.split(":")[0]] += 1
            rejected_records.append(rec)
        canonical_train, ck_rejected, ck_gate_report = run_curatorkit_stage1_gates_and_recovery(
            canonical_train,
            max_solution_chars=4500,
            require_answer_sentinel=True,
            enable_llm_gates=False,
            repair_fn=_deterministic_repair,
        )
    else:
        canonical_train = deterministic_curated_pool
    stage_timings["llm_refinement_sec"] = round(time.perf_counter() - t0, 3)

    # 8. Write Stage 1 artifacts, rejected log, CuratorKIT provenance bundle, and SHA-256 checksums
    t0 = time.perf_counter()
    cand_file = resolve_project_path(canonical_candidates_path)
    rej_file = resolve_project_path(rejected_output_path)
    pool_file = resolve_project_path(canonical_train_250_path)
    manifest_file = resolve_project_path(manifest_output_path)
    checksums_file = resolve_project_path(checksums_output_path)

    save_problems_jsonl(unique_book_candidates, cand_file)
    save_problems_jsonl(canonical_train, pool_file)

    rej_file.parent.mkdir(parents=True, exist_ok=True)
    with rej_file.open("w", encoding="utf-8") as f:
        for rec in rejected_records:
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")

    counts_by_domain = dict(sorted(Counter(p.domain for p in canonical_train).items()))
    counts_by_source = dict(sorted(Counter(p.source for p in canonical_train).items()))

    tracked_files = [
        resolve_project_path(train_seeds_path),
        resolve_project_path(validation_path),
        resolve_project_path(test_path),
        resolve_project_path(initial_book_dataset_path),
        raw_pool_file,
        det_curated_file,
        cand_file,
        rej_file,
        pool_file,
    ]

    ck_bundle_paths = write_curatorkit_provenance_bundle(
        pool_file.parent,
        stage_id="02_canonical_train",
        accepted_problems=canonical_train,
        rejected_records=[r.to_dict() for r in rejected_records] + ck_rejected,
        gate_report=ck_gate_report,
        summary_dict={
            "counts_by_domain": counts_by_domain,
            "counts_by_source": counts_by_source,
            "stage_timings_sec": stage_timings,
        },
        tracked_files=tracked_files,
        generation_model=llm_model_name,
        runtime_backend=llm_backend,
        wall_clock_seconds=time.perf_counter() - t_pipeline_start,
    )
    stage_timings["provenance_and_save_sec"] = round(time.perf_counter() - t0, 3)
    total_elapsed = round(time.perf_counter() - t_pipeline_start, 3)

    output_paths = {
        "leak_free_raw_pool": str(raw_pool_file),
        "manual_deterministic_only_pool_e1": str(manual_det_file),
        "deterministic_curated_pool_e2": str(det_curated_file),
        "canonical_candidates": str(cand_file),
        "rejected_candidates": str(rej_file),
        "canonical_train_e3": str(pool_file),
        "canonical_train": str(pool_file),
        "canonical_train_250": str(pool_file),
        "manifest": str(manifest_file),
        "curatorkit_manifest": str(ck_bundle_paths["manifest"]),
        "curatorkit_rejected": str(ck_bundle_paths["rejected"]),
        "dataset_card": str(ck_bundle_paths["dataset_card"]),
        "lexsi_provenance": str(ck_bundle_paths["lexsi_provenance"]),
        "checksums": str(checksums_file),
    }

    summary = CanonicalPoolSummary(
        physgym_seeds_total=len(train_seeds),
        physgym_seeds_accepted=len(accepted_seeds),
        physgym_seeds_rejected=len(train_seeds) - len(accepted_seeds),
        initial_book_problems_total=len(book_initial),
        leak_free_raw_pool_count=len(leak_free_raw_pool),
        candidates_after_subproblem_split=len(split_candidates),
        candidates_passing_acceptance_gates=len(gate_passed_candidates),
        candidates_purged_by_anti_leakage=len(leaked_matches),
        candidates_purged_by_within_pool_dedup=len(pool_dups),
        canonical_candidates_count=len(unique_book_candidates),
        deterministic_curated_pool_count=len(deterministic_curated_pool),
        canonical_train_count=len(canonical_train),
        canonical_train_250_count=len(canonical_train),
        counts_by_domain=counts_by_domain,
        counts_by_source=counts_by_source,
        rejection_reasons=dict(sorted(rejection_reasons.items())),
        llm_refinement_applied=use_llm_refinement,
        stage_timings_sec=stage_timings,
        total_elapsed_sec=total_elapsed,
        output_paths=output_paths,
    )

    manifest_payload = {
        "schema_version": 1,
        "stage": "02_canonical_train",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "author": "vital-ar",
        "target_total_cap": target_total,
        "curatorkit_gate_report": ck_gate_report.to_dict(),
        "stage_timings_sec": stage_timings,
        "total_elapsed_sec": total_elapsed,
        "acceptance_criteria": {
            "logical_step_range": [min_logical_steps, max_logical_steps],
            "anti_leakage_cosine_threshold": anti_leakage_cosine_threshold,
            "single_target_only": True,
            "reject_diagrams_and_external_refs": True,
            "schema_fields": [
                "problem_id",
                "domain",
                "problem_statement",
                "reference_solution",
                "source",
                "original_id",
            ],
        },
        "summary": summary.to_dict(),
    }
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # Write SHA-256 checksums.txt
    tracked_files = [
        resolve_project_path(train_seeds_path),
        resolve_project_path(validation_path),
        resolve_project_path(test_path),
        resolve_project_path(initial_book_dataset_path),
        raw_pool_file,
        manual_det_file,
        det_curated_file,
        cand_file,
        rej_file,
        pool_file,
        manifest_file,
    ]
    checksum_lines = []
    for tf in tracked_files:
        if tf.is_file():
            checksum_lines.append(f"{_compute_file_sha256(tf)}  {tf.name}")
    checksums_file.parent.mkdir(parents=True, exist_ok=True)
    checksums_file.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    logger.info(
        "Stage 1 Canonical Pool complete in %.2fs (timings=%s): %d problems (%s) written to %s",
        total_elapsed,
        stage_timings,
        len(canonical_train),
        counts_by_domain,
        pool_file,
    )
    return canonical_train, summary
