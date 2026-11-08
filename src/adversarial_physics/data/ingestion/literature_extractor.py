"""Stage 1.2: Literature and textbook physics problem extraction & filtering via CuratorKIT / Aya-Expanse."""
from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from adversarial_physics.data.ingestion.book_downloader import (
    DEFAULT_DOWNLOAD_DIR,
    download_books,
    resolve_book_path,
)
from adversarial_physics.data.ingestion.book_loader import (
    BookChunk,
    load_book_segments,
    load_books_from_sources,
)
from adversarial_physics.data.ingestion.solution_mapping import (
    LinkedProblem,
    LinkingReport,
    ProblemBookLayout,
    SolutionEntry,
    SolutionManualLayout,
    build_linking_report,
    format_paired_text,
    link_problems_to_solutions,
    parse_solution_manual,
    segment_problems,
)
from adversarial_physics.data.processing.normalization import normalize_domain
from adversarial_physics.data.schemas.problem import CanonicalProblem, save_problems_jsonl
from adversarial_physics.utils.config import load_yaml, resolve_project_path
from adversarial_physics.utils.logging import get_logger
from adversarial_physics.utils.runtime import (
    AYA_EXPANSE_8B,
    BackendName,
    build_aya_chat_generator,
)

logger = get_logger("stage_1_2_literature_extractor")

DEFAULT_DIAGRAM_PATTERNS: tuple[str, ...] = (
    r"\bfigure\s+(?:\d+|[a-z]{1,2}\b)",
    r"\bfig\.\s*(?:\d+|[a-z]{1,2}\b)",
    r"\bas\s+shown\s+in\s+(?:the\s+)?(?:figure|diagram|graph|picture|sketch|illustration|image|photo)",
    r"\b(?:the\s+)?(?:figure|diagram|graph|photo|picture|illustration)\s+(?:shows|depicts|illustrates)\b",
    r"\b(?:in|from|on|below|above)\s+the\s+(?:accompanying\s+|inset\s+)?(?:figure|diagram|graph|photo|table|chart)\b",
    r"\brefer(?:ring)?\s+to\s+(?:the\s+)?(?:figure|diagram|graph|table|chart)",
    r"\b(?:draw|sketch|redraw|trace)\s+(?:a\s+|the\s+|all\s+)?(?:graph|plot|plots|ray\s+diagram|diagram|shape|trajectory|diffraction\s+pattern|four\s+panels)",
    r"\bshown\s+below\b",
    r"\bsee\s+(?:the\s+)?(?:figure|diagram|graph)\b",
)

_MULTIPART_MARKER_RE = re.compile(
    r"(?:^|\s)(?:\([a-d]\)|\bpart\s+[a-d]\b)",
    flags=re.IGNORECASE,
)

EXTRACTION_SYSTEM_PROMPT = """You are an expert physics curriculum designer.
Extract standalone university-level physics problems from the provided textbook chunk and return a JSON list of problem objects.

Strict Extraction & Formatting Rules (Stage 1.2):
1. 1-to-10 Logical Step Derivations: Select problems solvable in 1 to 10 logical physical/mathematical operations (counting distinct reasoning steps rather than displayed equation lines) so the complete derivation fits within 1536 output tokens.
2. No Diagrams or Missing External References: Reject any problem that depends on a visual figure, diagram, graph, table, or another problem/page (e.g., "refer to Figure 3.2", "in problem 5 on page 1028").
3. Split Multi-Part Questions: If a textbook question asks for multiple targets (e.g., "(a) find acceleration and (b) find tension"), split them into standalone, single-target problems only when subproblems are independently solvable.
4. Valid Domains: Classify each problem into one of: MECHANICS, ELECTROMAGNETISM, THERMODYNAMICS, OPTICS, MODERN_PHYSICS, RELATIVITY.
5. Official Solutions: If the excerpt contains an "OFFICIAL SOLUTION" block, it is ground truth from the textbook author. Base `reference_solution` on it and never contradict it.

Return ONLY a valid JSON array where each element has the keys:
- "domain": string (one of the 6 domains)
- "derivation_steps": integer (estimated number of logical physical/mathematical operations, 1 to 10)
- "problem_statement": string (self-contained single-target problem statement)
- "reference_solution": string (clear step-by-step derivation ending with ANSWER: <final answer>)
"""


@dataclass(frozen=True)
class RejectedLiteratureCandidate:
    """Record of a candidate textbook problem rejected during Stage 1.2 filtering."""

    candidate_index: int
    book_id: str
    chunk_id: str
    reason: str
    raw_candidate: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LiteratureExtractionSummary:
    """Summary of a Stage 1.2 textbook extraction run."""

    books_loaded: int
    chunks_processed: int
    candidates_extracted: int
    accepted_count: int
    rejected_count: int
    accepted_by_domain: dict[str, int]
    rejection_reasons: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def is_diagram_dependent(
    text: str,
    *,
    custom_patterns: list[str] | tuple[str, ...] | None = None,
) -> bool:
    """Return True if a problem statement or solution references a visual diagram/figure."""
    patterns = custom_patterns if custom_patterns is not None else DEFAULT_DIAGRAM_PATTERNS
    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return True
    return False


def has_unsplit_multipart(problem_statement: str) -> bool:
    """Return True if a problem statement still contains multiple sub-question markers like (a) and (b)."""
    matches = _MULTIPART_MARKER_RE.findall(problem_statement)
    unique_markers = {m.strip().lower() for m in matches}
    return ("(a)" in unique_markers and "(b)" in unique_markers) or (
        "part a" in unique_markers and "part b" in unique_markers
    )


def estimate_derivation_steps(solution_text: str, explicit_steps: int | None = None) -> int:
    """Estimate the number of derivation steps in a reference solution."""
    if explicit_steps is not None and explicit_steps > 0:
        return int(explicit_steps)
    lines = [line.strip() for line in solution_text.splitlines() if line.strip()]
    eq_lines = [line for line in lines if "=" in line]
    if eq_lines:
        return len(eq_lines)
    sentences = [s.strip() for s in re.split(r"[.!?]+", solution_text) if s.strip()]
    return max(1, len(sentences))


def validate_and_build_canonical_problem(
    candidate: dict[str, Any],
    *,
    problem_id: str,
    book_id: str,
    chunk_id: str,
    min_steps: int = 1,
    max_steps: int = 10,
    max_solution_chars: int = 4500,
    reject_diagrams: bool = True,
    reject_unsplit_multipart: bool = True,
    diagram_patterns: list[str] | None = None,
) -> tuple[CanonicalProblem | None, str | None]:
    """Apply Stage 1.2 filters to a raw candidate dictionary.

    Returns:
        Tuple of (CanonicalProblem or None, rejection_reason or None).
    """
    statement = str(candidate.get("problem_statement") or "").strip()
    solution = str(candidate.get("reference_solution") or "").strip()

    if not statement or not solution:
        return None, "missing_required_fields"

    if reject_diagrams and (
        is_diagram_dependent(statement, custom_patterns=diagram_patterns)
        or is_diagram_dependent(solution, custom_patterns=diagram_patterns)
    ):
        return None, "diagram_dependent"

    if reject_unsplit_multipart and has_unsplit_multipart(statement):
        return None, "unsplit_multipart_question"

    explicit_steps = candidate.get("derivation_steps")
    steps = estimate_derivation_steps(
        solution,
        int(explicit_steps) if isinstance(explicit_steps, (int, float)) else None,
    )
    if steps < min_steps or steps > max_steps:
        return None, f"derivation_steps_out_of_bounds ({steps} not in [{min_steps}, {max_steps}])"

    if len(solution) > max_solution_chars:
        return None, f"solution_too_long ({len(solution)} chars > {max_solution_chars})"

    raw_domain = str(candidate.get("domain") or "MECHANICS")
    domain = normalize_domain(raw_domain, canonicalize=True)

    problem = CanonicalProblem(
        problem_id=problem_id,
        domain=domain,
        problem_statement=statement,
        reference_solution=solution,
        source="literature_curated",
        original_id=f"{book_id}:{chunk_id}",
    )
    return problem, None


def _parse_json_candidates(raw_response: str) -> list[dict[str, Any]]:
    """Parse a JSON array of candidate problem dictionaries from an LLM response."""
    cleaned = raw_response.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start != -1 and end != -1 and end > start:
            parsed = json.loads(cleaned[start : end + 1])
        else:
            raise

    if isinstance(parsed, dict):
        if "problems" in parsed and isinstance(parsed["problems"], list):
            return [x for x in parsed["problems"] if isinstance(x, dict)]
        return [parsed]
    if isinstance(parsed, list):
        return [x for x in parsed if isinstance(x, dict)]
    return []


def build_aya_chunk_extractor(
    *,
    model_name: str | None = AYA_EXPANSE_8B,
    backend: BackendName = "auto",
    temperature: float = 0.2,
    max_new_tokens: int = 1536,
) -> Callable[[BookChunk], list[dict[str, Any]]]:
    """Create a callable that extracts candidate physics problems from a BookChunk via Aya-Expanse (8B by default)."""
    chat_fn = build_aya_chat_generator(
        task="literature_extraction",
        model_name=model_name,
        backend=backend,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )

    def _extract(chunk: BookChunk) -> list[dict[str, Any]]:
        domain_note = f"\nPreferred domain hint: {chunk.domain_hint}" if chunk.domain_hint else ""
        user_prompt = (
            f"BOOK ID: {chunk.book_id} | CHUNK ID: {chunk.chunk_id}{domain_note}\n\n"
            f"TEXTBOOK EXCERPT:\n{chunk.text}"
        )
        raw_text = chat_fn(EXTRACTION_SYSTEM_PROMPT, user_prompt)
        return _parse_json_candidates(raw_text or "[]")

    return _extract


def _format_domain_short(domain: str) -> str:
    mapping = {
        "MECHANICS": "mech",
        "ELECTROMAGNETISM": "em",
        "THERMODYNAMICS": "thermo",
        "OPTICS": "optics",
        "MODERN_PHYSICS": "modern",
        "RELATIVITY": "rel",
    }
    return mapping.get(domain.upper(), "phys")


def extract_literature_problems(
    chunks: list[BookChunk],
    *,
    chunk_extractor: Callable[[BookChunk], list[dict[str, Any]]] | None = None,
    target_count: int = 229,
    min_steps: int = 1,
    max_steps: int = 10,
    max_solution_chars: int = 4500,
    reject_diagrams: bool = True,
    reject_unsplit_multipart: bool = True,
    id_prefix: str = "lit",
) -> tuple[list[CanonicalProblem], list[RejectedLiteratureCandidate], LiteratureExtractionSummary]:
    """Process textbook chunks into verified CanonicalProblem records for Stage 1.2."""
    accepted: list[CanonicalProblem] = []
    rejected: list[RejectedLiteratureCandidate] = []
    accepted_by_domain: dict[str, int] = {}
    rejection_reasons: dict[str, int] = {}

    candidate_counter = 0
    chunks_processed = 0
    seen_books = {c.book_id for c in chunks}

    for chunk in chunks:
        if len(accepted) >= target_count:
            break
        chunks_processed += 1

        raw_record = chunk.metadata.get("raw_record") if chunk.metadata else None
        if isinstance(raw_record, dict) and "problem_statement" in raw_record and chunk_extractor is None:
            raw_candidates = [raw_record]
        elif chunk_extractor is not None:
            raw_candidates = chunk_extractor(chunk)
        else:
            logger.warning(
                "Skipping chunk %s: no chunk_extractor provided and chunk is not a pre-extracted candidate.",
                chunk.chunk_id,
            )
            continue

        for cand in raw_candidates:
            if len(accepted) >= target_count:
                break
            candidate_counter += 1
            raw_domain = normalize_domain(str(cand.get("domain") or chunk.domain_hint or "MECHANICS"), canonicalize=True)
            domain_slug = _format_domain_short(raw_domain)
            problem_id = str(cand.get("problem_id") or f"{id_prefix}_{domain_slug}_{len(accepted) + 1:04d}")
            problem_ref = str(chunk.metadata.get("problem_ref") or "").strip() if chunk.metadata else ""

            problem, reject_reason = validate_and_build_canonical_problem(
                cand,
                problem_id=problem_id,
                book_id=chunk.book_id,
                chunk_id=chunk.chunk_id,
                min_steps=min_steps,
                max_steps=max_steps,
                max_solution_chars=max_solution_chars,
                reject_diagrams=reject_diagrams,
                reject_unsplit_multipart=reject_unsplit_multipart,
            )

            if problem is not None and problem_ref:
                problem = CanonicalProblem(
                    **{**problem.to_dict(), "original_id": problem_ref}
                )

            if problem is not None:
                accepted.append(problem)
                accepted_by_domain[problem.domain] = accepted_by_domain.get(problem.domain, 0) + 1
            else:
                reason_key = (reject_reason or "unknown").split(":")[0]
                rejection_reasons[reason_key] = rejection_reasons.get(reason_key, 0) + 1
                rejected.append(
                    RejectedLiteratureCandidate(
                        candidate_index=candidate_counter,
                        book_id=chunk.book_id,
                        chunk_id=chunk.chunk_id,
                        reason=reject_reason or "unknown",
                        raw_candidate=cand,
                    )
                )

    summary = LiteratureExtractionSummary(
        books_loaded=len(seen_books),
        chunks_processed=chunks_processed,
        candidates_extracted=candidate_counter,
        accepted_count=len(accepted),
        rejected_count=len(rejected),
        accepted_by_domain=accepted_by_domain,
        rejection_reasons=rejection_reasons,
    )
    return accepted, rejected, summary


def _save_linking_outputs(
    reports: list[LinkingReport],
    linked: list[LinkedProblem],
    pairing_cfg: dict[str, Any],
) -> None:
    """Persist problem<->solution linking coverage and pairs (for inspection / reuse)."""
    report_path = pairing_cfg.get("linking_report_path")
    if report_path and reports:
        out = resolve_project_path(report_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps([r.to_dict() for r in reports], indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Saved linking report to %s", out)
    pairs_path = pairing_cfg.get("linked_pairs_path")
    if pairs_path and linked:
        out = resolve_project_path(pairs_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as file:
            for item in linked:
                file.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
        logger.info("Saved %d linked problem/solution pairs to %s", len(linked), out)


def _solution_layout_for(entry: dict[str, Any]) -> SolutionManualLayout:
    return SolutionManualLayout.from_dict(entry.get("layout"))


def _problem_layout_for(entry: dict[str, Any]) -> ProblemBookLayout:
    return ProblemBookLayout.from_dict(entry.get("layout"))


def build_paired_problem_chunks(
    sources: list[dict[str, Any]],
    *,
    download_dir: str | Path = DEFAULT_DOWNLOAD_DIR,
    require_official_solution: bool = True,
    shuffle_seed: int | None = 42,
) -> tuple[list[BookChunk], list[LinkingReport], list[LinkedProblem], set[str]]:
    """Segment problem books into individual problems and attach official solutions."""
    enabled = [e for e in sources if e.get("enabled", True)]
    by_id = {str(e.get("book_id")): e for e in enabled if e.get("book_id")}

    def local_file(entry: dict[str, Any]) -> Path | None:
        path = resolve_book_path(entry, download_dir=download_dir)
        return path if path is not None and path.is_file() else None

    referenced = {str(e.get("solution_book_id")) for e in enabled if e.get("solution_book_id")}
    solution_indexes: dict[str, dict[str, dict[tuple[int, int], SolutionEntry]]] = {}
    for sol_id in sorted(referenced):
        sol_entry = by_id.get(sol_id)
        if sol_entry is None or str(sol_entry.get("role", "")).lower() != "solution_book":
            logger.warning("solution_book_id %r is not an enabled 'solution_book' entry; skipping pairing.", sol_id)
            continue
        sol_path = local_file(sol_entry)
        if sol_path is None:
            logger.warning("Solution book %r is not on disk yet (download it first); skipping pairing.", sol_id)
            continue
        segments = load_book_segments(
            sol_path,
            file_format=sol_entry.get("format"),
            start_page=int(sol_entry.get("start_page") or 1),
            end_page=int(sol_entry["end_page"]) if sol_entry.get("end_page") is not None else None,
        )
        solution_indexes[sol_id] = parse_solution_manual(
            segments,
            solution_book_id=sol_id,
            section_titles=list(sol_entry.get("sections") or []),
            layout=_solution_layout_for(sol_entry),
        )
        counts = {title or "<all>": len(v) for title, v in solution_indexes[sol_id].items()}
        logger.info("Parsed solution book %s: %s", sol_id, counts)

    chunks: list[BookChunk] = []
    reports: list[LinkingReport] = []
    all_linked: list[LinkedProblem] = []
    handled: set[str] = set(solution_indexes)

    for entry in enabled:
        book_id = str(entry.get("book_id") or "")
        sol_id = str(entry.get("solution_book_id") or "")
        if str(entry.get("role", "")).lower() != "problem_book" or sol_id not in solution_indexes:
            continue
        book_path = local_file(entry)
        if book_path is None:
            logger.warning("Problem book %r is not on disk yet; skipping.", book_id)
            continue

        segments = load_book_segments(
            book_path,
            file_format=entry.get("format"),
            start_page=int(entry.get("start_page") or 1),
            end_page=int(entry["end_page"]) if entry.get("end_page") is not None else None,
        )
        section = entry.get("solution_section")
        problems = segment_problems(
            segments,
            book_id=book_id,
            solution_book_id=sol_id,
            solution_section=section,
            layout=_problem_layout_for(entry),
        )
        linked = link_problems_to_solutions(problems, solution_indexes)
        report = build_linking_report(linked, book_id=book_id, solution_book_id=sol_id, solution_section=section)
        reports.append(report)
        all_linked.extend(linked)
        handled.add(book_id)
        logger.info(
            "Linked %s -> %s[%s]: %d problems, %d matched, %d unmatched",
            book_id, sol_id, section, report.problems_found, report.matched, report.unmatched,
        )

        title = str(entry.get("title") or book_id)
        for item in linked:
            if require_official_solution and not item.matched:
                continue
            p = item.problem
            chunks.append(
                BookChunk(
                    chunk_id=f"{book_id}_ch{p.chapter:02d}_p{p.number:03d}",
                    book_id=book_id,
                    source_path=str(book_path),
                    format=str(entry.get("format") or book_path.suffix.lstrip(".")),
                    text=format_paired_text(item, book_title=title),
                    role="problem_book",
                    solution_book_id=sol_id,
                    page_start=p.page_start,
                    page_end=p.page_end,
                    domain_hint=entry.get("domain"),
                    metadata={
                        "title": title,
                        "url": str(entry.get("url") or ""),
                        "problem_ref": p.problem_ref,
                        "chapter": p.chapter,
                        "problem_number": p.number,
                        "solution_section": section,
                        "solution_key": item.solution.key if item.solution else None,
                        "solution_page": item.solution.page_start if item.solution else None,
                        "inline_solution_page": p.inline_solution_page,
                    },
                )
            )

    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(chunks)
    return chunks, reports, all_linked, handled


def run_literature_extraction_from_config(
    config_path: str | Path = "configs/data/book_ingestion.yaml",
    *,
    book_paths: list[str] | None = None,
    book_ids: list[str] | None = None,
    target_count: int | None = None,
    min_steps: int | None = None,
    max_steps: int | None = None,
    require_official_solution: bool | None = None,
    download: bool | None = None,
    backend: BackendName = "auto",
    model_name: str | None = None,
    output_jsonl_path: str | Path | None = None,
    rejected_jsonl_path: str | Path | None = None,
    chunk_extractor: Callable[[BookChunk], list[dict[str, Any]]] | None = None,
    dry_run: bool = False,
) -> tuple[list[CanonicalProblem], list[RejectedLiteratureCandidate], LiteratureExtractionSummary]:
    """Load book ingestion configuration, ingest PDF/DOCX/JSONL books, and extract verified problems."""
    cfg = load_yaml(config_path)
    sources_cfg = list(cfg.get("books", []))
    chunking_cfg = cfg.get("chunking", {})
    filters_cfg = cfg.get("filtering", {})
    extraction_cfg = cfg.get("extraction", {})
    output_cfg = cfg.get("output", {})
    download_cfg = cfg.get("download", {}) or {}
    pairing_cfg = cfg.get("pairing", {}) or {}
    runtime_cfg = cfg.get("runtime", {}) or {}
    download_dir = str(download_cfg.get("dir") or DEFAULT_DOWNLOAD_DIR)

    if book_paths:
        sources_cfg = [
            {
                "book_id": Path(p).stem,
                "path": p,
                "url": "",
                "format": Path(p).suffix.lstrip(".").lower() or "pdf",
                "enabled": True,
            }
            for p in book_paths
        ]

    if book_ids:
        wanted = set(book_ids)
        wanted |= {
            str(e.get("solution_book_id"))
            for e in sources_cfg
            if e.get("book_id") in wanted and e.get("solution_book_id")
        }
        sources_cfg = [{**e, "enabled": True} for e in sources_cfg if e.get("book_id") in wanted]

    eff_download = download if download is not None else bool(download_cfg.get("enabled", False))
    if eff_download and not dry_run:
        download_books(sources_cfg, download_dir=download_dir, timeout=float(download_cfg.get("timeout_s", 120)))

    eff_require_solution = (
        require_official_solution
        if require_official_solution is not None
        else bool(pairing_cfg.get("require_official_solution", True))
    )
    seed_value = pairing_cfg.get("shuffle_seed", 42)
    paired_chunks, linking_reports, linked_problems, handled_ids = build_paired_problem_chunks(
        sources_cfg,
        download_dir=download_dir,
        require_official_solution=eff_require_solution,
        shuffle_seed=int(seed_value) if seed_value is not None else None,
    )
    _save_linking_outputs(linking_reports, linked_problems, pairing_cfg)

    plain_sources = [
        e
        for e in sources_cfg
        if str(e.get("book_id")) not in handled_ids and str(e.get("role", "standalone")).lower() != "solution_book"
    ]
    plain_chunks = load_books_from_sources(
        plain_sources,
        default_chunk_size_chars=int(chunking_cfg.get("chunk_size_chars", 4000)),
        default_chunk_overlap_chars=int(chunking_cfg.get("chunk_overlap_chars", 400)),
        skip_empty_paths=True,
        skip_missing_files=True,
        download_dir=download_dir,
    )
    chunks = paired_chunks + plain_chunks

    if not chunks:
        logger.info(
            "No books loaded yet (blank paths/URLs or files not downloaded). "
            "Run `python scripts/02_extract_literature.py --download-only`, set paths in %s, "
            "or pass --book-path to extract problems.",
            config_path,
        )
        empty_summary = LiteratureExtractionSummary(
            books_loaded=0,
            chunks_processed=0,
            candidates_extracted=0,
            accepted_count=0,
            rejected_count=0,
            accepted_by_domain={},
            rejection_reasons={},
        )
        return [], [], empty_summary

    eff_target = target_count if target_count is not None else int(extraction_cfg.get("target_problem_count", 229))
    eff_min_steps = min_steps if min_steps is not None else int(filters_cfg.get("min_derivation_steps", 1))
    eff_max_steps = max_steps if max_steps is not None else int(filters_cfg.get("max_derivation_steps", 10))

    if chunk_extractor is None and not dry_run:
        has_raw_text_chunks = any(
            not (isinstance(c.metadata.get("raw_record"), dict) and "problem_statement" in c.metadata["raw_record"])
            for c in chunks
        )
        if has_raw_text_chunks:
            eff_backend = backend if backend != "auto" else str(runtime_cfg.get("backend", "auto"))  # type: ignore[assignment]
            eff_model = model_name or str(extraction_cfg.get("teacher_model", AYA_EXPANSE_8B))
            chunk_extractor = build_aya_chunk_extractor(
                model_name=eff_model,
                backend=eff_backend,  # type: ignore[arg-type]
                temperature=float(extraction_cfg.get("temperature", 0.2)),
            )

    accepted, rejected, summary = extract_literature_problems(
        chunks,
        chunk_extractor=chunk_extractor,
        target_count=eff_target,
        min_steps=eff_min_steps,
        max_steps=eff_max_steps,
        max_solution_chars=int(filters_cfg.get("max_solution_chars", 4500)),
        reject_diagrams=bool(filters_cfg.get("reject_diagram_dependent", True)),
        reject_unsplit_multipart=bool(filters_cfg.get("split_or_reject_multipart", True)),
        id_prefix=str(extraction_cfg.get("id_prefix", "lit")),
    )

    eff_out = output_jsonl_path or output_cfg.get("extracted_problems_path")
    if eff_out and accepted:
        out_path = resolve_project_path(eff_out)
        save_problems_jsonl(accepted, out_path)
        logger.info("Saved %d accepted literature problems to %s", len(accepted), out_path)

    eff_rej = rejected_jsonl_path or output_cfg.get("rejected_problems_path")
    if eff_rej and rejected:
        rej_path = resolve_project_path(eff_rej)
        rej_path.parent.mkdir(parents=True, exist_ok=True)
        with rej_path.open("w", encoding="utf-8") as file:
            for item in rejected:
                file.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
        logger.info("Saved %d rejected candidates to %s", len(rejected), rej_path)

    return accepted, rejected, summary
