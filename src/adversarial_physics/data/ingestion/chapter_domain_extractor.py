"""Stage 1.2: Manual chapter-name-to-domain mapping and initial book dataset builder.

All 97 chapter exercise PDFs across the 5 downloaded Crowell books under ``data/chapters/<book_id>/``
are manually classified—strictly by their chapter title—into one of the 6 canonical physics domains:
  * ``MECHANICS``
  * ``ELECTROMAGNETISM``
  * ``THERMODYNAMICS``
  * ``OPTICS``
  * ``MODERN_PHYSICS``
  * ``RELATIVITY``
or placed into ``SIDE`` when the chapter title is a general introduction, unit conversion/scaling,
pure math/calculus review, or general study of randomness that does not belong to a specific physics domain.

Chapters assigned to ``SIDE`` are excluded when extracting problems and pairing them with their official
worked solutions from ``<book_id>_solutions.pdf`` to form the initial book dataset
(``data/literature/initial_book_dataset.jsonl``).
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from adversarial_physics.data.curation.curatorkit_stage1 import apply_curatorkit_text_cleaner
from adversarial_physics.data.ingestion.book_loader import load_pdf_pages
from adversarial_physics.data.ingestion.solution_mapping import (
    ExtractedProblem,
    LinkedProblem,
    ProblemBookLayout,
    SolutionEntry,
    SolutionManualLayout,
    link_problems_to_solutions,
    parse_solution_manual,
    segment_problems,
)
from adversarial_physics.data.schemas.problem import CanonicalProblem, save_problems_jsonl
from adversarial_physics.utils.config import resolve_project_path
from adversarial_physics.utils.logging import get_logger

logger = get_logger("stage_1_2_chapter_domain_extractor")

VALID_CANONICAL_DOMAINS: frozenset[str] = frozenset(
    {
        "MECHANICS",
        "ELECTROMAGNETISM",
        "THERMODYNAMICS",
        "OPTICS",
        "MODERN_PHYSICS",
        "RELATIVITY",
    }
)
SIDE_DOMAIN = "SIDE"

DEFAULT_INITIAL_BOOK_DATASET_PATH = "data/literature/initial_book_dataset.jsonl"
DEFAULT_CHAPTER_DOMAIN_SUMMARY_PATH = "data/literature/initial_book_dataset_summary.json"

# ======================================================================================
# Manual Chapter-Name -> Domain Mapping (decided strictly by chapter title)
# ======================================================================================
MANUAL_CHAPTER_DOMAIN_MAP: dict[str, dict[str, str]] = {
    "crowell_conceptual_physics": {
        "01_conservation_of_mass_and_energy": "MECHANICS",
        "02_conservation_of_momentum": "MECHANICS",
        "03_conservation_of_angular_momentum": "MECHANICS",
        "04_relativity": "RELATIVITY",
        "05_electricity": "ELECTROMAGNETISM",
        "06_fields": "ELECTROMAGNETISM",
        "07_the_ray_model_of_light": "OPTICS",
        "08_waves": "MECHANICS",
    },
    "crowell_fields_and_circuits": {
        "01_the_electric_and_magnetic_fields": "ELECTROMAGNETISM",
        "02_gausss_law": "ELECTROMAGNETISM",
        "03_models_of_matter": "MODERN_PHYSICS",
        "04_the_electric_potential": "ELECTROMAGNETISM",
        "05_electromagnetism": "ELECTROMAGNETISM",
        "06_radiation": "ELECTROMAGNETISM",
        "07_more_about_relativity": "RELATIVITY",
        "08_electrical_resistance": "ELECTROMAGNETISM",
        "09_parallel_and_series_circuits": "ELECTROMAGNETISM",
        "10_iterated_integrals": "SIDE",
        "11_sources_of_magnetism": "ELECTROMAGNETISM",
        "12_review_of_oscillations_resonance_and_complex_numbers": "SIDE",
        "13_ac_circuits": "ELECTROMAGNETISM",
        "14_impedance": "ELECTROMAGNETISM",
        "15_stokess_theorem": "SIDE",
        "17_relativity": "RELATIVITY",
    },
    "crowell_mechanics": {
        "00_introduction_and_review": "SIDE",
        "01_scaling_and_estimation": "SIDE",
        "02_velocity_and_relative_motion": "MECHANICS",
        "03_acceleration_and_free_fall": "MECHANICS",
        "04_force_and_motion": "MECHANICS",
        "05_analysis_of_forces": "MECHANICS",
        "06_newtons_laws_in_three_dimensions": "MECHANICS",
        "07_vectors": "SIDE",
        "08_vectors_and_motion": "MECHANICS",
        "09_circular_motion": "MECHANICS",
        "10_gravity": "MECHANICS",
        "11_conservation_of_energy": "MECHANICS",
        "12_simplifying_the_energy_zoo": "MECHANICS",
        "13_work_the_transfer_of_mechanical_energy": "MECHANICS",
        "14_conservation_of_momentum": "MECHANICS",
        "15_conservation_of_angular_momentum": "MECHANICS",
        "16_vibrations": "MECHANICS",
        "17_resonance": "MECHANICS",
    },
    "crowell_modern_physics": {
        "01_time": "RELATIVITY",
        "02_waves": "MECHANICS",
        "03_electromagnetic_waves": "ELECTROMAGNETISM",
        "04_the_lorentz_transformation": "RELATIVITY",
        "05_waves_done_medium_well": "MECHANICS",
        "06_relativistic_energy_and_momentum": "RELATIVITY",
        "07_statistics_and_the_ideal_gas": "THERMODYNAMICS",
        "08_the_macroscopic_picture": "THERMODYNAMICS",
        "09_entropy": "THERMODYNAMICS",
        "10_images_qualitatively": "OPTICS",
        "11_images_quantiitatively": "OPTICS",
        "12_wave_optics": "OPTICS",
        "13_the_atom_and_the_nucleus": "MODERN_PHYSICS",
        "14_probability_distributions_and_a_first_glimpse_of_quantum_physics": "MODERN_PHYSICS",
        "15_light_as_a_particle": "MODERN_PHYSICS",
        "16_matter_as_a_wave": "MODERN_PHYSICS",
        "17_the_schrodinger_equation": "MODERN_PHYSICS",
        "18_quantization_of_angular_momentum": "MODERN_PHYSICS",
    },
    "light_and_matter_problems": {
        "00_introduction_and_review": "SIDE",
        "01_scaling_and_estimation": "SIDE",
        "02_velocity_and_relative_motion": "MECHANICS",
        "03_acceleration_and_free_fall": "MECHANICS",
        "04_force_and_motion": "MECHANICS",
        "05_analysis_of_forces": "MECHANICS",
        "06_newtons_laws_in_three_dimensions": "MECHANICS",
        "07_vectors": "SIDE",
        "08_vectors_and_motion": "MECHANICS",
        "09_circular_motion": "MECHANICS",
        "10_gravity": "MECHANICS",
        "11_conservation_of_energy": "MECHANICS",
        "12_simplifying_the_energy_zoo": "MECHANICS",
        "13_work_the_transfer_of_mechanical_energy": "MECHANICS",
        "14_conservation_of_momentum": "MECHANICS",
        "15_conservation_of_angular_momentum": "MECHANICS",
        "16_thermodynamics": "THERMODYNAMICS",
        "17_vibrations": "MECHANICS",
        "18_resonance": "MECHANICS",
        "19_free_waves": "MECHANICS",
        "20_bounded_waves": "MECHANICS",
        "21_electricity_and_circuits": "ELECTROMAGNETISM",
        "22_the_nonmechanical_universe": "ELECTROMAGNETISM",
        "23_relativity_and_magnetism": "ELECTROMAGNETISM",
        "24_electromagnetism": "ELECTROMAGNETISM",
        "25_capacitance_and_inductance": "ELECTROMAGNETISM",
        "26_the_atom_and_e_mc_2": "MODERN_PHYSICS",
        "27_general_relativity": "RELATIVITY",
        "28_the_ray_model_of_light": "OPTICS",
        "29_images_by_reflection": "OPTICS",
        "30_images_quantitatively": "OPTICS",
        "31_refraction": "OPTICS",
        "32_wave_optics": "OPTICS",
        "33_rules_of_randomness": "SIDE",
        "34_light_as_a_particle": "MODERN_PHYSICS",
        "35_matter_as_a_wave": "MODERN_PHYSICS",
        "36_the_atom": "MODERN_PHYSICS",
    },
}

_DOMAIN_SLUG: dict[str, str] = {
    "MECHANICS": "mech",
    "ELECTROMAGNETISM": "em",
    "THERMODYNAMICS": "thermo",
    "OPTICS": "optics",
    "MODERN_PHYSICS": "modern",
    "RELATIVITY": "rel",
}


@dataclass(frozen=True)
class InitialBookDatasetSummary:
    """Summary of the initial book dataset extracted from ``data/chapters/``."""

    books_processed: int
    included_chapters_count: int
    excluded_side_chapters_count: int
    total_problems_in_included_chapters: int
    matched_problems_with_solutions: int
    unmatched_problems_skipped: int
    curatorkit_hygiene_applied: bool
    counts_by_domain: dict[str, int] = field(default_factory=dict)
    counts_by_book: dict[str, int] = field(default_factory=dict)
    excluded_side_chapters: list[str] = field(default_factory=list)
    stage_timings_sec: dict[str, float] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    output_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_problems_from_chapter_pdf(
    chapter_pdf_path: str | Path,
    *,
    book_id: str,
    chapter_number: int,
    solution_book_id: str | None = None,
    layout: ProblemBookLayout | None = None,
) -> list[ExtractedProblem]:
    """Extract all numbered problems from a single per-chapter exercises PDF file.

    Because each ``<XX>_<chapter_title>.pdf`` in ``data/chapters/<book_id>/`` already contains
    only the ``Problems`` pages for ``chapter_number``, we prepend synthetic header markers
    ``Chapter <chapter_number>`` and ``Problems`` so :func:`segment_problems` starts immediately
    on page 1 even when a chapter's ``Problems`` heading was at the bottom of the preceding page.
    """
    pages = load_pdf_pages(chapter_pdf_path)
    if not pages:
        return []
    first_page_num, first_text = pages[0]
    synthetic_header = f"Chapter {chapter_number}\nProblems\n{first_text}"
    patched_pages: list[tuple[int | None, str]] = [(first_page_num, synthetic_header)] + [
        (p, t) for p, t in pages[1:]
    ]
    return segment_problems(
        patched_pages,
        book_id=book_id,
        solution_book_id=solution_book_id,
        solution_section=None,
        layout=layout,
    )


def parse_book_solutions_pdf(
    solutions_pdf_path: str | Path,
    *,
    solution_book_id: str,
    layout: SolutionManualLayout | None = None,
) -> dict[str, dict[tuple[int, int], SolutionEntry]]:
    """Parse a per-book ``<book_id>_solutions.pdf`` into ``{"": {(chapter, number): SolutionEntry}}``."""
    pages = load_pdf_pages(solutions_pdf_path)
    return parse_solution_manual(
        pages,
        solution_book_id=solution_book_id,
        section_titles=None,
        layout=layout,
    )


def build_initial_book_dataset(
    *,
    chapters_dir: str | Path = "data/chapters",
    output_path: str | Path = DEFAULT_INITIAL_BOOK_DATASET_PATH,
    summary_path: str | Path | None = DEFAULT_CHAPTER_DOMAIN_SUMMARY_PATH,
    chapter_domain_map: dict[str, dict[str, str]] | None = None,
    require_matched_solution: bool = True,
    deduplicate_statements: bool = True,
    id_prefix: str = "book",
) -> tuple[list[CanonicalProblem], InitialBookDatasetSummary]:
    """Build the initial book dataset from ``data/chapters/<book_id>/`` using the manual chapter-domain map.

    Steps:
      1. Iterate over each book folder in ``data/chapters/<book_id>/``.
      2. Look up each ``<XX>_<chapter_title>.pdf`` in ``MANUAL_CHAPTER_DOMAIN_MAP``.
      3. Skip any chapter mapped to ``SIDE`` (e.g., general introduction, scaling/units, pure math review).
      4. Extract individual numbered problems from each included chapter PDF and link them to their
         corresponding solution in ``<book_id>_solutions.pdf`` by ``(chapter, problem_number)``.
      5. Build ``CanonicalProblem`` entries with the chapter's assigned domain and save to ``output_path``.
    """
    t_start = time.perf_counter()
    stage_timings: dict[str, float] = {}
    root_dir = resolve_project_path(chapters_dir)
    domain_lookup = chapter_domain_map or MANUAL_CHAPTER_DOMAIN_MAP

    canonical_problems: list[CanonicalProblem] = []
    seen_statements: set[str] = set()
    counts_by_domain: Counter[str] = Counter()
    counts_by_book: Counter[str] = Counter()
    excluded_side: list[str] = []

    books_processed = 0
    included_chapters = 0
    total_extracted = 0
    matched_count = 0
    unmatched_count = 0

    t_extract = time.perf_counter()
    for book_id in sorted(domain_lookup.keys()):
        book_dir = root_dir / book_id
        if not book_dir.is_dir():
            logger.warning("Chapter folder not found for book %r at %s; skipping.", book_id, book_dir)
            continue

        sol_pdf = book_dir / f"{book_id}_solutions.pdf"
        if not sol_pdf.is_file():
            logger.warning("Solutions PDF not found for book %r at %s; skipping.", book_id, sol_pdf)
            continue

        books_processed += 1
        sol_id = f"{book_id}_solutions"
        sol_index = parse_book_solutions_pdf(sol_pdf, solution_book_id=sol_id)
        chapter_map = domain_lookup[book_id]

        for ch_stem, assigned_domain in sorted(chapter_map.items()):
            domain_upper = assigned_domain.strip().upper()
            if domain_upper == SIDE_DOMAIN or domain_upper not in VALID_CANONICAL_DOMAINS:
                excluded_side.append(f"{book_id}/{ch_stem}")
                continue

            ch_pdf = book_dir / f"{ch_stem}.pdf"
            if not ch_pdf.is_file():
                logger.warning("Expected chapter PDF %s not found; skipping.", ch_pdf)
                continue

            m = re.match(r"^(\d{1,2})_", ch_stem)
            if not m:
                continue
            chapter_num = int(m.group(1))
            included_chapters += 1

            extracted = extract_problems_from_chapter_pdf(
                ch_pdf,
                book_id=book_id,
                chapter_number=chapter_num,
                solution_book_id=sol_id,
            )
            total_extracted += len(extracted)
            linked: list[LinkedProblem] = link_problems_to_solutions(extracted, {sol_id: sol_index})

            for item in linked:
                if not item.matched or item.solution is None:
                    unmatched_count += 1
                    if require_matched_solution:
                        continue
                stmt = item.problem.text.strip()
                sol_text = item.solution.text.strip() if item.solution else ""
                if not stmt or (require_matched_solution and not sol_text):
                    unmatched_count += 1
                    continue

                if deduplicate_statements:
                    norm_stmt = re.sub(r"\s+", " ", stmt.lower())
                    if norm_stmt in seen_statements:
                        continue
                    seen_statements.add(norm_stmt)

                matched_count += 1
                counts_by_domain[domain_upper] += 1
                counts_by_book[book_id] += 1
                domain_slug = _DOMAIN_SLUG.get(domain_upper, "phys")
                problem_id = f"{id_prefix}_{domain_slug}_{len(canonical_problems) + 1:04d}"

                canonical_problems.append(
                    CanonicalProblem(
                        problem_id=problem_id,
                        domain=domain_upper,
                        problem_statement=stmt,
                        reference_solution=sol_text,
                        source="book_initial",
                        original_id=f"{book_id}:{ch_stem}:p{item.problem.number:03d}",
                    )
                )
    stage_timings["pdf_extraction_and_linking_sec"] = round(time.perf_counter() - t_extract, 3)

    t_clean = time.perf_counter()
    canonical_problems = apply_curatorkit_text_cleaner(canonical_problems)
    stage_timings["curatorkit_text_cleaner_sec"] = round(time.perf_counter() - t_clean, 3)

    resolved_out = resolve_project_path(output_path)
    save_problems_jsonl(canonical_problems, resolved_out)
    elapsed_sec = round(time.perf_counter() - t_start, 3)
    logger.info(
        "Saved %d canonical book problems across %d domains to %s in %.2fs",
        len(canonical_problems),
        len(counts_by_domain),
        resolved_out,
        elapsed_sec,
    )

    summary = InitialBookDatasetSummary(
        books_processed=books_processed,
        included_chapters_count=included_chapters,
        excluded_side_chapters_count=len(excluded_side),
        total_problems_in_included_chapters=total_extracted,
        matched_problems_with_solutions=len(canonical_problems),
        unmatched_problems_skipped=unmatched_count,
        curatorkit_hygiene_applied=True,
        counts_by_domain=dict(sorted(counts_by_domain.items())),
        counts_by_book=dict(sorted(counts_by_book.items())),
        excluded_side_chapters=excluded_side,
        stage_timings_sec=stage_timings,
        elapsed_seconds=elapsed_sec,
        output_path=str(resolved_out),
    )

    if summary_path is not None:
        resolved_sum = resolve_project_path(summary_path)
        resolved_sum.parent.mkdir(parents=True, exist_ok=True)
        resolved_sum.write_text(json.dumps(summary.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    return canonical_problems, summary
