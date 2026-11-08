"""Stage 1.2: Split downloaded textbooks into per-chapter exercise PDFs (named by book chapter title) and book solution PDFs.

For each downloaded problem book (e.g., ``data/books/crowell_mechanics.pdf``), creates a subfolder
``data/chapters/<book_id>/`` containing:
  * ``<XX>_<chapter_title_slug>.pdf`` (e.g. ``01_scaling_and_estimation.pdf``, ``02_velocity_and_relative_motion.pdf``)
    — the exact PDF pages of that chapter's ``Problems`` section, named after the book chapter title.
  * ``<book_id>_solutions.pdf`` — the exact PDF pages from the shared all-in-one solution manual
    (`light_and_matter_solutions.pdf`) corresponding to this book's `solution_section`.
"""
from __future__ import annotations

import logging
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from adversarial_physics.data.ingestion.book_downloader import DEFAULT_DOWNLOAD_DIR, resolve_book_path
from adversarial_physics.data.ingestion.book_loader import load_pdf_pages
from adversarial_physics.data.ingestion.solution_mapping import (
    ProblemBookLayout,
    SolutionManualLayout,
    parse_solution_manual,
    segment_problems,
)
from adversarial_physics.utils.config import load_yaml, resolve_project_path
from adversarial_physics.utils.logging import get_logger

logger = get_logger("stage_1_2_pdf_chapter_splitter")

DEFAULT_CHAPTERS_DIR = "data/chapters"

_LIGATURES = {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl", "\u00ad": ""}
_LIGATURE_RE = re.compile("|".join(map(re.escape, _LIGATURES)))


@dataclass(frozen=True)
class ChapterPdfSlice:
    """Metadata for a single exported chapter exercises PDF."""

    chapter: int
    chapter_title: str
    start_page: int
    end_page: int
    problem_count: int
    pdf_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BookChapterSplitResult:
    """Summary of chapter exercise PDFs and solution slice PDF exported for one book."""

    book_id: str
    source_pdf: str
    output_dir: str
    chapters: list[ChapterPdfSlice] = field(default_factory=list)
    solutions_pdf: str | None = None
    solutions_start_page: int | None = None
    solutions_end_page: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def slugify_chapter_title(title: str) -> str:
    """Convert a book chapter title into a clean snake_case filename slug."""
    cleaned = _LIGATURE_RE.sub(lambda m: _LIGATURES[m.group(0)], title)
    # Collapse LaTeX umlauts rendered by pypdf as " ¨o" or "¨o" -> "o"
    cleaned = re.sub(r"\s*\u00a8\s*([A-Za-z])", r"\1", cleaned)
    cleaned = cleaned.replace("\u2019", "").replace("'", "")
    slug = re.sub(r"[^a-z0-9]+", "_", cleaned.lower()).strip("_")
    return slug or "chapter"


def extract_book_chapter_titles(pages: list[tuple[int, str]]) -> dict[int, str]:
    """Extract ``{chapter_number: chapter_title}`` from the book's Brief Contents or chapter openers."""
    titles: dict[int, str] = {}

    # 1. Parse from "Brief Contents" (present in all Crowell textbooks within the first 15 pages)
    for _, text in pages[:15]:
        if "Brief Contents" not in text:
            continue
        lines = [
            _LIGATURE_RE.sub(lambda m: _LIGATURES[m.group(0)], line.strip())
            for line in text.splitlines()
            if line.strip()
        ]
        idx = 0
        while idx < len(lines):
            line = lines[idx]
            if line.startswith("For a semester-length") or line.startswith("Contents"):
                break
            match = re.match(r"^(\d{1,2})\s+(?:\u22c6\s*)?([A-Z].*)$", line)
            if match:
                chapter_no = int(match.group(1))
                rest = match.group(2).strip()
                while not re.search(r"\s+\d{1,4}$", rest) and idx + 1 < len(lines):
                    idx += 1
                    sep = "" if rest.endswith("-") else " "
                    rest = (rest.rstrip("-") + sep + lines[idx]).strip()
                title = re.sub(r"\s+\d{1,4}$", "", rest).strip()
                title = re.sub(r"\s*\(optional[^)]*\)", "", title, flags=re.IGNORECASE).strip()
                title = re.sub(r"\s*\u00a8\s*([A-Za-z])", r"\1", title)
                title = re.sub(r"\s+", " ", title)
                if title:
                    titles[chapter_no] = title
            idx += 1
        if titles:
            return titles

    # 2. Fallback: scan chapter opener pages ("Chapter N\n<Title>\nN.1 ...")
    for _, text in pages:
        lines = [
            _LIGATURE_RE.sub(lambda m: _LIGATURES[m.group(0)], line.strip())
            for line in text.splitlines()
            if line.strip()
        ]
        for idx, line in enumerate(lines):
            m = re.match(r"^Chapter\s+(\d{1,2})$", line)
            if m and idx + 1 < len(lines):
                ch = int(m.group(1))
                if ch in titles:
                    continue
                title_parts: list[str] = []
                j = idx + 1
                while j < len(lines) and j <= idx + 3:
                    cand = lines[j]
                    if re.match(rf"^{ch}\.\d+\b", cand) or cand.startswith("Chapter "):
                        break
                    title_parts.append(cand)
                    j += 1
                if title_parts:
                    titles[ch] = " ".join(title_parts).strip()
    return titles


def write_pdf_page_range(
    source_pdf: str | Path,
    dest_pdf: str | Path,
    *,
    start_page: int,
    end_page: int,
) -> Path:
    """Write 1-indexed inclusive page range ``[start_page, end_page]`` from ``source_pdf`` to ``dest_pdf``."""
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    from pypdf import PdfReader, PdfWriter  # type: ignore[import-not-found]

    src_path = Path(source_pdf)
    out_path = Path(dest_pdf)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    reader = PdfReader(str(src_path))
    total = len(reader.pages)
    first_idx = max(0, start_page - 1)
    last_idx = min(total, end_page)
    if first_idx >= last_idx:
        raise ValueError(
            f"Invalid page range [{start_page}, {end_page}] for {src_path} (total pages: {total})"
        )

    writer = PdfWriter()
    for idx in range(first_idx, last_idx):
        writer.add_page(reader.pages[idx])

    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    with tmp_path.open("wb") as file:
        writer.write(file)
    tmp_path.replace(out_path)
    return out_path


def detect_chapter_exercise_ranges(
    pages: list[tuple[int, str]],
    *,
    book_id: str,
    layout: ProblemBookLayout | None = None,
) -> list[tuple[int, int, int, int]]:
    """Return ``(chapter, start_page, end_page, problem_count)`` for every chapter's Problems section."""
    lay = layout or ProblemBookLayout()
    problems = segment_problems(pages, book_id=book_id, layout=lay)
    by_chapter: dict[int, list[int]] = {}
    counts: dict[int, int] = {}
    for prob in problems:
        counts[prob.chapter] = counts.get(prob.chapter, 0) + 1
        bucket = by_chapter.setdefault(prob.chapter, [])
        if prob.page_start is not None:
            bucket.append(prob.page_start)
        if prob.page_end is not None:
            bucket.append(prob.page_end)

    page_text_map = dict(pages)
    start_re = re.compile(lay.problems_start_pattern, flags=re.IGNORECASE)

    ranges: list[tuple[int, int, int, int]] = []
    for chapter in sorted(by_chapter):
        ch_pages = by_chapter[chapter]
        if not ch_pages:
            continue
        p_start = min(ch_pages)
        p_end = max(ch_pages)
        prev_text = page_text_map.get(p_start - 1, "")
        if prev_text and start_re.search(prev_text) and not start_re.search(page_text_map.get(p_start, "")):
            p_start -= 1
        ranges.append((chapter, p_start, p_end, counts[chapter]))
    return ranges


def detect_solution_section_ranges(
    pages: list[tuple[int, str]],
    *,
    solution_book_id: str,
    section_titles: list[str],
    layout: SolutionManualLayout | None = None,
) -> dict[str, tuple[int, int]]:
    """Return ``{section_title: (start_page, end_page)}`` (1-indexed inclusive) for a shared solution manual."""
    parsed = parse_solution_manual(
        pages,
        solution_book_id=solution_book_id,
        section_titles=section_titles,
        layout=layout,
    )
    raw_starts: dict[str, int] = {}
    raw_ends: dict[str, int] = {}
    for title, entries in parsed.items():
        s_pages = [e.page_start for e in entries.values() if e.page_start is not None]
        e_pages = [e.page_end for e in entries.values() if e.page_end is not None]
        if s_pages and e_pages:
            raw_starts[title] = min(s_pages)
            raw_ends[title] = max(e_pages)

    ordered = sorted(raw_starts.items(), key=lambda item: item[1])
    ranges: dict[str, tuple[int, int]] = {}
    for idx, (title, start_pg) in enumerate(ordered):
        if idx + 1 < len(ordered):
            next_start = ordered[idx + 1][1]
            end_pg = max(raw_ends[title], next_start - 1)
        else:
            end_pg = raw_ends[title]
        ranges[title] = (start_pg, end_pg)
    return ranges


def split_books_into_chapter_pdfs(
    config_path: str | Path = "configs/data/book_ingestion.yaml",
    *,
    chapters_dir: str | Path = DEFAULT_CHAPTERS_DIR,
    download_dir: str | Path | None = None,
    book_ids: list[str] | None = None,
    include_disabled: bool = True,
) -> list[BookChapterSplitResult]:
    """Split all downloaded PDF problem books into per-chapter exercise PDFs (named after book chapter titles) and book solution PDFs.

    Creates ``data/chapters/<book_id>/`` for each downloaded problem book containing:
      * ``<XX>_<chapter_title_slug>.pdf`` for every chapter with exercises
      * ``<book_id>_solutions.pdf`` containing the slice of the solution manual for this book
    """
    cfg = load_yaml(config_path)
    sources = list(cfg.get("books", []))
    dl_dir = str(download_dir or (cfg.get("download", {}) or {}).get("dir") or DEFAULT_DOWNLOAD_DIR)
    out_root = resolve_project_path(chapters_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    wanted = set(book_ids or [])

    solution_ranges_by_id: dict[str, tuple[Path, dict[str, tuple[int, int]]]] = {}
    for entry in sources:
        if str(entry.get("role", "")).lower() != "solution_book":
            continue
        sol_id = str(entry.get("book_id") or "")
        sol_path = resolve_book_path(entry, download_dir=dl_dir)
        if sol_path is None or not sol_path.is_file() or sol_path.suffix.lower() != ".pdf":
            continue
        sol_pages = load_pdf_pages(sol_path)
        sec_titles = list(entry.get("sections") or [])
        sec_ranges = detect_solution_section_ranges(
            sol_pages,
            solution_book_id=sol_id,
            section_titles=sec_titles,
            layout=SolutionManualLayout.from_dict(entry.get("layout")),
        )
        norm_map = {k.strip().lower(): v for k, v in sec_ranges.items()}
        solution_ranges_by_id[sol_id] = (sol_path, norm_map)
        logger.info("Detected solution manual page ranges for %s: %s", sol_id, sec_ranges)

    results: list[BookChapterSplitResult] = []
    for entry in sources:
        if str(entry.get("role", "")).lower() == "solution_book":
            continue
        if not include_disabled and not entry.get("enabled", True):
            continue
        book_id = str(entry.get("book_id") or "")
        if wanted and book_id not in wanted:
            continue

        book_path = resolve_book_path(entry, download_dir=dl_dir)
        if book_path is None or not book_path.is_file() or book_path.suffix.lower() != ".pdf":
            continue

        book_out_dir = out_root / book_id
        if book_out_dir.exists():
            shutil.rmtree(book_out_dir)
        book_out_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Extracting chapter titles and exercise pages from %s (%s)...", book_id, book_path.name)
        pages = load_pdf_pages(
            book_path,
            start_page=int(entry.get("start_page") or 1),
            end_page=int(entry["end_page"]) if entry.get("end_page") is not None else None,
        )
        chapter_titles = extract_book_chapter_titles(pages)
        ch_ranges = detect_chapter_exercise_ranges(
            pages,
            book_id=book_id,
            layout=ProblemBookLayout.from_dict(entry.get("layout")),
        )

        chapter_slices: list[ChapterPdfSlice] = []
        for chapter, p_start, p_end, prob_count in ch_ranges:
            raw_title = chapter_titles.get(chapter, f"chapter_{chapter:02d}")
            title_slug = slugify_chapter_title(raw_title)
            filename = f"{chapter:02d}_{title_slug}.pdf" if chapter in chapter_titles else f"{title_slug}.pdf"
            ch_pdf = book_out_dir / filename
            write_pdf_page_range(book_path, ch_pdf, start_page=p_start, end_page=p_end)
            chapter_slices.append(
                ChapterPdfSlice(
                    chapter=chapter,
                    chapter_title=raw_title,
                    start_page=p_start,
                    end_page=p_end,
                    problem_count=prob_count,
                    pdf_path=str(ch_pdf),
                )
            )

        sol_pdf_str: str | None = None
        sol_start: int | None = None
        sol_end: int | None = None
        sol_id = str(entry.get("solution_book_id") or "")
        sol_section = str(entry.get("solution_section") or "").strip().lower()
        if sol_id in solution_ranges_by_id:
            sol_src_path, sec_map = solution_ranges_by_id[sol_id]
            page_span = sec_map.get(sol_section)
            if page_span is not None:
                sol_start, sol_end = page_span
                sol_dest = book_out_dir / f"{book_id}_solutions.pdf"
                write_pdf_page_range(sol_src_path, sol_dest, start_page=sol_start, end_page=sol_end)
                sol_pdf_str = str(sol_dest)

        logger.info(
            "Exported %d chapter exercise PDFs (and solutions=%s) to %s",
            len(chapter_slices),
            bool(sol_pdf_str),
            book_out_dir,
        )
        results.append(
            BookChapterSplitResult(
                book_id=book_id,
                source_pdf=str(book_path),
                output_dir=str(book_out_dir),
                chapters=chapter_slices,
                solutions_pdf=sol_pdf_str,
                solutions_start_page=sol_start,
                solutions_end_page=sol_end,
            )
        )

    return results
