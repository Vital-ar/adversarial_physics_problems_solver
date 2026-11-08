"""Stage 1.2: deterministic problem <-> solution mapping for textbooks with a separate solution manual.

Designed for Benjamin Crowell's open-source books (lightandmatter.com), where:

* Each problem book (Light and Matter, Mechanics, Fields and Circuits, Modern Physics,
  Conceptual Physics, ...) ends every chapter with a ``Problems`` section whose items are
  numbered ``1, 2, 3, ...`` and restart in every chapter. Even pages carry a running header
  ``Chapter N``.
* ONE shared solution manual ("Solutions for Conceptual Physics, Light and Matter, Mechanics,
  Simple Nature, Fields and Circuits, and Modern Physics", archive.org item ``lm_solutions``)
  is split into one section per book (``1 Light and Matter``, ``3 Mechanics``, ...). Inside a
  section each solution starts with ``<chapter>-<problem>`` (e.g. ``24-4``).

Mapping key::

    (solution_book_id, solution_section, chapter, problem_number)

The problem book declares ``solution_book_id`` + ``solution_section`` in
``configs/data/book_ingestion.yaml``; chapter and problem number are recovered from the text.

All regexes live in :class:`ProblemBookLayout` / :class:`SolutionManualLayout`, so other books
with a different layout can be supported by overriding them from YAML (``layout:`` key).
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

Segment = tuple[int | None, str]  # (page_number or None, text)


# --------------------------------------------------------------------------------------
# Layout definitions (regexes) - defaults match Crowell's LaTeX templates
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ProblemBookLayout:
    """Regexes describing how chapters and end-of-chapter problems look in a problem book."""

    # Running header / chapter title line, e.g. "Chapter 3" or "38 Chapter 0 Introduction".
    chapter_pattern: str = r"^\s*(?:\d+\s+)?Chapter\s+(\d{1,2})\b"
    # Start of an end-of-chapter problem set ("Problems" heading, or the "Key" legend).
    problems_start_pattern: str = (
        r"^\s*Problems\s*(?:Key)?\s*$|computerized answer check is available online"
    )
    # Lines that terminate a problem set (lab exercises, appendices, back matter).
    problems_end_pattern: str = (
        r"^\s*(?:Exercise\s+\d+[A-Za-z]?\s*:|Appendix\b|Solutions to [Ss]elected|Answers to\b|Photo [Cc]redits)"
    )
    # Problem start: number at line start followed by a capital letter, "(", quote, or "["
    problem_start_pattern: str = r"^\s*(\d{1,3})\s+(?=[A-Z(\[\"'\u201c])"
    # Lines to drop inside problem sets (legend of the symbol key, bare page numbers, headers).
    noise_line_pattern: str = (
        r"^\s*(?:Key|Problems|\d{1,4}|.*computerized answer check.*|.*requires calculus.*|"
        r".*A difficult problem.*)\s*$"
    )
    # Reference to a solution printed in the back of the problem book itself.
    inline_solution_pattern: str = r"Solution,\s*p\.\s*(\d+)"
    # Maximum allowed gap between consecutive problem numbers (tolerates OCR/extraction misses).
    max_number_gap: int = 3

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ProblemBookLayout:
        if not data:
            return cls()
        allowed = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in allowed})


@dataclass(frozen=True)
class SolutionManualLayout:
    """Regexes describing a solution manual split into per-book sections."""

    # Solution entry start, e.g. "24-4 (a) Using the equation..." -> chapter=24, problem=4
    entry_pattern: str = r"^\s*(\d{1,2})\s*[-\u2013]\s*(\d{1,3})\b\s*(.*)$"
    # Section header template; "{title}" is replaced by each escaped section title.
    section_header_template: str = r"^\s*(?:\d{{1,2}}\s+|chapter\s+\d{{1,2}}\.?\s+)?{title}\s*$"
    # Maximum chapter jump between consecutive entries (rejects false hits inside math).
    max_chapter_gap: int = 3

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> SolutionManualLayout:
        if not data:
            return cls()
        allowed = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in allowed})


# --------------------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ExtractedProblem:
    """A single end-of-chapter problem segmented from a problem book."""

    book_id: str
    chapter: int
    number: int
    text: str
    page_start: int | None = None
    page_end: int | None = None
    solution_book_id: str | None = None
    solution_section: str | None = None
    inline_solution_page: int | None = None

    @property
    def key(self) -> str:
        return f"{self.chapter}-{self.number}"

    @property
    def problem_ref(self) -> str:
        """Stable identifier, e.g. ``light_and_matter_problems:ch02_p005``."""
        return f"{self.book_id}:ch{self.chapter:02d}_p{self.number:03d}"

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "key": self.key, "problem_ref": self.problem_ref}


@dataclass(frozen=True)
class SolutionEntry:
    """A single worked solution parsed from a solution manual section."""

    solution_book_id: str
    section: str
    chapter: int
    number: int
    text: str
    page_start: int | None = None
    page_end: int | None = None

    @property
    def key(self) -> str:
        return f"{self.chapter}-{self.number}"

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "key": self.key}


@dataclass(frozen=True)
class LinkedProblem:
    """A problem paired with its official solution (``solution`` is None when unmatched)."""

    problem: ExtractedProblem
    solution: SolutionEntry | None = None

    @property
    def matched(self) -> bool:
        return self.solution is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem": self.problem.to_dict(),
            "solution": self.solution.to_dict() if self.solution else None,
            "matched": self.matched,
        }


@dataclass(frozen=True)
class LinkingReport:
    """Coverage statistics for one problem book."""

    book_id: str
    solution_book_id: str | None
    solution_section: str | None
    problems_found: int
    matched: int
    unmatched: int
    unmatched_with_inline_solution: int
    chapters: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _iter_lines(segments: list[Segment]):
    for page, text in segments:
        for line in text.splitlines():
            yield page, line


_LIGATURES = {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl", "\u00ad": ""}
_LIGATURE_RE = re.compile("|".join(map(re.escape, _LIGATURES)))


def _clean_block(lines: list[str]) -> str:
    text = "\n".join(line.rstrip() for line in lines)
    # Replace PDF typographic ligatures only (NFKC would also turn "10²" into "102").
    text = _LIGATURE_RE.sub(lambda m: _LIGATURES[m.group(0)], text)
    # Re-join words hyphenated across line breaks ("cal-\nculator" -> "calculator").
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", title).strip().lower()


# --------------------------------------------------------------------------------------
# Problem book segmentation
# --------------------------------------------------------------------------------------
def segment_problems(
    segments: list[Segment],
    *,
    book_id: str,
    solution_book_id: str | None = None,
    solution_section: str | None = None,
    layout: ProblemBookLayout | None = None,
) -> list[ExtractedProblem]:
    """Split a problem book (page/paragraph segments) into numbered end-of-chapter problems.

    State machine:
      * ``Chapter N`` lines update the current chapter (outside problem sets).
      * A ``Problems`` heading / key legend opens a problem set for the current chapter.
      * Inside a set, a line ``<n> <Capital...>`` starts problem ``n`` only if ``n`` follows
        the previous number within ``max_number_gap`` (rejects page numbers and math).
      * A chapter header with a different number or an end marker closes the set.
    """
    lay = layout or ProblemBookLayout()
    chapter_re = re.compile(lay.chapter_pattern)
    start_re = re.compile(lay.problems_start_pattern, flags=re.IGNORECASE)
    end_re = re.compile(lay.problems_end_pattern)
    problem_re = re.compile(lay.problem_start_pattern)
    noise_re = re.compile(lay.noise_line_pattern, flags=re.IGNORECASE)
    inline_re = re.compile(lay.inline_solution_pattern)

    problems: list[ExtractedProblem] = []
    seen_keys: set[tuple[int, int]] = set()

    current_chapter: int | None = None
    in_problems = False
    set_chapter: int | None = None
    last_number = 0
    cur_number: int | None = None
    cur_lines: list[str] = []
    cur_pages: list[int] = []

    def flush() -> None:
        nonlocal cur_number, cur_lines, cur_pages
        if cur_number is not None and set_chapter is not None:
            text = _clean_block(cur_lines)
            key = (set_chapter, cur_number)
            if text and key not in seen_keys:
                seen_keys.add(key)
                inline = inline_re.search(text)
                problems.append(
                    ExtractedProblem(
                        book_id=book_id,
                        chapter=set_chapter,
                        number=cur_number,
                        text=text,
                        page_start=min(cur_pages) if cur_pages else None,
                        page_end=max(cur_pages) if cur_pages else None,
                        solution_book_id=solution_book_id,
                        solution_section=solution_section,
                        inline_solution_page=int(inline.group(1)) if inline else None,
                    )
                )
        cur_number, cur_lines, cur_pages = None, [], []

    def close_set() -> None:
        nonlocal in_problems, set_chapter, last_number
        flush()
        in_problems, set_chapter, last_number = False, None, 0

    for page, line in _iter_lines(segments):
        stripped = line.strip()
        if not stripped:
            if cur_number is not None:
                cur_lines.append("")
            continue

        chapter_match = chapter_re.match(stripped) if len(stripped) < 120 else None
        if chapter_match:
            chapter_no = int(chapter_match.group(1))
            if in_problems and chapter_no != set_chapter:
                close_set()
            current_chapter = chapter_no
            continue  # running header, never part of a problem

        if start_re.search(stripped):
            if not in_problems and current_chapter is not None:
                in_problems, set_chapter, last_number = True, current_chapter, 0
            continue

        if not in_problems:
            continue

        if end_re.match(stripped):
            close_set()
            continue

        problem_match = problem_re.match(stripped)
        if problem_match:
            number = int(problem_match.group(1))
            if last_number < number <= last_number + lay.max_number_gap:
                flush()
                cur_number, last_number = number, number
                cur_lines = [stripped[problem_match.end(1):].strip()]
                cur_pages = [page] if page is not None else []
                continue

        if cur_number is None or noise_re.match(stripped):
            continue
        cur_lines.append(stripped)
        if page is not None:
            cur_pages.append(page)

    close_set()
    return problems


# --------------------------------------------------------------------------------------
# Solution manual parsing
# --------------------------------------------------------------------------------------
def parse_solution_manual(
    segments: list[Segment],
    *,
    solution_book_id: str,
    section_titles: list[str] | None = None,
    layout: SolutionManualLayout | None = None,
) -> dict[str, dict[tuple[int, int], SolutionEntry]]:
    """Parse a solution manual into ``{section_title: {(chapter, number): SolutionEntry}}``.

    If ``section_titles`` is empty/None the whole manual is treated as a single section
    named ``""`` (use this for a solution book that serves exactly one problem book).

    Entries must appear in increasing ``(chapter, number)`` order inside a section; out-of-order
    matches (e.g. "3-4" inside an equation) are treated as ordinary solution text.
    Header lines from the table of contents are harmless: switching sections only resets the
    ordering state when the section actually changes.
    """
    lay = layout or SolutionManualLayout()
    entry_re = re.compile(lay.entry_pattern)
    titles = list(section_titles or [])
    header_res = [
        (
            title,
            re.compile(
                lay.section_header_template.format(title=re.escape(title).replace(r"\ ", r"\s+")),
                flags=re.IGNORECASE,
            ),
        )
        for title in titles
    ]

    index: dict[str, dict[tuple[int, int], SolutionEntry]] = {t: {} for t in titles} if titles else {"": {}}
    current_section: str | None = None if titles else ""
    last_key: tuple[int, int] = (-1, -1)
    cur_key: tuple[int, int] | None = None
    cur_lines: list[str] = []
    cur_pages: list[int] = []

    def flush() -> None:
        nonlocal cur_key, cur_lines, cur_pages
        if cur_key is not None and current_section is not None:
            text = _clean_block(cur_lines)
            bucket = index.setdefault(current_section, {})
            if text and cur_key not in bucket:
                bucket[cur_key] = SolutionEntry(
                    solution_book_id=solution_book_id,
                    section=current_section,
                    chapter=cur_key[0],
                    number=cur_key[1],
                    text=text,
                    page_start=min(cur_pages) if cur_pages else None,
                    page_end=max(cur_pages) if cur_pages else None,
                )
        cur_key, cur_lines, cur_pages = None, [], []

    for page, line in _iter_lines(segments):
        stripped = line.strip()
        if not stripped:
            if cur_key is not None:
                cur_lines.append("")
            continue

        switched = False
        for title, header_re in header_res:
            if header_re.match(stripped):
                if title != current_section:
                    flush()
                    current_section = title
                    last_key = (-1, -1)
                switched = True
                break
        if switched:
            continue

        if current_section is None:
            continue

        entry_match = entry_re.match(stripped)
        if entry_match:
            key = (int(entry_match.group(1)), int(entry_match.group(2)))
            in_order = key > last_key
            small_jump = last_key[0] < 0 or key[0] - last_key[0] <= lay.max_chapter_gap
            if in_order and small_jump:
                flush()
                cur_key, last_key = key, key
                cur_lines = [entry_match.group(3).strip()]
                cur_pages = [page] if page is not None else []
                continue

        if cur_key is not None:
            cur_lines.append(stripped)
            if page is not None:
                cur_pages.append(page)

    flush()
    return index


# --------------------------------------------------------------------------------------
# Linking
# --------------------------------------------------------------------------------------
def link_problems_to_solutions(
    problems: list[ExtractedProblem],
    solution_indexes: dict[str, dict[str, dict[tuple[int, int], SolutionEntry]]],
) -> list[LinkedProblem]:
    """Attach each problem to its solution via ``(solution_book_id, section, chapter, number)``.

    ``solution_indexes`` maps ``solution_book_id`` -> output of :func:`parse_solution_manual`.
    Section titles are compared case/whitespace-insensitively.
    """
    linked: list[LinkedProblem] = []
    for problem in problems:
        solution: SolutionEntry | None = None
        sections = solution_indexes.get(problem.solution_book_id or "")
        if sections:
            wanted = _normalize_title(problem.solution_section or "")
            bucket = next(
                (entries for title, entries in sections.items() if _normalize_title(title) == wanted),
                None,
            )
            if bucket is None and len(sections) == 1:
                bucket = next(iter(sections.values()))
            if bucket is not None:
                solution = bucket.get((problem.chapter, problem.number))
        linked.append(LinkedProblem(problem=problem, solution=solution))
    return linked


def build_linking_report(
    linked: list[LinkedProblem],
    *,
    book_id: str,
    solution_book_id: str | None,
    solution_section: str | None,
) -> LinkingReport:
    matched = sum(1 for item in linked if item.matched)
    unmatched_inline = sum(
        1 for item in linked if not item.matched and item.problem.inline_solution_page is not None
    )
    return LinkingReport(
        book_id=book_id,
        solution_book_id=solution_book_id,
        solution_section=solution_section,
        problems_found=len(linked),
        matched=matched,
        unmatched=len(linked) - matched,
        unmatched_with_inline_solution=unmatched_inline,
        chapters=sorted({item.problem.chapter for item in linked}),
    )


def format_paired_text(item: LinkedProblem, *, book_title: str = "") -> str:
    """Render a problem + official solution pair as the text block sent to the teacher LLM."""
    p = item.problem
    header = f"SOURCE: {book_title or p.book_id} | Chapter {p.chapter}, Problem {p.number}"
    parts = [header, "", "ORIGINAL PROBLEM:", p.text]
    if item.solution is not None:
        parts += ["", "OFFICIAL SOLUTION (ground truth from the author's solution manual):", item.solution.text]
    return "\n".join(parts)
