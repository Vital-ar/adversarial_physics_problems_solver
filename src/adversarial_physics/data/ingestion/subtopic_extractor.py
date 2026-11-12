"""Zero-LLM Textbook Subtopic Extraction, Cross-Book Deduplication & Retrieval (`subtopic_extractor.py`).

Extracts clean subtopic theory/explanation chunks from the downloaded Crowell physics textbooks
(excluding ``Problems`` exercise pages and figure/graph-dependent paragraphs), deduplicates
near-identical paragraphs across books sharing the same physics domain using ``CuratorKIT``, and
provides zero-LLM chapter-scoped subtopic mapping via ``curatorkit.diagnostic.retriever.PostHocRetriever``.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from curatorkit.connectors.pdf import PDFChunker
from curatorkit.normalizers.clean import TextCleaner
from curatorkit.normalizers.dedup import ExactDeduplicator, MinHashDeduplicator
from curatorkit.schema import DataSample

from adversarial_physics.data.ingestion.book_downloader import DEFAULT_DOWNLOAD_DIR, resolve_book_path
from adversarial_physics.data.ingestion.book_loader import load_pdf_pages
from adversarial_physics.data.ingestion.chapter_domain_extractor import (
    MANUAL_CHAPTER_DOMAIN_MAP,
    SIDE_DOMAIN,
    VALID_CANONICAL_DOMAINS,
)
from adversarial_physics.data.ingestion.pdf_chapter_splitter import (
    extract_book_chapter_titles,
    slugify_chapter_title,
)
from adversarial_physics.data.schemas.problem import CanonicalProblem
from adversarial_physics.utils.config import load_yaml, resolve_project_path
from adversarial_physics.utils.logging import get_logger

logger = get_logger("subtopic_extractor")

DEFAULT_SUBTOPIC_CHUNKS_PATH = "data/literature/subtopic_chunks.jsonl"
DEFAULT_SUBTOPIC_SUMMARY_PATH = "data/literature/subtopic_chunks_summary.json"

_LIGATURES = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\u00ad": "",
}
_LIGATURE_RE = re.compile("|".join(map(re.escape, _LIGATURES)))

_VISUAL_CUE_RE = re.compile(
    r"\b(?:"
    r"fig(?:ure)?\.?\s*\d+|figures?\s+[a-z0-9]|"
    r"as\s+shown\s+(?:in|at|below|above)|"
    r"shown\s+(?:in\s+the\s+)?(?:figure|diagram|graph|photo|drawing|sketch)|"
    r"diagram\s+(?:below|above|at\s+right|at\s+left)|"
    r"graph\s+(?:below|above|at\s+right|at\s+left)|"
    r"at\s+(?:the\s+)?(?:upper|lower)\s+(?:left|right)|"
    r"(?:top|bottom|left|right)\s+(?:photo|panel|inset)|"
    r"in\s+the\s+photo|"
    r"see\s+(?:figure|table)\s+\d+"
    r")\b",
    flags=re.IGNORECASE,
)

_SECTION_HEADING_RE = re.compile(r"^(\d{1,2}\.\d{1,2})\s+(?:\u22c6\s*)?([A-Z][^\n]{2,100})$")
_TOC_SECTION_RE = re.compile(r"^(\d{1,2}\.\d{1,2})\s+(?:\u22c6\s*)?(.+?)(?:\s*\.{2,}\s*|\s+)(\d{1,4})$")
_RUNNING_HEADER_RE = re.compile(
    r"^(?:(?:\d{1,4}\s+)?Chapter\s+\d{1,2}(?:\s+.*)?|Section\s+\d+\.\d+(?:\s+.*)?|Summary|Problems|Discussion\s+question\b.*)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SubtopicChunk:
    """Clean, deduplicated textbook subtopic chunk linked to book, chapter, and physics domain."""

    chunk_id: str
    book_id: str
    chapter: int
    chapter_slug: str
    domain: str
    section_id: str
    section_title: str
    text: str
    token_count: int
    start_page: int
    end_page: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SubtopicChunk:
        return cls(
            chunk_id=str(data["chunk_id"]),
            book_id=str(data["book_id"]),
            chapter=int(data["chapter"]),
            chapter_slug=str(data["chapter_slug"]),
            domain=str(data["domain"]),
            section_id=str(data["section_id"]),
            section_title=str(data["section_title"]),
            text=str(data["text"]),
            token_count=int(data["token_count"]),
            start_page=int(data["start_page"]),
            end_page=int(data["end_page"]),
        )


def save_subtopic_chunks_jsonl(chunks: list[SubtopicChunk], path: str | Path) -> Path:
    """Persist ``SubtopicChunk`` records as UTF-8 JSONL."""
    out = resolve_project_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(ch.to_dict(), ensure_ascii=False) + "\n")
    return out


def load_subtopic_chunks_jsonl(path: str | Path) -> list[SubtopicChunk]:
    """Load ``SubtopicChunk`` records from JSONL."""
    in_path = resolve_project_path(path)
    chunks: list[SubtopicChunk] = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                chunks.append(SubtopicChunk.from_dict(json.loads(stripped)))
    return chunks


def _clean_pdf_text(text: str) -> str:
    cleaned = _LIGATURE_RE.sub(lambda m: _LIGATURES[m.group(0)], text or "")
    cleaned = re.sub(r"-\s*\n\s*", "", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    return cleaned.strip()


_EQUATION_FRAGMENT_RE = re.compile(
    r"(?:[=<>≈∝±]|\\frac|\\sqrt|\^|\d+(?:\.\d+)?\s*(?:m/s|kg|N|J|W|Pa|Hz|V|A|C|K|eV)\b)"
)


def strip_visual_paragraphs(text: str, *, min_chars: int = 80) -> str:
    """Drop caption fragments and image/graph-dependent paragraphs while keeping short physics equations."""
    raw_paras = re.split(r"\n{2,}|(?<=[.!?])\s+(?=[A-Z])", text or "")
    kept: list[str] = []
    for para in raw_paras:
        p = re.sub(r"\s+", " ", para).strip()
        if not p:
            continue
        if _VISUAL_CUE_RE.search(p):
            continue
        # Drop figure labels like "a / The ..." or "(a) A ..."
        if re.match(r"^(?:[a-z]\s*/\s*|\([a-z]\)\s+A\s)", p):
            continue
        is_equation = bool(_EQUATION_FRAGMENT_RE.search(p)) and len(p) >= 5
        if len(p) < min_chars and not is_equation:
            continue
        kept.append(p)
    return "\n\n".join(kept).strip()


def parse_detailed_contents(pages: list[tuple[int, str]]) -> dict[str, tuple[str, int]]:
    """Parse ``{section_id: (section_title, book_page)}`` from the PDF's ``Contents`` pages."""
    sections: dict[str, tuple[str, int]] = {}
    in_contents = False
    for page_no, raw_text in pages[:22]:
        text = _clean_pdf_text(raw_text)
        if not in_contents and re.search(r"^Contents\s*$", text, flags=re.MULTILINE):
            in_contents = True
        if not in_contents:
            continue
        if page_no > 4 and re.search(r"^Chapter\s+0?\b", text, flags=re.MULTILINE):
            break
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        idx = 0
        while idx < len(lines):
            ln = lines[idx]
            m_start = re.match(r"^(\d{1,2}\.\d{1,2})\s+(?:\u22c6\s*)?(.+)$", ln)
            if m_start:
                sec_id = m_start.group(1)
                combined = ln
                while not re.search(r"\d{1,4}$", combined) and idx + 1 < len(lines):
                    idx += 1
                    combined = f"{combined} {lines[idx]}"
                m_full = _TOC_SECTION_RE.match(combined)
                if m_full:
                    title = re.sub(r"\s*\.{2,}\s*", "", m_full.group(2)).strip(" .")
                    page_num = int(m_full.group(3))
                    if title:
                        sections[sec_id] = (title, page_num)
            idx += 1
    return sections


def _resolve_chapter_domain(book_id: str, chapter_no: int, chapter_title: str) -> str:
    """Resolve canonical domain from ``MANUAL_CHAPTER_DOMAIN_MAP`` by chapter number/title."""
    book_map = MANUAL_CHAPTER_DOMAIN_MAP.get(book_id, {})
    slug = slugify_chapter_title(chapter_title)
    key_exact = f"{chapter_no:02d}_{slug}"
    if key_exact in book_map:
        return book_map[key_exact]
    prefix = f"{chapter_no:02d}_"
    for k, dom in book_map.items():
        if k.startswith(prefix):
            return dom
    return SIDE_DOMAIN


def extract_book_subtopic_chunks(
    pdf_path: str | Path,
    book_id: str,
    *,
    max_chunk_tokens: int = 380,
    overlap_tokens: int = 40,
) -> list[SubtopicChunk]:
    """Extract clean theory chunks grouped by chapter and subtopic from one textbook PDF."""
    pages = load_pdf_pages(pdf_path)
    if not pages:
        return []

    chapter_titles = extract_book_chapter_titles(pages)
    toc_sections = parse_detailed_contents(pages)
    logger.info(
        "Extracting subtopics from %s (%d pages, %d chapters, %d TOC sections)...",
        book_id,
        len(pages),
        len(chapter_titles),
        len(toc_sections),
    )

    # Walk body pages after Contents, excluding chapter Problems pages and back-matter
    current_chapter: int | None = None
    in_problems_section = False
    chapter_blocks: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for pdf_page_no, raw_text in pages:
        if pdf_page_no <= 10:
            # Contents / front-matter pages
            if "Chapter 0" not in raw_text and "Chapter 1" not in raw_text:
                continue

        cleaned = _clean_pdf_text(raw_text)
        lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
        if not lines:
            continue

        # Detect stop markers in back matter (Answers / Index / Credits)
        joined_head = "\n".join(lines[:4])
        if re.search(r"^(?:Answers\s+to\s+Selected\s+Problems|Photo\s+Credits|Index)\b", joined_head, re.I):
            if current_chapter is not None:
                break

        # Detect "Chapter N" transitions anywhere on the page (captions can push title below line 6)
        found_standalone_ch = False
        for ln in lines[:24]:
            m_ch = re.match(r"^Chapter\s+(\d{1,2})\b$", ln)
            if m_ch:
                current_chapter = int(m_ch.group(1))
                in_problems_section = False
                found_standalone_ch = True
                break
        if not found_standalone_ch:
            for ln in lines:
                m_hdr = re.match(r"^\d{1,4}\s+Chapter\s+(\d{1,2})\b", ln)
                if m_hdr:
                    hdr_ch = int(m_hdr.group(1))
                    if current_chapter is None or hdr_ch != current_chapter:
                        current_chapter = hdr_ch
                        in_problems_section = False
                    break

        if current_chapter is None:
            continue

        # Skip Problems pages and Solutions / Homework pages
        if any(re.match(r"^Problems(?:\s+for\s+Chapter\s+\d+)?$", ln, re.I) for ln in lines[:5]):
            in_problems_section = True
        if in_problems_section:
            continue

        # Build heading/paragraph blocks for CuratorKIT PDFChunker (preserving newlines for short equation lines)
        para_buf: list[str] = []

        def _flush_para_buf() -> None:
            if not para_buf:
                return
            joined_lines: list[str] = []
            for raw_ln in para_buf:
                # Isolate equation-bearing lines with blank lines so strip_visual_paragraphs retains them intact
                if len(raw_ln) < 80 and _EQUATION_FRAGMENT_RE.search(raw_ln) and not _VISUAL_CUE_RE.search(raw_ln):
                    joined_lines.append(f"\n{raw_ln}\n")
                else:
                    joined_lines.append(raw_ln)
            text_block = re.sub(r"\n{3,}", "\n\n", " ".join(joined_lines).replace(" \n", "\n").replace("\n ", "\n")).strip()
            if text_block:
                chapter_blocks[current_chapter].append(
                    {"type": "paragraph", "text": text_block, "page": pdf_page_no}
                )
            para_buf.clear()

        for ln in lines:
            if _RUNNING_HEADER_RE.match(ln) or re.match(r"^\d{1,4}$", ln):
                continue
            m_sec = _SECTION_HEADING_RE.match(ln)
            if m_sec:
                _flush_para_buf()
                sec_id = m_sec.group(1)
                sec_title = toc_sections.get(sec_id, (m_sec.group(2), pdf_page_no))[0]
                chapter_blocks[current_chapter].append(
                    {"type": "heading", "text": f"{sec_id} {sec_title}", "page": pdf_page_no}
                )
            elif re.match(r"^Summary$", ln, re.I):
                _flush_para_buf()
                chapter_blocks[current_chapter].append(
                    {"type": "heading", "text": f"{current_chapter}.99 Chapter Summary", "page": pdf_page_no}
                )
            else:
                para_buf.append(ln)

        _flush_para_buf()

    chunker = PDFChunker(
        strategy="heading",
        max_tokens=max_chunk_tokens,
        overlap_tokens=overlap_tokens,
        extract_tables=False,
        min_section_tokens=35,
    )

    chunks: list[SubtopicChunk] = []
    for ch_no, blocks in sorted(chapter_blocks.items()):
        ch_title = chapter_titles.get(ch_no, f"Chapter {ch_no}")
        domain = _resolve_chapter_domain(book_id, ch_no, ch_title)
        if domain not in VALID_CANONICAL_DOMAINS:
            continue

        ch_slug = f"{ch_no:02d}_{slugify_chapter_title(ch_title)}"
        raw_sections = chunker.chunk(blocks)
        for idx, sec in enumerate(raw_sections):
            heading = str(sec.get("heading") or f"{ch_no}.0 {ch_title}")
            cleaned_body = strip_visual_paragraphs(str(sec.get("text") or ""))
            if not cleaned_body or len(cleaned_body.split()) < 45:
                continue

            sec_parts = heading.split(" ", 1)
            sec_id = sec_parts[0] if re.match(r"^\d+\.\d+$", sec_parts[0]) else f"{ch_no}.0"
            sec_title = sec_parts[1].strip() if len(sec_parts) > 1 else ch_title
            page_no = int(sec.get("page") or 1)
            digest = hashlib.sha1(f"{book_id}:{ch_slug}:{idx}:{cleaned_body[:120]}".encode()).hexdigest()[:10]
            chunks.append(
                SubtopicChunk(
                    chunk_id=f"{book_id}:{ch_slug}:{sec_id}:{digest}",
                    book_id=book_id,
                    chapter=ch_no,
                    chapter_slug=ch_slug,
                    domain=domain,
                    section_id=sec_id,
                    section_title=sec_title,
                    text=cleaned_body,
                    token_count=len(cleaned_body.split()),
                    start_page=page_no,
                    end_page=page_no,
                )
            )

    logger.info("Extracted %d clean theory chunks from %s.", len(chunks), book_id)
    return chunks


def deduplicate_subtopic_chunks_within_domain(
    chunks: list[SubtopicChunk],
    *,
    minhash_threshold: float = 0.78,
) -> tuple[list[SubtopicChunk], int]:
    """Run CuratorKIT ``TextCleaner``, ``ExactDeduplicator``, and ``MinHashDeduplicator`` per domain."""
    if not chunks:
        return [], 0

    by_domain: dict[str, list[SubtopicChunk]] = defaultdict(list)
    for ch in chunks:
        by_domain[ch.domain].append(ch)

    surviving: list[SubtopicChunk] = []
    removed = 0
    for domain, dom_chunks in sorted(by_domain.items()):
        samples = [
            DataSample(
                id=ch.chunk_id,
                source_uri=f"{ch.book_id}:{ch.chapter_slug}",
                instruction=ch.section_title,
                input=ch.text,
                output=ch.text,
                task_type="corpus",
                metadata={"chunk_idx": i},
            )
            for i, ch in enumerate(dom_chunks)
        ]
        cleaner = TextCleaner(fields=["instruction", "output"])
        cleaned_samples = cleaner.run(samples)
        after_exact = ExactDeduplicator().run(cleaned_samples)
        after_minhash = MinHashDeduplicator(
            threshold=minhash_threshold,
            ngram=3,
            num_perm=128,
            seed=42,
        ).run(after_exact)

        surviving_ids = {s.id for s in after_minhash}
        for ch, smp in zip(dom_chunks, samples):
            if smp.id in surviving_ids:
                surviving.append(ch)
            else:
                removed += 1
        logger.info(
            "Subtopic dedup [%s]: %d -> %d chunks (-%d duplicates).",
            domain,
            len(dom_chunks),
            len(dom_chunks) - sum(1 for c in dom_chunks if c.chunk_id not in surviving_ids),
            sum(1 for c in dom_chunks if c.chunk_id not in surviving_ids),
        )
    return surviving, removed


def build_subtopic_corpus(
    *,
    book_config_path: str | Path = "configs/data/book_ingestion.yaml",
    download_dir: str | Path = DEFAULT_DOWNLOAD_DIR,
    output_path: str | Path = DEFAULT_SUBTOPIC_CHUNKS_PATH,
    summary_path: str | Path = DEFAULT_SUBTOPIC_SUMMARY_PATH,
    max_chunk_tokens: int = 380,
    minhash_threshold: float = 0.78,
) -> tuple[list[SubtopicChunk], dict[str, Any]]:
    """Extract & CuratorKIT-deduplicate subtopic chunks across all configured physics textbooks."""
    cfg = load_yaml(resolve_project_path(book_config_path))
    all_raw: list[SubtopicChunk] = []
    per_book_raw: dict[str, int] = {}

    for book_cfg in cfg.get("books", []):
        if book_cfg.get("role") not in ("problem_book", "problems") or not book_cfg.get("enabled", True):
            continue
        book_id = str(book_cfg["book_id"])
        pdf_path = resolve_book_path(book_cfg, download_dir=resolve_project_path(download_dir))
        if pdf_path is None or not pdf_path.is_file():
            logger.warning("Book PDF not found for %s at %s; skipping.", book_id, pdf_path)
            continue
        extracted = extract_book_subtopic_chunks(
            pdf_path,
            book_id,
            max_chunk_tokens=max_chunk_tokens,
        )
        per_book_raw[book_id] = len(extracted)
        all_raw.extend(extracted)

    deduped, removed_count = deduplicate_subtopic_chunks_within_domain(
        all_raw,
        minhash_threshold=minhash_threshold,
    )
    out_file = save_subtopic_chunks_jsonl(deduped, output_path)

    by_domain: dict[str, int] = defaultdict(int)
    for ch in deduped:
        by_domain[ch.domain] += 1

    summary = {
        "raw_chunks_count": len(all_raw),
        "deduped_chunks_count": len(deduped),
        "removed_duplicates": removed_count,
        "per_book_raw": per_book_raw,
        "by_domain": dict(sorted(by_domain.items())),
        "output_path": str(out_file),
    }
    sum_file = resolve_project_path(summary_path)
    sum_file.parent.mkdir(parents=True, exist_ok=True)
    sum_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return deduped, summary


def parse_problem_book_and_chapter(problem: CanonicalProblem) -> tuple[str | None, str | None]:
    """Parse ``(book_id, chapter_slug)`` from ``CanonicalProblem.original_id`` if present."""
    orig = str(problem.original_id or "")
    if ":" not in orig:
        return None, None
    parts = orig.split(":")
    if len(parts) < 2:
        return None, None
    return parts[0], parts[1]


def _normalize_topic_slug(chapter_slug: str | None) -> str:
    if not chapter_slug:
        return ""
    return re.sub(r"^\d+_", "", str(chapter_slug).strip().lower())


# Explicit cross-book chapter topic aliases for chapters in `light_and_matter_problems` (ch 21-36)
# and sibling Crowell volumes whose section prose lives in `crowell_fields_and_circuits` or `crowell_modern_physics`.
CHAPTER_TOPIC_ALIASES: dict[tuple[str, str], tuple[tuple[str, str], ...]] = {
    ("light_and_matter_problems", "21_electricity_and_circuits"): (
        ("crowell_fields_and_circuits", "01_the_electric_and_magnetic_fields"),
        ("crowell_fields_and_circuits", "04_the_electric_potential"),
        ("crowell_fields_and_circuits", "08_electrical_resistance"),
        ("crowell_fields_and_circuits", "09_parallel_and_series_circuits"),
        ("crowell_conceptual_physics", "05_electricity"),
    ),
    ("light_and_matter_problems", "22_the_nonmechanical_universe"): (
        ("crowell_fields_and_circuits", "01_the_electric_and_magnetic_fields"),
        ("crowell_fields_and_circuits", "02_gausss_law"),
        ("crowell_fields_and_circuits", "04_the_electric_potential"),
    ),
    ("light_and_matter_problems", "23_circuits"): (
        ("crowell_fields_and_circuits", "08_electrical_resistance"),
        ("crowell_fields_and_circuits", "09_parallel_and_series_circuits"),
    ),
    ("light_and_matter_problems", "24_electromagnetism"): (
        ("crowell_fields_and_circuits", "05_electromagnetism"),
        ("crowell_fields_and_circuits", "06_radiation"),
        ("crowell_fields_and_circuits", "11_sources_of_magnetism"),
    ),
    ("light_and_matter_problems", "25_capacitance_and_inductance"): (
        ("crowell_fields_and_circuits", "13_ac_circuits"),
        ("crowell_fields_and_circuits", "14_impedance"),
        ("crowell_fields_and_circuits", "04_the_electric_potential"),
        ("crowell_fields_and_circuits", "05_electromagnetism"),
    ),
    ("light_and_matter_problems", "26_the_atom_and_e_mc_2"): (
        ("crowell_fields_and_circuits", "03_models_of_matter"),
        ("crowell_fields_and_circuits", "07_more_about_relativity"),
        ("crowell_modern_physics", "06_relativistic_energy_and_momentum"),
        ("crowell_modern_physics", "13_the_atom_and_the_nucleus"),
    ),
    ("light_and_matter_problems", "27_general_relativity"): (
        ("crowell_modern_physics", "01_time"),
        ("crowell_modern_physics", "04_the_lorentz_transformation"),
        ("crowell_modern_physics", "06_relativistic_energy_and_momentum"),
        ("crowell_fields_and_circuits", "07_more_about_relativity"),
        ("crowell_fields_and_circuits", "17_relativity"),
    ),
    ("light_and_matter_problems", "28_the_ray_model_of_light"): (
        ("crowell_modern_physics", "10_images_qualitatively"),
        ("crowell_conceptual_physics", "07_the_ray_model_of_light"),
    ),
    ("light_and_matter_problems", "29_images"): (
        ("crowell_modern_physics", "10_images_qualitatively"),
        ("crowell_modern_physics", "11_images_quantiitatively"),
    ),
    ("light_and_matter_problems", "30_images_quantitatively"): (
        ("crowell_modern_physics", "11_images_quantiitatively"),
        ("crowell_modern_physics", "10_images_qualitatively"),
    ),
    ("light_and_matter_problems", "31_refraction"): (
        ("crowell_modern_physics", "10_images_qualitatively"),
        ("crowell_modern_physics", "11_images_quantiitatively"),
        ("crowell_modern_physics", "12_wave_optics"),
        ("crowell_conceptual_physics", "07_the_ray_model_of_light"),
    ),
    ("light_and_matter_problems", "32_wave_optics"): (
        ("crowell_modern_physics", "12_wave_optics"),
    ),
    ("light_and_matter_problems", "33_optics"): (
        ("crowell_modern_physics", "10_images_qualitatively"),
        ("crowell_modern_physics", "11_images_quantiitatively"),
        ("crowell_modern_physics", "12_wave_optics"),
    ),
    ("light_and_matter_problems", "34_light_as_a_particle"): (
        ("crowell_modern_physics", "15_light_as_a_particle"),
    ),
    ("light_and_matter_problems", "35_matter_as_a_wave"): (
        ("crowell_modern_physics", "16_matter_as_a_wave"),
        ("crowell_modern_physics", "17_the_schrodinger_equation"),
    ),
    ("light_and_matter_problems", "36_atoms"): (
        ("crowell_modern_physics", "17_the_schrodinger_equation"),
        ("crowell_modern_physics", "18_quantization_of_angular_momentum"),
    ),
}

_STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "and", "for", "with", "that", "from", "this", "what", "when", "where",
        "which", "find", "calculate", "determine", "show", "give", "given", "into",
        "about", "does", "have", "are", "was", "were", "will", "would", "can",
    }
)


def _extract_physics_keywords(text: str) -> set[str]:
    """Extract normalized domain terms and symbol tokens for hybrid lexical + embedding scoring."""
    tokens = re.findall(r"[A-Za-z_]{3,}|\\[A-Za-z]+", (text or "").lower())
    return {t.lstrip("\\") for t in tokens if len(t) >= 3 and t not in _STOPWORDS}


class SubtopicMapper:
    """Zero-LLM chapter-scoped subtopic retriever backed by ``CuratorKIT`` ``PostHocRetriever``."""

    def __init__(
        self,
        chunks: list[SubtopicChunk],
        *,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.chunks = chunks
        self.embedding_model = embedding_model
        self._chunk_texts = [f"{c.section_id} {c.section_title}\n{c.text}" for c in chunks]
        self._chunk_keywords = [_extract_physics_keywords(f"{c.section_title} {c.text}") for c in chunks]
        self._title_keywords = [_extract_physics_keywords(c.section_title) for c in chunks]
        self._is_summary = [
            c.section_id.endswith(".99") or "summary" in c.section_title.lower()
            for c in chunks
        ]
        self._by_book_chapter: dict[tuple[str, str], list[int]] = defaultdict(list)
        self._by_domain_topic: dict[tuple[str, str], list[int]] = defaultdict(list)
        self._by_domain: dict[str, list[int]] = defaultdict(list)
        for idx, ch in enumerate(chunks):
            self._by_book_chapter[(ch.book_id, ch.chapter_slug)].append(idx)
            topic = _normalize_topic_slug(ch.chapter_slug)
            if topic:
                self._by_domain_topic[(ch.domain, topic)].append(idx)
            self._by_domain[ch.domain].append(idx)
        self._retriever: Any = None

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path = DEFAULT_SUBTOPIC_CHUNKS_PATH,
        *,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> SubtopicMapper:
        return cls(load_subtopic_chunks_jsonl(path), embedding_model=embedding_model)

    def _ensure_retriever(self) -> Any:
        if self._retriever is None and self._chunk_texts:
            from curatorkit.diagnostic.retriever import PostHocRetriever

            self._retriever = PostHocRetriever(self._chunk_texts, model=self.embedding_model)
        return self._retriever

    def _select_candidate_indices(self, problem: CanonicalProblem) -> tuple[list[int], int]:
        """Select candidate chunk indices using 3-tier chapter -> alias/topic -> domain scoping."""
        book_id, chapter_slug = parse_problem_book_and_chapter(problem)
        topic = _normalize_topic_slug(chapter_slug)

        candidate_indices: list[int] = []
        match_tier = 3
        if book_id and chapter_slug and (book_id, chapter_slug) in self._by_book_chapter:
            candidate_indices = list(self._by_book_chapter[(book_id, chapter_slug)])
            match_tier = 1
        elif book_id and chapter_slug and (book_id, chapter_slug) in CHAPTER_TOPIC_ALIASES:
            alias_hits: list[int] = []
            for target_key in CHAPTER_TOPIC_ALIASES[(book_id, chapter_slug)]:
                alias_hits.extend(self._by_book_chapter.get(target_key, ()))
            if alias_hits:
                # Deduplicate while preserving order
                candidate_indices = list(dict.fromkeys(alias_hits))
                match_tier = 2
        if not candidate_indices and topic and (problem.domain, topic) in self._by_domain_topic:
            candidate_indices = list(self._by_domain_topic[(problem.domain, topic)])
            match_tier = 2
        if not candidate_indices and problem.domain in self._by_domain:
            candidate_indices = list(self._by_domain[problem.domain])
            match_tier = 3
        return candidate_indices, match_tier

    def find_subtopics_with_tiers(
        self,
        problem: CanonicalProblem,
        *,
        top_k: int = 1,
    ) -> list[tuple[SubtopicChunk, int]]:
        """Return ``[(SubtopicChunk, match_tier), ...]`` ranked by CuratorKIT embeddings + lexical physics overlap."""
        candidate_indices, match_tier = self._select_candidate_indices(problem)
        if not candidate_indices:
            return []

        # Prefer focused section prose over catch-all end-of-chapter summaries when enough sections exist
        non_summary = [idx for idx in candidate_indices if not self._is_summary[idx]]
        if len(non_summary) >= top_k:
            candidate_indices = non_summary

        if len(candidate_indices) <= top_k:
            return [(self.chunks[i], match_tier) for i in candidate_indices]

        import numpy as np

        retriever = self._ensure_retriever()
        query = f"{problem.problem_statement}\n{problem.reference_solution[:400]}".strip()
        q_emb = retriever.model.encode([query], normalize_embeddings=True)
        sub_index = retriever.index[candidate_indices]
        emb_scores = np.asarray(sub_index @ q_emb.T, dtype=np.float32).flatten()

        q_kw = _extract_physics_keywords(query)
        hybrid_scores = np.empty_like(emb_scores)
        denom = max(1, len(q_kw))
        for local_i, global_idx in enumerate(candidate_indices):
            c_kw = self._chunk_keywords[global_idx]
            t_kw = self._title_keywords[global_idx]
            lex_overlap = len(q_kw & c_kw) / denom
            title_bonus = 0.08 * (len(q_kw & t_kw) / max(1, len(t_kw))) if t_kw else 0.0
            summary_penalty = -0.10 if self._is_summary[global_idx] else 0.0
            hybrid_scores[local_i] = 0.78 * float(emb_scores[local_i]) + 0.22 * lex_overlap + title_bonus + summary_penalty

        top_local = np.argsort(hybrid_scores)[::-1][:top_k]
        return [(self.chunks[candidate_indices[int(loc)]], match_tier) for loc in top_local]

    def find_subtopics(self, problem: CanonicalProblem, *, top_k: int = 1) -> list[SubtopicChunk]:
        """Return ``top_k`` best-matching subtopic chunks scoped to the problem's chapter (or domain)."""
        return [chunk for chunk, _ in self.find_subtopics_with_tiers(problem, top_k=top_k)]
