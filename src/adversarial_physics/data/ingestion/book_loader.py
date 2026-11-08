"""Book and document loader supporting PDF, DOCX, TXT, and JSONL files for Stage 1.2."""
from __future__ import annotations

import json
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


@dataclass(frozen=True)
class BookChunk:
    """A text chunk extracted from a textbook or problem book (PDF, DOCX, TXT, or JSONL)."""

    chunk_id: str
    book_id: str
    source_path: str
    format: str
    text: str
    role: str = "standalone"  # "problem_book" | "solution_book" | "standalone"
    solution_book_id: str | None = None  # Reference linking a problem book to its solution book
    page_start: int | None = None
    page_end: int | None = None
    domain_hint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_pdf_pages(
    path: str | Path,
    *,
    start_page: int = 1,
    end_page: int | None = None,
) -> list[tuple[int, str]]:
    """Extract (page_number, text) tuples from a PDF file (1-indexed page numbers).

    Tries `pypdf` first, then `pymupdf` (`fitz`), then `pdfplumber`.
    """
    pdf_path = Path(path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # 1. Try pypdf
    try:
        import logging

        from pypdf import PdfReader  # type: ignore[import-not-found]

        # pypdf logs a very long "fontTools is required..." warning per font; text is still fine.
        logging.getLogger("pypdf").setLevel(logging.ERROR)
        reader = PdfReader(str(pdf_path))
        total_pages = len(reader.pages)
        first_idx = max(0, start_page - 1)
        last_idx = min(total_pages, end_page) if end_page is not None else total_pages
        pages: list[tuple[int, str]] = []
        for idx in range(first_idx, last_idx):
            page_text = (reader.pages[idx].extract_text() or "").strip()
            if page_text:
                pages.append((idx + 1, page_text))
        return pages
    except ImportError:
        pass

    # 2. Try PyMuPDF (fitz)
    try:
        import fitz  # type: ignore[import-not-found]

        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        first_idx = max(0, start_page - 1)
        last_idx = min(total_pages, end_page) if end_page is not None else total_pages
        pages = []
        for idx in range(first_idx, last_idx):
            page_text = doc[idx].get_text("text").strip()
            if page_text:
                pages.append((idx + 1, page_text))
        doc.close()
        return pages
    except ImportError:
        pass

    # 3. Try pdfplumber
    try:
        import pdfplumber  # type: ignore[import-not-found]

        pages = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            total_pages = len(pdf.pages)
            first_idx = max(0, start_page - 1)
            last_idx = min(total_pages, end_page) if end_page is not None else total_pages
            for idx in range(first_idx, last_idx):
                page_text = (pdf.pages[idx].extract_text() or "").strip()
                if page_text:
                    pages.append((idx + 1, page_text))
        return pages
    except ImportError as exc:
        raise ImportError(
            "Loading PDF books requires 'pypdf', 'pymupdf', or 'pdfplumber'. "
            "Install via: pip install pypdf"
        ) from exc


def _extract_docx_with_stdlib_zip(docx_path: Path) -> list[str]:
    """Fallback DOCX reader using Python's built-in zipfile + xml.etree (zero extra dependencies)."""
    paragraphs: list[str] = []
    with zipfile.ZipFile(docx_path, "r") as archive:
        if "word/document.xml" not in archive.namelist():
            raise ValueError(f"Invalid .docx file (missing word/document.xml): {docx_path}")
        xml_bytes = archive.read("word/document.xml")

    root = ET.fromstring(xml_bytes)
    # WordprocessingML paragraph tag ends with '}p' and text runs end with '}t'
    for elem in root.iter():
        if elem.tag.endswith("}p"):
            texts = [node.text for node in elem.iter() if node.tag.endswith("}t") and node.text]
            paragraph = "".join(texts).strip()
            if paragraph:
                paragraphs.append(paragraph)
    return paragraphs


def load_docx_paragraphs(path: str | Path) -> list[str]:
    """Extract non-empty paragraphs and table rows from a .docx file.

    Uses `python-docx` when installed, with a seamless stdlib XML fallback.
    """
    docx_path = Path(path)
    if not docx_path.is_file():
        raise FileNotFoundError(f"DOCX file not found: {docx_path}")

    try:
        import docx  # type: ignore[import-not-found]

        document = docx.Document(str(docx_path))
        blocks: list[str] = []
        for para in document.paragraphs:
            cleaned = para.text.strip()
            if cleaned:
                blocks.append(cleaned)
        for table in document.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if cells:
                    blocks.append(" | ".join(cells))
        return blocks
    except ImportError:
        return _extract_docx_with_stdlib_zip(docx_path)


def chunk_text_segments(
    segments: list[tuple[int | None, str]],
    *,
    book_id: str,
    source_path: str,
    file_format: str,
    role: str = "standalone",
    solution_book_id: str | None = None,
    chunk_size_chars: int = 4000,
    chunk_overlap_chars: int = 400,
    domain_hint: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[BookChunk]:
    """Group page or paragraph segments into overlapping BookChunk windows."""
    if chunk_size_chars <= 0:
        raise ValueError("chunk_size_chars must be positive")
    if chunk_overlap_chars < 0 or chunk_overlap_chars >= chunk_size_chars:
        raise ValueError("chunk_overlap_chars must be in [0, chunk_size_chars)")

    chunks: list[BookChunk] = []
    buffer_texts: list[str] = []
    buffer_pages: list[int] = []
    buffer_len = 0
    chunk_index = 0

    def flush_buffer() -> None:
        nonlocal chunk_index
        if not buffer_texts:
            return
        combined = "\n\n".join(buffer_texts).strip()
        if not combined:
            return
        page_start = min(buffer_pages) if buffer_pages else None
        page_end = max(buffer_pages) if buffer_pages else None
        chunks.append(
            BookChunk(
                chunk_id=f"{book_id}_chunk_{chunk_index:04d}",
                book_id=book_id,
                source_path=source_path,
                format=file_format,
                text=combined,
                role=role,
                solution_book_id=solution_book_id,
                page_start=page_start,
                page_end=page_end,
                domain_hint=domain_hint,
                metadata=dict(metadata or {}),
            )
        )
        chunk_index += 1

    for page_num, seg_text in segments:
        cleaned = seg_text.strip()
        if not cleaned:
            continue

        # If a single segment is larger than chunk_size_chars, split it by paragraphs or slices
        if len(cleaned) > chunk_size_chars:
            flush_buffer()
            buffer_texts.clear()
            buffer_pages.clear()
            buffer_len = 0

            step = chunk_size_chars - chunk_overlap_chars
            for start in range(0, len(cleaned), step):
                sub_text = cleaned[start : start + chunk_size_chars].strip()
                if sub_text:
                    chunks.append(
                        BookChunk(
                            chunk_id=f"{book_id}_chunk_{chunk_index:04d}",
                            book_id=book_id,
                            source_path=source_path,
                            format=file_format,
                            text=sub_text,
                            role=role,
                            solution_book_id=solution_book_id,
                            page_start=page_num,
                            page_end=page_num,
                            domain_hint=domain_hint,
                            metadata=dict(metadata or {}),
                        )
                    )
                    chunk_index += 1
            continue

        if buffer_len + len(cleaned) + 2 > chunk_size_chars and buffer_texts:
            flush_buffer()
            # Retain trailing segments up to chunk_overlap_chars
            retained_texts: list[str] = []
            retained_pages: list[int] = []
            retained_len = 0
            for prev_text, prev_page in zip(reversed(buffer_texts), reversed(buffer_pages)):
                if retained_len + len(prev_text) <= chunk_overlap_chars:
                    retained_texts.insert(0, prev_text)
                    if prev_page is not None:
                        retained_pages.insert(0, prev_page)
                    retained_len += len(prev_text) + 2
                else:
                    break
            buffer_texts = retained_texts
            buffer_pages = retained_pages
            buffer_len = retained_len

        buffer_texts.append(cleaned)
        if page_num is not None:
            buffer_pages.append(page_num)
        buffer_len += len(cleaned) + 2

    flush_buffer()
    return chunks


def load_book_segments(
    path: str | Path,
    *,
    file_format: str | None = None,
    start_page: int = 1,
    end_page: int | None = None,
) -> list[tuple[int | None, str]]:
    """Load raw text segments from a .pdf (one per page), .docx (one per paragraph) or .txt/.md file.

    Segments are ``(page_number_or_None, text)`` tuples with original line breaks preserved,
    which is what the line-based problem/solution parser in ``solution_mapping`` needs.
    """
    book_path = Path(path)
    if not book_path.is_file():
        raise FileNotFoundError(f"Book file not found: {book_path}")
    resolved_format = (file_format or book_path.suffix.lstrip(".")).lower()

    if resolved_format == "pdf":
        return [(p, t) for p, t in load_pdf_pages(book_path, start_page=start_page, end_page=end_page)]
    if resolved_format == "docx":
        return [(None, p) for p in load_docx_paragraphs(book_path)]
    if resolved_format in {"txt", "md"}:
        raw_text = book_path.read_text(encoding="utf-8", errors="replace")
        # Form feeds (\f) mark page breaks in some text exports; keep them as page boundaries.
        pages = raw_text.split("\f")
        if len(pages) > 1:
            return [(idx + 1, page) for idx, page in enumerate(pages) if page.strip()]
        return [(None, blk.strip()) for blk in raw_text.split("\n\n") if blk.strip()]
    raise ValueError(
        f"Unsupported segment format {resolved_format!r} for {book_path}. Supported: 'pdf', 'docx', 'txt', 'md'."
    )


def load_book_chunks(
    path: str | Path,
    *,
    book_id: str | None = None,
    file_format: str | None = None,
    role: str = "standalone",
    solution_book_id: str | None = None,
    start_page: int = 1,
    end_page: int | None = None,
    chunk_size_chars: int = 4000,
    chunk_overlap_chars: int = 400,
    domain_hint: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[BookChunk]:
    """Load a textbook or problem file (.pdf, .docx, .txt, .md, or .jsonl) into BookChunk records."""
    book_path = Path(path)
    if not book_path.is_file():
        raise FileNotFoundError(f"Book file not found: {book_path}")

    resolved_format = (file_format or book_path.suffix.lstrip(".")).lower()
    resolved_book_id = book_id or book_path.stem

    if resolved_format in {"pdf", "docx", "txt", "md"}:
        segments = load_book_segments(
            book_path, file_format=resolved_format, start_page=start_page, end_page=end_page
        )
        return chunk_text_segments(
            segments,
            book_id=resolved_book_id,
            source_path=str(book_path),
            file_format=resolved_format,
            role=role,
            solution_book_id=solution_book_id,
            chunk_size_chars=chunk_size_chars,
            chunk_overlap_chars=chunk_overlap_chars,
            domain_hint=domain_hint,
            metadata=metadata,
        )

    if resolved_format in {"jsonl", "json"}:
        chunks: list[BookChunk] = []
        with book_path.open("r", encoding="utf-8") as file:
            for idx, line in enumerate(file):
                stripped = line.strip()
                if not stripped:
                    continue
                record = json.loads(stripped)
                text = str(
                    record.get("text")
                    or record.get("content")
                    or record.get("problem_statement")
                    or json.dumps(record, ensure_ascii=False)
                )
                chunks.append(
                    BookChunk(
                        chunk_id=str(record.get("chunk_id") or f"{resolved_book_id}_chunk_{idx:04d}"),
                        book_id=resolved_book_id,
                        source_path=str(book_path),
                        format="jsonl",
                        text=text,
                        role=str(record.get("role") or role),
                        solution_book_id=record.get("solution_book_id") or solution_book_id,
                        page_start=record.get("page_start"),
                        page_end=record.get("page_end"),
                        domain_hint=record.get("domain") or domain_hint,
                        metadata={**(metadata or {}), "raw_record": record},
                    )
                )
        return chunks

    raise ValueError(
        f"Unsupported book format {resolved_format!r} for {book_path}. "
        "Supported formats: 'pdf', 'docx', 'txt', 'md', 'jsonl'."
    )


def load_books_from_sources(
    sources: list[dict[str, Any]],
    *,
    default_chunk_size_chars: int = 4000,
    default_chunk_overlap_chars: int = 400,
    skip_empty_paths: bool = True,
    skip_missing_files: bool = False,
    download_dir: str | Path = "data/books",
) -> list[BookChunk]:
    """Load BookChunks from a list of book source dicts defined in YAML config.

    Entries whose `path` and `url` are blank (`""` or `None`) are skipped gracefully
    when `skip_empty_paths=True`. Entries with only a `url` are read from their default
    download location `<download_dir>/<book_id>.<format>` (see `book_downloader`).
    """
    from adversarial_physics.data.ingestion.book_downloader import resolve_book_path

    all_chunks: list[BookChunk] = []
    for index, entry in enumerate(sources):
        if not entry.get("enabled", True):
            continue
        raw_url = str(entry.get("url") or "").strip()
        local_path = resolve_book_path(entry, download_dir=download_dir)

        if local_path is None:
            if skip_empty_paths:
                continue
            raise ValueError(f"Book source entry at index {index} has neither 'path' nor 'url' set.")

        if not local_path.is_file():
            if skip_missing_files:
                continue
            raise FileNotFoundError(
                f"Book {entry.get('book_id', index)!r} not found at {local_path}. "
                + (
                    "Download it first: python scripts/02_extract_literature.py --download-only"
                    if raw_url
                    else "Fix its 'path' in the book ingestion config."
                )
            )

        book_chunks = load_book_chunks(
            local_path,
            book_id=entry.get("book_id") or f"book_{index + 1:02d}",
            file_format=entry.get("format"),
            role=str(entry.get("role") or "standalone"),
            solution_book_id=entry.get("solution_book_id"),
            start_page=int(entry.get("start_page") or 1),
            end_page=int(entry.get("end_page")) if entry.get("end_page") is not None else None,
            chunk_size_chars=int(entry.get("chunk_size_chars") or default_chunk_size_chars),
            chunk_overlap_chars=int(entry.get("chunk_overlap_chars") or default_chunk_overlap_chars),
            domain_hint=entry.get("domain"),
            metadata={
                "url": raw_url,
                "title": entry.get("title", ""),
                "solution_book_id": entry.get("solution_book_id"),
                "role": entry.get("role", "standalone"),
            },
        )
        all_chunks.extend(book_chunks)

    return all_chunks
