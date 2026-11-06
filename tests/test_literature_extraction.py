"""Unit tests for Stage 1.2 book loader (PDF/DOCX), solution mapping, offline code check, and literature extraction filters."""
from __future__ import annotations

import zipfile
from pathlib import Path

from adversarial_physics.data.curation.offline_code_check import verify_reference_code
from adversarial_physics.data.ingestion.book_downloader import resolve_book_path
from adversarial_physics.data.ingestion.book_loader import (
    load_book_chunks,
    load_books_from_sources,
    load_docx_paragraphs,
)
from adversarial_physics.data.ingestion.literature_extractor import (
    build_paired_problem_chunks,
    extract_literature_problems,
    has_unsplit_multipart,
    is_diagram_dependent,
    run_literature_extraction_from_config,
    validate_and_build_canonical_problem,
)
from adversarial_physics.data.ingestion.solution_mapping import (
    build_linking_report,
    link_problems_to_solutions,
    parse_solution_manual,
    segment_problems,
)
from adversarial_physics.utils.config import load_book_ingestion_config


def _create_minimal_docx(path: Path, paragraphs: list[str]) -> Path:
    """Create a valid minimal .docx archive containing the given paragraphs."""
    body_xml = "".join(
        f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p>" for text in paragraphs
    )
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body_xml}</w:body>"
        "</w:document>"
    )
    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    )
    rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="word/document.xml"/>'
        "</Relationships>"
    )
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("[Content_Types].xml", content_types_xml)
        zf.writestr("_rels/.rels", rels_xml)
        zf.writestr("word/document.xml", document_xml)
    return path


def test_load_docx_paragraphs_and_chunks(tmp_path: Path):
    docx_file = _create_minimal_docx(
        tmp_path / "sample_problems.docx",
        [
            "Problem 1: A block of mass 2.0 kg slides down a frictionless incline of height 5.0 m.",
            "Find its speed at the bottom using conservation of energy.",
        ],
    )
    paragraphs = load_docx_paragraphs(docx_file)
    assert len(paragraphs) == 2
    assert "block of mass 2.0 kg" in paragraphs[0]

    chunks = load_book_chunks(docx_file, book_id="sample_docx", chunk_size_chars=500, chunk_overlap_chars=50)
    assert len(chunks) == 1
    assert chunks[0].format == "docx"
    assert chunks[0].book_id == "sample_docx"


def test_book_ingestion_config_has_download_urls_and_skips_when_not_downloaded():
    cfg = load_book_ingestion_config("configs/data/book_ingestion.yaml")
    assert cfg["dataset_id"] == "literature_book_ingestion"

    by_id = {b["book_id"]: b for b in cfg["books"]}
    assert by_id["light_and_matter_solutions"]["role"] == "solution_book"
    assert by_id["light_and_matter_solutions"]["url"].endswith("solns-all.pdf")
    assert "Light and Matter" in by_id["light_and_matter_solutions"]["sections"]
    assert "Mechanics" in by_id["light_and_matter_solutions"]["sections"]

    for pid, sec, pdf_name in [
        ("light_and_matter_problems", "Light and Matter", "lm.pdf"),
        ("crowell_mechanics", "Mechanics", "me.pdf"),
        ("crowell_fields_and_circuits", "Fields and Circuits", "fac.pdf"),
        ("crowell_modern_physics", "Modern Physics", "mod.pdf"),
    ]:
        entry = by_id[pid]
        assert entry["role"] == "problem_book"
        assert entry["solution_book_id"] == "light_and_matter_solutions"
        assert entry["solution_section"] == sec
        assert entry["url"].endswith(pdf_name)
        resolved = resolve_book_path(entry)
        assert resolved is not None and resolved.name == f"{pid}.pdf"

    chunks = load_books_from_sources(cfg["books"], skip_empty_paths=True, skip_missing_files=True)
    assert chunks == []

    accepted, rejected, summary = run_literature_extraction_from_config(
        "configs/data/book_ingestion.yaml", dry_run=True
    )
    assert accepted == []
    assert rejected == []
    assert summary.books_loaded == 0


def test_multi_book_to_single_solution_manual_mapping(tmp_path: Path):
    # Simulate two separate Crowell problem books (Light and Matter & Mechanics)
    # and one shared all-in-one solution manual containing sections for both.
    lm_book = tmp_path / "lm.txt"
    lm_book.write_text(
        "\n".join(
            [
                "Chapter 2",
                "Problems",
                "Key",
                "1 A solid sphere of mass 2.0 kg and radius 0.15 m rolls without slipping down an incline of height 1.4 m. Find its translational speed at the bottom.",
                "2 A car accelerates uniformly from rest to 20 m/s in 5.0 s. Find the distance traveled.",
            ]
        ),
        encoding="utf-8",
    )

    me_book = tmp_path / "me.txt"
    me_book.write_text(
        "\n".join(
            [
                "Chapter 1",
                "Problems",
                "Key",
                "1 The speed of light is 3.0x10^8 m/s. Convert this to km/s.",
            ]
        ),
        encoding="utf-8",
    )

    sol_manual = tmp_path / "solns_all.txt"
    sol_manual.write_text(
        "\n".join(
            [
                "Contents",
                "1 Light and Matter",
                "3 Mechanics",
                "",
                "1 Light and Matter",
                "2-1 By conservation of energy, mgh = (7/10)mv^2, so v = sqrt(10gh/7) = 4.43 m/s.",
                "2-2 Using x = (1/2)at^2 with a = 4.0 m/s^2 gives x = 50 m.",
                "",
                "3 Mechanics",
                "1-1 Dividing by 1000 m/km gives 3.0x10^5 km/s.",
            ]
        ),
        encoding="utf-8",
    )

    sources = [
        {
            "book_id": "lm_solutions",
            "role": "solution_book",
            "format": "txt",
            "path": str(sol_manual),
            "sections": ["Light and Matter", "Mechanics"],
            "enabled": True,
        },
        {
            "book_id": "lm_problems",
            "role": "problem_book",
            "title": "Light and Matter",
            "format": "txt",
            "path": str(lm_book),
            "domain": "MECHANICS",
            "solution_book_id": "lm_solutions",
            "solution_section": "Light and Matter",
            "enabled": True,
        },
        {
            "book_id": "me_problems",
            "role": "problem_book",
            "title": "Mechanics",
            "format": "txt",
            "path": str(me_book),
            "domain": "MECHANICS",
            "solution_book_id": "lm_solutions",
            "solution_section": "Mechanics",
            "enabled": True,
        },
    ]

    paired_chunks, reports, linked, handled = build_paired_problem_chunks(
        sources, require_official_solution=True, shuffle_seed=None
    )
    assert len(paired_chunks) == 3
    assert handled == {"lm_solutions", "lm_problems", "me_problems"}
    by_chunk_id = {c.chunk_id: c for c in paired_chunks}

    ch2_p1 = by_chunk_id["lm_problems_ch02_p001"]
    assert "solid sphere of mass 2.0 kg" in ch2_p1.text
    assert "OFFICIAL SOLUTION" in ch2_p1.text
    assert "sqrt(10gh/7) = 4.43 m/s" in ch2_p1.text
    assert ch2_p1.metadata["problem_ref"] == "lm_problems:ch02_p001"
    assert ch2_p1.metadata["solution_key"] == "2-1"

    me_ch1_p1 = by_chunk_id["me_problems_ch01_p001"]
    assert "3.0x10^5 km/s" in me_ch1_p1.text
    assert me_ch1_p1.metadata["problem_ref"] == "me_problems:ch01_p001"

    # Verify that extract_literature_problems preserves problem_ref in CanonicalProblem.original_id
    def fake_extractor(chunk):
        if chunk.chunk_id == "lm_problems_ch02_p001":
            return [
                {
                    "domain": "MECHANICS",
                    "derivation_steps": 3,
                    "problem_statement": "A solid sphere of mass 2.0 kg and radius 0.15 m rolls without slipping down an incline of height 1.4 m. Find its translational speed at the bottom.",
                    "reference_solution": "By conservation of energy, mgh = 7/10 mv^2, so v = sqrt(10gh/7) = 4.43 m/s.",
                    "reference_answer": "4.43 m/s",
                    "reference_equation": "sqrt(10 * g * h / 7)",
                    "reference_code": "def solve(m=2.0, r=0.15, h=1.4, g=9.8):\n    return (10 * g * h / 7) ** 0.5",
                }
            ]
        return []

    accepted, rejected, _ = extract_literature_problems(paired_chunks, chunk_extractor=fake_extractor)
    assert len(accepted) == 1
    assert accepted[0].original_id == "lm_problems:ch02_p001"
    assert rejected == []


def test_offline_code_verification_passes_valid_blueprint_problem():
    res = verify_reference_code(
        reference_code="def solve(m=2.0, r=0.15, h=1.4, g=9.8):\n    return (10 * g * h / 7) ** 0.5",
        reference_answer="4.43 m/s",
        reference_equation="sqrt(10 * g * h / 7)",
        rel_tol=0.01,
    )
    assert res.passed is True
    assert res.relative_error is not None and res.relative_error < 0.01


def test_stage_1_2_filters_reject_diagrams_multipart_and_bad_code():
    assert is_diagram_dependent("Calculate the tension as shown in Figure 3.2.") is True
    assert is_diagram_dependent("A solid sphere rolls without slipping down an incline.") is False

    assert has_unsplit_multipart("Find (a) the acceleration of the block and (b) the tension in the cord.") is True
    assert has_unsplit_multipart("Find the translational speed at the bottom.") is False

    valid_candidate = {
        "domain": "MECHANICS",
        "derivation_steps": 3,
        "problem_statement": "A solid sphere of mass 2.0 kg and radius 0.15 m rolls without slipping down an incline of height 1.4 m. Find its translational speed at the bottom.",
        "reference_solution": "By conservation of mechanical energy, mgh = 1/2 mv^2 + 1/2 I*omega^2. Since I = 2/5 mr^2 and omega = v/r, mgh = 7/10 mv^2. Solving for v gives v = sqrt(10gh/7) = 4.43 m/s.",
        "reference_answer": "4.43 m/s",
        "reference_equation": "sqrt(10 * g * h / 7)",
        "reference_code": "def solve(m=2.0, r=0.15, h=1.4, g=9.8):\n    return (10 * g * h / 7) ** 0.5",
    }
    problem, reason, check = validate_and_build_canonical_problem(
        valid_candidate,
        problem_id="lit_mech_0142",
        book_id="openstax_vol1",
        chunk_id="chunk_0001",
    )
    assert problem is not None
    assert reason is None
    assert check is not None and check.passed is True
    assert problem.source == "literature_curated"

    wrong_code_candidate = {
        **valid_candidate,
        "reference_code": "def solve(m=2.0, r=0.15, h=1.4, g=9.8):\n    return (2 * g * h) ** 0.5",
    }
    bad_problem, bad_reason, _ = validate_and_build_canonical_problem(
        wrong_code_candidate,
        problem_id="lit_mech_0143",
        book_id="openstax_vol1",
        chunk_id="chunk_0001",
    )
    assert bad_problem is None
    assert bad_reason is not None and "offline_code_verification_failed" in bad_reason
