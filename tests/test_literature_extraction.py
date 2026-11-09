"""Unit tests for Stage 1.2 book loader (PDF/DOCX), solution mapping, Aya-Expanse runtime selection, and literature extraction filters."""
from __future__ import annotations

import zipfile
from pathlib import Path

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
from adversarial_physics.data.ingestion.chapter_domain_extractor import (
    MANUAL_CHAPTER_DOMAIN_MAP,
    SIDE_DOMAIN,
    VALID_CANONICAL_DOMAINS,
    build_initial_book_dataset,
)
from adversarial_physics.data.ingestion.pdf_chapter_splitter import (
    extract_book_chapter_titles,
    slugify_chapter_title,
)
from adversarial_physics.utils.config import load_book_ingestion_config
from adversarial_physics.utils.runtime import (
    AYA_EXPANSE_8B,
    AYA_EXPANSE_32B,
    RuntimeConfig,
    detect_runtime_backend,
    select_aya_expanse_model,
)


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


def test_book_ingestion_config_has_download_urls_and_aya_model():
    cfg = load_book_ingestion_config("configs/data/book_ingestion.yaml")
    assert cfg["dataset_id"] == "literature_book_ingestion"
    assert cfg["extraction"]["teacher_model"] == AYA_EXPANSE_8B
    assert cfg["output"]["repo_id"].startswith("vital-ar/")

    by_id = {b["book_id"]: b for b in cfg["books"]}
    assert by_id["light_and_matter_solutions"]["role"] == "solution_book"
    assert by_id["light_and_matter_solutions"]["url"].endswith("solns-all.pdf")

    for pid, sec, pdf_name in [
        ("light_and_matter_problems", "Light and Matter", "lm.pdf"),
        ("crowell_mechanics", "Mechanics", "me.pdf"),
        ("crowell_fields_and_circuits", "Fields and Circuits", "fac.pdf"),
        ("crowell_modern_physics", "Modern Physics", "mod.pdf"),
        ("crowell_conceptual_physics", "Conceptual Physics", "cp.pdf"),
    ]:
        entry = by_id[pid]
        assert entry["role"] == "problem_book"
        assert entry["solution_book_id"] == "light_and_matter_solutions"
        assert entry["solution_section"] == sec
        assert entry["url"].endswith(pdf_name)
        resolved = resolve_book_path(entry)
        assert resolved is not None and resolved.name == f"{pid}.pdf"


def test_aya_expanse_model_selector_and_multi_backend_runtime():
    cpu_rt = detect_runtime_backend("local_cpu")
    assert cpu_rt.backend == "local_cpu"
    assert cpu_rt.device == "cpu"

    zerogpu_rt = detect_runtime_backend("hf_zerogpu")
    assert zerogpu_rt.backend in ("hf_gpu", "hf_zerogpu")
    assert zerogpu_rt.device in ("cuda", "hf_remote_gpu")

    kaggle_big = RuntimeConfig(
        backend="kaggle_gpu",
        execution_mode="local_cuda",
        device="cuda",
        torch_dtype="float16",
        load_in_4bit=True,
        device_map="auto",
        is_kaggle=True,
        is_hf_gpu=False,
        cuda_available=True,
        accelerate_available=True,
        vram_gb=30.0,
    )
    model_extract, _ = select_aya_expanse_model("literature_extraction", runtime=kaggle_big)
    assert model_extract == AYA_EXPANSE_8B

    model_flaw_gpu, _ = select_aya_expanse_model("adversarial_flaw_synthesis", runtime=kaggle_big)
    assert model_flaw_gpu == AYA_EXPANSE_8B

    model_flaw_cpu, _ = select_aya_expanse_model("adversarial_flaw_synthesis", runtime=cpu_rt)
    assert model_flaw_cpu == AYA_EXPANSE_8B


def test_multi_book_to_single_solution_manual_mapping(tmp_path: Path):
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

    def fake_extractor(chunk):
        if chunk.chunk_id == "lm_problems_ch02_p001":
            return [
                {
                    "domain": "MECHANICS",
                    "derivation_steps": 3,
                    "problem_statement": "A solid sphere of mass 2.0 kg and radius 0.15 m rolls without slipping down an incline of height 1.4 m. Find its translational speed at the bottom.",
                    "reference_solution": "By conservation of energy, mgh = 7/10 mv^2, so v = sqrt(10gh/7) = 4.43 m/s.",
                }
            ]
        return []

    accepted, rejected, _ = extract_literature_problems(paired_chunks, chunk_extractor=fake_extractor)
    assert len(accepted) == 1
    assert accepted[0].original_id == "lm_problems:ch02_p001"
    assert set(accepted[0].to_dict().keys()) == {
        "problem_id",
        "domain",
        "problem_statement",
        "reference_solution",
        "source",
        "original_id",
    }
    assert rejected == []


def test_stage_1_2_filters_reject_diagrams_and_multipart():
    assert is_diagram_dependent("Calculate the tension as shown in Figure 3.2.") is True
    assert is_diagram_dependent("A solid sphere rolls without slipping down an incline.") is False

    assert has_unsplit_multipart("Find (a) the acceleration of the block and (b) the tension in the cord.") is True
    assert has_unsplit_multipart("Find the translational speed at the bottom.") is False

    valid_candidate = {
        "domain": "MECHANICS",
        "derivation_steps": 3,
        "problem_statement": "A solid sphere of mass 2.0 kg and radius 0.15 m rolls without slipping down an incline of height 1.4 m. Find its translational speed at the bottom.",
        "reference_solution": "By conservation of mechanical energy, mgh = 1/2 mv^2 + 1/2 I*omega^2. Since I = 2/5 mr^2 and omega = v/r, mgh = 7/10 mv^2. Solving for v gives v = sqrt(10gh/7) = 4.43 m/s.",
    }
    problem, reason = validate_and_build_canonical_problem(
        valid_candidate,
        problem_id="lit_mech_0142",
        book_id="crowell_mechanics",
        chunk_id="chunk_0001",
    )
    assert problem is not None
    assert reason is None
    assert problem.source == "literature_curated"

    diagram_candidate = {
        **valid_candidate,
        "problem_statement": "Find the tension as shown in Figure 4.1.",
    }
    bad_problem, bad_reason = validate_and_build_canonical_problem(
        diagram_candidate,
        problem_id="lit_mech_0143",
        book_id="crowell_mechanics",
        chunk_id="chunk_0001",
    )
    assert bad_problem is None
    assert bad_reason == "diagram_dependent"


def test_extract_book_chapter_titles_and_slugify():
    pages = [
        (
            5,
            "Brief Contents\n0 Introduction and review 11\n1 Scaling and estimation 37\n"
            "Motion in one dimension\n2 Velocity and relative motion 59\n"
            "6 Newton’s laws in\nthree dimensions 191\n",
        )
    ]
    titles = extract_book_chapter_titles(pages)
    assert titles[0] == "Introduction and review"
    assert titles[1] == "Scaling and estimation"
    assert titles[2] == "Velocity and relative motion"
    assert titles[6] == "Newton’s laws in three dimensions"
    assert f"06_{slugify_chapter_title(titles[6])}.pdf" == "06_newtons_laws_in_three_dimensions.pdf"


def test_manual_chapter_domain_map_and_initial_book_dataset():
    # Every manually classified chapter must map to one of the 6 canonical domains or SIDE
    all_assigned = {
        domain
        for book_map in MANUAL_CHAPTER_DOMAIN_MAP.values()
        for domain in book_map.values()
    }
    assert all_assigned == (VALID_CANONICAL_DOMAINS | {SIDE_DOMAIN})

    # Verify non-physics / general-math chapters are assigned to SIDE
    assert MANUAL_CHAPTER_DOMAIN_MAP["crowell_mechanics"]["00_introduction_and_review"] == SIDE_DOMAIN
    assert MANUAL_CHAPTER_DOMAIN_MAP["crowell_mechanics"]["01_scaling_and_estimation"] == SIDE_DOMAIN
    assert MANUAL_CHAPTER_DOMAIN_MAP["crowell_mechanics"]["07_vectors"] == SIDE_DOMAIN
    assert MANUAL_CHAPTER_DOMAIN_MAP["crowell_fields_and_circuits"]["10_iterated_integrals"] == SIDE_DOMAIN
    assert MANUAL_CHAPTER_DOMAIN_MAP["crowell_fields_and_circuits"]["15_stokess_theorem"] == SIDE_DOMAIN
    assert MANUAL_CHAPTER_DOMAIN_MAP["light_and_matter_problems"]["33_rules_of_randomness"] == SIDE_DOMAIN

    # Verify physics chapters map to their respective canonical domain
    assert MANUAL_CHAPTER_DOMAIN_MAP["crowell_mechanics"]["08_vectors_and_motion"] == "MECHANICS"
    assert MANUAL_CHAPTER_DOMAIN_MAP["crowell_fields_and_circuits"]["05_electromagnetism"] == "ELECTROMAGNETISM"
    assert MANUAL_CHAPTER_DOMAIN_MAP["crowell_modern_physics"]["01_time"] == "RELATIVITY"
    assert MANUAL_CHAPTER_DOMAIN_MAP["crowell_modern_physics"]["15_light_as_a_particle"] == "MODERN_PHYSICS"
    assert MANUAL_CHAPTER_DOMAIN_MAP["crowell_modern_physics"]["07_statistics_and_the_ideal_gas"] == "THERMODYNAMICS"
    assert MANUAL_CHAPTER_DOMAIN_MAP["crowell_conceptual_physics"]["07_the_ray_model_of_light"] == "OPTICS"


