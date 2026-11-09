"""Unit tests for Stage 1.3 & 1.4 anti-leakage deduplication, acceptance gates, independent subproblem splitting, and canonical_train_250 construction."""
from __future__ import annotations

from pathlib import Path

from adversarial_physics.data.curation.curatorkit_stage1 import (
    run_curatorkit_stage1_gates_and_recovery,
    write_curatorkit_provenance_bundle,
)
from adversarial_physics.data.processing.canonical_pool_builder import (
    build_aya_candidate_validator,
    build_stage1_canonical_pool,
    clean_book_problem_text,
    clean_book_solution_text,
    estimate_logical_steps,
    evaluate_self_containment_and_acceptance,
    extract_final_answer_from_solution,
    select_balanced_canonical_pool,
    split_independent_subproblems,
)
from adversarial_physics.data.processing.deduplication import (
    deduplicate_candidate_pool,
    run_anti_leakage_gate,
)
from adversarial_physics.data.schemas.problem import CanonicalProblem, save_problems_jsonl


def test_rejects_book_modern_0986_and_external_references():
    p_0986 = CanonicalProblem(
        problem_id="book_modern_0986",
        domain="MODERN_PHYSICS",
        problem_statement=(
            "Hydrogen is the only element whose energy levels can be expressed exactly in an equation. "
            "Calculate the ratio λE/λF of the wavelengths of the transitions labeled E and F in problem 2 on p. 1028. "
            "Express your answer as an exact fraction, not a decimal approximation."
        ),
        reference_solution=(
            "All of the constants cancel out, so we only need the n-dependent part of the expressions. "
            "The result is λE/λF = EF/EE = 2^-2 - 5^-2 / 2^-2 - 4^-2 = 2825"
        ),
        source="book_initial",
        original_id="light_and_matter_problems:36_the_atom:p011",
    )
    ok, reason, _ = evaluate_self_containment_and_acceptance(p_0986)
    assert ok is False
    assert reason in ("external_problem_or_page_dependency", "corrupted_unrecovered_math_fraction")


def test_independent_subproblem_splitting_keeps_part_a_and_drops_dependent_part_c():
    multi_prob = CanonicalProblem(
        problem_id="book_modern_0974",
        domain="MODERN_PHYSICS",
        problem_statement=(
            "A free electron that contributes to the current in an ohmic material typically has a speed of 10^5 m/s.\n"
            "(a) Estimate its de Broglie wavelength, in nm. √\n"
            "(b) If a computer memory chip contains 10^8 electric circuits in a 1 cm^2 area, estimate the linear size, in nm, of one such circuit. √\n"
            "(c) Based on your answers from parts a and b, does an electrical engineer need to worry about diffraction?"
        ),
        reference_solution=(
            "(a) λ = h/p = h/mv = 10^-8 m = 10 nm.\n"
            "(b) The area of one circuit would be 10^-8 cm^2, which is a square 1000 nm on a side.\n"
            "(c) No, because the circuit is much larger than the wavelength."
        ),
        source="book_initial",
        original_id="light_and_matter_problems:35_matter_as_a_wave:p003",
    )
    split_items = split_independent_subproblems(multi_prob)
    # Parts (a) and (b) are independently solvable; part (c) explicitly depends on "your answers from parts a and b"
    assert len(split_items) == 2
    assert split_items[0].original_id == "light_and_matter_problems:35_matter_as_a_wave:p003:part_a"
    assert split_items[1].original_id == "light_and_matter_problems:35_matter_as_a_wave:p003:part_b"
    assert "ANSWER:" in split_items[0].reference_solution
    assert "ANSWER:" in split_items[1].reference_solution


def test_anti_leakage_gate_purges_val_and_test_matches():
    val_prob = CanonicalProblem(
        problem_id="physgym_0039",
        domain="MECHANICS",
        problem_statement=(
            "A rectangular wooden block of height H and density rho_1 is gently placed on the water surface "
            "of density rho_2. Gently press down on the wooden block and release it. Determine the period of motion T."
        ),
        reference_solution="T = 2*pi*sqrt(rho_1 * H / (rho_2 * g))\nANSWER: 2*pi*sqrt(rho_1*H/(rho_2*g))",
        source="physgym",
        original_id=39,
    )
    leaked_candidate = CanonicalProblem(
        problem_id="book_mech_0001",
        domain="MECHANICS",
        problem_statement=(
            "A rectangular wooden block of height H and density rho_1 is placed gently on a water surface "
            "of density rho_2. After pressing down on the wooden block and releasing it, find the period of motion T."
        ),
        reference_solution="T = 2*pi*sqrt(rho_1 * H / (rho_2 * g))\nANSWER: 2*pi*sqrt(rho_1*H/(rho_2*g))",
        source="book_initial",
        original_id="crowell_mechanics:16_vibrations:p001",
    )
    clean_candidate = CanonicalProblem(
        problem_id="book_mech_0002",
        domain="MECHANICS",
        problem_statement=(
            "A solid sphere of mass 2.0 kg and radius 0.15 m rolls without slipping down an incline of height 1.4 m. "
            "Find its translational speed at the bottom."
        ),
        reference_solution="By energy conservation, mgh = 7/10 mv^2, so v = sqrt(10gh/7) = 4.43 m/s.\nANSWER: 4.43 m/s",
        source="book_initial",
        original_id="crowell_mechanics:15_conservation_of_angular_momentum:p002",
    )

    accepted, leaked = run_anti_leakage_gate(
        [leaked_candidate, clean_candidate],
        val_problems=[val_prob],
        test_problems=[],
        cosine_threshold=0.85,
    )
    assert len(accepted) == 1
    assert accepted[0].problem_id == "book_mech_0002"
    assert len(leaked) == 1
    assert leaked[0].candidate_id == "book_mech_0001"
    assert leaked[0].matched_problem_id == "physgym_0039"


def test_curatorkit_stage1_gates_adaptive_recovery_and_provenance(tmp_path: Path):
    missing_sentinel_prob = CanonicalProblem(
        problem_id="lit_mech_0001",
        domain="MECHANICS",
        problem_statement="A 2.0 kg block accelerates at 3.0 m/s^2. Find the net force.",
        reference_solution="By Newton's second law, F = m * a = 2.0 * 3.0 = 6.0 N.",
        source="literature_curated",
        original_id="crowell_mechanics:04_force_and_motion:p001",
    )

    def repair_fn(prob: CanonicalProblem, _diag: str) -> CanonicalProblem:
        return CanonicalProblem(
            problem_id=prob.problem_id,
            domain=prob.domain,
            problem_statement=prob.problem_statement,
            reference_solution=clean_book_solution_text(prob.reference_solution, ensure_answer_sentinel=True),
            source=prob.source,
            original_id=prob.original_id,
        )

    accepted, rejected, report = run_curatorkit_stage1_gates_and_recovery(
        [missing_sentinel_prob],
        require_answer_sentinel=True,
        repair_fn=repair_fn,
    )
    assert len(accepted) == 1
    assert rejected == []
    assert report.adaptive_recovery_attempted == 1
    assert report.adaptive_recovery_succeeded == 1
    assert "ANSWER: 6.0 N" in accepted[0].reference_solution

    bundle = write_curatorkit_provenance_bundle(
        tmp_path / "processed",
        stage_id="02_canonical_train_250",
        accepted_problems=accepted,
        rejected_records=rejected,
        gate_report=report,
        summary_dict={"counts_by_domain": {"MECHANICS": 1}},
        tracked_files=[],
    )
    assert bundle["manifest"].is_file()
    assert bundle["rejected"].is_file()
    assert bundle["lexsi_provenance"].is_file()
    assert bundle["checksums"].is_file()


def test_select_balanced_canonical_pool_backfills_when_llm_rejects_candidate():
    # Suppose we have 3 valid candidates in MECHANICS and target_total = 2,
    # and the LLM validator rejects candidate #1. It should automatically backfill with candidate #3!
    candidates = [
        CanonicalProblem(
            problem_id=f"book_mech_{i:04d}",
            domain="MECHANICS",
            problem_statement=f"A block of mass {i} kg slides down a frictionless ramp of height 2.0 m. Find its speed.",
            reference_solution="v = sqrt(2*g*h) = 6.26 m/s\nANSWER: 6.26 m/s",
            source="book_initial",
            original_id=f"crowell_mechanics:11_conservation_of_energy:p00{i}",
        )
        for i in (1, 2, 3)
    ]

    def fake_llm_validator(prob: CanonicalProblem):
        if str(prob.original_id).endswith("p001"):
            return None, "llm_rejected:missing_context"
        return prob, None

    selected, llm_rejected = select_balanced_canonical_pool(
        accepted_physgym_seeds=[],
        accepted_book_candidates=candidates,
        target_total=2,
        domain_weights={"MECHANICS": 1.0},
        candidate_validator=fake_llm_validator,
    )
    assert len(selected) == 2
    assert [p.original_id for p in selected] == [
        "crowell_mechanics:11_conservation_of_energy:p002",
        "crowell_mechanics:11_conservation_of_energy:p003",
    ]
    assert len(llm_rejected) == 1
    assert llm_rejected[0].original_id == "crowell_mechanics:11_conservation_of_energy:p001"


def test_rescues_false_llm_rejection_and_preserves_decimal_answers(tmp_path: Path):
    assert extract_final_answer_from_solution("By/Bx = tan 59 deg = 1.7") == "1.7"
    assert (
        extract_final_answer_from_solution("j = 1.00e3 / area = 1.82× 106 A/m2.")
        == "1.82× 106 A/m2"
    )

    seed_prob = CanonicalProblem(
        problem_id="physgym_0043",
        domain="ELECTROMAGNETISM",
        problem_statement="Two dielectric materials with base areas A have capacitances C1 and C2 in series. Find C.",
        reference_solution="Using series capacitance formula:\nC = C1*C2/(C1+C2)\nANSWER: C1*C2/(C1+C2)",
        source="physgym_seed",
        original_id=43,
    )

    def fake_chat_false_reject(_sys: str, _usr: str) -> str:
        return (
            "[[REJECT: The provided solution is complete and follows a logical step-by-step derivation. "
            "No further simplification or rejection is required.]]"
        )

    validator = build_aya_candidate_validator(
        checkpoint_path=tmp_path / "ckpt.jsonl",
        chat_fn=fake_chat_false_reject,
    )
    validated, reason = validator(seed_prob)
    assert validated is not None
    assert reason is None
    assert "[[REJECT" not in validated.reference_solution
    assert "ANSWER: C1*C2/(C1+C2)" in validated.reference_solution


def test_hf_gpu_backend_resolution_and_timing():
    from adversarial_physics.utils.runtime import RECOMMENDED_HF_GPU_FLAVOR, detect_runtime_backend

    cfg = detect_runtime_backend("hf_gpu")
    assert cfg.backend == "hf_gpu"
    assert cfg.is_hf_gpu is True
    assert cfg.recommended_hf_hardware == RECOMMENDED_HF_GPU_FLAVOR
    if not cfg.cuda_available:
        assert cfg.execution_mode == "hf_remote_api"


def test_uncapped_selection_retains_all_valid_candidates_over_250():
    # Verify that when target_total=None (default), all >250 valid candidates are retained
    seeds = [
        CanonicalProblem(
            problem_id=f"physgym_{i:04d}",
            domain="MECHANICS",
            problem_statement=f"Seed problem {i}: A mass of {i} kg moves at 2 m/s. Find kinetic energy.",
            reference_solution=f"K = 0.5 * {i} * 4 = {2*i} J\nANSWER: {2*i} J",
            source="physgym_seed",
            original_id=i,
        )
        for i in range(1, 13)
    ]
    book_cands = [
        CanonicalProblem(
            problem_id=f"book_mech_{i:04d}",
            domain="MECHANICS",
            problem_statement=f"Book problem {i}: A particle of mass {i} kg accelerates at 3 m/s^2. Find force.",
            reference_solution=f"F = m*a = {3*i} N\nANSWER: {3*i} N",
            source="book_initial",
            original_id=f"crowell_mechanics:04_force_and_motion:p{i:04d}",
        )
        for i in range(1, 301)
    ]
    selected, rejected = select_balanced_canonical_pool(
        accepted_physgym_seeds=seeds,
        accepted_book_candidates=book_cands,
        target_total=None,
    )
    assert len(selected) == 312
    assert rejected == []





