"""Unit tests for subtopic extraction/mapping, strict CuratorKIT Part 1 & 2, synthetic E2 generation, and composite checkpointing."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from adversarial_physics.data.curation.curatorkit_stage1 import (
    PART1_STRICT_CONFIG,
    PART2_STRICT_CONFIG,
    check_strict_physics_hygiene,
    run_curatorkit_stage1_gates_and_recovery,
)
from adversarial_physics.data.generation.synthetic_problem_generator import (
    build_seed_style_prompt,
    parse_tagged_generation,
    verify_sympy_consistency,
)
from adversarial_physics.data.ingestion.subtopic_extractor import (
    SubtopicChunk,
    SubtopicMapper,
    deduplicate_subtopic_chunks_within_domain,
    parse_detailed_contents,
    strip_visual_paragraphs,
)
from adversarial_physics.data.schemas.problem import CanonicalProblem
from adversarial_physics.training.run_aligntune import (
    compute_composite_checkpoint_score,
    evaluate_task_accuracy,
    select_best_epoch_checkpoint,
)


def test_parse_contents_and_strip_visual_paragraphs() -> None:
    pages = [
        (
            6,
            "Contents\n"
            "15 Conservation of Angular Momentum\n"
            "15.4 Torque: the rate of transfer of angular momentum . . . 442\n"
            "15.5 Statics . . . 449\n",
        ),
        (11, "Chapter 0\nIntroduction\n"),
    ]
    toc = parse_detailed_contents(pages)
    assert toc["15.4"] == ("Torque: the rate of transfer of angular momentum", 442)
    assert toc["15.5"] == ("Statics", 449)

    prose = (
        "Torque is defined as the cross product of displacement r and applied force F, giving tau = r * F * sin(theta). "
        "This quantity governs rotational acceleration.\n\n"
        "As shown in figure 14 at right, the crane arm rotates clockwise when the cable snaps."
    )
    cleaned = strip_visual_paragraphs(prose, min_chars=40)
    assert "Torque is defined" in cleaned
    assert "figure 14" not in cleaned


def test_dedup_chunks_and_chapter_scoped_subtopic_mapper() -> None:
    c1 = SubtopicChunk(
        chunk_id="crowell_mechanics:15_conservation_of_angular_momentum:15.4:a",
        book_id="crowell_mechanics",
        chapter=15,
        chapter_slug="15_conservation_of_angular_momentum",
        domain="MECHANICS",
        section_id="15.4",
        section_title="Torque",
        text=(
            "Torque is the rotational analog of force, defined by tau = r * F * sin(theta) where r is the lever arm "
            "and theta is the angle between displacement and force vectors."
        ),
        token_count=30,
        start_page=442,
        end_page=444,
    )
    c1_dup = SubtopicChunk(
        chunk_id="crowell_conceptual_physics:03_conservation_of_angular_momentum:3.2:b",
        book_id="crowell_conceptual_physics",
        chapter=3,
        chapter_slug="03_conservation_of_angular_momentum",
        domain="MECHANICS",
        section_id="3.2",
        section_title="Torque",
        text=(
            "Torque is the rotational analog of force, defined by tau = r * F * sin(theta) where r is the lever arm "
            "and theta is the angle between displacement and force vectors."
        ),
        token_count=30,
        start_page=88,
        end_page=89,
    )
    c2 = SubtopicChunk(
        chunk_id="crowell_mechanics:15_conservation_of_angular_momentum:15.5:c",
        book_id="crowell_mechanics",
        chapter=15,
        chapter_slug="15_conservation_of_angular_momentum",
        domain="MECHANICS",
        section_id="15.5",
        section_title="Statics",
        text="In static equilibrium both the net force vector and net torque vector vanish simultaneously.",
        token_count=15,
        start_page=449,
        end_page=450,
    )

    deduped, removed = deduplicate_subtopic_chunks_within_domain([c1, c1_dup, c2], minhash_threshold=0.75)
    assert removed == 1
    assert len(deduped) == 2

    class _FakeRetriever:
        def __init__(self, texts: list[str], model: str = "") -> None:
            del model
            self.texts = texts
            self.index = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        class model:
            @staticmethod
            def encode(queries: list[str], normalize_embeddings: bool = True) -> np.ndarray:
                del normalize_embeddings
                return np.array([[1.0, 0.0] if "torque" in queries[0].lower() else [0.0, 1.0]], dtype=np.float32)

    mapper = SubtopicMapper(deduped)
    prob = CanonicalProblem(
        problem_id="lit_mech_0173",
        domain="MECHANICS",
        problem_statement="A 50 cm wrench loosens a bolt. What is the maximum torque exerted by 55 kg?",
        reference_solution="tau = r * m * g = 270 N*m\nANSWER: 270 N*m",
        source="literature_curated",
        original_id="crowell_mechanics:15_conservation_of_angular_momentum:p004",
    )
    with patch("curatorkit.diagnostic.retriever.PostHocRetriever", _FakeRetriever):
        hits = mapper.find_subtopics(prob, top_k=1)
    assert len(hits) == 1
    assert hits[0].section_id == "15.4"


def test_parse_tagged_generation_and_sympy_oracle() -> None:
    anchor = CanonicalProblem(
        problem_id="lit_mech_0173",
        domain="MECHANICS",
        problem_statement="You hang 55 kg on a 50 cm wrench. Find max torque.",
        reference_solution="tau = r m g = 270 N*m\nANSWER: 270 N*m",
        source="literature_curated",
        original_id="crowell_mechanics:15_conservation_of_angular_momentum:p004",
    )
    seed = CanonicalProblem(
        problem_id="physgym_0039",
        domain="MECHANICS",
        problem_statement="A wooden block of height $H$ and density $\\rho_1$ floats in water $\\rho_2$. Find period $T$.",
        reference_solution="$$T = 2\\pi\\sqrt{\\frac{\\rho_1 H}{\\rho_2 g}}$$",
        source="physgym",
        original_id=39,
    )
    prompt = build_seed_style_prompt(anchor, [], seed)
    assert "===PROBLEM===" in prompt
    assert "CHECK_VALUES:" in prompt

    good_output = (
        "===PROBLEM===\n"
        "A uniform rigid wrench of length $L$ is attached to a bolt at origin $O$. A mechanic of mass $m$ hangs "
        "vertically from the free end in uniform gravitational acceleration $g$. Determine the maximum torque $\\tau_{\\max}$.\n"
        "===SOLUTION===\n"
        "The lever arm has length $r = L$ and the downward gravitational force has magnitude $F = mg$. Maximum torque occurs when $\\theta = \\pi/2$.\n"
        "Thus $\\tau_{\\max} = L m g$.\n"
        "ANSWER: L m g\n"
        "CHECK_VALUES: L=0.50, m=55, g=9.81"
    )
    parsed = parse_tagged_generation(good_output)
    assert parsed is not None
    assert parsed.answer_value == "L m g"
    assert parsed.check_values == {"L": 0.50, "m": 55.0, "g": 9.81}

    ok_sym, reason_sym = verify_sympy_consistency(parsed, anchor)
    assert ok_sym is True, reason_sym

    # Wrong formula must fail SymPy substitution check against Crowell 270 N*m anchor
    bad_output = good_output.replace("ANSWER: L m g", "ANSWER: L / (m * g)")
    parsed_bad = parse_tagged_generation(bad_output)
    assert parsed_bad is not None
    ok_bad, _ = verify_sympy_consistency(parsed_bad, anchor)
    assert ok_bad is False


def test_strict_curatorkit_hygiene_and_fail_closed_judges() -> None:
    prose_only = CanonicalProblem(
        problem_id="p_prose",
        domain="MECHANICS",
        problem_statement="Explain qualitatively why heavier carts require noticeably greater horizontal pushing effort to accelerate along a smooth level floor.",
        reference_solution=(
            "Because inertial mass measures resistance to acceleration according to Newtonian dynamics, "
            "a heavier cart with greater mass requires a larger push to achieve the same speed change.\n"
            "ANSWER: inertia increases"
        ),
        source="book_initial",
        original_id="x:1",
    )
    ok_p, r_p = check_strict_physics_hygiene(prose_only, PART1_STRICT_CONFIG)
    assert ok_p is False and r_p == "non_quantitative_prose_only"

    multi_ans = CanonicalProblem(
        problem_id="p_multi",
        domain="MECHANICS",
        problem_statement="Find velocity v after t = 2 s starting from rest at constant acceleration a = 3 m/s^2 along a line.",
        reference_solution="Step 1: v = a*t = 3 * 2 = 6 m/s. Wait, maybe check again step by step.\nANSWER: 5 m/s\nActually:\nANSWER: 6 m/s",
        source="book_initial",
        original_id="x:2",
    )
    ok_m, r_m = check_strict_physics_hygiene(multi_ans, PART1_STRICT_CONFIG)
    assert ok_m is False and "multiple_answer_sentinels" in r_m

    valid_prob = CanonicalProblem(
        problem_id="p_valid",
        domain="MECHANICS",
        problem_statement=(
            "A block of mass $m = 4\\,\\text{kg}$ accelerates from rest under a constant horizontal force $F = 12\\,\\text{N}$ "
            "on a frictionless table for time $t = 5\\,\\text{s}$. Determine the final speed $v$ of the block."
        ),
        reference_solution=(
            "By Newton's second law along the horizontal direction, the net force equals $m a$, so the constant "
            "acceleration of the block is $a = F / m = 12 / 4 = 3\\,\\text{m/s}^2$. Since the block starts from rest "
            "with initial velocity $v_0 = 0$, constant-acceleration kinematics gives $v = v_0 + a t = 0 + 3 \\times 5 = 15\\,\\text{m/s}$.\n"
            "ANSWER: 15 m/s"
        ),
        source="synthetic_seed_subtopic",
        original_id="x:3",
    )
    ok_v, r_v = check_strict_physics_hygiene(valid_prob, PART1_STRICT_CONFIG)
    assert ok_v is True, r_v

    # Part 2 judge returns invalid garbage -> on_judge_error="fail" must REJECT instead of passing!
    passed, rejected, report = run_curatorkit_stage1_gates_and_recovery(
        [valid_prob],
        strict_config=PART2_STRICT_CONFIG,
        chat_fn=lambda _sys, _usr: "INVALID UNPARSEABLE JUDGE RESPONSE",
    )
    assert len(passed) == 0
    assert report.hallucination_gate_failed + report.reward_gate_failed >= 1
    assert len(rejected) >= 1


def test_task_accuracy_physgym_and_composite_checkpoint_selection(tmp_path: Path) -> None:
    # Verify evaluate_task_accuracy handles raw PhysGym $$...$$ / \boxed{} references
    acc_res = evaluate_task_accuracy(
        prediction="Using SHM kinematics...\nANSWER: 2*pi*sqrt(rho_1*H/(rho_2*g))",
        reference="Period is given by\n$$\nT = 2\\pi\\sqrt{\\frac{\\rho_1 H}{\\rho_2 g}}\n$$",
        is_verifier=False,
    )
    assert acc_res["final_answer_accuracy"] == 1.0

    # Verify composite checkpoint selection chooses checkpoint-20 (higher val_accuracy outweighs tiny eval_loss bump)
    out_dir = tmp_path / "solver_e2_lora"
    ckpt10 = out_dir / "checkpoint-10"
    ckpt20 = out_dir / "checkpoint-20"
    ckpt10.mkdir(parents=True)
    ckpt20.mkdir(parents=True)
    (ckpt10 / "adapter_model.safetensors").write_text("epoch1", encoding="utf-8")
    (ckpt20 / "adapter_model.safetensors").write_text("epoch2_winner", encoding="utf-8")

    s1 = compute_composite_checkpoint_score(0.95, 0.10, alpha_loss=1.0, k_accuracy=1.5)  # 0.80
    s2 = compute_composite_checkpoint_score(0.97, 0.25, alpha_loss=1.0, k_accuracy=1.5)  # 0.595 (better)
    assert s2 < s1

    report = select_best_epoch_checkpoint(
        out_dir,
        alpha_loss=1.0,
        k_accuracy=1.5,
        precomputed_results=[
            {"checkpoint_dir": str(ckpt10), "step": 10, "eval_loss": 0.95, "val_accuracy": 0.10},
            {"checkpoint_dir": str(ckpt20), "step": 20, "eval_loss": 0.97, "val_accuracy": 0.25},
        ],
    )
    assert report is not None
    assert Path(report["winner"]["checkpoint_dir"]).name == "checkpoint-20"
    assert (out_dir / "final_adapter" / "adapter_model.safetensors").read_text(encoding="utf-8") == "epoch2_winner"
    saved = json.loads((out_dir / "checkpoint_selection_summary.json").read_text(encoding="utf-8"))
    assert saved["winner"]["step"] == 20


def test_prepare_e2_llm_generation_requests_zero_llm(tmp_path: Path) -> None:
    from adversarial_physics.data.generation.synthetic_problem_generator import (
        prepare_e2_llm_generation_requests,
    )
    from adversarial_physics.data.schemas.problem import save_problems_jsonl

    books_path = tmp_path / "clean_books.jsonl"
    seeds_path = tmp_path / "train_seeds.jsonl"
    subtopics_path = tmp_path / "subtopics.jsonl"
    prompts_out = tmp_path / "ready_prompts.jsonl"
    summary_out = tmp_path / "ready_summary.json"

    save_problems_jsonl(
        [
            CanonicalProblem(
                problem_id="lit_mech_01",
                domain="MECHANICS",
                problem_statement=(
                    "A block of mass $m = 4\\,\\text{kg}$ accelerates from rest under a constant horizontal force $F = 12\\,\\text{N}$ "
                    "on a smooth horizontal table for $t = 5\\,\\text{s}$. Determine the final speed $v$."
                ),
                reference_solution=(
                    "By Newton's second law $a = F / m = 12 / 4 = 3\\,\\text{m/s}^2$, and from constant-acceleration kinematics "
                    "$v = a t = 3 \\times 5 = 15\\,\\text{m/s}$.\nANSWER: 15 m/s"
                ),
                source="literature_curated",
                original_id="crowell_mechanics:01_motion:p001",
            )
        ],
        books_path,
    )
    save_problems_jsonl(
        [
            CanonicalProblem(
                problem_id="physgym_0001",
                domain="MECHANICS",
                problem_statement="A particle of mass $M$ moves under constant force $F_0$ for time $T$. Find $v(T)$.",
                reference_solution="$$v(T) = \\frac{F_0 T}{M}$$",
                source="physgym",
                original_id=1,
            )
        ],
        seeds_path,
    )
    subtopic = SubtopicChunk(
        chunk_id="crowell_mechanics:01_motion:1.2:a",
        book_id="crowell_mechanics",
        chapter=1,
        chapter_slug="01_motion",
        domain="MECHANICS",
        section_id="1.2",
        section_title="Newton's Second Law",
        text="Acceleration is proportional to net horizontal force divided by inertial mass, $a = F / m$, giving linear velocity growth $v = v_0 + a t$.",
        token_count=24,
        start_page=10,
        end_page=12,
    )
    subtopics_path.write_text(json.dumps(subtopic.to_dict()) + "\n", encoding="utf-8")

    records, summary = prepare_e2_llm_generation_requests(
        clean_books_path=books_path,
        subtopic_chunks_path=subtopics_path,
        train_seeds_path=seeds_path,
        output_path=prompts_out,
        summary_out=summary_out,
    )
    assert summary["ready_prompt_count"] == 1
    assert summary["subtopic_matched_count"] == 1
    assert summary["by_domain"] == {"MECHANICS": 1}
    assert len(records) == 1
    assert records[0]["matched_seed_id"] == "physgym_0001"
    assert records[0]["matched_subtopic"]["section_id"] == "1.2"
    assert "===PROBLEM===" in records[0]["user_prompt"]
    assert prompts_out.is_file()
    assert summary_out.is_file()
