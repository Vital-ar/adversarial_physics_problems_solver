"""Regression tests for all 10 approved v2 pipeline fixes & Lexsi framework integrations."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from adversarial_physics.data.curation.answer_validator import (
    _is_purely_numeric_answer,
    validate_candidate_answer,
)
from adversarial_physics.data.curation.curatorkit_runner import (
    curate_and_deduplicate_stage2_trajectories,
)
from adversarial_physics.data.curation.curatorkit_stage1 import (
    PART2_STRICT_CONFIG,
    problem_to_datasample,
    run_curatorkit_stage1_gates_and_recovery,
)
from adversarial_physics.data.curation.directive_leak_check import (
    sanitize_directive_text,
    validate_verifier_directive,
)
from adversarial_physics.data.generation.candidate_generator import (
    generate_controlled_error_trajectories,
    generate_minority_alternative_solutions,
)
from adversarial_physics.data.generation.synthetic_problem_generator import (
    generate_synthetic_e2_and_e3_pools,
    parse_tagged_generation,
)
from adversarial_physics.data.ingestion.subtopic_extractor import (
    SubtopicChunk,
    SubtopicMapper,
    strip_visual_paragraphs,
)
from adversarial_physics.data.schemas.problem import CanonicalProblem, save_problems_jsonl
from adversarial_physics.data.schemas.trajectory import (
    VERDICT_CHALLENGE,
    CandidateSolution,
    format_solver_prompt,
    save_candidates_jsonl,
)
from adversarial_physics.evaluation import run_physics_eval_audit
from adversarial_physics.training.run_aligntune import (
    batched_evaluate_solver_split,
    select_best_epoch_checkpoint,
)


def test_answer_validator_no_substring_match_and_trig_sympy() -> None:
    # Fix 3a: Numeric substring "20" inside "200" must NOT pass!
    res_bad = validate_candidate_answer("Final result:\nANSWER: 20 m/s", "Compute.\nANSWER: 200 m/s")
    assert res_bad.is_equivalent is False

    # Fix 3b: Symbolic trig/log expressions must preserve function names and validate via SymPy
    assert _is_purely_numeric_answer(r"2 \sin(\theta)") is False
    res_trig = validate_candidate_answer(
        r"ANSWER: v_0 * sin(theta)",
        r"ANSWER: v_0 \sin\theta",
    )
    assert res_trig.is_equivalent is True
    assert res_trig.match_method in ("sympy_symbolic", "aligntune_sympy")

    res_ln = validate_candidate_answer(r"ANSWER: \ln(2)", r"ANSWER: log(2)")
    assert res_ln.is_equivalent is True


def test_directive_leak_scientific_notation_and_short_symbolic() -> None:
    # Fix 10a: Scientific notation / comma leak of 1200 -> caught and redacted
    prob = "Find the work done in joules if mass moves up the ramp."
    ref = "Compute work.\nANSWER: 1200 J"
    raw_dir = "Check conservation of mechanical energy so you obtain 1.2e3 J."
    val_res = validate_verifier_directive(
        f"{raw_dir}\n{VERDICT_CHALLENGE}",
        problem_statement=prob,
        reference_solution=ref,
        expected_verdict=VERDICT_CHALLENGE,
    )
    assert val_res.is_valid is False
    assert val_res.leaks_final_answer is True

    san = sanitize_directive_text(
        raw_dir,
        problem_statement=prob,
        reference_solution=ref,
        expected_verdict=VERDICT_CHALLENGE,
    )
    assert "1.2e3" not in san
    assert "the target value" in san

    # Fix 10b: Short symbolic answer leak (< 12 normalized chars, e.g., `m v^2 / r`) is caught
    prob_sym = "A particle of mass m moves in a circle of radius r at speed v. Find centripetal force."
    ref_sym = "By circular motion dynamics.\nANSWER: m v^2 / r"
    leak_sym = validate_verifier_directive(
        f"Use radial dynamics giving m*v^2/r directly.\n{VERDICT_CHALLENGE}",
        problem_statement=prob_sym,
        reference_solution=ref_sym,
        expected_verdict=VERDICT_CHALLENGE,
    )
    assert leak_sym.is_valid is False
    assert leak_sym.leaks_final_answer is True


def test_problem_to_datasample_no_self_grounding_in_hallucination_gate() -> None:
    # Fix 4: `input` must NOT equal `reference_solution` when no source_chunk exists
    prob = CanonicalProblem(
        problem_id="syn_hal_01",
        domain="MECHANICS",
        problem_statement=(
            "A block of mass $m = 4\\,\\text{kg}$ accelerates from rest under a constant horizontal force "
            "$F = 12\\,\\text{N}$ on a smooth floor for $t = 5\\,\\text{s}$. Find the speed $v$."
        ),
        reference_solution=(
            "Step 1: $a = F / m = 12 / 4 = 3\\,\\text{m/s}^2$. Step 2: $v = a t = 15\\,\\text{m/s}$.\n"
            "ANSWER: 15 m/s"
        ),
        source="synthetic_seed_subtopic",
        original_id="crowell_mechanics:04_force_and_motion:p001",
    )
    s_no_ctx = problem_to_datasample(prob)
    assert s_no_ctx.input == ""
    assert "source_chunk" not in s_no_ctx.metadata

    s_with_ctx = problem_to_datasample(prob, source_chunk="Textbook says $F = m a$.")
    assert s_with_ctx.input == "Textbook says $F = m a$."
    assert s_with_ctx.metadata["source_chunk"] == "Textbook says $F = m a$."

    # Verify HallucinationGate receives real external source context and rejects low grounding score
    seen_prompts: list[str] = []

    def _judge(_sys: str, usr: str) -> str:
        seen_prompts.append(usr)
        if "grounding_score" in usr:
            return '{"grounding_score": 0.25, "verdict": "hallucinated"}'
        return '{"overall_score": 0.95}'

    passed, rejected, report = run_curatorkit_stage1_gates_and_recovery(
        [prob],
        strict_config=PART2_STRICT_CONFIG,
        source_contexts={prob.problem_id: "Conservation of linear momentum states p = m v."},
        chat_fn=_judge,
    )
    assert len(passed) == 0
    assert report.hallucination_gate_failed == 1
    assert any(r["gate"] == "HallucinationGate" for r in rejected)
    assert any("Conservation of linear momentum states p = m v." in p for p in seen_prompts)


def test_parse_tagged_generation_non_greedy_and_e2_seed_retention(tmp_path: Path) -> None:
    # Fix 5: Body text containing lowercase "answer:" earlier must NOT be wiped out
    tagged = (
        "===PROBLEM===\n"
        "A spring with stiffness k and mass m oscillates horizontally with amplitude A without friction.\n"
        "===SOLUTION===\n"
        "Step 1: Find the restoring force; this gives an intermediate answer: $F = -k x$.\n"
        "Step 2: Apply energy conservation to solve for maximum speed $v_{\\max} = A \\sqrt{k/m}$.\n"
        "ANSWER: A * sqrt(k / m)"
    )
    parsed = parse_tagged_generation(tagged)
    assert parsed is not None
    assert "Step 2: Apply energy conservation" in parsed.reference_solution
    assert parsed.answer_value == "A * sqrt(k / m)"

    # Fix 6: E2 & E3 retain clean physgym_seeds AND exclude SymPy/hygiene-broken synthetic items from E2
    clean_seed = CanonicalProblem(
        problem_id="physgym_0010",
        domain="ELECTROMAGNETISM",
        problem_statement="A point charge $q$ sits at distance $r$ from origin in vacuum. Determine electric potential $V(r)$ at $r$ with $V(\\infty)=0$.",
        reference_solution="Integrating the radial electric field $E = k_e q / r^2$ from infinity gives $V(r) = k_e q / r$.\nANSWER: k_e * q / r",
        source="physgym_seed",
        original_id=10,
    )
    book_prob = CanonicalProblem(
        problem_id="lit_mech_01",
        domain="MECHANICS",
        problem_statement="A wrench of length $0.50\\,\\text{m}$ has force from $55\\,\\text{kg}$ mass applied perpendicularly. Calculate maximum torque $\\tau$.",
        reference_solution="Using $\\tau = r m g = 0.50 \\times 55 \\times 9.81 = 269.78\\,\\text{N}\\cdot\\text{m}$.\nANSWER: 269.78 N*m",
        source="literature_curated",
        original_id="crowell_mechanics:15_conservation_of_angular_momentum:p004",
    )
    clean_books_p = tmp_path / "clean_books.jsonl"
    seeds_p = tmp_path / "train_seeds.jsonl"
    subtopics_p = tmp_path / "subtopics.jsonl"
    val_p = tmp_path / "val.jsonl"
    test_p = tmp_path / "test.jsonl"
    ckpt_p = tmp_path / "ckpt.jsonl"
    raw_p = tmp_path / "raw_syn.jsonl"
    e2_p = tmp_path / "e2.jsonl"
    e3_p = tmp_path / "e3.jsonl"
    sum_p = tmp_path / "summary.json"

    save_problems_jsonl([clean_seed, book_prob], clean_books_p)
    save_problems_jsonl([clean_seed], seeds_p)
    save_problems_jsonl([], val_p)
    save_problems_jsonl([], test_p)
    sub = SubtopicChunk(
        chunk_id="crowell_mechanics:15_conservation_of_angular_momentum:15.4:a",
        book_id="crowell_mechanics",
        chapter=15,
        chapter_slug="15_conservation_of_angular_momentum",
        domain="MECHANICS",
        section_id="15.4",
        section_title="Torque",
        text="Torque equals lever arm multiplied by perpendicular force component, $\\tau = r F \\sin\\theta$.",
        token_count=20,
        start_page=440,
        end_page=441,
    )
    subtopics_p.write_text(json.dumps(sub.to_dict()) + "\n", encoding="utf-8")

    # LLM generates a BAD synthetic output (fails SymPy check against 269.78 anchor!)
    bad_syn_resp = (
        "===PROBLEM===\n"
        "A rigid wrench of length $L$ supports mass $m$ perpendicular to pivot $O$ under gravity $g$. Compute $\\tau$.\n"
        "===SOLUTION===\n"
        "Using wrong formula $\\tau = L / (m g)$ gives the incorrect expression.\n"
        "ANSWER: L / (m * g)\n"
        "CHECK_VALUES: L=0.50, m=55, g=9.81"
    )

    class _MockEmbDedup:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def index_existing(self, *args, **kwargs) -> None:
            pass

        def _embed(self, texts: list[str]) -> np.ndarray:
            return np.zeros((len(texts), 4), dtype=np.float32)

        def _check_against_index(self, _row: np.ndarray) -> tuple[bool, float]:
            return False, 0.0

    with patch("curatorkit.normalizers.embedding_dedup.EmbeddingDeduplicator", _MockEmbDedup):
        e2_pool, e3_pool, _ = generate_synthetic_e2_and_e3_pools(
            clean_books_path=clean_books_p,
            subtopic_chunks_path=subtopics_p,
            train_seeds_path=seeds_p,
            validation_path=val_p,
            test_path=test_p,
            checkpoint_path=ckpt_p,
            synthetic_raw_out=raw_p,
            e2_pool_out=e2_p,
            e3_pool_out=e3_p,
            summary_out=sum_p,
            enable_part2_llm_judges=False,
            chat_fn=lambda _s, _u: bad_syn_resp,
        )

    e2_ids = {p.problem_id for p in e2_pool}
    e3_ids = {p.problem_id for p in e3_pool}
    # Clean physgym_seed MUST survive into both E2 and E3; SymPy-failed syn_seed_lit_mech_01 MUST stay out of E2!
    assert "physgym_0010" in e2_ids
    assert "physgym_0010" in e3_ids
    assert "syn_seed_lit_mech_01" not in e2_ids


def test_subtopic_equation_preservation_alias_and_summary_filter() -> None:
    # Fix 7: Short equations kept while visual figure fragments dropped
    mixed = (
        "When an ideal spring obeys Hooke's law, restoring force is opposite displacement.\n\n"
        "F = -k x\n\n"
        "See Figure 3b at right.\n\n"
        "short non math fragment"
    )
    cleaned = strip_visual_paragraphs(mixed, min_chars=60)
    assert "F = -k x" in cleaned
    assert "Figure 3b" not in cleaned
    assert "short non math fragment" not in cleaned

    # Fix 8: Alias Tier 2 mapping (`25_capacitance_and_inductance` -> `crowell_fields_and_circuits` ch 13/14)
    # + Chapter Summary exclusion + hybrid lexical reranking
    c_summary = SubtopicChunk(
        chunk_id="crowell_fields_and_circuits:13_ac_circuits:13.99:s",
        book_id="crowell_fields_and_circuits",
        chapter=13,
        chapter_slug="13_ac_circuits",
        domain="ELECTROMAGNETISM",
        section_id="13.99",
        section_title="Chapter Summary",
        text="AC circuits include capacitance, inductance, resonance, reactance, and impedance Z.",
        token_count=15,
        start_page=310,
        end_page=311,
    )
    c_sec = SubtopicChunk(
        chunk_id="crowell_fields_and_circuits:13_ac_circuits:13.2:b",
        book_id="crowell_fields_and_circuits",
        chapter=13,
        chapter_slug="13_ac_circuits",
        domain="ELECTROMAGNETISM",
        section_id="13.2",
        section_title="Capacitance and inductance",
        text="An inductor opposes changes in current via $V = L dI/dt$ while a capacitor stores energy $U = \\frac{1}{2} C V^2$.",
        token_count=22,
        start_page=298,
        end_page=300,
    )
    c_unrelated_em = SubtopicChunk(
        chunk_id="crowell_fields_and_circuits:02_gausss_law:2.1:c",
        book_id="crowell_fields_and_circuits",
        chapter=2,
        chapter_slug="02_gausss_law",
        domain="ELECTROMAGNETISM",
        section_id="2.1",
        section_title="Gauss law",
        text="Electric flux through a closed Gaussian surface equals enclosed charge over epsilon_0.",
        token_count=16,
        start_page=40,
        end_page=42,
    )

    class _TieEmbRetriever:
        def __init__(self, texts: list[str], model: str = "") -> None:
            del model
            self.index = np.ones((len(texts), 2), dtype=np.float32) * 0.5

        class model:
            @staticmethod
            def encode(queries: list[str], normalize_embeddings: bool = True) -> np.ndarray:
                del queries, normalize_embeddings
                return np.array([[0.5, 0.5]], dtype=np.float32)

    mapper = SubtopicMapper([c_summary, c_sec, c_unrelated_em])
    prob = CanonicalProblem(
        problem_id="lit_em_cap",
        domain="ELECTROMAGNETISM",
        problem_statement="An inductor of inductance L carries current I. Find stored magnetic energy and dI/dt voltage.",
        reference_solution="V = L dI/dt\nANSWER: L dI/dt",
        source="literature_curated",
        original_id="light_and_matter_problems:25_capacitance_and_inductance:p004",
    )
    with patch("curatorkit.diagnostic.retriever.PostHocRetriever", _TieEmbRetriever):
        hits = mapper.find_subtopics_with_tiers(prob, top_k=1)
    assert len(hits) == 1
    chunk_hit, tier = hits[0]
    assert tier == 2
    assert chunk_hit.section_id == "13.2"


def test_candidate_generator_and_curatorkit_runner_no_fake_overrides(tmp_path: Path) -> None:
    # Fix 9: Retry instead of blind `-()` / `ANSWER:` overwrite; fix `RejectedSample` AttributeError
    canon = CanonicalProblem(
        problem_id="p_opt_1",
        domain="OPTICS",
        problem_statement="Light enters glass ($n = 1.5$) from air ($n = 1.0$) at $30^\\circ$. Find $\\sin\\theta_2$.",
        reference_solution="Step 1: Snell's law $1.0 \\sin 30^\\circ = 1.5 \\sin\\theta_2$.\nANSWER: 0.333",
        source="literature_curated",
        original_id="crowell:10:p1",
    )
    canon_path = tmp_path / "canonical.jsonl"
    save_problems_jsonl([canon], canon_path)

    # First flawed attempt accidentally returns 0.333 (correct!); retry returns genuinely flawed 0.75
    flaw_calls = {"n": 0}

    def _flaw_chat(_sys: str, _usr: str) -> str:
        flaw_calls["n"] += 1
        if flaw_calls["n"] == 1:
            return "Step 1: Snell's law.\nANSWER: 0.333"
        return "Step 1: Inverted Snell's ratio $1.5 \\sin 30^\\circ$.\nANSWER: 0.75"

    err_cands = generate_controlled_error_trajectories(
        canonical_train_path=canon_path,
        checkpoint_path=tmp_path / "err_ckpt.jsonl",
        output_path=tmp_path / "err.jsonl",
        chat_fn=_flaw_chat,
    )
    assert flaw_calls["n"] == 2
    assert len(err_cands) == 1
    assert "0.75" in err_cands[0].solution_text
    assert "-(0.333)" not in err_cands[0].solution_text

    # Sol2 attempt returns wrong answer twice -> rejected/excluded instead of overwriting ANSWER:
    sol2_cands = generate_minority_alternative_solutions(
        canonical_train_path=canon_path,
        checkpoint_path=tmp_path / "sol2_ckpt.jsonl",
        output_path=tmp_path / "sol2.jsonl",
        chat_fn=lambda _s, _u: "Step 1: Wrong approach.\nANSWER: 9.99",
    )
    assert len(sol2_cands) == 0

    # CuratorKIT runner: short schema fail (tests RejectedSample.id / rejection_reason!) + bad Sol2 + correct synthetic error
    sol2_in = tmp_path / "run_sol2.jsonl"
    flawed_in = tmp_path / "run_flawed.jsonl"
    tiny_in = tmp_path / "run_tiny.jsonl"
    save_candidates_jsonl(
        [
            CandidateSolution(
                problem_id="p_opt_1",
                candidate_id="p_opt_1_sol2_bad",
                solution_text="Step 1: Compute wrong value step carefully.\nStep 2: Multiply incorrectly.\nANSWER: 8.88",
                is_correct=True,
                domain="OPTICS",
            )
        ],
        sol2_in,
    )
    save_candidates_jsonl(
        [
            CandidateSolution(
                problem_id="p_opt_1",
                candidate_id="p_opt_1_err_acc_ok",
                solution_text="Step 1: Apply Snell's law accurately.\nANSWER: 0.333",
                is_correct=False,
                domain="OPTICS",
                source="aya_expanse_8b_synthetic_error",
            ),
            CandidateSolution(
                problem_id="p_opt_1",
                candidate_id="p_opt_1_err_short",
                solution_text="",
                is_correct=False,
                domain="OPTICS",
                source="aya_expanse_8b_synthetic_error",
            ),
        ],
        flawed_in,
    )
    save_candidates_jsonl([], tiny_in)

    rejected_p = tmp_path / "rej.jsonl"
    curated_sol2, surviving, _ = curate_and_deduplicate_stage2_trajectories(
        canonical_train_path=canon_path,
        solution2_input_path=sol2_in,
        flawed_8b_input_path=flawed_in,
        tiny_aya_input_path=tiny_in,
        raw_synthetic_output_path=tmp_path / "raw_cand.jsonl",
        curated_sol2_output_path=tmp_path / "out_sol2.jsonl",
        tiny_aya_correct_output_path=tmp_path / "out_tiny_ok.jsonl",
        surviving_flawed_output_path=tmp_path / "out_surv.jsonl",
        rejected_output_path=rejected_p,
        summary_output_path=tmp_path / "run_sum.json",
    )
    assert len(curated_sol2) == 0
    assert len(surviving) == 0
    rej_rows = [json.loads(ln) for ln in rejected_p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    reasons = {r["reason"] for r in rej_rows}
    assert any("solution2_answer_mismatch" in r for r in reasons)
    assert any("synthetic_error_solved_correctly" in r for r in reasons)
    assert any("curatorkit_schema_gate:" in r for r in reasons)


def test_batched_eval_raw_prompt_checkpoint_swap_and_auditkit_bridge(tmp_path: Path) -> None:
    # Fix 1 & 2: Raw AlignTune training prompt format by default + Peft adapter swapping without base stacking
    val_p = tmp_path / "validation.jsonl"
    stmt = "A ball falls from height 20 m. Find impact speed."
    val_p.write_text(
        json.dumps(
            {
                "problem_id": "v1",
                "domain": "MECHANICS",
                "problem_statement": stmt,
                "reference_solution": "v = sqrt(2 g h) = 19.8 m/s\nANSWER: 19.8 m/s",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    seen_texts: list[str] = []

    class _FakeTok:
        padding_side = "right"
        pad_token_id = 0
        eos_token = "</s>"
        eos_token_id = 2

        def apply_chat_template(self, *args, **kwargs):
            raise AssertionError("Raw AlignTune evaluation must not invoke apply_chat_template by default")

        def __call__(self, texts: list[str], **_kwargs):
            seen_texts.extend(texts)
            return {"input_ids": torch.tensor([[1, 2, 3]] * len(texts))}

        def decode(self, _ids, skip_special_tokens: bool = True) -> str:
            del skip_special_tokens
            return "Using kinematics...\nANSWER: 19.8 m/s"

    class _FakePeftModel:
        peft_config = {"default": object()}
        active_adapter = "default"
        device = torch.device("cpu")
        training = False

        def __init__(self) -> None:
            self.loaded: list[str] = []
            self.activated: list[str] = []
            self.deleted: list[str] = []

        def eval(self):
            return self

        def parameters(self):
            return iter([torch.zeros(1)])

        def load_adapter(self, _path: str, adapter_name: str) -> None:
            self.loaded.append(adapter_name)

        def set_adapter(self, adapter_name: str) -> None:
            self.active_adapter = adapter_name
            self.activated.append(adapter_name)

        def delete_adapter(self, adapter_name: str) -> None:
            self.deleted.append(adapter_name)

        def generate(self, input_ids: torch.Tensor, **_kwargs) -> torch.Tensor:
            out = torch.full((input_ids.shape[0], input_ids.shape[1] + 2), 9)
            return out

    fake_peft = _FakePeftModel()
    tok = _FakeTok()
    eval_res = batched_evaluate_solver_split(fake_peft, tok, split_path=val_p, batch_size=2)
    assert eval_res["correct"] == 1
    assert seen_texts == [f"{format_solver_prompt(stmt, [])}\n\n"]

    # AuditKIT / AlignTune bridge check (`19.8 m/s` vs `19.8 m/s` -> exact_normalized)
    audit = run_physics_eval_audit(eval_res, experiment_name="test_exp")
    assert audit["match_method_breakdown"] == {"exact_normalized": 1}
    assert "avg_response_tokens" in audit["generation_hygiene"]

    # Multi-checkpoint selection on existing PeftModel uses load_adapter/set_adapter & restores 'default'
    out_dir = tmp_path / "run_ckpts"
    (out_dir / "checkpoint-10").mkdir(parents=True)
    (out_dir / "checkpoint-20").mkdir(parents=True)
    (out_dir / "checkpoint-10" / "adapter_model.safetensors").write_text("c10", encoding="utf-8")
    (out_dir / "checkpoint-20" / "adapter_model.safetensors").write_text("c20", encoding="utf-8")
    report = select_best_epoch_checkpoint(
        out_dir,
        base_model=fake_peft,
        tokenizer=tok,
        validation_path=val_p,
    )
    assert report is not None
    assert len(fake_peft.loaded) == 2
    assert fake_peft.active_adapter == "default"
    assert sorted(fake_peft.deleted) == sorted(fake_peft.loaded)
