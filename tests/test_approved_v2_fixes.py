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
    _normalize_step_lines_and_answer,
    _sanitize_flawed_trajectory_text,
    curate_and_deduplicate_stage2_trajectories,
)
from adversarial_physics.data.curation.curatorkit_stage1 import (
    PART2_STRICT_CONFIG,
    STAGE1_PHYSICS_REFINE_PROMPT,
    problem_to_datasample,
    run_curatorkit_stage1_gates_and_recovery,
)
from adversarial_physics.data.curation.dataset_exporter import export_all_stage2_sft_datasets
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


def test_train_adapter_no_undefined_vars_and_03b_endpoint_url_arg(tmp_path: Path) -> None:
    import importlib.util
    from types import SimpleNamespace
    from adversarial_physics.training.run_aligntune import train_adapter

    # Regression check for `is_verifier` and `dataset_dir` inside `train_adapter` (post-training checkpoint selection)
    out_dir = tmp_path / "adapter_out"
    val_dir = tmp_path / "ds_dir"
    val_dir.mkdir(parents=True)
    val_p = val_dir / "validation.jsonl"
    val_p.write_text(
        json.dumps(
            {
                "problem_id": "v1",
                "domain": "MECHANICS",
                "problem_statement": "A ball drops from 20 m. Find v.",
                "reference_solution": "ANSWER: 19.8 m/s",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class _MockTrainer:
        config = SimpleNamespace(
            logging=SimpleNamespace(output_dir=str(out_dir)),
            train=SimpleNamespace(
                epochs=1,
                per_device_batch_size=1,
                gradient_accumulation_steps=1,
                learning_rate=2e-4,
                completion_only_loss=True,
                early_stopping_patience=2,
                early_stopping_threshold=0.005,
            ),
            model=SimpleNamespace(name_or_path="CohereLabs/tiny-aya-global", peft=SimpleNamespace(rank=16, alpha=32)),
            dataset=SimpleNamespace(dataset_name=str(val_dir)),
        )
        model = object()
        tokenizer = object()

        def train(self):
            return {"train_loss": 0.5}

        def save_model(self, output_dir: str = "") -> str:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            return output_dir

    cfg = {
        "hardware": {"effective_batch_size": 1, "per_device_train_batch_size": 1, "gradient_accumulation_steps": 1},
        "training_hyperparameters": {"logging_dir": str(tmp_path / "runs")},
        "checkpoint_selection": {"enabled": True},
        "validation_dataset": {"local_path": str(tmp_path / "missing_val.jsonl")},
        "datasets": {"solver_e2_curatorkit_nollm": {"role": "solver", "local_dir": str(val_dir)}},
    }
    seen_val_paths: list[str] = []

    def _fake_select(**kwargs):
        seen_val_paths.append(str(kwargs["validation_path"]))
        return {"winner": {"checkpoint_dir": str(out_dir / "checkpoint-1"), "composite_score": 0.1}}

    with (
        patch("adversarial_physics.training.run_aligntune.create_aligntune_sft_trainer", return_value=_MockTrainer()),
        patch("adversarial_physics.training.run_aligntune.select_best_epoch_checkpoint", side_effect=_fake_select),
    ):
        res = train_adapter("solver_e2_curatorkit_nollm", cfg, output_dir_override=str(out_dir))
    assert seen_val_paths == [str(val_p)]
    assert res["checkpoint_selection"]["composite_score"] == 0.1

    # Verify `scripts/03b_generate_synthetic_e2.py` passes `--endpoint-url` cleanly
    spec = importlib.util.spec_from_file_location(
        "script_03b_mod",
        Path(__file__).resolve().parents[1] / "scripts" / "03b_generate_synthetic_e2.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    captured: dict[str, object] = {}

    def _fake_gen(**kwargs):
        captured.update(kwargs)
        return [], [], SimpleNamespace(to_dict=lambda: {"ok": True})

    with patch.object(mod, "generate_synthetic_e2_and_e3_pools", side_effect=_fake_gen):
        rc = mod.main(["--backend", "hf_gpu", "--endpoint-url", "https://example.endpoints.huggingface.cloud"])
    assert rc == 0
    assert captured["endpoint_url"] == "https://example.endpoints.huggingface.cloud"


def test_sanitize_flawed_trajectory_strips_meta_giveaways_and_answer_notes() -> None:
    raw_flawed = (
        "1. **Following the reference solution:** Set $F = ma$.\n"
        "2. **Incorrect Time Dependency:** **Flaw:** We incorrectly drop the cosine factor.\n"
        "3. **Step 3: Final Calculation:** Simplifying, we arrive at a **flawed final answer**: $v = 12$.\n"
        "**ANSWER: 12** (Note: This answer is flawed due to the injected error (cos skipped) in step 2.)\n"
        "<|im_end|>\n"
        "ANSWER: **Note:** This derivation intentionally omits a term."
    )
    cleaned, is_usable = _sanitize_flawed_trajectory_text(raw_flawed)
    assert is_usable is True
    assert "<|im_end|>" not in cleaned
    for cue in ("flaw", "flawed", "incorrect", "injected", "reference solution", "intentionally"):
        assert cue not in cleaned.lower(), f"Meta cue {cue!r} leaked in: {cleaned!r}"
    assert cleaned.endswith("\nANSWER: 12")

    # Preserve clean numeric/scientific answers on Sol2 normalization
    sol2_norm = _normalize_step_lines_and_answer("1. Solve.\n10. **ANSWER: 1.5e-19 C** (Verified)")
    assert sol2_norm.endswith("ANSWER: 1.5e-19 C (Verified)")
    sol2_num = _normalize_step_lines_and_answer("1. Solve.\nANSWER: 32 cm")
    assert sol2_num.endswith("ANSWER: 32 cm")


def test_post_refinement_contract_gate_and_llm_max_token_override() -> None:
    from curatorkit.llm.base import BaseLLM, LLMResponse

    base_prob = CanonicalProblem(
        problem_id="lit_mech_refine_01",
        domain="MECHANICS",
        problem_statement=(
            "A block of mass $m = 3.0\\,\\text{kg}$ accelerates uniformly from rest under a constant "
            "horizontal net force $F = 18.0\\,\\text{N}$ over $t = 2.5\\,\\text{s}$. Calculate its final speed $v$."
        ),
        reference_solution=(
            "By Newton's second law $a = F / m = 18.0 / 3.0 = 6.0\\,\\text{m/s}^2$, so $v = a t = 15.0\\,\\text{m/s}$.\n"
            "ANSWER: 15.0 m/s"
        ),
        source="literature_curated",
        original_id="crowell:04:p01",
    )

    class _MockStage1LLM(BaseLLM):
        def __init__(self, refine_text: str) -> None:
            super().__init__(model="mock-aya-8b", temperature=0.0, max_tokens=512, max_retries=0)
            self.refine_text = refine_text
            self.calls: list[tuple[int, str]] = []

        def _call(self, messages: list[dict[str, str]], **kwargs: object) -> LLMResponse:
            max_toks = int(kwargs.get("max_tokens", self.max_tokens))
            user_txt = "\n".join(m["content"] for m in messages)
            self.calls.append((max_toks, user_txt))
            if "Rewrite the solution to specifically improve" in user_txt:
                return LLMResponse(text=self.refine_text, model=self.model)
            if "grounding_score" in user_txt:
                return LLMResponse(text='{"grounding_score": 0.95, "verdict": "grounded"}', model=self.model)
            # Fail initial concise draft on RewardGate, pass refined draft on re-eval
            if "18.0 / 3.0 = 6.0" in user_txt and "Step 2:" not in user_txt and "grounding" not in user_txt:
                return LLMResponse(
                    text='{"overall_score": 0.50, "dimension_scores": {"helpfulness": 0.5, "coherence": 0.5}, "weaknesses": "too brief"}',
                    model=self.model,
                )
            return LLMResponse(
                text='{"overall_score": 0.94, "dimension_scores": {"helpfulness": 0.94, "coherence": 0.94}, "weaknesses": ""}',
                model=self.model,
            )

    # 1. Broken rewrite missing ANSWER: must be rejected by PostRefinementGate even if RewardGate scores 0.94
    broken_llm = _MockStage1LLM(
        "Step 1: Compute acceleration $a = F / m = 6.0\\,\\text{m/s}^2$. "
        "Step 2: Multiply by time to get $v = 15.0\\,\\text{m/s}$ without an answer sentinel line."
    )
    passed_bad, rejected_bad, rep_bad = run_curatorkit_stage1_gates_and_recovery(
        [base_prob],
        strict_config=PART2_STRICT_CONFIG,
        llm=broken_llm,
    )
    assert passed_bad == []
    assert rep_bad.adaptive_recovery_attempted == 1
    assert rep_bad.adaptive_recovery_succeeded == 0
    assert any(r["gate"] == "PostRefinementGate" and "missing ANSWER:" in r["reason"] for r in rejected_bad)
    # Verify refiner got max_tokens >= 1536 while sharing the same LLM instance
    refine_calls = [(mt, txt) for mt, txt in broken_llm.calls if "Reference Context:" in txt]
    assert len(refine_calls) == 1
    assert refine_calls[0][0] == 1536
    assert "Reference Context:" in refine_calls[0][1]
    assert STAGE1_PHYSICS_REFINE_PROMPT.splitlines()[0].split("{axis}")[0] in refine_calls[0][1]

    # 2. Truncated \left without \right or meta "source passage" cue must also be rejected
    meta_llm = _MockStage1LLM(
        "Step 1: Grounded in the source passage, acceleration is $a = \\left(\\frac{18}{3} = 6\\,\\text{m/s}^2$ "
        "and speed is $15.0\\,\\text{m/s}$.\nANSWER: 15.0 m/s"
    )
    passed_meta, rejected_meta, _ = run_curatorkit_stage1_gates_and_recovery(
        [base_prob],
        strict_config=PART2_STRICT_CONFIG,
        llm=meta_llm,
    )
    assert passed_meta == []
    assert any(r["gate"] == "PostRefinementGate" for r in rejected_meta)

    # 3. Clean refined solution with `**Answer:**` gets canonicalized to `ANSWER:` and accepted
    good_llm = _MockStage1LLM(
        "Step 1: Using Newton's second law, $a = F / m = 18.0\\,\\text{N} / 3.0\\,\\text{kg} = 6.0\\,\\text{m/s}^2$.\n"
        "Step 2: Using constant-acceleration kinematics from rest, $v = a t = 6.0 \\times 2.5 = 15.0\\,\\text{m/s}$.\n"
        "**Answer: 15.0 m/s**"
    )
    passed_good, rejected_good, rep_good = run_curatorkit_stage1_gates_and_recovery(
        [base_prob],
        strict_config=PART2_STRICT_CONFIG,
        llm=good_llm,
    )
    assert len(passed_good) == 1
    assert rejected_good == []
    assert rep_good.adaptive_recovery_succeeded == 1
    assert passed_good[0].reference_solution.endswith("\nANSWER: 15.0 m/s")


def test_dataset_exporter_enforces_answer_sentinel_and_handles_orphans(tmp_path: Path) -> None:
    from adversarial_physics.data.schemas.trajectory import save_candidates_jsonl

    valid_p = CanonicalProblem(
        problem_id="p_ok",
        domain="MECHANICS",
        problem_statement="A 2 kg block slides with constant force 10 N for 3 s starting from rest. Find final speed.",
        reference_solution=(
            "Using Newton's second law $a = F/m = 10/2 = 5\\,\\text{m/s}^2$, final speed $v = a t = 15\\,\\text{m/s}$.\n"
            "ANSWER: 15 m/s"
        ),
        source="literature_curated",
        original_id="c:1",
    )
    missing_ans_p = CanonicalProblem(
        problem_id="p_no_ans",
        domain="MECHANICS",
        problem_statement="A 4 kg mass moves at 3 m/s on a horizontal smooth floor. Determine its kinetic energy.",
        reference_solution=(
            "Using the kinetic energy definition $K = \\frac{1}{2} m v^2 = 0.5 \\times 4 \\times 9 = 18\\,\\text{J}$ "
            "without writing any answer marker line."
        ),
        source="literature_curated",
        original_id="c:2",
    )
    val_p = CanonicalProblem(
        problem_id="val_1",
        domain="MECHANICS",
        problem_statement="A 1 kg ball drops freely from 5 m height. Find impact speed.",
        reference_solution="Using energy conservation $v = \\sqrt{2 g h} = 9.9\\,\\text{m/s}$.\nANSWER: 9.9 m/s",
        source="physgym",
        original_id="v:1",
    )

    e1_p = tmp_path / "e1.jsonl"
    e2_p = tmp_path / "e2.jsonl"
    e3_p = tmp_path / "e3.jsonl"
    val_file = tmp_path / "val.jsonl"
    test_file = tmp_path / "test.jsonl"
    sol2_file = tmp_path / "sol2.jsonl"
    surv_file = tmp_path / "surv.jsonl"
    dir_file = tmp_path / "dirs.jsonl"

    save_problems_jsonl([valid_p], e1_p)
    save_problems_jsonl([valid_p], e2_p)
    save_problems_jsonl([valid_p, missing_ans_p], e3_p)
    save_problems_jsonl([val_p], val_file)
    save_problems_jsonl([val_p], test_file)

    # Orphan Sol2 pointing to an upstream-dropped ID `p_dropped` must NOT cause KeyError
    save_candidates_jsonl(
        [
            CandidateSolution(
                problem_id="p_dropped",
                candidate_id="p_dropped_sol2",
                solution_text="Alternative valid path giving exact velocity result cleanly step by step.\nANSWER: 12 m/s",
                is_correct=True,
                domain="MECHANICS",
                error_type=None,
                paired_directive=None,
                source="aya_8b_alt",
            )
        ],
        sol2_file,
    )
    save_candidates_jsonl([], surv_file)
    dir_file.write_text("", encoding="utf-8")

    summary = export_all_stage2_sft_datasets(
        e1_pool_path=e1_p,
        e2_pool_path=e2_p,
        e3_pool_path=e3_p,
        validation_path=val_file,
        test_path=test_file,
        curated_sol2_path=sol2_file,
        surviving_flawed_path=surv_file,
        directives_input_path=dir_file,
        sft_output_root=tmp_path / "sft",
    )
    assert summary.solver_e3_canonical_llm_count == 1
    assert summary.solver_e4_adversarial_total_count == 1
    assert summary.solver_e4_turn0_minority_sol2_count == 0
    e3_rows = [json.loads(line) for line in (tmp_path / "sft" / "solver_e3_canonical_llm" / "train.jsonl").open(encoding="utf-8")]
    assert len(e3_rows) == 1
    assert e3_rows[0]["problem_id"] == "p_ok"

    import hashlib

    sft_root = tmp_path / "sft"
    expected_dirs = [
        "solver_e1_manual_det",
        "solver_e2_curatorkit_nollm",
        "solver_e3_canonical_llm",
        "solver_e4_adversarial",
        "verifier_e4_curated",
    ]
    for ds_name in expected_dirs:
        ds_dir = sft_root / ds_name
        readme_path = ds_dir / "README.md"
        assert readme_path.is_file(), f"Missing README.md for {ds_name}"
        card = readme_path.read_text(encoding="utf-8")
        assert card.startswith("---\nlicense: cc-by-sa-4.0\n")
        assert "configs:\n  - config_name: default" in card
        assert "path: train.jsonl" in card
        if ds_name == "verifier_e4_curated":
            assert "path: validation.jsonl" not in card
        else:
            assert "path: validation.jsonl" in card
            assert "path: test.jsonl" in card

        checksum_lines = [
            ln.strip()
            for ln in (ds_dir / "checksums.txt").read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        tracked_names = set()
        for ln in checksum_lines:
            digest, fname = ln.split(None, 1)
            tracked_names.add(fname)
            actual = hashlib.sha256((ds_dir / fname).read_bytes()).hexdigest()
            assert actual == digest, f"Checksum mismatch for {ds_name}/{fname}"
        assert "README.md" in tracked_names

    e3_card = (sft_root / "solver_e3_canonical_llm" / "README.md").read_text(encoding="utf-8")
    assert "`train.jsonl` | **1**" in e3_card
    assert '"problem_id": "p_ok"' in e3_card


def test_push_to_hub_uploads_winning_final_adapter_and_metadata(tmp_path: Path) -> None:
    from types import SimpleNamespace
    from unittest.mock import patch
    from adversarial_physics.training.run_aligntune import train_adapter

    out_dir = tmp_path / "solver_out"
    val_file = tmp_path / "validation.jsonl"
    val_file.write_text(
        json.dumps(
            {
                "problem_id": "v1",
                "domain": "MECHANICS",
                "problem_statement": "A 1 kg ball drops from 5 m. Find impact speed.",
                "reference_solution": "ANSWER: 9.9 m/s",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    # checkpoint-10 has lower eval_loss (0.8), checkpoint-20 has higher val_accuracy (0.75 vs 0.0 -> wins composite score)
    for step, ckpt_tag in ((10, "ckpt_10"), (20, "ckpt_20")):
        ckpt_dir = out_dir / f"checkpoint-{step}"
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "adapter_config.json").write_text(json.dumps({"tag": ckpt_tag}), encoding="utf-8")
    (out_dir / "trainer_state.json").write_text(
        json.dumps(
            {
                "log_history": [
                    {"step": 10, "eval_loss": 0.8},
                    {"step": 20, "eval_loss": 1.0},
                ]
            }
        ),
        encoding="utf-8",
    )

    class _MockModel:
        def __init__(self) -> None:
            self.ram_push_called = False

        def push_to_hub(self, *args: object, **kwargs: object) -> None:
            self.ram_push_called = True
            raise AssertionError("trainer.model.push_to_hub should not bypass final_adapter upload")

    class _MockTok:
        def save_pretrained(self, save_dir: str) -> None:
            Path(save_dir, "tokenizer_config.json").write_text('{"tok": "ok"}', encoding="utf-8")

    class _MockPeft:
        def __init__(self, first_ckpt: str) -> None:
            self.active = first_ckpt

        def load_adapter(self, ckpt_dir: str, adapter_name: str = "") -> None:
            self._loaded = ckpt_dir

        def set_adapter(self, adapter_name: str) -> None:
            self.active = getattr(self, "_loaded", adapter_name)

        def unload(self) -> None:
            pass

    mock_model = _MockModel()

    class _MockTrainer:
        config = SimpleNamespace(
            logging=SimpleNamespace(output_dir=str(out_dir)),
            train=SimpleNamespace(
                epochs=2,
                per_device_batch_size=1,
                gradient_accumulation_steps=1,
                learning_rate=2e-4,
                completion_only_loss=True,
                early_stopping_patience=2,
                early_stopping_threshold=0.005,
            ),
            model=SimpleNamespace(name_or_path="CohereLabs/tiny-aya-global", peft=SimpleNamespace(rank=16, alpha=32)),
            dataset=SimpleNamespace(dataset_name=str(tmp_path)),
        )
        model = mock_model
        tokenizer = None
        processing_class = _MockTok()

        def train(self):
            return {"train_loss": 0.5}

        def save_model(self, output_dir: str = "") -> str:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            # Simulate initial eval_loss-best weights + AlignTune provenance in final_adapter before ckpt selection
            Path(output_dir, "adapter_config.json").write_text(json.dumps({"tag": "ckpt_10"}), encoding="utf-8")
            Path(output_dir, "lexsi_provenance.json").write_text('{"schema": "lexsi.provenance/1"}', encoding="utf-8")
            return output_dir

    eval_calls = iter(
        [
            {"total": 38, "correct": 0, "accuracy": 0.0},
            {"total": 38, "correct": 29, "accuracy": 0.7632},
        ]
    )
    created_repos: list[tuple[str, str]] = []
    uploaded_folders: list[dict[str, str]] = []

    class _MockHfApi:
        def __init__(self, token=None) -> None:
            self.token = token

        def create_repo(self, *, repo_id: str, repo_type: str = "model", exist_ok: bool = False) -> None:
            created_repos.append((repo_id, repo_type))

        def upload_folder(self, *, folder_path: str, repo_id: str, repo_type: str = "model", commit_message: str = "") -> None:
            uploaded_folders.append(
                {
                    "folder_path": folder_path,
                    "repo_id": repo_id,
                    "repo_type": repo_type,
                    "adapter_tag": json.loads(Path(folder_path, "adapter_config.json").read_text(encoding="utf-8"))["tag"],
                    "has_tok": str(Path(folder_path, "tokenizer_config.json").is_file()),
                    "has_prov": str(Path(folder_path, "lexsi_provenance.json").is_file()),
                    "has_summary": str(Path(folder_path, "training_summary.json").is_file()),
                }
            )

    cfg = {
        "hardware": {"effective_batch_size": 1, "per_device_train_batch_size": 1, "gradient_accumulation_steps": 1},
        "training_hyperparameters": {"logging_dir": str(tmp_path / "runs")},
        "checkpoint_selection": {"enabled": True, "composite_alpha_loss": 1.0, "composite_k_accuracy": 1.5},
        "validation_dataset": {"local_path": str(val_file)},
        "datasets": {
            "solver_e3_canonical_llm": {
                "role": "solver",
                "adapter_name": "solver_e3_canonical_llm_v2_lora",
                "output_repo_id": "vital-ar/adversarial-physics-solver-e3-canonical-llm-v2-lora",
                "local_dir": str(tmp_path),
            }
        },
    }

    with (
        patch("adversarial_physics.training.run_aligntune.create_aligntune_sft_trainer", return_value=_MockTrainer()),
        patch("peft.PeftModel.from_pretrained", side_effect=lambda base, ckpt, **kw: _MockPeft(ckpt)),
        patch("adversarial_physics.training.run_aligntune.batched_evaluate_solver_split", side_effect=lambda *a, **kw: next(eval_calls)),
        patch("huggingface_hub.HfApi", _MockHfApi),
    ):
        res = train_adapter(
            "solver_e3_canonical_llm",
            cfg,
            output_dir_override=str(out_dir),
            push_to_hub=True,
            hf_token="hf_test_tok",
        )

    assert mock_model.ram_push_called is False
    assert created_repos == [("vital-ar/adversarial-physics-solver-e3-canonical-llm-v2-lora", "model")]
    assert len(uploaded_folders) == 1
    up = uploaded_folders[0]
    assert up["folder_path"] == str(out_dir / "final_adapter")
    assert up["adapter_tag"] == "ckpt_20"
    assert up["has_tok"] == "True"
    assert up["has_prov"] == "True"
    assert up["has_summary"] == "True"
    assert res["checkpoint_selection"]["step"] == 20
    # Final summary saved on disk must match the composite winner
    disk_summary = json.loads((out_dir / "final_adapter" / "training_summary.json").read_text(encoding="utf-8"))
    assert disk_summary["checkpoint_selection"]["step"] == 20
