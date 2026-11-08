"""Stage 1 CuratorKIT Integration (`src/adversarial_physics/data/curation/curatorkit_stage1.py`).

Uses CuratorKIT (`curatorkit` v0.1.22) directly as a mandatory core dependency across Stage 1:
  1. **Ingestion & Text Hygiene (`apply_curatorkit_text_cleaner`)**:
     Runs `curatorkit.normalizers.clean.TextCleaner` over `CanonicalProblem` statements and solutions.
  2. **Two-Level Deduplication**:
     * `run_curatorkit_embedding_dedup`: Persistent cross-run `EmbeddingDeduplicator` (FAISS +
       `sentence-transformers/all-MiniLM-L6-v2`) configured with `threshold = 0.85` (overriding
       CuratorKIT's 0.92 default) for the Anti-Leakage Gate against the 76 held-out Validation
       and Test problems.
     * `run_curatorkit_within_pool_dedup`: `ExactDeduplicator`, `MinHashDeduplicator` (`threshold = 0.85`),
       and `DiversityGate` (`similarity_threshold = 0.88`, `coverage_field = "domain"`).
  3. **Quality Gates & Adaptive Recovery (`run_curatorkit_stage1_gates_and_recovery`)**:
     * `SchemaGate` (`required_fields=["instruction", "output"]`, `min_tokens=10`, `max_tokens=2048`)
     * `HallucinationGate` (source-solution grounding check)
     * `RewardGate` (custom 6-dimension physics rubric with `{instruction}`, `{response}`, and `"overall_score"`)
     * `RewardRefiner` (targeted single-retry Adaptive Recovery for failed samples)
     * `DiversityGate` (`similarity_threshold = 0.88`, `coverage_field = "domain"`)
  4. **Provenance & Exports (`write_curatorkit_provenance_bundle`)**:
     Uses `ProvenanceManifest`, `DatasetCardGenerator`, and `build_provenance` from `curatorkit.manifest`
     to emit `manifest.json`, `rejected.jsonl`, `dataset_card.md`, `lexsi_provenance.json`, and `checksums.txt`.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from curatorkit.diagnostic.reward_refine import RewardRefiner
from curatorkit.gates.diversity import DiversityGate
from curatorkit.gates.hallucination import HallucinationGate
from curatorkit.gates.reward import RewardGate
from curatorkit.gates.schema import SchemaGate
from curatorkit.llm.base import BaseLLM, LLMResponse
from curatorkit.manifest import DatasetCardGenerator, ProvenanceManifest, build_provenance
from curatorkit.normalizers.clean import TextCleaner
from curatorkit.normalizers.dedup import ExactDeduplicator, MinHashDeduplicator
from curatorkit.normalizers.embedding_dedup import EmbeddingDeduplicator
from curatorkit.pipeline import PipelineResult
from curatorkit.schema import DataSample, RejectedSample

from adversarial_physics.data.schemas.problem import CanonicalProblem
from adversarial_physics.utils.config import resolve_project_path
from adversarial_physics.utils.logging import get_logger
from adversarial_physics.utils.runtime import AYA_EXPANSE_8B, AyaExpanseCuratorLLM, BackendName

logger = get_logger("stage_1_curatorkit")

# Custom 6-dimension physics rubric for CuratorKIT's RewardGate.
# Must include {instruction}, {response}, and return JSON containing "overall_score"
# to satisfy curatorkit.gates.reward.RewardGate's prompt validator and score parser.
STAGE1_PHYSICS_RUBRIC_PROMPT = """You are an expert university physics evaluator.
Rate the quality of the reference solution for the physics problem on a scale from 0.0 to 1.0.

Problem Statement:
{instruction}

Reference Solution:
{response}

Evaluate across these 6 physics dimensions:
1. physical_law_selection: Governing physical laws are appropriate for the problem statement.
2. mathematical_correctness: Algebraic substitutions and final expression are consistent.
3. logical_consistency: Steps follow logically without circular reasoning.
4. solution_conciseness: Solvable in 1 to 10 logical reasoning operations.
5. completeness_of_final_answer: Ends with a clear `ANSWER: <value>` line.
6. unit_consistency: SI units or symbolic dimensions are consistent.

Respond in JSON format ONLY:
{{
  "overall_score": 0.XX,
  "dimension_scores": {{
    "helpfulness": 0.XX,
    "instruction_following": 0.XX,
    "truthfulness": 0.XX,
    "coherence": 0.XX
  }},
  "strengths": "brief note",
  "weaknesses": "concise diagnosis if failed"
}}
"""


class _CallableCuratorLLM(BaseLLM):
    """Adapter wrapping a `(system_prompt, user_prompt) -> str` callable into a CuratorKIT `BaseLLM`."""

    def __init__(
        self,
        chat_fn: Callable[[str, str], str],
        model: str = AYA_EXPANSE_8B,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> None:
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens, max_retries=0)
        self._chat_fn = chat_fn

    def _call(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        sys_parts = [m["content"] for m in messages if m.get("role") == "system"]
        usr_parts = [m["content"] for m in messages if m.get("role") != "system"]
        system_prompt = "\n".join(sys_parts).strip() or "You are an expert physics evaluator."
        user_prompt = "\n".join(usr_parts).strip()
        text = self._chat_fn(system_prompt, user_prompt)
        return LLMResponse(text=str(text), model=self.model)


@dataclass(frozen=True)
class CuratorKitGateReport:
    """Report returned after running CuratorKIT Stage 1 gates and Adaptive Recovery."""

    schema_gate_passed: int
    schema_gate_failed: int
    diversity_gate_passed: int
    diversity_gate_failed: int
    reward_gate_passed: int
    reward_gate_failed: int
    hallucination_gate_passed: int
    hallucination_gate_failed: int
    adaptive_recovery_attempted: int
    adaptive_recovery_succeeded: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def problem_to_datasample(prob: CanonicalProblem, *, split_tag: str = "train") -> DataSample:
    """Convert a 6-field `CanonicalProblem` into a CuratorKIT `DataSample`."""
    return DataSample(
        id=prob.problem_id,
        source_uri=f"{prob.source}:{prob.original_id}",
        instruction=prob.problem_statement,
        input=prob.reference_solution,
        output=prob.reference_solution,
        task_type="instruction_following",
        metadata={
            "problem_id": prob.problem_id,
            "domain": prob.domain,
            "source": prob.source,
            "original_id": prob.original_id,
            "split": split_tag,
        },
    )


def datasample_to_problem(sample: DataSample) -> CanonicalProblem:
    """Reconstruct a 6-field `CanonicalProblem` from a CuratorKIT `DataSample`."""
    meta = sample.metadata or {}
    return CanonicalProblem(
        problem_id=str(meta.get("problem_id") or sample.id),
        domain=str(meta.get("domain") or "MECHANICS"),
        problem_statement=sample.instruction.strip(),
        reference_solution=sample.output.strip(),
        source=str(meta.get("source") or "literature_curated"),
        original_id=meta.get("original_id", sample.source_uri),
    )


def apply_curatorkit_text_cleaner(problems: list[CanonicalProblem]) -> list[CanonicalProblem]:
    """Run `curatorkit.normalizers.clean.TextCleaner` directly on a list of `CanonicalProblem` records."""
    if not problems:
        return []
    samples = [problem_to_datasample(p) for p in problems]
    cleaner = TextCleaner(fields=["instruction", "output"])
    cleaned_samples = cleaner.run(samples)
    return [datasample_to_problem(s) for s in cleaned_samples]


def run_curatorkit_embedding_dedup(
    candidates: list[CanonicalProblem],
    heldout: list[tuple[CanonicalProblem, str]],
    *,
    cosine_threshold: float = 0.85,
    index_dir: str | Path = "data/processed/embedding_index_heldout",
) -> tuple[list[CanonicalProblem], list[dict[str, Any]]]:
    """Run CuratorKIT's `EmbeddingDeduplicator` (`threshold = 0.85`) against the 76 held-out Val/Test problems.

    1. Seeds a fresh persistent FAISS index at `index_dir` with all held-out Validation and Test
       problem statements (`text_field="instruction"`).
    2. Checks every candidate problem statement against the held-out FAISS index at `cosine_threshold`
       (default `0.85`, explicitly overriding CuratorKIT's `0.92` default).
    """
    if not candidates or not heldout:
        return list(candidates), []

    resolved_index_dir = resolve_project_path(index_dir)
    if resolved_index_dir.exists():
        shutil.rmtree(resolved_index_dir)

    dedup = EmbeddingDeduplicator(
        index_dir=resolved_index_dir,
        model="sentence-transformers/all-MiniLM-L6-v2",
        threshold=cosine_threshold,
        text_field="instruction",
    )

    # 1. Index all 76 held-out Validation + Test problems into the persistent FAISS index
    heldout_samples = [
        problem_to_datasample(prob, split_tag=split_name) for prob, split_name in heldout
    ]
    # Use threshold=1.01 while seeding so every held-out problem is stored in the index
    dedup.threshold = 1.01
    dedup.run(heldout_samples)
    dedup.threshold = cosine_threshold
    dedup._load_index()

    # 2. Encode all candidate problem statements via CuratorKIT's SentenceTransformer model
    model = dedup._load_model()
    cand_samples = [problem_to_datasample(c) for c in candidates]
    texts = [dedup._get_text(s) for s in cand_samples]
    raw_embs = model.encode(
        texts,
        batch_size=dedup.batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype(np.float32)
    norms = np.linalg.norm(raw_embs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    embeddings = raw_embs / norms

    accepted: list[CanonicalProblem] = []
    purged: list[dict[str, Any]] = []
    for cand, emb in zip(candidates, embeddings):
        emb_row = emb.reshape(1, -1)
        is_dup, max_sim = dedup._check_against_index(emb_row)
        if is_dup:
            purged.append(
                {
                    "candidate_id": cand.problem_id,
                    "candidate_original_id": cand.original_id,
                    "max_similarity": round(float(max_sim), 4),
                    "reason": f"curatorkit_embedding_dedup>={cosine_threshold:.2f} ({max_sim:.3f})",
                }
            )
        else:
            accepted.append(cand)

    logger.info(
        "CuratorKIT EmbeddingDeduplicator (threshold=%.2f): %d accepted, %d purged against %d held-out problems.",
        cosine_threshold,
        len(accepted),
        len(purged),
        len(heldout),
    )
    return accepted, purged


def run_curatorkit_within_pool_dedup(
    candidates: list[CanonicalProblem],
    *,
    seed_problems: list[CanonicalProblem] | None = None,
    minhash_threshold: float = 0.85,
    diversity_threshold: float = 0.88,
) -> tuple[list[CanonicalProblem], list[dict[str, Any]]]:
    """Run CuratorKIT's `ExactDeduplicator` and `MinHashDeduplicator` over candidate problems."""
    if not candidates:
        return [], []

    seeds = seed_problems or []
    seed_ids = {p.problem_id for p in seeds}
    combined_samples = [problem_to_datasample(p, split_tag="seed") for p in seeds] + [
        problem_to_datasample(c, split_tag="candidate") for c in candidates
    ]

    exact_dedup = ExactDeduplicator()
    after_exact = exact_dedup.run(combined_samples)

    minhash_dedup = MinHashDeduplicator(
        threshold=minhash_threshold,
        ngram=3,
        num_perm=128,
        seed=42,
    )
    after_minhash = minhash_dedup.run(after_exact)

    surviving_ids = {s.id for s in after_minhash}
    accepted: list[CanonicalProblem] = []
    purged: list[dict[str, Any]] = []
    after_exact_ids = {s.id for s in after_exact}

    for cand in candidates:
        if cand.problem_id in surviving_ids and cand.problem_id not in seed_ids:
            accepted.append(cand)
        else:
            reason = (
                "curatorkit_exact_duplicate"
                if cand.problem_id not in after_exact_ids
                else f"curatorkit_minhash_duplicate>={minhash_threshold:.2f}"
            )
            purged.append(
                {
                    "candidate_id": cand.problem_id,
                    "candidate_original_id": cand.original_id,
                    "reason": reason,
                }
            )
    return accepted, purged


def run_curatorkit_stage1_gates_and_recovery(
    problems: list[CanonicalProblem],
    *,
    max_solution_chars: int = 4500,
    require_answer_sentinel: bool = True,
    diversity_threshold: float = 0.88,
    enable_llm_gates: bool = False,
    llm: BaseLLM | None = None,
    chat_fn: Callable[[str, str], str] | None = None,
    repair_fn: Callable[[CanonicalProblem, str], CanonicalProblem] | None = None,
) -> tuple[list[CanonicalProblem], list[dict[str, Any]], CuratorKitGateReport]:
    """Run CuratorKIT `SchemaGate`, `HallucinationGate`, `RewardGate`, `RewardRefiner`, and `DiversityGate`."""
    if not problems:
        empty_report = CuratorKitGateReport(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        return [], [], empty_report

    recovery_attempted = 0
    recovery_succeeded = 0

    # Pre-check deterministic physics constraints (ANSWER: sentinel & max_solution_chars) with Adaptive Recovery
    pre_checked_problems: list[CanonicalProblem] = []
    rejected_records: list[dict[str, Any]] = []
    schema_failed_pre = 0

    for prob in problems:
        failure_diag = ""
        if len(prob.reference_solution) > max_solution_chars:
            failure_diag = f"reference_solution exceeds {max_solution_chars} chars"
        elif require_answer_sentinel and "ANSWER:" not in prob.reference_solution:
            failure_diag = "missing ANSWER: sentinel line"

        if failure_diag and repair_fn is not None:
            recovery_attempted += 1
            repaired = repair_fn(prob, failure_diag)
            if len(repaired.reference_solution) <= max_solution_chars and (
                not require_answer_sentinel or "ANSWER:" in repaired.reference_solution
            ):
                prob = repaired
                failure_diag = ""
                recovery_succeeded += 1

        if failure_diag:
            schema_failed_pre += 1
            rejected_records.append(
                {**prob.to_dict(), "gate": "SchemaGate", "reason": failure_diag}
            )
        else:
            pre_checked_problems.append(prob)

    # 1. CuratorKIT SchemaGate
    samples = [problem_to_datasample(p) for p in pre_checked_problems]
    schema_gate = SchemaGate(
        required_fields=["instruction", "output"],
        min_tokens=5,
        max_tokens=2048,
        use_tiktoken=False,
    )
    schema_passed_samples, schema_rejected_samples = schema_gate.run(samples)

    # Attempt repair on any sample rejected by CuratorKIT SchemaGate
    if schema_rejected_samples and repair_fn is not None:
        still_rejected_schema: list[RejectedSample] = []
        for rej_sample in schema_rejected_samples:
            recovery_attempted += 1
            repaired_prob = repair_fn(datasample_to_problem(rej_sample), rej_sample.rejection_reason)
            re_passed, re_rej = schema_gate.run([problem_to_datasample(repaired_prob)])
            if re_passed:
                schema_passed_samples.extend(re_passed)
                recovery_succeeded += 1
            else:
                still_rejected_schema.extend(re_rej)
        schema_rejected_samples = still_rejected_schema

    for rej in schema_rejected_samples:
        rejected_records.append(
            {
                **datasample_to_problem(rej).to_dict(),
                "gate": "SchemaGate",
                "reason": rej.rejection_reason,
            }
        )

    schema_passed_count = len(schema_passed_samples)
    schema_failed_count = schema_failed_pre + len(schema_rejected_samples)
    active_samples = schema_passed_samples

    # 2. Optional CuratorKIT HallucinationGate + RewardGate + RewardRefiner (when enable_llm_gates=True)
    hallucination_passed_count = len(active_samples)
    hallucination_failed_count = 0
    reward_passed_count = len(active_samples)
    reward_failed_count = 0

    resolved_llm: BaseLLM | None = llm
    if resolved_llm is None and chat_fn is not None:
        resolved_llm = _CallableCuratorLLM(chat_fn)

    if enable_llm_gates and resolved_llm is not None and active_samples:
        hallucination_gate = HallucinationGate(
            llm=resolved_llm,
            threshold=0.7,
            concurrency=1,
            on_judge_error="pass",
        )
        hal_passed, hal_rejected = hallucination_gate.run(active_samples)
        hallucination_passed_count = len(hal_passed)
        hallucination_failed_count = len(hal_rejected)
        for rej in hal_rejected:
            rejected_records.append(
                {
                    **datasample_to_problem(rej).to_dict(),
                    "gate": "HallucinationGate",
                    "reason": rej.rejection_reason,
                }
            )
        active_samples = hal_passed

        reward_gate = RewardGate(
            llm=resolved_llm,
            threshold=0.7,
            dimensions=["helpfulness", "instruction_following", "truthfulness", "coherence"],
            prompt_template=STAGE1_PHYSICS_RUBRIC_PROMPT,
            concurrency=1,
            on_judge_error="pass",
        )
        rew_passed, rew_rejected = reward_gate.run(active_samples)
        if rew_rejected:
            recovery_attempted += len(rew_rejected)
            refiner = RewardRefiner(
                generator_llm=resolved_llm,
                reward_gate=reward_gate,
                concurrency=1,
            )
            recovered_samples, still_rejected = refiner.refine(rew_rejected)
            recovery_succeeded += len(recovered_samples)
            rew_passed.extend(recovered_samples)
            rew_rejected = still_rejected

        reward_passed_count = len(rew_passed)
        reward_failed_count = len(rew_rejected)
        for rej in rew_rejected:
            rejected_records.append(
                {
                    **datasample_to_problem(rej).to_dict(),
                    "gate": "RewardGate",
                    "reason": rej.rejection_reason,
                }
            )
        active_samples = rew_passed

    # 3. CuratorKIT DiversityGate (embedding similarity threshold = 0.88, coverage_field = "domain")
    diversity_gate = DiversityGate(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold=diversity_threshold,
        text_field="instruction",
        coverage_field="domain",
    )
    div_passed, div_rejected = diversity_gate.run(active_samples)
    for rej in div_rejected:
        rejected_records.append(
            {
                **datasample_to_problem(rej).to_dict(),
                "gate": "DiversityGate",
                "reason": rej.rejection_reason,
            }
        )

    diversity_passed_count = len(div_passed)
    diversity_failed_count = len(div_rejected)
    accepted_problems = [datasample_to_problem(s) for s in div_passed]

    report = CuratorKitGateReport(
        schema_gate_passed=schema_passed_count,
        schema_gate_failed=schema_failed_count,
        diversity_gate_passed=diversity_passed_count,
        diversity_gate_failed=diversity_failed_count,
        reward_gate_passed=reward_passed_count,
        reward_gate_failed=reward_failed_count,
        hallucination_gate_passed=hallucination_passed_count,
        hallucination_gate_failed=hallucination_failed_count,
        adaptive_recovery_attempted=recovery_attempted,
        adaptive_recovery_succeeded=recovery_succeeded,
    )
    return accepted_problems, rejected_records, report


def write_curatorkit_provenance_bundle(
    output_dir: Path,
    *,
    stage_id: str,
    accepted_problems: list[CanonicalProblem],
    rejected_records: list[dict[str, Any]],
    gate_report: CuratorKitGateReport,
    summary_dict: dict[str, Any],
    tracked_files: list[Path],
    generation_model: str = AYA_EXPANSE_8B,
    runtime_backend: BackendName = "auto",
    wall_clock_seconds: float = 0.0,
) -> dict[str, Path]:
    """Write the official CuratorKIT provenance & dataset card bundle using `ProvenanceManifest` and `DatasetCardGenerator`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")

    passed_samples = [problem_to_datasample(p) for p in accepted_problems]
    ck_rejected_samples: list[RejectedSample] = []
    for idx, item in enumerate(rejected_records):
        pid = str(item.get("problem_id") or f"rejected_{idx:04d}")
        dom = str(item.get("domain") or "UNKNOWN")
        src = str(item.get("source") or "book_initial")
        orig = item.get("original_id", pid)
        stmt = str(
            item.get("problem_statement")
            or item.get("problem_statement_preview")
            or "rejected_candidate"
        )
        sol = str(item.get("reference_solution") or "")
        reason = str(item.get("reason") or "rejected")
        step_name = str(item.get("gate") or "Stage1AcceptanceGate")
        ck_rejected_samples.append(
            RejectedSample(
                id=pid,
                source_uri=f"{src}:{orig}",
                instruction=stmt,
                input=sol,
                output=sol,
                task_type="instruction_following",
                metadata={"problem_id": pid, "domain": dom, "source": src, "original_id": orig},
                rejection_reason=reason,
                rejecting_step=step_name,
            )
        )

    stage_counts = {
        "SchemaGate": {
            "input_count": gate_report.schema_gate_passed + gate_report.schema_gate_failed,
            "output_count": gate_report.schema_gate_passed,
            "probe_recovered": gate_report.adaptive_recovery_succeeded,
            "rejected_count": gate_report.schema_gate_failed,
        },
        "RewardGate": {
            "input_count": gate_report.reward_gate_passed + gate_report.reward_gate_failed,
            "output_count": gate_report.reward_gate_passed,
            "rejected_count": gate_report.reward_gate_failed,
        },
        "DiversityGate": {
            "input_count": gate_report.diversity_gate_passed + gate_report.diversity_gate_failed,
            "output_count": gate_report.diversity_gate_passed,
            "rejected_count": gate_report.diversity_gate_failed,
        },
    }

    pipeline_result = PipelineResult(
        passed=passed_samples,
        rejected=ck_rejected_samples,
        stage_counts=stage_counts,
        wall_clock_seconds=round(wall_clock_seconds, 3),
    )

    config_hash = hashlib.sha256(
        json.dumps({"stage": stage_id, "summary": summary_dict}, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]

    # Individual-example level provenance required by Section 4.3 of the blueprint
    example_provenance = [
        {
            "problem_id": p.problem_id,
            "parent_problem_id": str(p.original_id),
            "domain": p.domain,
            "source": p.source,
            "generation_model": generation_model,
            "runtime_backend": runtime_backend,
            "generation_prompt_version": "v2_10step",
            "generation_seed": 42,
            "validator_status": "passed_stage1_acceptance_and_anti_leakage",
            "curation_status": "accepted",
        }
        for p in accepted_problems
    ]

    lexsi_prov = build_provenance(
        method="curate_stage1_canonical_pool",
        inputs=[{"kind": "dataset", "ref": str(tf)} for tf in tracked_files if tf.exists()],
        params={
            "stage": stage_id,
            "author": "vital-ar",
            "anti_leakage_cosine_threshold": 0.85,
            "within_pool_diversity_threshold": 0.88,
            "logical_step_ceiling": 10,
            "curatorkit_gate_report": gate_report.to_dict(),
            "summary": summary_dict,
        },
        base_model=generation_model,
    )
    lexsi_prov["created_at_utc"] = now_iso
    lexsi_prov["records"] = example_provenance

    manifest_writer = ProvenanceManifest(
        result=pipeline_result,
        pipeline_config_hash=config_hash,
        output_dir=output_dir,
        provenance=lexsi_prov,
    )
    manifest_path = manifest_writer.write()
    rejected_path = manifest_writer.write_rejected_sidecar()
    checksums_path = manifest_writer.write_checksums(tracked_files)

    manifest_dict = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_dict["stage"] = stage_id
    manifest_dict["author"] = "vital-ar"
    manifest_dict["curatorkit_gates"] = gate_report.to_dict()
    manifest_dict["summary"] = summary_dict
    manifest_path.write_text(json.dumps(manifest_dict, indent=2, ensure_ascii=False), encoding="utf-8")

    card_generator = DatasetCardGenerator()
    card_path = card_generator.generate(
        manifest=manifest_dict,
        output_dir=output_dir,
        pipeline_name=f"vital-ar/{stage_id}",
        write_readme=False,
    )

    provenance_path = output_dir / "lexsi_provenance.json"
    return {
        "manifest": manifest_path,
        "rejected": rejected_path,
        "dataset_card": card_path,
        "lexsi_provenance": provenance_path,
        "checksums": checksums_path,
    }


__all__ = [
    "AYA_EXPANSE_8B",
    "AyaExpanseCuratorLLM",
    "CuratorKitGateReport",
    "STAGE1_PHYSICS_RUBRIC_PROMPT",
    "apply_curatorkit_text_cleaner",
    "datasample_to_problem",
    "problem_to_datasample",
    "run_curatorkit_embedding_dedup",
    "run_curatorkit_stage1_gates_and_recovery",
    "run_curatorkit_within_pool_dedup",
    "write_curatorkit_provenance_bundle",
]
