"""Stage 2.5 SFT Dataset Builder & CuratorKIT Provenance Exporter (`dataset_exporter.py`).

Constructs, curates with `CuratorKIT`, and exports all 5 progressive SFT datasets for `AlignTune`
training and separate Hugging Face Hub dataset repositories:
  1. `solver_e1_manual_det/` — Manual deterministic Python rules only, before `CuratorKIT` Part-1/Part-2 gates
  2. `solver_e2_curatorkit_nollm/` — `CuratorKIT` Part-1 deterministic gates (+ SymPy-verified subtopic synthetics)
  3. `solver_e3_canonical_llm/` — `CuratorKIT` Part-2 LLM-curated canonical pool
  4. `solver_e4_adversarial/` — Multi-turn hint-conditioned + minority-domain Turn-0 Solver SFT
  5. `verifier_e4_curated/` — Domain-balanced `[[APPROVE]]` + sanitized synthetic & real `[[CHALLENGE]]` Verifier SFT

Each dataset directory receives `train.jsonl` (plus `validation.jsonl` and `test.jsonl` for Solver datasets),
`manifest.json`, `lexsi_provenance.json`, Hugging Face `README.md`, and `checksums.txt` via `ProvenanceManifest`.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from curatorkit.gates.schema import SchemaGate
from curatorkit.manifest import ProvenanceManifest, build_provenance
from curatorkit.normalizers.clean import TextCleaner
from curatorkit.normalizers.dedup import ExactDeduplicator
from curatorkit.pipeline import PipelineResult
from curatorkit.schema import DataSample, RejectedSample

from adversarial_physics.data.curation.curatorkit_runner import _normalize_step_lines_and_answer
from adversarial_physics.data.curation.directive_leak_check import (
    sanitize_directive_text,
    validate_verifier_directive,
)
from adversarial_physics.data.schemas.problem import CanonicalProblem, load_problems_jsonl
from adversarial_physics.data.schemas.trajectory import (
    VERDICT_APPROVE,
    VERDICT_CHALLENGE,
    CandidateSolution,
    SolverTrajectory,
    VerifierTrajectory,
    format_solver_prompt,
    format_verifier_prompt,
    load_candidates_jsonl,
)
from adversarial_physics.utils.config import resolve_project_path
from adversarial_physics.utils.logging import get_logger
from adversarial_physics.utils.runtime import AYA_EXPANSE_8B

logger = get_logger("stage_2_dataset_exporter")

DEFAULT_E1_MANUAL_DET_POOL_PATH = "data/intermediate/01a_manual_deterministic_only_pool.jsonl"
DEFAULT_E2_CURATORKIT_NOLLM_POOL_PATH = "data/intermediate/01b_deterministic_curated_pool.jsonl"
DEFAULT_E3_CANONICAL_TRAIN_PATH = "data/processed/canonical_train.jsonl"
DEFAULT_VALIDATION_PATH = "data/splits/validation.jsonl"
DEFAULT_TEST_PATH = "data/splits/test.jsonl"
DEFAULT_CURATED_SOL2_PATH = "data/synthetic/02d_curated_solution2.jsonl"
DEFAULT_SURVIVING_FLAWED_PATH = "data/synthetic/02f_surviving_flawed_trajectories.jsonl"
DEFAULT_DIRECTIVES_INPUT_PATH = "data/synthetic/02g_generated_directives.jsonl"
DEFAULT_SFT_OUTPUT_ROOT = "data/sft"


@dataclass(frozen=True)
class Stage2ExportSummary:
    """Complete export summary across all 5 progressive SFT datasets."""

    solver_e1_manual_det_count: int
    solver_e2_curatorkit_nollm_count: int
    solver_e3_canonical_llm_count: int
    solver_e4_adversarial_total_count: int
    solver_e4_turn0_sol1_count: int
    solver_e4_turn0_minority_sol2_count: int
    solver_e4_turn1_one_hint_count: int
    solver_e4_turn2_two_hints_count: int
    verifier_e4_total_count: int
    verifier_e4_challenged_count: int
    verifier_e4_approved_count: int
    verifier_e4_approvals_by_domain: dict[str, int]
    verifier_e4_challenges_by_source: dict[str, int]
    directives_valid_count: int
    directives_invalid_count: int
    output_files: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _problems_to_turn0_solver_trajectories(
    problems: list[CanonicalProblem],
    *,
    dataset_tier: str,
) -> list[SolverTrajectory]:
    rows: list[SolverTrajectory] = []
    for p in problems:
        prompt = format_solver_prompt(p.problem_statement, [])
        rows.append(
            SolverTrajectory(
                trajectory_id=f"{dataset_tier}:{p.problem_id}:turn0",
                problem_id=p.problem_id,
                domain=p.domain,
                turn=0,
                accumulated_directives=[],
                prompt=prompt,
                target_output=p.reference_solution,
                source_candidate_ids=[f"{p.problem_id}_sol1"],
                dataset_tier=dataset_tier,
            )
        )
    return rows


def _write_jsonl_rows(rows: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


_REPO_URL = "https://github.com/vital-ar/adversarial_physics_problems_solver"
_REPO_BADGE_URL = (
    "https://img.shields.io/badge/GitHub-"
    "vital--ar%2Fadversarial__physics__problems__solver-blue?logo=github"
)
_SOURCE_LABELS: dict[str, str] = {
    "physgym": "PhysGym training seeds",
    "physgym_seed": "PhysGym training seeds",
    "book_initial": "Crowell textbook problems",
    "literature_curated": "Curated Crowell textbook problems",
    "synthetic_seed_subtopic": "Subtopic-grounded SymPy-verified synthetic problems",
}


@dataclass(frozen=True)
class _DatasetCardSpec:
    """Static metadata template for rendering a Hugging Face dataset card (`README.md`)."""

    folder_name: str
    title: str
    pretty_name: str
    tier_label: str
    task_categories: tuple[str, ...]
    tags: tuple[str, ...]
    v2_hf_repo_id: str
    target_model: str
    pipeline_section_title: str


@dataclass
class _CuratedDatasetResult:
    """In-memory handle for a curated dataset awaiting final card & checksum generation."""

    dataset_key: str
    train_path: Path
    dataset_dir: Path
    dataset_name: str
    curated_rows: list[dict[str, Any]]
    val_rows: list[dict[str, Any]] | None
    test_rows: list[dict[str, Any]] | None
    stage_counts: dict[str, Any]
    manifest_writer: ProvenanceManifest
    tracked_files: list[Path]


_DATASET_CARD_SPECS: dict[str, _DatasetCardSpec] = {
    "solver_e1_manual_det": _DatasetCardSpec(
        folder_name="solver_e1_manual_det",
        title="Adversarial Physics Solver — Experiment E1 (`solver_e1_manual_det`)",
        pretty_name="Adversarial Physics Solver E1 — Manual Deterministic Rules Only (Pre-CuratorKIT Baseline)",
        tier_label="**Dataset 1 (`Model 1 / E1`)**",
        task_categories=("text-generation", "question-answering"),
        tags=(
            "physics",
            "sft",
            "curatorkit",
            "aligntune",
            "auditkit",
            "lexsi",
            "tiny-aya-global",
            "ablation-e1",
        ),
        v2_hf_repo_id="vital-ar/physics-solver-e1-manual-rules-v2",
        target_model="`CohereLabs/tiny-aya-global` (fine-tuned via `AlignTune`)",
        pipeline_section_title="Manual Deterministic Rules Applied (Pre-`CuratorKIT` Baseline)",
    ),
    "solver_e2_curatorkit_nollm": _DatasetCardSpec(
        folder_name="solver_e2_curatorkit_nollm",
        title="Adversarial Physics Solver — Experiment E2 (`solver_e2_curatorkit_nollm`)",
        pretty_name="Adversarial Physics Solver E2 — CuratorKIT Part-1 Curation & SymPy-Verified Synthetic Augmentation",
        tier_label="**Dataset 2 (`Model 2 / E2`)**",
        task_categories=("text-generation", "question-answering"),
        tags=(
            "physics",
            "sft",
            "curatorkit",
            "aligntune",
            "auditkit",
            "lexsi",
            "aya-expanse-8b",
            "tiny-aya-global",
            "sympy",
            "ablation-e2",
        ),
        v2_hf_repo_id="vital-ar/physics-solver-e2-curatorkit-nollm-v2",
        target_model="`CohereLabs/tiny-aya-global` (fine-tuned via `AlignTune`)",
        pipeline_section_title="`CuratorKIT` Part-1 Gates & Subtopic-Grounded Synthetic Augmentation",
    ),
    "solver_e3_canonical_llm": _DatasetCardSpec(
        folder_name="solver_e3_canonical_llm",
        title="Adversarial Physics Solver — Experiment E3 (`solver_e3_canonical_llm`)",
        pretty_name="Adversarial Physics Solver E3 — CuratorKIT Part-2 LLM-Curated Canonical SFT",
        tier_label="**Dataset 3 (`Model 3 / E3`)**",
        task_categories=("text-generation", "question-answering"),
        tags=(
            "physics",
            "sft",
            "curatorkit",
            "aligntune",
            "auditkit",
            "lexsi",
            "aya-expanse-8b",
            "tiny-aya-global",
            "ablation-e3",
        ),
        v2_hf_repo_id="vital-ar/physics-solver-e3-canonical-llm-v2",
        target_model="`CohereLabs/tiny-aya-global` (fine-tuned via `AlignTune`)",
        pipeline_section_title="`CuratorKIT` Part-2 LLM Quality Gates & Adaptive Recovery",
    ),
    "solver_e4_adversarial": _DatasetCardSpec(
        folder_name="solver_e4_adversarial",
        title="Adversarial Physics Solver — Experiment E4 (`solver_e4_adversarial`)",
        pretty_name="Adversarial Physics Solver E4 — Multi-Turn Hint-Conditioned & Minority-Enriched SFT",
        tier_label="**Dataset 4A (`Model 4A / E4 Solver`)**",
        task_categories=("text-generation", "question-answering"),
        tags=(
            "physics",
            "sft",
            "curatorkit",
            "aligntune",
            "auditkit",
            "lexsi",
            "aya-expanse-8b",
            "tiny-aya-global",
            "adversarial-reasoning",
            "ablation-e4",
        ),
        v2_hf_repo_id="vital-ar/physics-solver-e4-adversarial-v2",
        target_model="`CohereLabs/tiny-aya-global` (**Model A — Solver**, fine-tuned via `AlignTune`)",
        pipeline_section_title="Multi-Turn Adversarial Trajectory Construction & Validation",
    ),
    "verifier_e4_curated": _DatasetCardSpec(
        folder_name="verifier_e4_curated",
        title="Adversarial Physics Verifier — Experiment E4 (`verifier_e4_curated`)",
        pretty_name="Adversarial Physics Verifier E4 — Balanced Critique, Challenge Directives & Anti-Leak SFT",
        tier_label="**Dataset 4B (`Model 4B / E4 Verifier`)**",
        task_categories=("text-generation", "text-classification"),
        tags=(
            "physics",
            "sft",
            "curatorkit",
            "aligntune",
            "auditkit",
            "lexsi",
            "aya-expanse-8b",
            "tiny-aya-global",
            "verifier",
            "adversarial-reasoning",
            "ablation-e4",
        ),
        v2_hf_repo_id="vital-ar/physics-verifier-e4-curated-v2",
        target_model="`CohereLabs/tiny-aya-global` (**Model B — Verifier**, fine-tuned via `AlignTune`)",
        pipeline_section_title="Verifier Trajectory Construction & Anti-Leak Sanitization",
    ),
}


def _pct(part: int, total: int) -> str:
    return "0.0%" if total <= 0 else f"{(part / total) * 100:.1f}%"


def _render_markdown_table(
    headers: list[str],
    alignments: list[str],
    rows: list[list[str]],
) -> str:
    align_tokens = {"left": ":---", "center": ":---:", "right": "---:"}
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(align_tokens.get(a, ":---") for a in alignments) + " |"
    body_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, sep_line, *body_lines])


def _truncate_text_preview(text: str, max_chars: int = 240) -> str:
    cleaned = text.strip()
    if len(cleaned) <= max_chars:
        return cleaned
    tail = ""
    last_line = cleaned.splitlines()[-1].strip()
    if last_line.upper().startswith("ANSWER:") or last_line in {VERDICT_APPROVE, VERDICT_CHALLENGE}:
        tail = f"\n...\n{last_line}"
    head_budget = max(60, max_chars - len(tail) - 4)
    return cleaned[:head_budget].rstrip() + "..." + tail


def _format_source_mix(source_counts: Counter[str]) -> str:
    if not source_counts:
        return "Curated training rows"
    parts = [
        f"`{count:,}` {_SOURCE_LABELS.get(src, src)}"
        for src, count in sorted(source_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    return " + ".join(parts)


def _summarize_stage_gates(stage_counts: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    tc = stage_counts.get("TextCleaner") or {}
    sg = stage_counts.get("SchemaGate") or {}
    ed = stage_counts.get("ExactDeduplicator") or {}
    ag = stage_counts.get("AnswerSentinelGate") or {}
    if tc:
        lines.append(
            f"- **`curatorkit.normalizers.clean.TextCleaner`:** Normalized whitespace and Unicode math/control characters across `{tc.get('input_count', 0):,}` rows."
        )
    if sg:
        lines.append(
            f"- **`curatorkit.gates.schema.SchemaGate`** (`min_tokens = 20`, `max_tokens = 8192`): `{sg.get('output_count', 0):,}` passed, `{sg.get('rejected_count', 0):,}` rejected."
        )
    if ed:
        lines.append(
            f"- **`curatorkit.normalizers.dedup.ExactDeduplicator`:** `{ed.get('output_count', 0):,}` unique records retained (`{ed.get('duplicates_removed', 0):,}` exact duplicates removed)."
        )
    if ag:
        lines.append(
            f"- **`AnswerSentinelGate`:** `{ag.get('output_count', 0):,}` passed with canonical `ANSWER:` sentinel (`{ag.get('rejected_count', 0):,}` rejected)."
        )
    return lines


def _build_schema_example_json(
    rows: list[dict[str, Any]],
    *,
    is_verifier: bool,
) -> str:
    if not rows:
        return "{}"
    sample = rows[0]
    if is_verifier:
        preferred = next((r for r in rows if r.get("verdict") == VERDICT_CHALLENGE), sample)
    else:
        preferred = next((r for r in rows if r.get("turn") == 2), sample)

    excluded_aliases = {"instruction", "input", "output"}
    compact: dict[str, Any] = {}
    for k, v in preferred.items():
        if k in excluded_aliases:
            continue
        if isinstance(v, str):
            compact[k] = _truncate_text_preview(v)
        elif isinstance(v, list):
            compact[k] = [_truncate_text_preview(str(item), 160) for item in v]
        else:
            compact[k] = v
    return json.dumps(compact, indent=2, ensure_ascii=False)


def _resolve_card_repo_id(spec: _DatasetCardSpec, out_root: Path, base_dataset_name: str) -> str:
    if "v2" in out_root.name.lower():
        return spec.v2_hf_repo_id
    return base_dataset_name


def _build_dataset_card_markdown(
    *,
    result: _CuratedDatasetResult,
    out_root: Path,
    tier_counts: dict[str, int],
    source_counts: Counter[str],
    upstream_domain_counts: Counter[str] | None = None,
    upstream_label: str = "",
    e4_extra: dict[str, int] | None = None,
    verifier_stats: dict[str, int] | None = None,
) -> str:
    spec = _DATASET_CARD_SPECS[result.dataset_key]
    hf_repo_id = _resolve_card_repo_id(spec, out_root, result.dataset_name)
    has_val = result.val_rows is not None
    has_test = result.test_rows is not None
    train_count = len(result.curated_rows)
    val_count = len(result.val_rows) if result.val_rows is not None else 0
    test_count = len(result.test_rows) if result.test_rows is not None else 0

    data_files_yaml = ["      - split: train", "        path: train.jsonl"]
    if has_val:
        data_files_yaml.extend(["      - split: validation", "        path: validation.jsonl"])
    if has_test:
        data_files_yaml.extend(["      - split: test", "        path: test.jsonl"])

    yaml_lines = [
        "---",
        "license: cc-by-sa-4.0",
        "task_categories:",
        *[f"  - {tc}" for tc in spec.task_categories],
        "language:",
        "  - en",
        "tags:",
        *[f"  - {tag}" for tag in spec.tags],
        f'pretty_name: "{spec.pretty_name}"',
        "configs:",
        "  - config_name: default",
        "    default: true",
        "    data_files:",
        *data_files_yaml,
        "---",
    ]

    e1_cnt = tier_counts.get("solver_e1_manual_det", 0)
    e2_cnt = tier_counts.get("solver_e2_curatorkit_nollm", 0)
    e3_cnt = tier_counts.get("solver_e3_canonical_llm", 0)
    e4s_cnt = tier_counts.get("solver_e4_adversarial", 0)
    e4v_cnt = tier_counts.get("verifier_e4_curated", 0)
    ablation_summary = (
        f"**`E1`** (`{e1_cnt:,}` problems) $\\rightarrow$ "
        f"**`E2`** (`{e2_cnt:,}` problems) $\\rightarrow$ "
        f"**`E3`** (`{e3_cnt:,}` problems) $\\rightarrow$ "
        f"**`E4 Solver`** (`{e4s_cnt:,}` trajectories) / "
        f"**`E4 Verifier`** (`{e4v_cnt:,}` trajectories)"
    )

    synth_cnt = source_counts.get("synthetic_seed_subtopic", 0)
    source_summary = _format_source_mix(source_counts)

    if result.dataset_key == "solver_e1_manual_det":
        overview_body = (
            f"This dataset represents the **pre-`CuratorKIT` Part-1/Part-2 manual deterministic baseline**: it contains "
            f"university-level physics problems passing deterministic Python splitting, regex hygiene, and sentinel "
            f"rules before `CuratorKIT`'s cross-split embedding anti-leakage deduplicator, MinHash deduplicator, "
            f"diversity gate, or teacher LLM curation are applied.\n\n"
            f"- **Role in Ablation Ladder:** {ablation_summary}.\n"
            f"- **Target Student Model:** {spec.target_model}.\n"
            f"- **Evaluation Benchmark:** Evaluated with `AuditKIT` on the locked `{val_count}`-problem Validation and `{test_count}`-problem Test splits from *PhysGym*."
        )
        pipeline_bullets = [
            "1. **PhysGym Seed & Crowell Textbook Extraction:** Retained bounded PhysGym training seeds and split independent single-target textbook subproblems.",
            "2. **Manual Deterministic Regex Hygiene:** Stripped PDF running headers/footers, enforced quantitative single-target formatting and explicit `ANSWER:` sentinels, and filtered figure-dependent or external cross-reference items.",
            "3. **What Is Still Present (Addressed in `E2`/`E3`):** Cross-problem phrasing overlap, un-augmented domain coverage gaps, and textbook solutions lacking full `CuratorKIT` `HallucinationGate`/`RewardGate` normalization.",
        ]
    elif result.dataset_key == "solver_e2_curatorkit_nollm":
        synth_note = (
            f"augmented with `{synth_cnt:,}` SymPy-verified subtopic-grounded synthetic problems generated by `CohereForAI/aya-expanse-8b` from extracted Crowell textbook subtopic chunks"
            if synth_cnt > 0
            else "without teacher LLM rewriting"
        )
        overview_body = (
            f"Derived from the deterministic `CuratorKIT` Part-1 filtering of the `{e1_cnt:,}`-problem `E1` pool and "
            f"{synth_note}, **`solver_e2_curatorkit_nollm`** isolates the impact of strict deterministic `CuratorKIT` "
            f"deduplication and verified symbolic augmentation **before** Stage-1 Part-2 teacher LLM quality gates "
            f"(`HallucinationGate`, `RewardGate`, `RewardRefiner`) are applied in `E3`.\n\n"
            f"- **Role in Ablation Ladder:** {ablation_summary}.\n"
            f"- **Target Student Model:** {spec.target_model}.\n"
            f"- **Evaluation Benchmark:** Evaluated with `AuditKIT` on the locked `{val_count}`-problem Validation and `{test_count}`-problem Test splits."
        )
        pipeline_bullets = [
            "1. **`curatorkit.normalizers.clean.TextCleaner` & `SchemaGate`:** Normalized Unicode math symbols/whitespace and enforced strict problem (`12–450` tokens) and solution (`20–1200` tokens) budgets with single quantitative `ANSWER:` sentinels and zero figure cues.",
            "2. **`curatorkit.normalizers.embedding_dedup.EmbeddingDeduplicator` (`threshold = 0.80`):** Enforced strict anti-leakage semantic exclusion against the locked Validation and Test splits (`sentence-transformers/all-MiniLM-L6-v2`).",
            "3. **`curatorkit.normalizers.dedup.ExactDeduplicator` & `MinHashDeduplicator` (`threshold = 0.75`):** Purged exact and near-duplicate textbook formulations within the pool.",
            "4. **`curatorkit.gates.diversity.DiversityGate` (`similarity_threshold = 0.82`, `coverage_field = \"domain\"`):** Balanced dense textbook clusters while preserving minority domain coverage.",
            (
                f"5. **SymPy-Verified Subtopic Synthetic Augmentation:** Added `{synth_cnt:,}` subtopic-grounded synthetic problems (`CohereForAI/aya-expanse-8b`) whose mathematical answers passed `SymPy` symbolic/numeric verification and all Part-1 deduplication and anti-leakage checks."
                if synth_cnt > 0
                else "5. **Zero Teacher Rewriting:** Retained raw curated formulations (`0` synthetic additions)."
            ),
        ]
    elif result.dataset_key == "solver_e3_canonical_llm":
        overview_body = (
            f"**`{spec.folder_name}`** filters and canonicalizes the `{e2_cnt:,}`-problem `E2` pool through **`CuratorKIT` Part-2 LLM Quality Gates** powered by `CohereForAI/aya-expanse-8b`, yielding `{train_count:,}` verified `Turn 0` physics training examples.\n\n"
            f"- **Role in Ablation Ladder:** {ablation_summary}.\n"
            f"- **Teacher / Judge Model:** `CohereForAI/aya-expanse-8b`.\n"
            f"- **Target Student Model:** {spec.target_model}."
        )
        pipeline_bullets = [
            "1. **`curatorkit.gates.hallucination.HallucinationGate` (`threshold = 0.85`):** Verified faithfulness and derivation grounding against each source problem and reference derivation.",
            "2. **`curatorkit.gates.reward.RewardGate` (`threshold = 0.82`, `on_judge_error = \"reject\"`):** Scored every problem–solution pair across a 6-dimension physics quality rubric.",
            "3. **`curatorkit.diagnostic.reward_refine.RewardRefiner` (`max_tokens = 1536`) + `PostRefinementGate`:** Regenerated borderline derivations with `STAGE1_PHYSICS_REFINE_PROMPT` and enforced deterministic contract checks (canonical `ANSWER:` sentinel, balanced `\\left`/`\\right` delimiters, single-target cleanliness, and zero prompt leakage) before re-judging.",
            "4. **`curatorkit.gates.diversity.DiversityGate` (`threshold = 0.82`, `coverage_field = \"domain\"`):** Enforced domain coverage and semantic non-redundancy across the final canonical training set.",
        ]
    elif result.dataset_key == "solver_e4_adversarial":
        ex = e4_extra or {}
        overview_body = (
            f"**`{spec.folder_name}`** is the final **Stage 2 Solver SFT dataset (`{train_count:,}` training trajectories)** used to train **Model A (Solver)** to:\n"
            f"1. Solve university physics problems zero-shot at **Turn 0** (`{ex.get('turn0_sol1', 0):,}` canonical Reference Solution 1 + `{ex.get('turn0_sol2', 0):,}` verified alternative derivations in minority domains).\n"
            f"2. Re-solve problems from scratch at **Turn 1 (`1` hint, `{ex.get('turn1', 0):,}` rows)** and **Turn 2 (`2` hints, `{ex.get('turn2', 0):,}` rows)** conditioned on self-contained `[[CHALLENGE]]` directives from **Model B (Verifier)**.\n\n"
            f"- **Role in Ablation Ladder:** {ablation_summary}.\n"
            f"- **Target Student Model:** {spec.target_model}."
        )
        pipeline_bullets = [
            "1. **Canonical & Minority-Domain Turn 0:** Paired all canonical problems with Reference Solution 1 and added verified alternative physical derivations (`Solution 2`) in `OPTICS`, `RELATIVITY`, and `THERMODYNAMICS` verified against ground-truth answers.",
            "2. **50% / 50% Multi-Turn Hint Split:** Partitioned the canonical pool cleanly so half receive `1` non-leaking `[[CHALLENGE]]` directive (`Turn 1`) and half receive `2` accumulated distinct directives (`Turn 2`), always targeting clean Reference Solution 1.",
            "3. **Anti-Leak Directive Validation (`validate_verifier_directive`):** Stripped and rejected any directive leaking the final answer, intermediate target values, or meta evaluation wording.",
        ]
    else:
        vs = verifier_stats or {}
        overview_body = (
            f"**`{spec.folder_name}`** is the final **Stage 2 Verifier SFT dataset (`{train_count:,}` trajectories)** used to train **Model B (Verifier)** to inspect a student's proposed physics derivation and emit either:\n"
            f"- A domain-calibrated acceptance ending with **`[[APPROVE]]`** (`{vs.get('approved', 0):,}` examples), or\n"
            f"- A concise, self-contained physical/mathematical directive ending with **`[[CHALLENGE]]`** (`{vs.get('challenged', 0):,}` examples across synthetic and real flawed student attempts).\n\n"
            f"- **Role in Ablation Ladder:** Paired companion critic for **`E4 Solver`** (`{e4s_cnt:,}` trajectories) inside the two-model adversarial loop.\n"
            f"- **Target Student Model:** {spec.target_model}."
        )
        pipeline_bullets = [
            "1. **Part 1 (`[[APPROVE]]`):** Domain-balanced verified correct derivations paired with `[[APPROVE]]` verdicts (with normalized step formatting and canonical `ANSWER:` lines).",
            "2. **Part 2 (Synthetic `[[CHALLENGE]]`):** Physics taxonomy flaw injections generated by `CohereForAI/aya-expanse-8b`, sanitized by `_sanitize_flawed_trajectory_text` to strip meta-giveaway words (`flaw`, `incorrect`, `injected`, etc.) and trailing answer annotations.",
            "3. **Part 3 (Real `[[CHALLENGE]]`):** Authentic flawed zero-shot student attempts generated by `CohereLabs/tiny-aya-global` (`is_correct == False`), filtered for semantic diversity and paired with non-leaking directives.",
            "4. **Directive Sanitization & Validation (`validate_verifier_directive`):** Guaranteed that every directive contains zero ground-truth answer leaks, ends cleanly with `\\n[[CHALLENGE]]` or `\\n[[APPROVE]]`, and stays within word budgets.",
        ]

    # Build Split Table
    split_rows = [
        [
            "`train`",
            "`train.jsonl`",
            f"**{train_count:,}**",
            source_summary
            if result.dataset_key in {"solver_e1_manual_det", "solver_e2_curatorkit_nollm", "solver_e3_canonical_llm"}
            else (
                f"`{e4_extra.get('turn0', 0):,}` Turn-0 + `{e4_extra.get('turn1', 0):,}` Turn-1 + `{e4_extra.get('turn2', 0):,}` Turn-2"
                if e4_extra is not None
                else f"`{(verifier_stats or {}).get('approved', 0):,}` `[[APPROVE]]` + `{(verifier_stats or {}).get('challenged', 0):,}` `[[CHALLENGE]]`"
            ),
        ]
    ]
    if has_val:
        split_rows.append(["`validation`", "`validation.jsonl`", f"**{val_count:,}**", "Locked PhysGym (`validation` split)"])
    if has_test:
        split_rows.append(["`test`", "`test.jsonl`", f"**{test_count:,}**", "Locked PhysGym (`test` split)"])
    splits_table = _render_markdown_table(
        ["Split", "File", "Examples", "Composition / Source"],
        ["left", "left", "right", "left"],
        split_rows,
    )

    # Build Domain Breakdown Table
    dom_counts = Counter(str(r.get("domain") or "UNKNOWN") for r in result.curated_rows)
    all_dom_keys = set(dom_counts.keys()) | (set(upstream_domain_counts.keys()) if upstream_domain_counts else set())
    domains_sorted = sorted(all_dom_keys, key=lambda d: (-dom_counts.get(d, 0), d))

    if result.dataset_key == "solver_e4_adversarial":
        dom_rows: list[list[str]] = []
        tot_t0_s1 = tot_t0_s2 = tot_t1 = tot_t2 = 0
        for dom in domains_sorted:
            d_rows = [r for r in result.curated_rows if r.get("domain") == dom]
            t0_s1 = sum(1 for r in d_rows if r.get("turn") == 0 and str(r.get("trajectory_id", "")).endswith(":turn0_sol1"))
            t0_s2 = sum(1 for r in d_rows if r.get("turn") == 0 and str(r.get("trajectory_id", "")).endswith(":turn0_sol2"))
            t1 = sum(1 for r in d_rows if r.get("turn") == 1)
            t2 = sum(1 for r in d_rows if r.get("turn") == 2)
            dtot = len(d_rows)
            tot_t0_s1 += t0_s1
            tot_t0_s2 += t0_s2
            tot_t1 += t1
            tot_t2 += t2
            dom_rows.append([f"`{dom}`", f"{t0_s1:,}", f"{t0_s2:,}", f"{t1:,}", f"{t2:,}", f"**{dtot:,}**", _pct(dtot, train_count)])
        dom_rows.append(["**Total**", f"**{tot_t0_s1:,}**", f"**{tot_t0_s2:,}**", f"**{tot_t1:,}**", f"**{tot_t2:,}**", f"**{train_count:,}**", "**100.0%**"])
        domain_table = _render_markdown_table(
            ["Physics Domain", "Turn 0 (`Sol 1`)", "Turn 0 (`Sol 2`)", "Turn 1 (`1 Hint`)", "Turn 2 (`2 Hints`)", "Total", "Share"],
            ["left", "right", "right", "right", "right", "right", "right"],
            dom_rows,
        )
    elif result.dataset_key == "verifier_e4_curated":
        dom_rows = []
        tot_p1 = tot_p2 = tot_p3 = 0
        for dom in domains_sorted:
            d_rows = [r for r in result.curated_rows if r.get("domain") == dom]
            p1 = sum(1 for r in d_rows if r.get("part") == "part_1_domain_balanced_approved")
            p2 = sum(1 for r in d_rows if r.get("part") == "part_2_synthetic_aya8b_challenged")
            p3 = sum(1 for r in d_rows if r.get("part") == "part_3_real_tiny_aya_challenged")
            dtot = len(d_rows)
            tot_p1 += p1
            tot_p2 += p2
            tot_p3 += p3
            dom_rows.append([f"`{dom}`", f"{p1:,}", f"{p2:,}", f"{p3:,}", f"**{dtot:,}**", _pct(dtot, train_count)])
        dom_rows.append(["**Total**", f"**{tot_p1:,}**", f"**{tot_p2:,}**", f"**{tot_p3:,}**", f"**{train_count:,}**", "**100.0%**"])
        domain_table = _render_markdown_table(
            ["Physics Domain", "`[[APPROVE]]` (Part 1)", "`[[CHALLENGE]]` Synth (Part 2)", "`[[CHALLENGE]]` Tiny-Aya (Part 3)", "Total", "Share"],
            ["left", "right", "right", "right", "right", "right"],
            dom_rows,
        )
    elif upstream_domain_counts is not None and upstream_label:
        up_total = sum(upstream_domain_counts.values())
        dom_rows = []
        for dom in domains_sorted:
            cnt = dom_counts[dom]
            up_cnt = upstream_domain_counts.get(dom, 0)
            delta = cnt - up_cnt
            delta_str = f"{delta:+,}" if delta != 0 else "0"
            dom_rows.append([f"`{dom}`", f"{up_cnt:,}", delta_str, f"**{cnt:,}**", _pct(cnt, train_count)])
        tot_delta = train_count - up_total
        dom_rows.append(["**Total**", f"**{up_total:,}**", f"**{tot_delta:+,}**", f"**{train_count:,}**", "**100.0%**"])
        domain_table = _render_markdown_table(
            ["Physics Domain", f"{upstream_label} Input", "Net Change", "Retained", "Share"],
            ["left", "right", "right", "right", "right"],
            dom_rows,
        )
    else:
        dom_rows = [[f"`{dom}`", f"{dom_counts[dom]:,}", _pct(dom_counts[dom], train_count)] for dom in domains_sorted]
        dom_rows.append(["**Total**", f"**{train_count:,}**", "**100.0%**"])
        domain_table = _render_markdown_table(
            ["Physics Domain", "Count", "Share"],
            ["left", "right", "right"],
            dom_rows,
        )

    gate_lines = _summarize_stage_gates(result.stage_counts)
    gates_block = "\n\n### Export-Time `CuratorKIT` Integrity Gates\n\n" + "\n".join(gate_lines) if gate_lines else ""

    schema_json = _build_schema_example_json(
        result.curated_rows,
        is_verifier=(result.dataset_key == "verifier_e4_curated"),
    )
    split_list_text = ", ".join(
        f"`{name}`"
        for name, present in (("train.jsonl", True), ("validation.jsonl", has_val), ("test.jsonl", has_test))
        if present
    )

    sections = [
        "\n".join(yaml_lines),
        f"# {spec.title}",
        f"[![GitHub Repository]({_REPO_BADGE_URL})]({_REPO_URL})\n\n"
        f"**Codebase & Pipeline Repository:** [vital-ar/adversarial_physics_problems_solver]({_REPO_URL})  \n"
        f"**Hugging Face Dataset ID:** `{hf_repo_id}`",
        f"## Dataset Overview\n\n**`{spec.folder_name}`** is {spec.tier_label} in the **Lexsi AI (`CuratorKIT` + `AlignTune` + `AuditKIT`)** progressive data-curation ablation study for the *Adversarial Physics Problem Solver–Verifier* system.\n\n{overview_body}",
        f"## Splits & Domain Distribution\n\n{splits_table}\n\n### Training Split Domain Breakdown (`{train_count:,}` Examples)\n\n{domain_table}",
        f"## {spec.pipeline_section_title}\n\n" + "\n".join(pipeline_bullets) + gates_block,
        f"## Schema\n\nEach record in {split_list_text} follows the canonical schema below (with `instruction`, `input`, and `output` convenience columns also included for direct `AlignTune` SFT loading):\n\n```json\n{schema_json}\n```",
        "## Provenance & Integrity Files\n\n"
        "- `manifest.json` — `CuratorKIT` stage gate counts and deduplication report.\n"
        "- `lexsi_provenance.json` — `lexsi.provenance/1` lineage record consumed by `AlignTune` and `AuditKIT`.\n"
        "- `checksums.txt` — SHA-256 digests of all split, metadata, and `README.md` files.\n"
        f"- **Project Repository & Codebase:** [{_REPO_URL}]({_REPO_URL})",
    ]
    return "\n\n".join(sections[:4]) + "\n\n---\n\n" + "\n\n---\n\n".join(sections[4:]) + "\n"


def _curate_and_write_dataset_artifacts(
    *,
    dataset_key: str,
    dataset_dir: Path,
    dataset_name: str,
    method: str,
    rows: list[dict[str, Any]],
    prompt_field: str = "prompt",
    output_field: str = "target_output",
    id_field: str = "trajectory_id",
    base_model: str | None = AYA_EXPANSE_8B,
    input_refs: list[Path] | None = None,
    extra_params: dict[str, Any] | None = None,
    val_rows: list[dict[str, Any]] | None = None,
    test_rows: list[dict[str, Any]] | None = None,
    require_answer_sentinel: bool = False,
) -> _CuratedDatasetResult:
    """Run CuratorKIT `TextCleaner`, `SchemaGate`, `ExactDeduplicator`, and optional `AnswerSentinelGate`
    over a dataset's rows, write `train.jsonl` (plus `validation.jsonl`/`test.jsonl` when provided),
    and emit `manifest.json` and `lexsi_provenance.json` via `ProvenanceManifest`.
    (Note: MinHashDeduplicator is intentionally applied earlier at the trajectory candidate stage in
    curatorkit_runner.py, not here, to prevent collapsing legitimate multi-turn prompt variants
    such as turn0, turn1, and turn2 of the same underlying physics problem).
    """
    t0 = time.monotonic()
    dataset_dir.mkdir(parents=True, exist_ok=True)

    samples = [
        DataSample(
            id=str(r.get(id_field) or f"row_{idx:04d}"),
            source_uri=f"{dataset_name}:{r.get('problem_id', idx)}",
            instruction=str(r.get(prompt_field) or ""),
            input=str(r.get("proposed_solution") or ""),
            output=str(r.get(output_field) or ""),
            task_type="instruction_following",
            metadata=dict(r),
        )
        for idx, r in enumerate(rows)
    ]

    cleaner = TextCleaner(fields=["instruction", "output"])
    schema_gate = SchemaGate(required_fields=["instruction", "output"], min_tokens=20, max_tokens=8192)
    exact_dedup = ExactDeduplicator()

    cleaned = cleaner.run(samples)
    passed, rejected = schema_gate.run(cleaned)
    deduped = exact_dedup.run(passed)
    exact_dups_removed = len(passed) - len(deduped)

    if require_answer_sentinel:
        sentinel_passed: list[DataSample] = []
        for s in deduped:
            if "ANSWER:" in str(s.output):
                sentinel_passed.append(s)
            else:
                rejected.append(
                    RejectedSample(
                        id=s.id,
                        source_uri=s.source_uri,
                        instruction=s.instruction,
                        input=s.input,
                        output=s.output,
                        task_type=s.task_type,
                        metadata=dict(s.metadata or {}),
                        rejection_reason="missing ANSWER: sentinel line",
                        rejecting_step="AnswerSentinelGate",
                    )
                )
        sentinel_rejected = len(deduped) - len(sentinel_passed)
        deduped = sentinel_passed
    else:
        sentinel_rejected = 0

    # Sync cleaned instruction/output back into row metadata
    curated_rows: list[dict[str, Any]] = []
    for s in deduped:
        row_dict = dict(s.metadata or {})
        row_dict[prompt_field] = s.instruction.strip()
        row_dict[output_field] = s.output.strip()
        curated_rows.append(row_dict)

    train_path = _write_jsonl_rows(curated_rows, dataset_dir / "train.jsonl")
    tracked_files = [train_path]

    if val_rows is not None:
        val_path = _write_jsonl_rows(val_rows, dataset_dir / "validation.jsonl")
        tracked_files.append(val_path)
    if test_rows is not None:
        test_path = _write_jsonl_rows(test_rows, dataset_dir / "test.jsonl")
        tracked_files.append(test_path)

    stage_counts = {
        "TextCleaner": {"input_count": len(samples), "output_count": len(cleaned)},
        "SchemaGate": {
            "input_count": len(cleaned),
            "output_count": len(passed),
            "rejected_count": len(rejected),
        },
        "ExactDeduplicator": {
            "input_count": len(passed),
            "output_count": len(passed) - exact_dups_removed,
            "duplicates_removed": exact_dups_removed,
        },
        **(
            {
                "AnswerSentinelGate": {
                    "input_count": len(passed) - exact_dups_removed,
                    "output_count": len(deduped),
                    "rejected_count": sentinel_rejected,
                }
            }
            if require_answer_sentinel
            else {}
        ),
    }

    pipeline_res = PipelineResult(
        passed=deduped,
        rejected=list(rejected),
        stage_counts=stage_counts,
        wall_clock_seconds=time.monotonic() - t0,
    )

    cfg_hash = hashlib.sha256(
        json.dumps({"dataset": dataset_name, "method": method, "rows": len(curated_rows)}, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]

    prov = build_provenance(
        method=method,
        inputs=[{"kind": "dataset", "ref": p.name} for p in (input_refs or []) if p.exists()],
        params={
            "dataset_name": dataset_name,
            "author": "vital-ar",
            "train_count": len(curated_rows),
            "validation_count": len(val_rows) if val_rows is not None else 0,
            "test_count": len(test_rows) if test_rows is not None else 0,
            **(extra_params or {}),
        },
        base_model=base_model,
    )

    manifest_writer = ProvenanceManifest(
        result=pipeline_res,
        pipeline_config_hash=cfg_hash,
        output_dir=dataset_dir,
        provenance=prov,
    )
    manifest_path = manifest_writer.write()
    tracked_files.extend([manifest_path, dataset_dir / "lexsi_provenance.json"])
    return _CuratedDatasetResult(
        dataset_key=dataset_key,
        train_path=train_path,
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        curated_rows=curated_rows,
        val_rows=val_rows,
        test_rows=test_rows,
        stage_counts=stage_counts,
        manifest_writer=manifest_writer,
        tracked_files=tracked_files,
    )


def _write_dataset_readme_and_checksums(
    result: _CuratedDatasetResult,
    readme_markdown: str,
) -> None:
    """Write the dataset's Hugging Face `README.md` card and finalize SHA-256 `checksums.txt`."""
    readme_path = result.dataset_dir / "README.md"
    readme_path.write_text(readme_markdown, encoding="utf-8")
    result.manifest_writer.write_checksums([*result.tracked_files, readme_path])


def _curate_and_write_dataset_provenance(
    *,
    dataset_dir: Path,
    dataset_name: str,
    method: str,
    rows: list[dict[str, Any]],
    prompt_field: str = "prompt",
    output_field: str = "target_output",
    id_field: str = "trajectory_id",
    base_model: str | None = AYA_EXPANSE_8B,
    input_refs: list[Path] | None = None,
    extra_params: dict[str, Any] | None = None,
    val_rows: list[dict[str, Any]] | None = None,
    test_rows: list[dict[str, Any]] | None = None,
    require_answer_sentinel: bool = False,
) -> tuple[Path, list[dict[str, Any]]]:
    """Curate rows and write `manifest.json`, `lexsi_provenance.json`, and `checksums.txt`."""
    res = _curate_and_write_dataset_artifacts(
        dataset_key=dataset_dir.name,
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        method=method,
        rows=rows,
        prompt_field=prompt_field,
        output_field=output_field,
        id_field=id_field,
        base_model=base_model,
        input_refs=input_refs,
        extra_params=extra_params,
        val_rows=val_rows,
        test_rows=test_rows,
        require_answer_sentinel=require_answer_sentinel,
    )
    res.manifest_writer.write_checksums(res.tracked_files)
    return res.train_path, res.curated_rows


def export_all_stage2_sft_datasets(
    *,
    e1_pool_path: str | Path = DEFAULT_E1_MANUAL_DET_POOL_PATH,
    e2_pool_path: str | Path = DEFAULT_E2_CURATORKIT_NOLLM_POOL_PATH,
    e3_pool_path: str | Path = DEFAULT_E3_CANONICAL_TRAIN_PATH,
    validation_path: str | Path = DEFAULT_VALIDATION_PATH,
    test_path: str | Path = DEFAULT_TEST_PATH,
    curated_sol2_path: str | Path = DEFAULT_CURATED_SOL2_PATH,
    surviving_flawed_path: str | Path = DEFAULT_SURVIVING_FLAWED_PATH,
    directives_input_path: str | Path = DEFAULT_DIRECTIVES_INPUT_PATH,
    sft_output_root: str | Path = DEFAULT_SFT_OUTPUT_ROOT,
) -> Stage2ExportSummary:
    """Build, curate with CuratorKIT, and export all 5 progressive SFT datasets (`E1`, `E2`, `E3`,
    `E4 Solver`, `E4 Verifier`) with per-dataset `manifest.json`, `lexsi_provenance.json`, and `checksums.txt`.
    """
    out_root = resolve_project_path(sft_output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    e1_file = resolve_project_path(e1_pool_path)
    e2_file = resolve_project_path(e2_pool_path)
    synth_e2_file = resolve_project_path("data/intermediate/01b_e2_books_plus_synthetic.jsonl")
    if e2_pool_path == DEFAULT_E2_CURATORKIT_NOLLM_POOL_PATH and synth_e2_file.is_file():
        e2_file = synth_e2_file
    e3_file = resolve_project_path(e3_pool_path)
    val_file = resolve_project_path(validation_path)
    test_file = resolve_project_path(test_path)
    sol2_file = resolve_project_path(curated_sol2_path)
    surv_file = resolve_project_path(surviving_flawed_path)
    dir_file = resolve_project_path(directives_input_path)

    e1_problems = load_problems_jsonl(e1_file)
    e2_problems = load_problems_jsonl(e2_file)
    e3_problems = load_problems_jsonl(e3_file)
    val_problems = load_problems_jsonl(val_file) if val_file.is_file() else []
    test_problems = load_problems_jsonl(test_file) if test_file.is_file() else []

    prob_map: dict[str, CanonicalProblem] = {p.problem_id: p for p in e3_problems}
    curated_sol2: list[CandidateSolution] = [
        c for c in load_candidates_jsonl(sol2_file) if c.problem_id in prob_map
    ]
    sol2_map: dict[str, str] = {c.candidate_id: c.solution_text for c in curated_sol2}

    surviving_flawed = load_candidates_jsonl(surv_file) if surv_file.is_file() else []
    surviving_map: dict[str, CandidateSolution] = {c.candidate_id: c for c in surviving_flawed}

    val_sft_rows = [
        r.to_dict() for r in _problems_to_turn0_solver_trajectories(val_problems, dataset_tier="validation")
    ]
    test_sft_rows = [
        r.to_dict() for r in _problems_to_turn0_solver_trajectories(test_problems, dataset_tier="test")
    ]

    # 0. Sanitize and validate all directives in `02g_generated_directives.jsonl` and sync with curated trajectories
    raw_directive_records: list[dict[str, Any]] = []
    if dir_file.is_file():
        with dir_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_directive_records.append(json.loads(line))

    directive_records: list[dict[str, Any]] = []
    valid_dir_count = 0
    invalid_dir_count = 0

    for rec in raw_directive_records:
        pid = str(rec.get("problem_id") or "")
        prob = prob_map.get(pid)
        if prob is None:
            continue
        cid = str(rec.get("candidate_id") or "")
        verdict = str(rec.get("verdict") or VERDICT_CHALLENGE).strip()

        # If this is a challenge record, only keep it if its trajectory survived in `02f_surviving_flawed_trajectories.jsonl`
        if verdict == VERDICT_CHALLENGE and str(rec.get("kind") or "") != "complementary_second_hint":
            if surviving_map and cid not in surviving_map:
                continue
            if cid in surviving_map:
                rec["proposed_solution"] = surviving_map[cid].solution_text
        elif verdict == VERDICT_APPROVE:
            base_cid = cid.removesuffix("_approve")
            if base_cid in sol2_map:
                rec["proposed_solution"] = sol2_map[base_cid]
            else:
                rec["proposed_solution"] = _normalize_step_lines_and_answer(prob.reference_solution)

        sanitized_full = sanitize_directive_text(
            str(rec.get("formatted_output") or rec.get("directive_body") or ""),
            expected_verdict=verdict,
            problem_statement=prob.problem_statement,
            reference_solution=prob.reference_solution,
        )
        val_res = validate_verifier_directive(
            sanitized_full,
            problem_statement=prob.problem_statement,
            reference_solution=prob.reference_solution,
            expected_verdict=verdict,
        )
        rec["directive_body"] = val_res.cleaned_directive_body
        rec["formatted_output"] = val_res.formatted_full_output
        rec["is_valid"] = val_res.is_valid
        if val_res.is_valid:
            valid_dir_count += 1
            directive_records.append(rec)
        else:
            invalid_dir_count += 1

    # Save cleaned `02g_generated_directives.jsonl`
    if directive_records:
        _write_jsonl_rows(directive_records, dir_file)

    # 1. Build & Export E1, E2, and E3 Turn-0 Solver SFT Datasets
    e1_solver_rows = [
        r.to_dict() for r in _problems_to_turn0_solver_trajectories(e1_problems, dataset_tier="E1_solver_manual_det")
    ]
    e2_solver_rows = [
        r.to_dict() for r in _problems_to_turn0_solver_trajectories(e2_problems, dataset_tier="E2_solver_curatorkit_nollm")
    ]
    e3_solver_rows = [
        r.to_dict() for r in _problems_to_turn0_solver_trajectories(e3_problems, dataset_tier="E3_solver_canonical_llm")
    ]

    e1_res = _curate_and_write_dataset_artifacts(
        dataset_key="solver_e1_manual_det",
        dataset_dir=out_root / "solver_e1_manual_det",
        dataset_name="vital-ar/physics-solver-e1-manual-rules",
        method="stage1_manual_deterministic_rules_only",
        rows=e1_solver_rows,
        base_model=None,
        input_refs=[e1_file],
        val_rows=val_sft_rows,
        test_rows=test_sft_rows,
        require_answer_sentinel=True,
    )
    e2_res = _curate_and_write_dataset_artifacts(
        dataset_key="solver_e2_curatorkit_nollm",
        dataset_dir=out_root / "solver_e2_curatorkit_nollm",
        dataset_name="vital-ar/physics-solver-e2-curatorkit-nollm",
        method="stage1_curatorkit_deterministic_gates",
        rows=e2_solver_rows,
        base_model=None,
        input_refs=[e2_file],
        val_rows=val_sft_rows,
        test_rows=test_sft_rows,
        require_answer_sentinel=True,
    )
    e3_res = _curate_and_write_dataset_artifacts(
        dataset_key="solver_e3_canonical_llm",
        dataset_dir=out_root / "solver_e3_canonical_llm",
        dataset_name="vital-ar/physics-solver-e3-canonical-aya8b",
        method="stage1_curatorkit_plus_aya8b_canonical",
        rows=e3_solver_rows,
        base_model=AYA_EXPANSE_8B,
        input_refs=[e3_file],
        val_rows=val_sft_rows,
        test_rows=test_sft_rows,
        require_answer_sentinel=True,
    )
    e1_out, e1_curated = e1_res.train_path, e1_res.curated_rows
    e2_out, e2_curated = e2_res.train_path, e2_res.curated_rows
    e3_out, e3_curated = e3_res.train_path, e3_res.curated_rows

    # 2. Build & Export E4 Solver SFT Dataset (`solver_e4_adversarial`: 0 hints + 50% 1-hint + 50% 2-hints)
    e4_solver_objs: list[SolverTrajectory] = []
    for p in e3_problems:
        e4_solver_objs.append(
            SolverTrajectory(
                trajectory_id=f"E4_solver:{p.problem_id}:turn0_sol1",
                problem_id=p.problem_id,
                domain=p.domain,
                turn=0,
                accumulated_directives=[],
                prompt=format_solver_prompt(p.problem_statement, []),
                target_output=p.reference_solution,
                source_candidate_ids=[f"{p.problem_id}_sol1"],
                dataset_tier="E4_solver_adversarial",
            )
        )
    sol2_kept_count = 0
    for s2 in curated_sol2:
        p = prob_map.get(s2.problem_id)
        if p is None:
            continue
        sol2_kept_count += 1
        e4_solver_objs.append(
            SolverTrajectory(
                trajectory_id=f"E4_solver:{p.problem_id}:turn0_sol2",
                problem_id=p.problem_id,
                domain=p.domain,
                turn=0,
                accumulated_directives=[],
                prompt=format_solver_prompt(p.problem_statement, []),
                target_output=s2.solution_text,
                source_candidate_ids=[s2.candidate_id],
                dataset_tier="E4_solver_adversarial",
            )
        )

    hints_by_prob: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for rec in directive_records:
        if rec.get("verdict") == VERDICT_CHALLENGE and rec.get("directive_body"):
            pid = str(rec["problem_id"])
            cid = str(rec["candidate_id"])
            hbody = str(rec["directive_body"]).strip()
            existing_bodies = [b for _, b in hints_by_prob[pid]]
            if hbody not in existing_bodies:
                hints_by_prob[pid].append((cid, hbody))

    two_hint_target = (len(e3_problems) + 1) // 2
    sorted_probs = sorted(
        e3_problems,
        key=lambda p: (-len(hints_by_prob.get(p.problem_id, [])), p.domain, p.problem_id),
    )
    two_hint_probs = sorted_probs[:two_hint_target]
    one_hint_probs = sorted_probs[two_hint_target:]

    turn1_count = 0
    for p in one_hint_probs:
        prob_hints = hints_by_prob.get(p.problem_id, [])
        if not prob_hints:
            continue
        cid1, h1 = prob_hints[0]
        e4_solver_objs.append(
            SolverTrajectory(
                trajectory_id=f"E4_solver:{p.problem_id}:turn1_1hint",
                problem_id=p.problem_id,
                domain=p.domain,
                turn=1,
                accumulated_directives=[h1],
                prompt=format_solver_prompt(p.problem_statement, [h1]),
                target_output=p.reference_solution,
                source_candidate_ids=[cid1],
                dataset_tier="E4_solver_adversarial",
            )
        )
        turn1_count += 1

    turn2_count = 0
    for p in two_hint_probs:
        prob_hints = hints_by_prob.get(p.problem_id, [])
        if not prob_hints:
            continue
        if len(prob_hints) >= 2:
            (cid1, h1), (cid2, h2) = prob_hints[0], prob_hints[1]
            directives = [h1, h2]
            cids = [cid1, cid2]
            turn_num = 2
        else:
            cid1, h1 = prob_hints[0]
            first_step_lines = [
                ln.strip()
                for ln in p.reference_solution.splitlines()
                if ln.strip() and not ln.strip().upper().startswith("ANSWER:")
            ]
            first_step = first_step_lines[0] if first_step_lines else "Apply conservation and boundary equations"
            h2_clean = sanitize_directive_text(
                f"Additionally, verify dimensional consistency and boundary substitutions when setting up the governing relation ({first_step[:140].rstrip('.')}).\n[[CHALLENGE]]",
                expected_verdict=VERDICT_CHALLENGE,
                problem_statement=p.problem_statement,
                reference_solution=p.reference_solution,
            ).replace(VERDICT_CHALLENGE, "").strip()
            directives = [h1, h2_clean]
            cids = [cid1, f"{p.problem_id}_second_constraint"]
            turn_num = 2
        e4_solver_objs.append(
            SolverTrajectory(
                trajectory_id=f"E4_solver:{p.problem_id}:turn2_2hints",
                problem_id=p.problem_id,
                domain=p.domain,
                turn=turn_num,
                accumulated_directives=directives,
                prompt=format_solver_prompt(p.problem_statement, directives),
                target_output=p.reference_solution,
                source_candidate_ids=cids,
                dataset_tier="E4_solver_adversarial",
            )
        )
        turn2_count += 1

    e4_solver_res = _curate_and_write_dataset_artifacts(
        dataset_key="solver_e4_adversarial",
        dataset_dir=out_root / "solver_e4_adversarial",
        dataset_name="vital-ar/physics-solver-e4-adversarial",
        method="stage2_curatorkit_adversarial_solver_sft",
        rows=[r.to_dict() for r in e4_solver_objs],
        base_model=AYA_EXPANSE_8B,
        input_refs=[e3_file, sol2_file, dir_file],
        extra_params={
            "turn0_sol1_count": len(e3_problems),
            "turn0_minority_sol2_count": sol2_kept_count,
            "turn1_one_hint_count": turn1_count,
            "turn2_two_hints_count": turn2_count,
        },
        val_rows=val_sft_rows,
        test_rows=test_sft_rows,
        require_answer_sentinel=True,
    )
    e4_solver_out, e4_solver_curated = e4_solver_res.train_path, e4_solver_res.curated_rows

    # 3. Build & Export E4 Verifier SFT Dataset (`verifier_e4_curated`)
    e4_verifier_objs: list[VerifierTrajectory] = []
    for rec in directive_records:
        kind = str(rec.get("kind") or "")
        if kind == "complementary_second_hint" or not str(rec.get("proposed_solution") or "").strip():
            continue
        pid = str(rec["problem_id"])
        prob = prob_map.get(pid)
        if prob is None:
            continue
        proposed_sol = str(rec["proposed_solution"]).strip()
        verdict = str(rec["verdict"]).strip()
        directive_body = str(rec["directive_body"]).strip()
        formatted_out = f"{directive_body}\n{verdict}"
        part_tag = (
            "part_1_domain_balanced_approved"
            if verdict == VERDICT_APPROVE
            else (
                "part_3_real_tiny_aya_challenged"
                if "tiny_aya" in str(rec.get("source") or "")
                else "part_2_synthetic_aya8b_challenged"
            )
        )
        e4_verifier_objs.append(
            VerifierTrajectory(
                trajectory_id=f"E4_verifier:{rec['candidate_id']}",
                problem_id=pid,
                domain=prob.domain,
                proposed_solution=proposed_sol,
                directive_text=directive_body,
                verdict=verdict,
                error_type=rec.get("error_type"),
                part=part_tag,
                prompt=format_verifier_prompt(prob.problem_statement, proposed_sol),
                target_output=formatted_out,
            )
        )

    e4_verifier_res = _curate_and_write_dataset_artifacts(
        dataset_key="verifier_e4_curated",
        dataset_dir=out_root / "verifier_e4_curated",
        dataset_name="vital-ar/physics-verifier-e4-curated",
        method="stage2_curatorkit_adversarial_verifier_sft",
        rows=[v.to_dict() for v in e4_verifier_objs],
        base_model=AYA_EXPANSE_8B,
        input_refs=[e3_file, sol2_file, surv_file, dir_file],
        extra_params={
            "challenged_count": sum(1 for v in e4_verifier_objs if v.verdict == VERDICT_CHALLENGE),
            "approved_count": sum(1 for v in e4_verifier_objs if v.verdict == VERDICT_APPROVE),
        },
    )
    e4_verifier_out, e4_verifier_curated = e4_verifier_res.train_path, e4_verifier_res.curated_rows

    challenged_rows = [r for r in e4_verifier_curated if r.get("verdict") == VERDICT_CHALLENGE]
    approved_rows = [r for r in e4_verifier_curated if r.get("verdict") == VERDICT_APPROVE]

    # 4. Generate dynamic Hugging Face `README.md` dataset cards & finalize SHA-256 `checksums.txt`
    tier_counts = {
        "solver_e1_manual_det": len(e1_curated),
        "solver_e2_curatorkit_nollm": len(e2_curated),
        "solver_e3_canonical_llm": len(e3_curated),
        "solver_e4_adversarial": len(e4_solver_curated),
        "verifier_e4_curated": len(e4_verifier_curated),
    }
    e1_src_map = {p.problem_id: p.source for p in e1_problems}
    e2_src_map = {p.problem_id: p.source for p in e2_problems}
    e3_src_map = {p.problem_id: p.source for p in e3_problems}
    e1_dom_counts = Counter(str(r.get("domain") or "UNKNOWN") for r in e1_curated)
    e2_dom_counts = Counter(str(r.get("domain") or "UNKNOWN") for r in e2_curated)
    e4_extra = {
        "turn0": sum(1 for r in e4_solver_curated if r.get("turn") == 0),
        "turn0_sol1": sum(1 for r in e4_solver_curated if r.get("turn") == 0 and str(r.get("trajectory_id", "")).endswith(":turn0_sol1")),
        "turn0_sol2": sum(1 for r in e4_solver_curated if r.get("turn") == 0 and str(r.get("trajectory_id", "")).endswith(":turn0_sol2")),
        "turn1": sum(1 for r in e4_solver_curated if r.get("turn") == 1),
        "turn2": sum(1 for r in e4_solver_curated if r.get("turn") == 2),
    }
    verifier_stats = {"approved": len(approved_rows), "challenged": len(challenged_rows)}

    card_configs: list[tuple[_CuratedDatasetResult, dict[str, Any]]] = [
        (
            e1_res,
            {"source_counts": Counter(e1_src_map.get(str(r.get("problem_id") or ""), "unknown") for r in e1_curated)},
        ),
        (
            e2_res,
            {
                "source_counts": Counter(e2_src_map.get(str(r.get("problem_id") or ""), "unknown") for r in e2_curated),
                "upstream_domain_counts": e1_dom_counts,
                "upstream_label": "`E1`",
            },
        ),
        (
            e3_res,
            {
                "source_counts": Counter(e3_src_map.get(str(r.get("problem_id") or ""), "unknown") for r in e3_curated),
                "upstream_domain_counts": e2_dom_counts,
                "upstream_label": "`E2`",
            },
        ),
        (
            e4_solver_res,
            {"source_counts": Counter(), "e4_extra": e4_extra},
        ),
        (
            e4_verifier_res,
            {"source_counts": Counter(), "verifier_stats": verifier_stats},
        ),
    ]
    for res, card_kwargs in card_configs:
        md = _build_dataset_card_markdown(
            result=res,
            out_root=out_root,
            tier_counts=tier_counts,
            **card_kwargs,
        )
        _write_dataset_readme_and_checksums(res, md)

    output_files = {
        "solver_e1_manual_det": str(e1_out),
        "solver_e2_curatorkit_nollm": str(e2_out),
        "solver_e3_canonical_llm": str(e3_out),
        "solver_e4_adversarial": str(e4_solver_out),
        "verifier_e4_curated": str(e4_verifier_out),
    }

    summary = Stage2ExportSummary(
        solver_e1_manual_det_count=len(e1_curated),
        solver_e2_curatorkit_nollm_count=len(e2_curated),
        solver_e3_canonical_llm_count=len(e3_curated),
        solver_e4_adversarial_total_count=len(e4_solver_curated),
        solver_e4_turn0_sol1_count=len(e3_problems),
        solver_e4_turn0_minority_sol2_count=len(curated_sol2),
        solver_e4_turn1_one_hint_count=turn1_count,
        solver_e4_turn2_two_hints_count=turn2_count,
        verifier_e4_total_count=len(e4_verifier_curated),
        verifier_e4_challenged_count=len(challenged_rows),
        verifier_e4_approved_count=len(approved_rows),
        verifier_e4_approvals_by_domain=dict(sorted(Counter(r["domain"] for r in approved_rows).items())),
        verifier_e4_challenges_by_source=dict(sorted(Counter(r["part"] for r in challenged_rows).items())),
        directives_valid_count=valid_dir_count,
        directives_invalid_count=invalid_dir_count,
        output_files=output_files,
    )

    summary_path = out_root / "stage2_export_summary.json"
    summary_path.write_text(json.dumps(summary.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(
        "Stage 2.5 Export complete: E1=%d | E2=%d | E3=%d | E4_Solver=%d (%d Sol1 + %d Sol2 + %d 1-hint + %d 2-hint) "
        "| E4_Verifier=%d (%d challenged + %d approved) | directives_valid=%d (invalid=%d)",
        summary.solver_e1_manual_det_count,
        summary.solver_e2_curatorkit_nollm_count,
        summary.solver_e3_canonical_llm_count,
        summary.solver_e4_adversarial_total_count,
        summary.solver_e4_turn0_sol1_count,
        summary.solver_e4_turn0_minority_sol2_count,
        summary.solver_e4_turn1_one_hint_count,
        summary.solver_e4_turn2_two_hints_count,
        summary.verifier_e4_total_count,
        summary.verifier_e4_challenged_count,
        summary.verifier_e4_approved_count,
        summary.directives_valid_count,
        summary.directives_invalid_count,
    )
    return summary
