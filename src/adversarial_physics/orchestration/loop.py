"""Multi-turn adversarial Solver (E4A) + Verifier (E4B) validation loop (Stage 5).

Executes up to ``max_iterations`` (default 3: Turn 0 initial solve with 0 hints,
plus up to 2 Verifier-guided revision turns with accumulated hints) over the
held-out PhysGym validation split, pulling missing validation data and LoRA
adapters automatically from Hugging Face Hub or deterministic PhysGym splits.
"""
from __future__ import annotations

import json
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from adversarial_physics.data.curation.answer_validator import (
    AnswerValidationResult,
    extract_answer_line,
    validate_candidate_answer,
)
from adversarial_physics.data.curation.directive_leak_check import sanitize_directive_text
from adversarial_physics.data.ingestion.physgym_loader import (
    DEFAULT_PHYSGYM_REPO,
    DEFAULT_PHYSGYM_REVISION,
    load_physgym_dataset,
)
from adversarial_physics.data.processing.canonical_pool_builder import (
    extract_final_answer_from_solution,
)
from adversarial_physics.data.schemas.problem import (
    CanonicalProblem,
    save_problems_jsonl,
)
from adversarial_physics.data.schemas.trajectory import (
    VERDICT_APPROVE,
    VERDICT_CHALLENGE,
    format_solver_prompt,
    format_verifier_prompt,
)
from adversarial_physics.utils.config import load_yaml, resolve_project_path
from adversarial_physics.utils.logging import get_logger

logger = get_logger("orchestration_loop")

DEFAULT_BASE_MODEL = "CohereLabs/tiny-aya-global"
DEFAULT_SOLVER_ADAPTER_REPO = "vital-ar/adversarial-physics-solver-e4-adversarial-lora"
DEFAULT_VERIFIER_ADAPTER_REPO = "vital-ar/adversarial-physics-verifier-e4-curated-lora"
DEFAULT_SOLVER_LOCAL_DIR = "models/adapters/solver_e4_adversarial_lora"
DEFAULT_VERIFIER_LOCAL_DIR = "models/adapters/verifier_e4_curated_lora"

DEFAULT_VALIDATION_PATH = "data/splits/validation.jsonl"
DEFAULT_SPLIT_MANIFEST_PATH = "configs/data/split_21_38_38.yaml"
DEFAULT_INGESTION_CONFIG_PATH = "configs/data/ingestion.yaml"
DEFAULT_VALIDATION_HF_REPOS: tuple[str, ...] = (
    "vital-ar/adversarial-physics-validation",
    "vital-ar/physics-solver-e4-adversarial",
    "vital-ar/adversarial-physics-splits",
)

PromptMode = str  # "raw" | "chat_template" | "alpaca"


@dataclass(frozen=True)
class ValidationDataLoadResult:
    """Loaded validation problems together with resolved local path and origin metadata."""

    problems: list[CanonicalProblem]
    local_path: Path
    source_used: str


@dataclass(frozen=True)
class ParsedVerifierOutput:
    """Structured parse of raw Model B (Verifier) text output."""

    verdict: str  # "APPROVE" | "CHALLENGE" | "UNKNOWN"
    directive: str
    raw_output: str
    had_sentinel: bool


@dataclass(frozen=True)
class TurnTrace:
    """Trace for one iteration (Solver generation + optional Verifier review)."""

    iteration: int
    turn: int
    accumulated_directives: list[str]
    solver_prompt: str
    solver_output: str
    solver_validation: dict[str, Any]
    solver_is_correct: bool
    solver_tokens: int
    solver_latency_s: float
    verifier_prompt: str | None = None
    verifier_output: str | None = None
    verifier_verdict: str | None = None
    verifier_directive: str | None = None
    verifier_is_decision_correct: bool | None = None
    verifier_tokens: int = 0
    verifier_latency_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProblemLoopResult:
    """Complete multi-turn validation outcome for one physics problem."""

    problem_id: str
    domain: str
    problem_statement: str
    reference_solution: str
    normalized_reference_answer: str
    iterations_used: int
    max_iterations: int
    stop_reason: str
    turn0_correct: bool
    final_correct: bool
    turn0_has_answer: bool
    final_has_answer: bool
    turn0_match_method: str
    final_match_method: str
    converted: bool
    regressed: bool
    total_latency_s: float
    total_tokens: int
    traces: list[TurnTrace]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["traces"] = [t.to_dict() for t in self.traces]
        return payload


def _resolve_hf_token(hf_token: str | None = None) -> str | None:
    return hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")


def extract_problem_statement_from_prompt(prompt_text: str) -> str:
    """Recover a bare problem statement from an exported SFT solver prompt string."""
    text = (prompt_text or "").strip()
    marker = "PROBLEM:\n"
    if marker in text:
        text = text.split(marker, 1)[1]
    constraints_marker = "\n\nCRITICAL PHYSICAL CONSTRAINTS TO FOLLOW:"
    if constraints_marker in text:
        text = text.split(constraints_marker, 1)[0]
    return text.strip()


def canonicalize_validation_record(row: dict[str, Any], index: int = 0) -> CanonicalProblem:
    """Convert either a ``CanonicalProblem`` row or an exported SFT row into ``CanonicalProblem``."""
    problem_id = str(row.get("problem_id") or row.get("id") or f"val_{index:04d}")
    domain = str(row.get("domain") or "UNKNOWN").strip().upper()

    problem_statement = str(row.get("problem_statement") or "").strip()
    if not problem_statement:
        raw_prompt = str(row.get("prompt") or row.get("instruction") or "")
        problem_statement = extract_problem_statement_from_prompt(raw_prompt)

    reference_solution = str(
        row.get("reference_solution") or row.get("target_output") or row.get("output") or ""
    ).strip()

    if not problem_statement or not reference_solution:
        raise ValueError(
            f"Validation record {problem_id!r} missing problem_statement or reference_solution"
        )

    return CanonicalProblem(
        problem_id=problem_id,
        domain=domain,
        problem_statement=problem_statement,
        reference_solution=reference_solution,
        source=str(row.get("source") or "physgym"),
        original_id=row.get("original_id"),
    )


def load_validation_problems_file(path: str | Path) -> list[CanonicalProblem]:
    """Load validation records from JSONL supporting both raw CanonicalProblem and SFT rows."""
    file_path = Path(path)
    problems: list[CanonicalProblem] = []
    with file_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if {"problem_id", "domain", "problem_statement", "reference_solution"}.issubset(row):
                problems.append(CanonicalProblem.from_dict(row))
            else:
                problems.append(canonicalize_validation_record(row, index=idx))
    return problems


def _download_validation_from_hf_hub(
    *,
    target_path: Path,
    repo_candidates: Sequence[str],
    token: str | None,
) -> ValidationDataLoadResult | None:
    """Attempt to download ``validation.jsonl`` from Hub repositories or datasets splits."""
    from huggingface_hub import hf_hub_download

    target_path.parent.mkdir(parents=True, exist_ok=True)
    filenames = ("validation.jsonl", "data/validation.jsonl", "data/splits/validation.jsonl")

    for repo_id in repo_candidates:
        if not repo_id:
            continue
        for filename in filenames:
            try:
                downloaded = Path(
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        repo_type="dataset",
                        token=token,
                    )
                )
                problems = load_validation_problems_file(downloaded)
                if problems:
                    save_problems_jsonl(problems, target_path)
                    logger.info(
                        f"Downloaded {len(problems)} validation problems from {repo_id}/{filename} -> {target_path}"
                    )
                    return ValidationDataLoadResult(
                        problems=problems,
                        local_path=target_path,
                        source_used=f"hf_hub:{repo_id}/{filename}",
                    )
            except Exception:
                continue

        # Fallback to `datasets.load_dataset` in case Hub dataset is stored in Parquet/Arrow format
        try:
            from datasets import load_dataset

            for split_name in ("validation", "train"):
                try:
                    ds = load_dataset(repo_id, split=split_name, token=token)
                except Exception:
                    continue
                if len(ds) == 0:
                    continue
                problems = [
                    canonicalize_validation_record(dict(row), index=i)
                    for i, row in enumerate(ds)
                ]
                save_problems_jsonl(problems, target_path)
                logger.info(
                    f"Loaded {len(problems)} validation problems from HF dataset {repo_id}[{split_name}] -> {target_path}"
                )
                return ValidationDataLoadResult(
                    problems=problems,
                    local_path=target_path,
                    source_used=f"hf_dataset:{repo_id}:{split_name}",
                )
        except Exception:
            continue
    return None


def _reconstruct_validation_from_physgym_split(
    *,
    target_path: Path,
    split_manifest_path: str | Path,
    ingestion_config_path: str | Path,
    token: str | None,
) -> ValidationDataLoadResult:
    """Deterministically reconstruct the exact 38 validation rows from PhysGym + committed split manifest."""
    manifest_file = resolve_project_path(split_manifest_path)
    manifest = load_yaml(manifest_file)
    val_ids = list((manifest.get("splits") or {}).get("val_ids") or [])
    if not val_ids:
        raise ValueError(f"Split manifest {manifest_file} contains no splits.val_ids")

    repo_id = DEFAULT_PHYSGYM_REPO
    revision = DEFAULT_PHYSGYM_REVISION
    split_name = "train"
    id_prefix = "physgym"
    canonicalize_domains = True
    domain_aliases: dict[str, str] | None = None

    ingestion_file = resolve_project_path(ingestion_config_path)
    if ingestion_file.is_file():
        ing_cfg = load_yaml(ingestion_file)
        src = ing_cfg.get("source") or {}
        norm = ing_cfg.get("normalization") or {}
        repo_id = str(src.get("repo_id") or repo_id)
        revision = str(src.get("revision") or revision)
        split_name = str(src.get("split") or split_name)
        id_prefix = str(norm.get("id_prefix") or id_prefix)
        canonicalize_domains = bool(norm.get("canonicalize_domains", True))
        raw_aliases = norm.get("domain_aliases")
        if isinstance(raw_aliases, dict):
            domain_aliases = {str(k): str(v) for k, v in raw_aliases.items()}

    all_problems, _ = load_physgym_dataset(
        repo_id=repo_id,
        revision=revision,
        split=split_name,
        canonicalize_domains=canonicalize_domains,
        domain_aliases=domain_aliases,
        id_prefix=id_prefix,
        resolve_hub_sha=False,
        token=token,
    )
    by_id = {p.problem_id: p for p in all_problems}
    val_problems = [by_id[pid] for pid in val_ids if pid in by_id]
    if len(val_problems) != len(val_ids):
        missing = [pid for pid in val_ids if pid not in by_id]
        raise RuntimeError(
            f"Reconstructed only {len(val_problems)}/{len(val_ids)} validation rows from {repo_id}@{revision}; "
            f"missing IDs: {missing[:5]}"
        )

    save_problems_jsonl(val_problems, target_path)
    logger.info(
        f"Reconstructed {len(val_problems)} held-out validation problems from {repo_id}@{revision} "
        f"via {manifest_file} -> {target_path}"
    )
    return ValidationDataLoadResult(
        problems=val_problems,
        local_path=target_path,
        source_used=f"physgym_split:{repo_id}@{revision[:10]}",
    )


def ensure_validation_dataset(
    local_path: str | Path = DEFAULT_VALIDATION_PATH,
    *,
    hf_repo_id: str | None = None,
    split_manifest_path: str | Path = DEFAULT_SPLIT_MANIFEST_PATH,
    ingestion_config_path: str | Path = DEFAULT_INGESTION_CONFIG_PATH,
    hf_token: str | None = None,
) -> ValidationDataLoadResult:
    """Load the validation split from disk or pull it automatically on Kaggle/remote hosts.

    Resolution order:
    1. ``local_path`` on disk (or ``data/splits/validation.jsonl`` / ``data/sft/solver_e4_adversarial/validation.jsonl``
       when using the default path).
    2. Hugging Face Hub validation/split repositories (``hf_repo_id`` if passed, plus configured project repos).
    3. Deterministic reconstruction from public ``YimengChen/PhysGym`` filtered by ``configs/data/split_21_38_38.yaml``.
    """
    resolved_target = resolve_project_path(local_path)
    if resolved_target.is_file():
        problems = load_validation_problems_file(resolved_target)
        if problems:
            logger.info(f"Loaded {len(problems)} validation problems from local file: {resolved_target}")
            return ValidationDataLoadResult(
                problems=problems,
                local_path=resolved_target,
                source_used=f"local:{resolved_target}",
            )

    if str(local_path).replace("\\", "/") == DEFAULT_VALIDATION_PATH:
        sft_fallback = resolve_project_path("data/sft/solver_e4_adversarial/validation.jsonl")
        if sft_fallback.is_file():
            problems = load_validation_problems_file(sft_fallback)
            if problems:
                save_problems_jsonl(problems, resolved_target)
                logger.info(
                    f"Loaded {len(problems)} validation problems from {sft_fallback} -> {resolved_target}"
                )
                return ValidationDataLoadResult(
                    problems=problems,
                    local_path=resolved_target,
                    source_used=f"local:{sft_fallback}",
                )

    token = _resolve_hf_token(hf_token)
    repo_candidates: list[str] = []
    if hf_repo_id:
        repo_candidates.append(hf_repo_id)
    for default_repo in DEFAULT_VALIDATION_HF_REPOS:
        if default_repo not in repo_candidates:
            repo_candidates.append(default_repo)

    hub_result = _download_validation_from_hf_hub(
        target_path=resolved_target,
        repo_candidates=repo_candidates,
        token=token,
    )
    if hub_result is not None:
        return hub_result

    logger.warning(
        "Could not pull validation split from configured HF Hub dataset repos; "
        "reconstructing deterministically from YimengChen/PhysGym + split manifest..."
    )
    return _reconstruct_validation_from_physgym_split(
        target_path=resolved_target,
        split_manifest_path=split_manifest_path,
        ingestion_config_path=ingestion_config_path,
        token=token,
    )


def _extract_last_boxed(text: str) -> str | None:
    """Extract the inner expression of the last balanced ``\\boxed{...}`` block."""
    token = "\\boxed"
    idx = text.rfind(token)
    if idx == -1:
        return None
    brace_start = text.find("{", idx + len(token))
    if brace_start == -1:
        return None
    depth = 0
    for pos in range(brace_start, len(text)):
        ch = text[pos]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start + 1 : pos].strip()
    return None


def _clean_extracted_math_answer(expr: str) -> str:
    """Clean LaTeX wrapper tokens that break answer comparison (\\boxed, \\tag, \\therefore, \\displaystyle)."""
    s = (expr or "").strip()
    boxed_inner = _extract_last_boxed(s)
    if boxed_inner:
        s = boxed_inner
    s = re.sub(r"\\tag\s*\{[^}]*\}", "", s)
    s = re.sub(r"\\(?:therefore|displaystyle|quad|qquad|hline)\b", "", s)
    s = s.replace(r"\;", " ").replace(r"\,", " ")
    s = re.sub(r"^\$+|\$+$", "", s).strip()
    s = re.sub(r"^\\\(\s*|\s*\\\)$", "", s).strip()
    s = re.sub(r"^\\\[\s*|\s*\\\]$", "", s).strip()
    s = re.sub(r"\s+", " ", s).strip(" .,;")
    return s


def prepare_reference_for_validation(reference_solution: str) -> tuple[str, str]:
    """Ensure ``reference_solution`` exposes an ``ANSWER:`` sentinel for deterministic scoring.

    Raw PhysGym validation rows store LaTeX prose ending in ``\\boxed{...}``, display math,
    or inline math without an explicit ``ANSWER:`` prefix. This synthesizes an ``ANSWER:``
    line strictly in memory without altering saved dataset files.
    """
    ref = (reference_solution or "").strip()
    has_ans, ans_val = extract_answer_line(ref)
    if has_ans:
        cleaned = _clean_extracted_math_answer(ans_val)
        return (f"{ref}\nANSWER: {cleaned}" if cleaned != ans_val else ref), cleaned

    # 1. Explicit \boxed{...} takes highest priority (even when commentary follows)
    extracted = _extract_last_boxed(ref) or ""

    # 2. Trailing inline math appearing AFTER the last display block (e.g., "... is $v=-\sqrt{...}$")
    if not extracted:
        last_display_end = max(ref.rfind("$$"), ref.rfind(r"\]"))
        tail_after_display = ref[last_display_end + 2 :] if last_display_end != -1 else ""
        if tail_after_display:
            inline_matches = re.findall(r"\$([^$]+)\$|\\\((.+?)\\\)", tail_after_display, flags=re.DOTALL)
            if inline_matches:
                last_inline = (inline_matches[-1][0] or inline_matches[-1][1]).strip()
                # Only prefer trailing inline math over the last display block when it contains a real expression/equation
                if (
                    "=" in last_inline
                    or "(" in last_inline
                    or bool(re.search(r"\\frac|\\sqrt|\^|\*|/|[+-]", last_inline))
                ):
                    extracted = last_inline

    # 3. Repository fallback extractor (display math / '=' / final line)
    if not extracted:
        extracted = extract_final_answer_from_solution(ref)

    cleaned = _clean_extracted_math_answer(extracted)
    if not cleaned:
        return ref, ""
    return f"{ref}\nANSWER: {cleaned}", cleaned


def prepare_candidate_for_validation(candidate_solution: str) -> str:
    """Clean known harmless artifacts (e.g., repeated ``ANSWER: ANSWER:`` or ``\\boxed{}``) before scoring."""
    text = (candidate_solution or "").strip()
    has_ans, ans_val = extract_answer_line(text)
    if not has_ans:
        return text
    cleaned = re.sub(r"^(?:\s*ANSWER\s*:\s*)+", "", ans_val, flags=re.IGNORECASE).strip()
    cleaned = _clean_extracted_math_answer(cleaned)
    if cleaned and cleaned != ans_val:
        return f"{text}\nANSWER: {cleaned}"
    return text


def score_solver_output(
    candidate_solution: str,
    reference_solution: str,
    *,
    rel_tol: float = 0.015,
) -> AnswerValidationResult:
    """Validate Solver output against raw or canonicalized ``reference_solution``."""
    ref_with_sentinel, _ = prepare_reference_for_validation(reference_solution)
    cand_prepared = prepare_candidate_for_validation(candidate_solution)
    return validate_candidate_answer(cand_prepared, ref_with_sentinel, rel_tol=rel_tol)


def format_model_prompt(
    raw_prompt: str,
    *,
    prompt_mode: PromptMode = "raw",
    tokenizer: Any | None = None,
) -> str:
    """Format a Stage 2 prompt for Stage 4 AlignTune ``instruction_following`` LoRA inference."""
    cleaned = raw_prompt.strip()
    if prompt_mode == "raw":
        # Matches `_normalize_dataset_jsonl_whitespace` in `run_aligntune.py`
        return f"{cleaned}\n\n"
    if prompt_mode == "alpaca":
        return f"### Instruction:\n{cleaned}\n\n### Response:\n"
    if prompt_mode == "chat_template":
        if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError("prompt_mode='chat_template' requires a tokenizer with apply_chat_template")
        return str(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": cleaned}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    raise ValueError(f"Unsupported prompt_mode: {prompt_mode!r}. Expected 'raw', 'chat_template', or 'alpaca'.")


def parse_verifier_response(
    raw_output: str,
    *,
    problem_statement: str = "",
) -> ParsedVerifierOutput:
    """Parse Verifier Model B output into verdict (`APPROVE` | `CHALLENGE` | `UNKNOWN`) + sanitized hint."""
    text = (raw_output or "").strip()
    matches = list(re.finditer(r"\[\[\s*(APPROVE|CHALLENGE)\s*\]\]", text, flags=re.IGNORECASE))
    if not matches:
        return ParsedVerifierOutput(
            verdict="UNKNOWN",
            directive=text,
            raw_output=raw_output,
            had_sentinel=False,
        )

    last_match = matches[-1]
    verdict = last_match.group(1).upper()
    body = text[: last_match.start()].strip()
    if not body:
        body = re.sub(r"\[\[\s*(?:APPROVE|CHALLENGE)\s*\]\]", "", text, flags=re.IGNORECASE).strip()

    expected_sentinel = VERDICT_APPROVE if verdict == "APPROVE" else VERDICT_CHALLENGE
    sanitized_with_sentinel = sanitize_directive_text(
        f"{body}\n{expected_sentinel}" if body else expected_sentinel,
        expected_verdict=expected_sentinel,
        problem_statement=problem_statement,
        reference_solution="",
    )
    directive = sanitized_with_sentinel.replace(expected_sentinel, "").strip()
    if not directive and body:
        directive = re.sub(r"\s+", " ", body).strip()

    return ParsedVerifierOutput(
        verdict=verdict,
        directive=directive,
        raw_output=raw_output,
        had_sentinel=True,
    )


def _has_local_adapter_weights(path: Path) -> bool:
    return path.is_dir() and (
        (path / "adapter_model.safetensors").is_file() or (path / "adapter_model.bin").is_file()
    )


def resolve_adapter_source(
    preferred: str | None,
    *,
    default_local_dir: str,
    default_hf_repo: str,
) -> str:
    """Resolve an adapter location: prefer local dir only when actual weight files exist, else HF Hub repo."""
    candidates: list[str] = []
    if preferred:
        p = resolve_project_path(preferred)
        if _has_local_adapter_weights(p):
            return str(p)
        if _has_local_adapter_weights(p / "final_adapter"):
            return str(p / "final_adapter")
        # If `preferred` looks like an HF repo ID ("org/repo") and is not a weightless local metadata dir:
        if not p.exists():
            return preferred
        logger.warning(
            f"Local adapter directory '{p}' has no weight files (adapter_model.safetensors/.bin); "
            f"falling back to Hugging Face Hub repo '{default_hf_repo}'."
        )
        return default_hf_repo

    candidates.extend([f"{default_local_dir}/final_adapter", default_local_dir])
    for cand in candidates:
        p = resolve_project_path(cand)
        if _has_local_adapter_weights(p):
            return str(p)
    return default_hf_repo


class DualLoRAEngine:
    """Single-base-model inference engine hot-swapping Solver (`E4A`) and Verifier (`E4B`) LoRA adapters."""

    def __init__(
        self,
        *,
        base_model_id: str = DEFAULT_BASE_MODEL,
        solver_adapter: str | None = None,
        verifier_adapter: str | None = None,
        prompt_mode: PromptMode = "raw",
        solver_max_new_tokens: int = 1024,
        verifier_max_new_tokens: int = 192,
        temperature_solver: float = 0.0,
        temperature_verifier: float = 0.0,
        device_map: str = "auto",
        dtype: str = "auto",
        stop_strings: Sequence[str] = (VERDICT_APPROVE, VERDICT_CHALLENGE),
        hf_token: str | None = None,
    ) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

        self.token = _resolve_hf_token(hf_token)
        self.base_model_id = base_model_id
        self.prompt_mode = prompt_mode
        self.solver_max_new_tokens = int(solver_max_new_tokens)
        self.verifier_max_new_tokens = int(verifier_max_new_tokens)
        self.temperature_solver = float(temperature_solver)
        self.temperature_verifier = float(temperature_verifier)
        self.stop_strings = tuple(s for s in stop_strings if s)

        self.solver_source = resolve_adapter_source(
            solver_adapter,
            default_local_dir=DEFAULT_SOLVER_LOCAL_DIR,
            default_hf_repo=DEFAULT_SOLVER_ADAPTER_REPO,
        )
        self.verifier_source = resolve_adapter_source(
            verifier_adapter,
            default_local_dir=DEFAULT_VERIFIER_LOCAL_DIR,
            default_hf_repo=DEFAULT_VERIFIER_ADAPTER_REPO,
        )

        cuda_avail = torch.cuda.is_available()
        resolved_dtype_str = dtype.lower()
        if resolved_dtype_str == "auto":
            if cuda_avail:
                torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                torch_dtype = torch.float32
        elif resolved_dtype_str == "bf16":
            torch_dtype = torch.bfloat16
        elif resolved_dtype_str == "fp16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        logger.info(
            f"Loading base model '{base_model_id}' (dtype={torch_dtype}, device_map={device_map})..."
        )
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.solver_source,
                trust_remote_code=True,
                token=self.token,
            )
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_id,
                trust_remote_code=True,
                token=self.token,
            )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch_dtype,
            device_map=(device_map if (cuda_avail and device_map != "cpu") else None),
            trust_remote_code=True,
            token=self.token,
        )
        logger.info(f"Attaching Solver LoRA ('solver') from: {self.solver_source}")
        self.model = PeftModel.from_pretrained(
            base_model,
            self.solver_source,
            adapter_name="solver",
            is_trainable=False,
            token=self.token,
        )
        logger.info(f"Attaching Verifier LoRA ('verifier') from: {self.verifier_source}")
        self.model.load_adapter(
            self.verifier_source,
            adapter_name="verifier",
            is_trainable=False,
            token=self.token,
        )
        self.model.eval()
        if hasattr(self.model, "config"):
            self.model.config.use_cache = True

        tokenizer_ref = self.tokenizer
        stop_strs = self.stop_strings

        class _SentinelStopCriteria(StoppingCriteria):
            def __init__(self, prompt_len: int) -> None:
                self.prompt_len = prompt_len

            def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
                if not stop_strs or input_ids.shape[1] <= self.prompt_len:
                    return False
                tail_start = max(self.prompt_len, input_ids.shape[1] - 12)
                tail_text = tokenizer_ref.decode(input_ids[0, tail_start:], skip_special_tokens=False)
                return any(sentinel in tail_text for sentinel in stop_strs)

        self._stop_criteria_cls = _SentinelStopCriteria
        self._stopping_criteria_list_cls = StoppingCriteriaList

    def _input_device(self) -> Any:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return "cpu"

    def _generate(
        self,
        *,
        adapter_name: str,
        raw_prompt: str,
        max_new_tokens: int,
        temperature: float,
        use_stop_strings: bool,
    ) -> tuple[str, int, float]:
        import torch

        t0 = time.monotonic()
        self.model.set_adapter(adapter_name)
        formatted_prompt = format_model_prompt(
            raw_prompt,
            prompt_mode=self.prompt_mode,
            tokenizer=self.tokenizer,
        )
        encoded = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        device = self._input_device()
        encoded = {k: v.to(device) for k, v in encoded.items()}
        prompt_len = int(encoded["input_ids"].shape[1])

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }
        if temperature > 0.0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.95
        else:
            gen_kwargs["do_sample"] = False

        if use_stop_strings and self.stop_strings:
            gen_kwargs["stopping_criteria"] = self._stopping_criteria_list_cls(
                [self._stop_criteria_cls(prompt_len)]
            )

        with torch.inference_mode():
            out_ids = self.model.generate(**encoded, **gen_kwargs)

        new_tokens = out_ids[0, prompt_len:]
        decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if use_stop_strings:
            for sentinel in self.stop_strings:
                idx = decoded.find(sentinel)
                if idx != -1:
                    decoded = decoded[: idx + len(sentinel)].strip()
                    break

        elapsed = time.monotonic() - t0
        return decoded, int(new_tokens.shape[0]), elapsed

    def generate_solver(self, raw_prompt: str) -> tuple[str, int, float]:
        """Run active ``solver`` adapter on a Turn-0 or Hinted-Revision prompt."""
        return self._generate(
            adapter_name="solver",
            raw_prompt=raw_prompt,
            max_new_tokens=self.solver_max_new_tokens,
            temperature=self.temperature_solver,
            use_stop_strings=False,
        )

    def generate_verifier(self, raw_prompt: str) -> tuple[str, int, float]:
        """Run active ``verifier`` adapter on a ``[problem, proposed_solution]`` prompt."""
        return self._generate(
            adapter_name="verifier",
            raw_prompt=raw_prompt,
            max_new_tokens=self.verifier_max_new_tokens,
            temperature=self.temperature_verifier,
            use_stop_strings=True,
        )


def _normalize_hint_key(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _safe_rate(num: int, den: int) -> float | None:
    return round(num / den, 6) if den > 0 else None


def _unpack_gen_result(res: str | tuple[str, int] | tuple[str, int, float]) -> tuple[str, int, float]:
    if isinstance(res, str):
        return res.strip(), max(1, len(res.split())), 0.0
    if len(res) == 2:
        return str(res[0]).strip(), int(res[1]), 0.0
    return str(res[0]).strip(), int(res[1]), float(res[2])


def run_single_problem_loop(
    problem: CanonicalProblem,
    *,
    solver_generate: Callable[[str], str | tuple[str, int] | tuple[str, int, float]],
    verifier_generate: Callable[[str], str | tuple[str, int] | tuple[str, int, float]],
    max_iterations: int = 3,
    verify_final_iteration: bool = True,
    stop_on_duplicate_hint: bool = True,
    rel_tol: float = 0.015,
) -> ProblemLoopResult:
    """Execute up to ``max_iterations`` Solver -> Verifier cycles on a single problem."""
    if max_iterations < 1:
        raise ValueError(f"max_iterations must be >= 1, got {max_iterations}")

    _, norm_ref_ans = prepare_reference_for_validation(problem.reference_solution)
    accumulated_directives: list[str] = []
    seen_hint_keys: set[str] = set()
    traces: list[TurnTrace] = []
    stop_reason = "max_iterations_reached"

    for iteration in range(1, max_iterations + 1):
        turn = iteration - 1
        directives_snapshot = list(accumulated_directives)
        solver_prompt = format_solver_prompt(problem.problem_statement, directives_snapshot)
        t_s0 = time.monotonic()
        solver_raw, solver_tokens, solver_lat = _unpack_gen_result(solver_generate(solver_prompt))
        if solver_lat == 0.0:
            solver_lat = time.monotonic() - t_s0

        val_res = score_solver_output(solver_raw, problem.reference_solution, rel_tol=rel_tol)

        should_verify = (iteration < max_iterations) or verify_final_iteration
        verifier_prompt: str | None = None
        verifier_raw: str | None = None
        verifier_verdict: str | None = None
        verifier_directive: str | None = None
        verifier_decision_ok: bool | None = None
        verifier_tokens = 0
        verifier_lat = 0.0

        if should_verify:
            verifier_prompt = format_verifier_prompt(problem.problem_statement, solver_raw)
            t_v0 = time.monotonic()
            verifier_raw, verifier_tokens, verifier_lat = _unpack_gen_result(verifier_generate(verifier_prompt))
            if verifier_lat == 0.0:
                verifier_lat = time.monotonic() - t_v0

            parsed = parse_verifier_response(verifier_raw, problem_statement=problem.problem_statement)
            verifier_verdict = parsed.verdict
            verifier_directive = parsed.directive
            if parsed.verdict == "APPROVE":
                verifier_decision_ok = bool(val_res.is_equivalent)
            elif parsed.verdict == "CHALLENGE":
                verifier_decision_ok = not bool(val_res.is_equivalent)

        traces.append(
            TurnTrace(
                iteration=iteration,
                turn=turn,
                accumulated_directives=directives_snapshot,
                solver_prompt=solver_prompt,
                solver_output=solver_raw,
                solver_validation=val_res.to_dict(),
                solver_is_correct=bool(val_res.is_equivalent),
                solver_tokens=solver_tokens,
                solver_latency_s=round(solver_lat, 4),
                verifier_prompt=verifier_prompt,
                verifier_output=verifier_raw,
                verifier_verdict=verifier_verdict,
                verifier_directive=verifier_directive,
                verifier_is_decision_correct=verifier_decision_ok,
                verifier_tokens=verifier_tokens,
                verifier_latency_s=round(verifier_lat, 4),
            )
        )

        if not should_verify:
            stop_reason = "max_iterations_reached"
            break

        if verifier_verdict == "APPROVE":
            stop_reason = "approved"
            break
        if verifier_verdict != "CHALLENGE":
            stop_reason = "verifier_no_sentinel"
            break
        if iteration >= max_iterations:
            stop_reason = "max_iterations_reached"
            break

        hint = (verifier_directive or "").strip()
        if not hint:
            stop_reason = "empty_directive"
            break
        hint_key = _normalize_hint_key(hint)
        if stop_on_duplicate_hint and hint_key in seen_hint_keys:
            stop_reason = "duplicate_hint_stopped"
            break

        seen_hint_keys.add(hint_key)
        accumulated_directives.append(hint)

    first = traces[0]
    last = traces[-1]
    turn0_ok = first.solver_is_correct
    final_ok = last.solver_is_correct

    return ProblemLoopResult(
        problem_id=problem.problem_id,
        domain=problem.domain,
        problem_statement=problem.problem_statement,
        reference_solution=problem.reference_solution,
        normalized_reference_answer=norm_ref_ans,
        iterations_used=len(traces),
        max_iterations=max_iterations,
        stop_reason=stop_reason,
        turn0_correct=turn0_ok,
        final_correct=final_ok,
        turn0_has_answer=bool(first.solver_validation.get("has_answer_sentinel", False)),
        final_has_answer=bool(last.solver_validation.get("has_answer_sentinel", False)),
        turn0_match_method=str(first.solver_validation.get("match_method", "unknown")),
        final_match_method=str(last.solver_validation.get("match_method", "unknown")),
        converted=(not turn0_ok) and final_ok,
        regressed=turn0_ok and (not final_ok),
        total_latency_s=round(sum(t.solver_latency_s + t.verifier_latency_s for t in traces), 4),
        total_tokens=sum(t.solver_tokens + t.verifier_tokens for t in traces),
        traces=traces,
    )


def compute_loop_metrics(
    results: Sequence[ProblemLoopResult],
    *,
    max_iterations: int = 3,
) -> dict[str, Any]:
    """Compute end-to-end Solver accuracy, conversion/regression rates, and Verifier safety diagnostics."""
    n = len(results)
    turn0_correct_n = sum(1 for r in results if r.turn0_correct)
    final_correct_n = sum(1 for r in results if r.final_correct)
    turn0_wrong_n = n - turn0_correct_n
    converted_n = sum(1 for r in results if r.converted)
    regressed_n = sum(1 for r in results if r.regressed)
    turn0_ans_n = sum(1 for r in results if r.turn0_has_answer)
    final_ans_n = sum(1 for r in results if r.final_has_answer)

    turn0_acc = _safe_rate(turn0_correct_n, n)
    final_acc = _safe_rate(final_correct_n, n)
    net_gain = round(final_acc - turn0_acc, 6) if (turn0_acc is not None and final_acc is not None) else None

    # Per-iteration cumulative accuracy (carries forward problems that already stopped earlier)
    per_iteration_acc: dict[str, dict[str, Any]] = {}
    for k in range(1, max_iterations + 1):
        k_correct = 0
        for r in results:
            trace = r.traces[min(k - 1, len(r.traces) - 1)]
            if trace.solver_is_correct:
                k_correct += 1
        per_iteration_acc[f"iter_{k}"] = {
            "correct": k_correct,
            "total": n,
            "accuracy": _safe_rate(k_correct, n),
        }

    # Verifier confusion matrix over all reviewed Solver outputs (positive class = CHALLENGE / flawed)
    tp = fp = tn = fn = unknown_v = 0
    for r in results:
        for t in r.traces:
            if t.verifier_verdict is None:
                continue
            if t.verifier_verdict == "CHALLENGE":
                if t.solver_is_correct:
                    fp += 1
                else:
                    tp += 1
            elif t.verifier_verdict == "APPROVE":
                if t.solver_is_correct:
                    tn += 1
                else:
                    fn += 1
            else:
                unknown_v += 1

    total_reviews = tp + fp + tn + fn + unknown_v
    precision = _safe_rate(tp, tp + fp)
    recall = _safe_rate(tp, tp + fn)
    f1 = (
        round((2 * precision * recall) / (precision + recall), 6)
        if (precision is not None and recall is not None and (precision + recall) > 0)
        else None
    )

    by_domain_raw: dict[str, list[ProblemLoopResult]] = defaultdict(list)
    for r in results:
        by_domain_raw[r.domain].append(r)

    per_domain: dict[str, dict[str, Any]] = {}
    for domain in sorted(by_domain_raw):
        d_rows = by_domain_raw[domain]
        dn = len(d_rows)
        d0 = sum(1 for x in d_rows if x.turn0_correct)
        df = sum(1 for x in d_rows if x.final_correct)
        d0_acc = _safe_rate(d0, dn)
        df_acc = _safe_rate(df, dn)
        per_domain[domain] = {
            "count": dn,
            "turn0_correct": d0,
            "turn0_accuracy": d0_acc,
            "final_correct": df,
            "final_accuracy": df_acc,
            "net_revision_gain": round(df_acc - d0_acc, 6) if (d0_acc is not None and df_acc is not None) else None,
            "converted": sum(1 for x in d_rows if x.converted),
            "regressed": sum(1 for x in d_rows if x.regressed),
        }

    stop_reasons = dict(Counter(r.stop_reason for r in results))
    final_match_methods = dict(Counter(r.final_match_method for r in results))

    return {
        "total_problems": n,
        "max_iterations": max_iterations,
        "solver": {
            "turn0_correct": turn0_correct_n,
            "turn0_accuracy": turn0_acc,
            "final_correct": final_correct_n,
            "final_accuracy": final_acc,
            "net_revision_gain": net_gain,
            "turn0_sentinel_compliance_rate": _safe_rate(turn0_ans_n, n),
            "final_sentinel_compliance_rate": _safe_rate(final_ans_n, n),
            "converted_count": converted_n,
            "turn0_wrong_count": turn0_wrong_n,
            "correction_conversion_rate": _safe_rate(converted_n, turn0_wrong_n),
            "regressed_count": regressed_n,
            "regression_rate": _safe_rate(regressed_n, turn0_correct_n),
            "per_iteration_cumulative": per_iteration_acc,
            "final_match_method_counts": final_match_methods,
        },
        "verifier": {
            "total_reviews": total_reviews,
            "tp_true_challenge": tp,
            "fp_false_challenge": fp,
            "tn_true_approve": tn,
            "fn_false_approve": fn,
            "unknown_no_sentinel": unknown_v,
            "verifier_sentinel_accuracy": _safe_rate(tp + tn, total_reviews),
            "verifier_challenge_precision": precision,
            "verifier_challenge_recall": recall,
            "verifier_challenge_f1": f1,
            "verifier_false_approval_rate": _safe_rate(fn, tp + fn),
            "verifier_false_challenge_rate": _safe_rate(fp, fp + tn),
            "verifier_overall_false_approval_rate": _safe_rate(fn, total_reviews),
            "verifier_approve_rate": _safe_rate(tn + fn, total_reviews),
            "verifier_challenge_rate": _safe_rate(tp + fp, total_reviews),
        },
        "loop_efficiency": {
            "mean_iterations_used": round(sum(r.iterations_used for r in results) / n, 4) if n else None,
            "mean_latency_s": round(sum(r.total_latency_s for r in results) / n, 4) if n else None,
            "total_latency_s": round(sum(r.total_latency_s for r in results), 4),
            "mean_tokens_per_problem": round(sum(r.total_tokens for r in results) / n, 2) if n else None,
            "stop_reason_counts": stop_reasons,
        },
        "per_domain": per_domain,
    }


def _load_existing_problem_results(jsonl_path: Path) -> list[ProblemLoopResult]:
    """Load previously saved ProblemLoopResult records when resuming an interrupted validation run."""
    if not jsonl_path.is_file():
        return []
    loaded: list[ProblemLoopResult] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            raw = json.loads(stripped)
            traces = [TurnTrace(**t) for t in raw.pop("traces", [])]
            loaded.append(ProblemLoopResult(**raw, traces=traces))
    return loaded


def run_adversarial_validation_loop(
    problems: Sequence[CanonicalProblem],
    *,
    engine: DualLoRAEngine | None = None,
    solver_generate: Callable[[str], str | tuple[str, int] | tuple[str, int, float]] | None = None,
    verifier_generate: Callable[[str], str | tuple[str, int] | tuple[str, int, float]] | None = None,
    max_iterations: int = 3,
    verify_final_iteration: bool = True,
    stop_on_duplicate_hint: bool = True,
    rel_tol: float = 0.015,
    output_dir: str | Path = "outputs/validation/e4_loop_iter3",
    resume: bool = False,
    metadata: dict[str, Any] | None = None,
    progress_callback: Callable[[int, int, ProblemLoopResult], None] | None = None,
) -> dict[str, Any]:
    """Run E4A + E4B multi-turn validation across ``problems``, writing JSONL + summary artifacts."""
    if engine is not None:
        solver_fn = engine.generate_solver
        verifier_fn = engine.generate_verifier
    elif solver_generate is not None and verifier_generate is not None:
        solver_fn = solver_generate
        verifier_fn = verifier_generate
    else:
        raise ValueError("Provide either `engine` or both `solver_generate` and `verifier_generate`.")

    out_path = resolve_project_path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    traces_jsonl = out_path / "results_per_problem.jsonl"
    summary_json = out_path / "summary.json"

    results: list[ProblemLoopResult] = []
    completed_ids: set[str] = set()
    if resume and traces_jsonl.is_file():
        results = _load_existing_problem_results(traces_jsonl)
        completed_ids = {r.problem_id for r in results}
        if completed_ids:
            logger.info(f"Resuming from {traces_jsonl}: skipping {len(completed_ids)} completed problems.")
    elif traces_jsonl.is_file():
        traces_jsonl.unlink()

    total = len(problems)
    with traces_jsonl.open("a", encoding="utf-8") as f:
        for idx, problem in enumerate(problems, start=1):
            if problem.problem_id in completed_ids:
                continue
            res = run_single_problem_loop(
                problem,
                solver_generate=solver_fn,
                verifier_generate=verifier_fn,
                max_iterations=max_iterations,
                verify_final_iteration=verify_final_iteration,
                stop_on_duplicate_hint=stop_on_duplicate_hint,
                rel_tol=rel_tol,
            )
            results.append(res)
            f.write(json.dumps(res.to_dict(), ensure_ascii=False) + "\n")
            f.flush()
            if progress_callback is not None:
                progress_callback(idx, total, res)

    metrics = compute_loop_metrics(results, max_iterations=max_iterations)

    def _build_turn_eval_rows(use_final: bool) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for res in results:
            if not res.traces:
                continue
            tr = res.traces[-1] if use_final else res.traces[0]
            val = dict(getattr(tr, "solver_validation", None) or {})
            source = "physgym_seed" if str(res.problem_id).startswith("physgym") else "curated_in_domain"
            rows.append(
                {
                    "problem_id": res.problem_id,
                    "domain": res.domain,
                    "source": source,
                    "is_equivalent": bool(val.get("is_equivalent", res.final_correct if use_final else res.turn0_correct)),
                    "match_method": str(val.get("match_method", res.final_match_method if use_final else res.turn0_match_method)),
                    "has_answer_sentinel": bool(
                        val.get("has_answer_sentinel", res.final_has_answer if use_final else res.turn0_has_answer)
                    ),
                    "unit_consistent": bool(val.get("unit_consistent", False)),
                    "predicted_answer": str(val.get("extracted_candidate_answer", "")),
                    "reference_answer": str(val.get("extracted_reference_answer", res.normalized_reference_answer)),
                    "prediction": str(getattr(tr, "solver_output", "") or ""),
                    "reference_solution": str(res.reference_solution or ""),
                }
            )
        return rows

    from adversarial_physics.evaluation.auditkit_bridge import run_physics_eval_audit

    exp_label = str((metadata or {}).get("experiment_name") or "e4_adversarial_loop")
    payload: dict[str, Any] = {
        "metadata": dict(metadata or {}),
        "metrics": metrics,
        "lexsi_audit": {
            "turn0": run_physics_eval_audit(
                {"per_sample": _build_turn_eval_rows(use_final=False)},
                experiment_name=f"{exp_label}_turn0",
            ),
            "final": run_physics_eval_audit(
                {"per_sample": _build_turn_eval_rows(use_final=True)},
                experiment_name=f"{exp_label}_final",
            ),
        },
        "artifacts": {
            "output_dir": str(out_path),
            "results_per_problem_jsonl": str(traces_jsonl),
            "summary_json": str(summary_json),
        },
    }
    if engine is not None:
        payload["metadata"].update(
            {
                "base_model": engine.base_model_id,
                "solver_adapter": engine.solver_source,
                "verifier_adapter": engine.verifier_source,
                "prompt_mode": engine.prompt_mode,
                "solver_max_new_tokens": engine.solver_max_new_tokens,
                "verifier_max_new_tokens": engine.verifier_max_new_tokens,
                "temperature_solver": engine.temperature_solver,
                "temperature_verifier": engine.temperature_verifier,
            }
        )

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return payload
