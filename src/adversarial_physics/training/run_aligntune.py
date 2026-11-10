"""AlignTune LoRA Post-Training Runner for Tiny-Aya-Global (Stage 4).

Trains parameter-efficient LoRA adapters for the 4 Physics Solver ablation tiers
(E1, E2, E3, E4A) and the Physics Verifier (E4B) on top of `CohereLabs/tiny-aya-global`
strictly using the mandatory Lexsi `aligntune` library (`create_sft_trainer`).

Key Design Features:
1. Parameterized YAML Config: Reads all defaults from `configs/training/aligntune_training.yaml`
   while supporting CLI overrides for rapid hyperparameter sweeps.
2. Clean Batch Sizing: Automatically computes:
       gradient_accumulation_steps = max(1, effective_batch_size // per_device_batch_size)
   With initial `effective_batch_size=16` and `per_device_train_batch_size=1`, this yields
   `gradient_accumulation_steps=16`. Multi-GPU hardware placement is handled automatically via `device_map="auto"`.
3. Masked Response-Only Cross-Entropy Loss: Configures AlignTune's `SFTTrainingConfig` with
   `task_type="instruction_following"`, `completion_only_loss=True`, `loss_type="nll"`, and
   `label_smoothing_factor=0.0`, masking prompt/instruction tokens with `-100` so gradients
   update strictly on physical derivations and terminal sentinels (`ANSWER:` or `[[CHALLENGE]]`/`[[APPROVE]]`).
4. Validation Early Stopping (`max_epochs=10`, `patience=2`, `threshold=0.005`): Evaluates at the
   end of every epoch on the 38 held-out PhysGym validation problems (Solvers) or a deterministic
   10% validation split (Verifier), restoring the best checkpoint (`load_best_model_at_end=True`).
5. Task Accuracy Evaluation Callback: During training, monitors `eval_loss` alongside end-to-end task accuracy
   (`eval_final_answer_accuracy` for Solvers via `validate_candidate_answer`;
   `eval_verifier_sentinel_accuracy` for Verifier via `TaskAccuracyCallback`), deferring the full 18-metric `AuditKIT`
   suite (identical for validation and test sets) to offline evaluation.
6. Automatic Hugging Face Hub Dataset Pull: If local train or validation dataset files are missing
   (e.g., on a fresh Kaggle instance), automatically pulls them from `vital-ar/<dataset_name>` on HF Hub.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml
from aligntune import create_sft_trainer


from adversarial_physics.data.curation.answer_validator import (
    extract_answer_line,
    validate_candidate_answer,
)

# Setup logger
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("run_aligntune")

# Canonical dataset alias map so both SFT directory names, HF Hub repo names, and Blueprint experiment IDs work
DATASET_ALIASES: Dict[str, str] = {
    "E1_solver_manual_deterministic_only": "solver_e1_manual_det",
    "E2_solver_curatorkit_no_llm": "solver_e2_curatorkit_nollm",
    "E3_solver_curatorkit_with_llm": "solver_e3_canonical_llm",
    "E4_solver_adversarial_curated": "solver_e4_adversarial",
    "E4_verifier_curated": "verifier_e4_curated",
    "e1": "solver_e1_manual_det",
    "e2": "solver_e2_curatorkit_nollm",
    "e3": "solver_e3_canonical_llm",
    "e4a": "solver_e4_adversarial",
    "e4b": "verifier_e4_curated",
    "physics-solver-e1-manual-rules": "solver_e1_manual_det",
    "vital-ar/physics-solver-e1-manual-rules": "solver_e1_manual_det",
    "physics-solver-e2-curatorkit-nollm": "solver_e2_curatorkit_nollm",
    "vital-ar/physics-solver-e2-curatorkit-nollm": "solver_e2_curatorkit_nollm",
    "physics-solver-e3-canonical-llm": "solver_e3_canonical_llm",
    "vital-ar/physics-solver-e3-canonical-llm": "solver_e3_canonical_llm",
    "physics-solver-e4-adversarial": "solver_e4_adversarial",
    "vital-ar/physics-solver-e4-adversarial": "solver_e4_adversarial",
    "physics-verifier-e4-curated": "verifier_e4_curated",
    "vital-ar/physics-verifier-e4-curated": "verifier_e4_curated",
}


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_dataset_key(dataset_arg: str, config: Dict[str, Any]) -> str:
    """Resolve CLI `--dataset` argument (or experiment ID alias) to a canonical dataset key."""
    datasets_cfg = config.get("datasets", {})
    if dataset_arg in datasets_cfg:
        return dataset_arg
    mapped = DATASET_ALIASES.get(dataset_arg)
    if mapped and mapped in datasets_cfg:
        return mapped
    # Also check `experiment_id`, `dataset_key`, or `hf_repo_id` fields inside datasets_cfg
    for key, info in datasets_cfg.items():
        if isinstance(info, dict):
            repo_id = info.get("hf_repo_id", "")
            if (
                info.get("experiment_id") == dataset_arg
                or info.get("dataset_key") == dataset_arg
                or repo_id == dataset_arg
                or (repo_id and repo_id.split("/")[-1] == dataset_arg)
            ):
                return key
    available = list(datasets_cfg.keys())
    raise ValueError(
        f"Dataset '{dataset_arg}' not defined in config. Available datasets: {available} "
        f"(or aliases: {list(DATASET_ALIASES.keys())})"
    )


def compute_gradient_accumulation_steps(
    effective_batch_size: int = 16,
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: Optional[int] = None,
) -> int:
    """Determine gradient accumulation steps directly from manual setting or batch config.

    If `gradient_accumulation_steps` is explicitly supplied, returns it directly.
    Otherwise computes: max(1, effective_batch_size // per_device_batch_size).
    """
    if gradient_accumulation_steps is not None:
        if gradient_accumulation_steps <= 0:
            raise ValueError(f"gradient_accumulation_steps must be positive, got {gradient_accumulation_steps}")
        logger.info(f"Using manual gradient_accumulation_steps={gradient_accumulation_steps}")
        return gradient_accumulation_steps

    if effective_batch_size <= 0:
        raise ValueError(f"effective_batch_size must be positive, got {effective_batch_size}")
    if per_device_batch_size <= 0:
        raise ValueError(f"per_device_batch_size must be positive, got {per_device_batch_size}")

    grad_accum_steps = max(1, effective_batch_size // per_device_batch_size)
    logger.info(
        f"Batch Configuration: Target Effective Batch={effective_batch_size} | "
        f"Per-Device Batch={per_device_batch_size} | "
        f"Computed Grad Accumulation Steps={grad_accum_steps}"
    )
    return grad_accum_steps


def ensure_dataset(
    local_path_str: str,
    hf_repo_id: Optional[str] = None,
    split_name: str = "train",
    hf_token: Optional[str] = None,
) -> str:
    """Ensure a specific JSONL dataset file exists locally, pulling from Hugging Face Hub if missing."""
    local_path = Path(local_path_str)
    if local_path.is_file():
        logger.info(f"Using local dataset file: {local_path.resolve()}")
        return str(local_path)

    if hf_repo_id:
        logger.warning(
            f"Local file '{local_path}' not found. Downloading '{split_name}.jsonl' "
            f"from Hugging Face Hub ({hf_repo_id})..."
        )
        try:
            from huggingface_hub import hf_hub_download

            filename = f"{split_name}.jsonl"
            local_path.parent.mkdir(parents=True, exist_ok=True)
            downloaded_path = hf_hub_download(
                repo_id=hf_repo_id,
                filename=filename,
                repo_type="dataset",
                local_dir=str(local_path.parent),
                token=hf_token or os.environ.get("HF_TOKEN"),
            )
            logger.info(f"Successfully downloaded {hf_repo_id}/{filename} -> {downloaded_path}")
            return str(downloaded_path)
        except Exception as exc:
            logger.error(f"Failed to download {split_name}.jsonl from HF Hub ({hf_repo_id}): {exc}")

    raise FileNotFoundError(
        f"Dataset file could not be found locally at '{local_path_str}' "
        f"and could not be downloaded from HF Hub ({hf_repo_id})."
    )


def _normalize_dataset_jsonl_whitespace(jsonl_path: Path) -> None:
    """Ensure prompt and instruction end with clean double newlines to avoid BPE boundary token merging."""
    if not jsonl_path.is_file():
        return
    try:
        modified = False
        rows: List[Dict[str, Any]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                for key in ("prompt", "instruction"):
                    if key in row and isinstance(row[key], str) and not row[key].endswith("\n\n"):
                        row[key] = row[key].rstrip() + "\n\n"
                        modified = True
                rows.append(row)
        if modified:
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            logger.info(f"Normalized boundary whitespace in {jsonl_path} ({len(rows)} samples)")
    except Exception as exc:
        logger.warning(f"Could not normalize whitespace in {jsonl_path}: {exc}")


def ensure_aligntune_dataset_directory(
    dataset_key: str,
    config: Dict[str, Any],
    hf_token: Optional[str] = None,
) -> str:
    """Ensure the AlignTune SFT dataset directory (`train.jsonl`, `validation.jsonl`, `README.md`) exists.

    If local files are absent (e.g., on Kaggle), pulls the repository snapshot from Hugging Face Hub
    (`vital-ar/<dataset_key>`) and ensures `validation.jsonl` is present for Solver datasets.
    """
    resolved_key = resolve_dataset_key(dataset_key, config)
    ds_info = config["datasets"][resolved_key]
    hf_repo_id = ds_info.get("hf_repo_id")
    is_verifier = ds_info.get("role") == "verifier"

    local_train_str = ds_info.get("local_path") or ds_info.get("path") or f"data/sft/{resolved_key}/train.jsonl"
    local_train_path = Path(local_train_str)
    local_dir = Path(ds_info.get("local_dir_sft") or local_train_path.parent)

    # 1. Ensure train.jsonl exists (pull snapshot or file from HF Hub if missing)
    if not local_train_path.is_file():
        if hf_repo_id:
            logger.warning(
                f"Local SFT directory '{local_dir}' missing train.jsonl. "
                f"Pulling dataset snapshot from Hugging Face Hub: {hf_repo_id}..."
            )
            try:
                from huggingface_hub import snapshot_download

                local_dir.mkdir(parents=True, exist_ok=True)
                snapshot_download(
                    repo_id=hf_repo_id,
                    repo_type="dataset",
                    local_dir=str(local_dir),
                    token=hf_token or os.environ.get("HF_TOKEN"),
                )
            except Exception as snap_exc:
                logger.warning(f"snapshot_download failed ({snap_exc}); falling back to ensure_dataset...")
                ensure_dataset(
                    local_path_str=str(local_train_path),
                    hf_repo_id=hf_repo_id,
                    split_name="train",
                    hf_token=hf_token,
                )
        else:
            raise FileNotFoundError(
                f"Local train dataset '{local_train_path}' not found and no hf_repo_id configured."
            )

    # 2. For Solver datasets, ensure validation.jsonl is present inside local_dir
    if not is_verifier:
        local_val_in_dir = local_dir / "validation.jsonl"
        if not local_val_in_dir.is_file():
            val_cfg = config.get("validation_dataset", {})
            canonical_val_str = val_cfg.get("local_path") or val_cfg.get("path") or "data/splits/validation.jsonl"
            canonical_val_path = Path(canonical_val_str)
            if canonical_val_path.is_file():
                local_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(canonical_val_path, local_val_in_dir)
                logger.info(f"Copied held-out validation split {canonical_val_path} -> {local_val_in_dir}")
            else:
                # Download from either dataset repo or validation repo on HF Hub
                val_repo = hf_repo_id or val_cfg.get("hf_repo_id")
                downloaded_val = ensure_dataset(
                    local_path_str=str(local_val_in_dir),
                    hf_repo_id=val_repo,
                    split_name="validation",
                    hf_token=hf_token,
                )
                if Path(downloaded_val).resolve() != local_val_in_dir.resolve():
                    shutil.copy2(downloaded_val, local_val_in_dir)

    # 3. Clean boundary whitespace so BPE token merging does not trigger TRL token mismatch warnings
    _normalize_dataset_jsonl_whitespace(local_train_path)
    local_val_path = local_dir / "validation.jsonl"
    if local_val_path.is_file():
        _normalize_dataset_jsonl_whitespace(local_val_path)

    return str(local_dir)


def format_alpaca_prompt(sample: Dict[str, Any]) -> str:
    """Format sample into instruction-following prompt with response sentinel."""
    instruction = (sample.get("instruction") or sample.get("prompt") or "").strip()
    input_text = (sample.get("input") or "").strip()
    output_text = (sample.get("output") or sample.get("target_output") or "").strip()

    if input_text:
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n{output_text}"
        )
    return f"### Instruction:\n{instruction}\n\n### Response:\n{output_text}"


def extract_answer_sentinel(text: str) -> Optional[str]:
    """Extract final answer value from text ending with `ANSWER: <value>`."""
    has_ans, ans = extract_answer_line(text or "")
    if has_ans:
        return ans
    match = re.search(r"ANSWER:\s*(.+)$", text or "", re.IGNORECASE | re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None


def extract_verifier_sentinel(text: str) -> Optional[str]:
    """Extract Verifier verdict sentinel (`APPROVE` or `CHALLENGE`)."""
    if not text:
        return None
    if "[[APPROVE]]" in text:
        return "APPROVE"
    if "[[CHALLENGE]]" in text:
        return "CHALLENGE"
    return None


def evaluate_task_accuracy(
    prediction: str,
    reference: str,
    is_verifier: bool = False,
) -> Dict[str, float]:
    """Compute task accuracy for one (prediction, reference) pair during validation.

    - Solver (`is_verifier=False`): `final_answer_accuracy` via `validate_candidate_answer`
      (exact normalized match, +/-1.5% numerical tolerance, SymPy symbolic equivalence, SI unit check).
    - Verifier (`is_verifier=True`): `verifier_sentinel_accuracy` (`[[APPROVE]]` vs `[[CHALLENGE]]` match).
    """
    if is_verifier:
        pred_verdict = extract_verifier_sentinel(prediction)
        ref_verdict = extract_verifier_sentinel(reference)
        correct = 1.0 if (pred_verdict is not None and pred_verdict == ref_verdict) else 0.0
        return {"verifier_sentinel_accuracy": correct}

    res = validate_candidate_answer(prediction or "", reference or "")
    return {"final_answer_accuracy": 1.0 if res.is_equivalent else 0.0}


def make_task_accuracy_metric_fn(
    is_verifier: bool = False,
) -> Callable[[str, str, str], Dict[str, float]]:
    """Create a callable compatible with AlignTune's `EvaluationConfig.custom_metrics`."""

    def _metric_fn(prediction: str, reference: str, instruction: str) -> Dict[str, float]:
        del instruction
        return evaluate_task_accuracy(
            prediction=prediction,
            reference=reference,
            is_verifier=is_verifier,
        )

    return _metric_fn


def build_aligntune_trainer_kwargs(
    dataset_key: str,
    config: Dict[str, Any],
    dataset_source: Optional[str] = None,
    effective_batch_size: Optional[int] = None,
    per_device_batch_size: Optional[int] = None,
    gradient_accumulation_steps: Optional[int] = None,
    epochs: Optional[int] = None,
    learning_rate: Optional[float] = None,
    lora_r: Optional[int] = None,
    output_dir_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Construct the exact keyword arguments for `aligntune.create_sft_trainer` from YAML config & overrides."""
    import torch

    resolved_key = resolve_dataset_key(dataset_key, config)
    ds_info = config["datasets"][resolved_key]
    is_verifier = ds_info.get("role") == "verifier"
    adapter_name = ds_info.get("adapter_name", f"{resolved_key}_lora")

    hw_cfg = config.get("hardware", {})
    train_cfg = config.get("training_hyperparameters", {})

    manual_accum = (
        gradient_accumulation_steps
        or train_cfg.get("gradient_accumulation_steps")
        or hw_cfg.get("gradient_accumulation_steps")
    )
    eff_batch = effective_batch_size if effective_batch_size is not None else hw_cfg.get("effective_batch_size", 16)
    per_device = (
        per_device_batch_size
        if per_device_batch_size is not None
        else hw_cfg.get("per_device_train_batch_size", 1)
    )
    eval_per_device = hw_cfg.get("per_device_eval_batch_size", per_device)

    grad_accum_steps = compute_gradient_accumulation_steps(
        effective_batch_size=eff_batch,
        per_device_batch_size=per_device,
        gradient_accumulation_steps=manual_accum,
    )

    num_epochs = epochs if epochs is not None else train_cfg.get("num_train_epochs", 10)

    if learning_rate is not None:
        lr = float(learning_rate)
    elif is_verifier:
        lr = float(train_cfg.get("learning_rate_verifier", 1.5e-4))
    else:
        lr = float(train_cfg.get("learning_rate_solver", 2.0e-4))

    lora_cfg = config.get("lora_configuration", {})
    r_val = lora_r if lora_r is not None else lora_cfg.get("r", 16)
    alpha_val = lora_cfg.get("lora_alpha", r_val * 2)
    target_modules = lora_cfg.get(
        "target_modules",
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    base_model_cfg = config.get("base_model", {})
    model_id = base_model_cfg.get("repo_id", "CohereLabs/tiny-aya-global")
    cuda_avail = torch.cuda.is_available()
    num_gpus = torch.cuda.device_count() if cuda_avail else 0
    bf16_ok = cuda_avail and torch.cuda.is_bf16_supported()
    cfg_precision = str(base_model_cfg.get("precision", "auto")).lower()
    if cfg_precision == "bf16" and not bf16_ok:
        precision = "fp16" if cuda_avail else "fp32"
    elif cfg_precision in ("fp16", "bf16", "fp32"):
        precision = cfg_precision
    else:
        precision = "bf16" if bf16_ok else ("fp16" if cuda_avail else "fp32")

    quantization: Dict[str, Any] = {}
    if cuda_avail and base_model_cfg.get("load_in_8bit", False):
        quantization = {"load_in_8bit": True}

    device_map = base_model_cfg.get("device_map")
    if device_map is None:
        device_map = "balanced" if num_gpus > 1 else "auto"

    loss_cfg = config.get("loss_and_collator", {})
    early_stop_cfg = config.get("early_stopping", {})
    accuracy_metric_cfg = config.get("task_accuracy_metric") or config.get("online_validation_metric", {})

    output_dir = output_dir_override or ds_info.get("checkpoint_dir") or f"models/adapters/{adapter_name}"
    resolved_source = dataset_source or ds_info.get("local_dir") or f"data/sft/{resolved_key}"
    val_split_ratio = ds_info.get("val_split_ratio", 0.10 if is_verifier else None)

    trainer_kwargs: Dict[str, Any] = {
        "model_name": model_id,
        "dataset_name": resolved_source,
        "backend": train_cfg.get("backend", "trl"),
        "output_dir": output_dir,
        "device_map": device_map,
        "num_epochs": num_epochs,
        "batch_size": per_device,
        "eval_batch_size": eval_per_device,
        "gradient_accumulation_steps": grad_accum_steps,
        "learning_rate": lr,
        "warmup_ratio": float(train_cfg.get("warmup_ratio", 0.10)),
        "weight_decay": float(train_cfg.get("weight_decay", 0.01)),
        "max_grad_norm": float(train_cfg.get("max_grad_norm", 1.0)),
        "optimizer": train_cfg.get("optimizer", "paged_adamw_8bit") if cuda_avail else "adamw_torch",
        "lr_scheduler": train_cfg.get("lr_scheduler_type", "cosine_with_restarts"),
        "max_seq_length": int(train_cfg.get("max_seq_length", 2048)),
        "seed": int(train_cfg.get("seed", 42)),
        # SFTModelConfig / PeftConfigData
        "precision": precision,
        "quantization": quantization,
        "use_unsloth": False,
        "use_gradient_checkpointing": bool(base_model_cfg.get("use_gradient_checkpointing", True)),
        "trust_remote_code": bool(base_model_cfg.get("trust_remote_code", True)),
        "use_peft": bool(lora_cfg.get("use_peft", True)),
        "lora_r": int(r_val),
        "lora_alpha": int(alpha_val),
        "lora_dropout": float(lora_cfg.get("lora_dropout", 0.05)),
        "lora_target_modules": list(target_modules),
        "lora_bias": str(lora_cfg.get("bias", "none")),
        "train_embeddings": bool(lora_cfg.get("train_embeddings", False)),
        # SFTDatasetConfig
        "task_type": train_cfg.get("task_type", "instruction_following"),
        "instruction_column": "instruction",
        "output_column": "output",
        "input_column": "input",
        "curator_schema_gate": False,
        "curator_clean": False,
        "curator_dedup": "none",
        "val_split_ratio": val_split_ratio,
        "split_seed": int(train_cfg.get("seed", 42)),
        # Masked Response-Only Loss (SFTTrainingConfig)
        "completion_only_loss": bool(loss_cfg.get("completion_only_loss", True)),
        "assistant_only_loss": bool(loss_cfg.get("assistant_only_loss", False)),
        "loss_type": str(loss_cfg.get("loss_type", "nll")),
        "label_smoothing_factor": float(loss_cfg.get("label_smoothing", 0.0)),
        "dataloader_num_workers": int(hw_cfg.get("dataloader_num_workers", 2)),
        # Early Stopping & Checkpointing
        "eval_strategy": str(early_stop_cfg.get("eval_strategy", "epoch")),
        "save_strategy": str(early_stop_cfg.get("save_strategy", "epoch")),
        "logging_steps": int(train_cfg.get("logging_steps", 10)),
        "early_stopping_patience": int(early_stop_cfg.get("early_stopping_patience", 2)),
        "early_stopping_threshold": float(early_stop_cfg.get("early_stopping_threshold", 0.005)),
        "load_best_model_at_end": bool(early_stop_cfg.get("load_best_model_at_end", True)),
        "metric_for_best_model": str(early_stop_cfg.get("metric_for_best_model", "eval_loss")),
        "greater_is_better": bool(early_stop_cfg.get("greater_is_better", False)),
        "loggers": ["tensorboard"],
        "report_to": "tensorboard",
        "logging_dir": str(train_cfg.get("logging_dir", "runs/aligntune")),
        "run_name": adapter_name,
        # SFTEvaluationConfig: disable heavy multi-metric suites during training; compute task accuracy callback
        "compute_perplexity": False,
        "compute_rouge": False,
        "compute_bleu": False,
        "compute_meteor": False,
        "compute_bertscore": False,
        "compute_semantic_similarity": False,
        "compute_codebleu": False,
        "max_samples_for_quality_metrics": int(accuracy_metric_cfg.get("max_eval_generation_samples", 38)),
        "custom_metrics": [make_task_accuracy_metric_fn(is_verifier=is_verifier)],
    }
    return trainer_kwargs


class TaskAccuracyCallback:
    """Hugging Face TrainerCallback that evaluates task accuracy alongside `eval_loss` after each epoch.

    - Solver (`is_verifier=False`): `eval_final_answer_accuracy`
    - Verifier (`is_verifier=True`): `eval_verifier_sentinel_accuracy`
    """

    def __init__(
        self,
        aligntune_trainer: Any,
        is_verifier: bool = False,
        max_eval_samples: int = 38,
        max_new_tokens: int = 512,
    ) -> None:
        self.aligntune_trainer = aligntune_trainer
        self.is_verifier = is_verifier
        self.max_eval_samples = max_eval_samples
        self.max_new_tokens = max_new_tokens
        self.metric_key = (
            "eval_verifier_sentinel_accuracy" if is_verifier else "eval_final_answer_accuracy"
        )

    def to_hf_callback(self) -> Any:
        """Create a `transformers.TrainerCallback` instance wrapping this evaluator."""
        import torch
        from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

        outer = self

        class _HFCallback(TrainerCallback):
            def on_evaluate(
                self,
                args: TrainingArguments,
                state: TrainerState,
                control: TrainerControl,
                metrics: Optional[Dict[str, float]] = None,
                **kwargs: Any,
            ) -> None:
                if metrics is None:
                    return
                model = kwargs.get("model") or getattr(outer.aligntune_trainer, "model", None)
                tokenizer = kwargs.get("processing_class") or kwargs.get("tokenizer") or getattr(
                    outer.aligntune_trainer, "tokenizer", None
                )
                eval_ds = getattr(outer.aligntune_trainer, "eval_dataset", None)
                if model is None or tokenizer is None or eval_ds is None or len(eval_ds) == 0:
                    return

                n_samples = min(len(eval_ds), outer.max_eval_samples)
                scores: List[float] = []
                was_training = model.training
                model.eval()
                try:
                    device = next(model.parameters()).device
                    with torch.no_grad():
                        for idx in range(n_samples):
                            row = eval_ds[idx]
                            prompt = row.get("prompt") or row.get("instruction") or ""
                            reference = row.get("completion") or row.get("output") or row.get("target_output") or ""
                            if not prompt or not reference:
                                continue
                            inputs = tokenizer(
                                prompt,
                                return_tensors="pt",
                                truncation=True,
                                max_length=2048,
                            )
                            inputs = {k: v.to(device) for k, v in inputs.items()}
                            gen_ids = model.generate(
                                **inputs,
                                max_new_tokens=outer.max_new_tokens,
                                do_sample=False,
                                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                            )
                            prompt_len = inputs["input_ids"].shape[1]
                            pred_text = tokenizer.decode(
                                gen_ids[0][prompt_len:], skip_special_tokens=True
                            ).strip()
                            res = evaluate_task_accuracy(
                                prediction=pred_text,
                                reference=reference,
                                is_verifier=outer.is_verifier,
                            )
                            scores.append(next(iter(res.values())))
                except Exception as exc:
                    logger.warning(f"TaskAccuracyCallback warning during generation: {exc}")
                finally:
                    if was_training:
                        model.train()

                if scores:
                    accuracy = float(sum(scores) / len(scores))
                    metrics[outer.metric_key] = accuracy
                    logger.info(
                        f"Epoch {state.epoch:.1f} Validation | "
                        f"eval_loss={metrics.get('eval_loss', float('nan')):.4f} | "
                        f"{outer.metric_key}={accuracy:.4f} ({sum(int(s) for s in scores)}/{len(scores)})"
                    )

        return _HFCallback()


def attach_early_stopping_and_accuracy_callbacks(
    trainer: Any,
    config: Dict[str, Any],
    is_verifier: bool = False,
) -> Any:
    """Attach `TaskAccuracyCallback` and `EarlyStoppingCallback` to AlignTune's `TRLSFTTrainer`."""
    from transformers import EarlyStoppingCallback

    early_stop_cfg = config.get("early_stopping", {})
    accuracy_metric_cfg = config.get("task_accuracy_metric") or config.get("online_validation_metric", {})

    # Ensure AlignTune's SFTEvaluationConfig also holds our custom metric callable
    if getattr(trainer, "config", None) is not None and getattr(trainer.config, "evaluation", None) is not None:
        trainer.config.evaluation.custom_metrics = [make_task_accuracy_metric_fn(is_verifier=is_verifier)]
        if getattr(trainer, "evaluator", None) is not None:
            trainer.evaluator.eval_config = trainer.config.evaluation

    max_eval_samples = int(accuracy_metric_cfg.get("max_eval_generation_samples", 38))
    max_new_tokens = int(
        accuracy_metric_cfg.get("verifier_max_new_tokens", 192)
        if is_verifier
        else accuracy_metric_cfg.get("solver_max_new_tokens", 512)
    )

    accuracy_cb = TaskAccuracyCallback(
        aligntune_trainer=trainer,
        is_verifier=is_verifier,
        max_eval_samples=max_eval_samples,
        max_new_tokens=max_new_tokens,
    ).to_hf_callback()

    early_stopping_cb: Optional[Any] = None
    if early_stop_cfg.get("enabled", True):
        early_stopping_cb = EarlyStoppingCallback(
            early_stopping_patience=int(early_stop_cfg.get("early_stopping_patience", 2)),
            early_stopping_threshold=float(early_stop_cfg.get("early_stopping_threshold", 0.005)),
        )

    orig_get_hf_callbacks = trainer.get_hf_callbacks

    def _enhanced_get_hf_callbacks() -> List[Any]:
        base_callbacks = list(orig_get_hf_callbacks())
        # Place accuracy_cb first so its metric is populated in `metrics` before HuggingFaceCallbackAdapter logs it
        combined: List[Any] = [accuracy_cb] + base_callbacks
        if early_stopping_cb is not None:
            combined.append(early_stopping_cb)
        return combined

    trainer.get_hf_callbacks = _enhanced_get_hf_callbacks
    return trainer


def create_aligntune_sft_trainer(
    dataset_key: str,
    config: Dict[str, Any],
    effective_batch_size: Optional[int] = None,
    per_device_batch_size: Optional[int] = None,
    gradient_accumulation_steps: Optional[int] = None,
    epochs: Optional[int] = None,
    learning_rate: Optional[float] = None,
    lora_r: Optional[int] = None,
    output_dir_override: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> Any:
    """Create and configure a native AlignTune `TRLSFTTrainer` instance via `aligntune.create_sft_trainer`."""
    resolved_key = resolve_dataset_key(dataset_key, config)
    ds_info = config["datasets"][resolved_key]
    is_verifier = ds_info.get("role") == "verifier"

    dataset_dir = ensure_aligntune_dataset_directory(
        dataset_key=resolved_key,
        config=config,
        hf_token=hf_token,
    )

    trainer_kwargs = build_aligntune_trainer_kwargs(
        dataset_key=resolved_key,
        config=config,
        dataset_source=dataset_dir,
        effective_batch_size=effective_batch_size,
        per_device_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        epochs=epochs,
        learning_rate=learning_rate,
        lora_r=lora_r,
        output_dir_override=output_dir_override,
    )

    logger.info(
        f"Creating mandatory AlignTune SFT trainer (`aligntune.create_sft_trainer`) "
        f"for dataset '{resolved_key}' (backend='{trainer_kwargs['backend']}', "
        f"device_map='{trainer_kwargs.get('device_map')}', "
        f"grad_accum={trainer_kwargs.get('gradient_accumulation_steps')})..."
    )
    trainer = create_sft_trainer(**trainer_kwargs)
    attach_early_stopping_and_accuracy_callbacks(
        trainer=trainer,
        config=config,
        is_verifier=is_verifier,
    )
    return trainer


def train_adapter(
    dataset_key: str,
    config: Dict[str, Any],
    effective_batch_size: Optional[int] = None,
    per_device_batch_size: Optional[int] = None,
    gradient_accumulation_steps: Optional[int] = None,
    epochs: Optional[int] = None,
    learning_rate: Optional[float] = None,
    lora_r: Optional[int] = None,
    output_dir_override: Optional[str] = None,
    push_to_hub: bool = False,
    hf_token: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute AlignTune LoRA fine-tuning for a specific dataset recipe (`E1`, `E2`, `E3`, `E4A`, or `E4B`)."""
    # Prevent CUDA memory fragmentation from causing OOM on large vocabulary logits
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    resolved_key = resolve_dataset_key(dataset_key, config)
    ds_info = config["datasets"][resolved_key]
    adapter_name = ds_info.get("adapter_name", f"{resolved_key}_lora")

    hw_cfg = config.get("hardware", {})
    train_cfg = config.get("training_hyperparameters", {})

    manual_accum = (
        gradient_accumulation_steps
        or train_cfg.get("gradient_accumulation_steps")
        or hw_cfg.get("gradient_accumulation_steps")
    )
    eff_batch = effective_batch_size if effective_batch_size is not None else hw_cfg.get("effective_batch_size", 16)
    per_device = (
        per_device_batch_size
        if per_device_batch_size is not None
        else hw_cfg.get("per_device_train_batch_size", 1)
    )
    grad_accum_steps = compute_gradient_accumulation_steps(
        effective_batch_size=eff_batch,
        per_device_batch_size=per_device,
        gradient_accumulation_steps=manual_accum,
    )

    trainer = create_aligntune_sft_trainer(
        dataset_key=resolved_key,
        config=config,
        effective_batch_size=eff_batch,
        per_device_batch_size=per_device,
        gradient_accumulation_steps=grad_accum_steps,
        epochs=epochs,
        learning_rate=learning_rate,
        lora_r=lora_r,
        output_dir_override=output_dir_override,
        hf_token=hf_token,
    )

    sft_cfg = getattr(trainer, "config", None)
    cfg_logging = getattr(sft_cfg, "logging", None)
    cfg_train = getattr(sft_cfg, "train", None)
    cfg_model = getattr(sft_cfg, "model", None)
    cfg_peft = getattr(cfg_model, "peft", None)

    if output_dir_override:
        output_dir = Path(output_dir_override)
    elif cfg_logging is not None and getattr(cfg_logging, "output_dir", None):
        output_dir = Path(cfg_logging.output_dir)
    else:
        output_dir = Path(f"models/adapters/{adapter_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Place a run initialization record so Git tracks this folder immediately
    init_meta = {
        "status": "training_started",
        "adapter_name": adapter_name,
        "dataset_key": resolved_key,
        "base_model": getattr(cfg_model, "name_or_path", "CohereLabs/tiny-aya-global"),
        "gradient_accumulation_steps": grad_accum_steps,
    }
    with open(output_dir / "training_started.json", "w", encoding="utf-8") as f:
        json.dump(init_meta, f, indent=2)

    log_epochs = getattr(cfg_train, "epochs", epochs or 10)
    log_per_device = getattr(cfg_train, "per_device_batch_size", per_device)
    log_grad_accum = getattr(cfg_train, "gradient_accumulation_steps", grad_accum_steps)
    log_lr = getattr(cfg_train, "learning_rate", 2e-4)

    logger.info(
        f"Starting AlignTune SFT training for '{adapter_name}' "
        f"(epochs={log_epochs}, per_device_batch={log_per_device}, "
        f"grad_accum={log_grad_accum}, lr={log_lr})..."
    )
    train_metrics = trainer.train()

    final_save_dir = Path(trainer.save_model(path=str(output_dir / "final_adapter")))
    summary = {
        "adapter_name": adapter_name,
        "dataset_key": resolved_key,
        "experiment_id": ds_info.get("experiment_id", resolved_key),
        "base_model": getattr(cfg_model, "name_or_path", "CohereLabs/tiny-aya-global"),
        "aligntune_backend": "trl",
        "effective_batch_size": eff_batch,
        "per_device_batch_size": log_per_device,
        "gradient_accumulation_steps": log_grad_accum,
        "learning_rate": log_lr,
        "lora_r": getattr(cfg_peft, "rank", lora_r or 16),
        "lora_alpha": getattr(cfg_peft, "alpha", (lora_r or 16) * 2),
        "max_epochs": log_epochs,
        "completion_only_loss": getattr(cfg_train, "completion_only_loss", True),
        "early_stopping_patience": getattr(cfg_train, "early_stopping_patience", 2),
        "early_stopping_threshold": getattr(cfg_train, "early_stopping_threshold", 0.005),
        "final_adapter_dir": str(final_save_dir),
        "train_metrics": train_metrics,
    }
    summary_path = final_save_dir / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    # Also write a copy at top level of adapter output dir for easy git tracking
    with open(output_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"AlignTune training completed! Final adapter & provenance saved to {final_save_dir}")

    if push_to_hub:
        target_repo = ds_info.get("output_repo_id") or ds_info.get("hf_repo_id")
        if target_repo and getattr(trainer, "model", None) is not None:
            token = hf_token or os.environ.get("HF_TOKEN")
            logger.info(f"Pushing trained LoRA adapter to Hugging Face Hub: {target_repo}...")
            trainer.model.push_to_hub(target_repo, token=token)
            if getattr(trainer, "tokenizer", None) is not None:
                trainer.tokenizer.push_to_hub(target_repo, token=token)

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser for `run_aligntune` and `scripts/05_train_aligntune.py`."""
    parser = argparse.ArgumentParser(
        description="Mandatory AlignTune LoRA Post-Training Runner for Physics Solver & Verifier (Stage 4)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help=(
            "Name of the dataset recipe to train: "
            "solver_e1_manual_det, solver_e2_curatorkit_nollm, "
            "solver_e3_canonical_llm, solver_e4_adversarial, or verifier_e4_curated"
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training/aligntune_training.yaml",
        help="Path to parameterized YAML training configuration file",
    )
    parser.add_argument(
        "--effective-batch-size",
        type=int,
        default=None,
        help="Target global effective batch size override (default from YAML: 16)",
    )
    parser.add_argument(
        "--per-device-batch-size",
        type=int,
        default=None,
        help="Per-device micro-batch size override (default from YAML: 1)",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="Manual gradient accumulation steps override (default from YAML: 16)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Maximum training epochs override (default from YAML: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate override (default: 2e-4 for Solvers, 1.5e-4 for Verifier)",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=None,
        help="LoRA rank override (default from YAML: 16)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory override for checkpoints and final LoRA adapter",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the trained LoRA adapter to Hugging Face Hub after training",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face API token (defaults to HF_TOKEN environment variable)",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    """CLI entrypoint for AlignTune LoRA post-training."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = load_yaml_config(args.config)

    return train_adapter(
        dataset_key=args.dataset,
        config=config,
        effective_batch_size=args.effective_batch_size,
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs=args.epochs,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        output_dir_override=args.output_dir,
        push_to_hub=args.push_to_hub,
        hf_token=args.hf_token,
    )


if __name__ == "__main__":
    main()
