"""AlignTune post-training package for Tiny-Aya-Global LoRA adapters (Stage 4)."""

from adversarial_physics.training.run_aligntune import (
    TaskAccuracyCallback,
    attach_early_stopping_and_accuracy_callbacks,
    build_aligntune_trainer_kwargs,
    compute_gradient_accumulation_steps,
    create_aligntune_sft_trainer,
    ensure_aligntune_dataset_directory,
    ensure_dataset,
    evaluate_task_accuracy,
    extract_answer_sentinel,
    extract_verifier_sentinel,
    format_alpaca_prompt,
    load_yaml_config,
    make_task_accuracy_metric_fn,
    resolve_dataset_key,
    train_adapter,
)

__all__ = [
    "TaskAccuracyCallback",
    "attach_early_stopping_and_accuracy_callbacks",
    "build_aligntune_trainer_kwargs",
    "compute_gradient_accumulation_steps",
    "create_aligntune_sft_trainer",
    "ensure_aligntune_dataset_directory",
    "ensure_dataset",
    "evaluate_task_accuracy",
    "extract_answer_sentinel",
    "extract_verifier_sentinel",
    "format_alpaca_prompt",
    "load_yaml_config",
    "make_task_accuracy_metric_fn",
    "resolve_dataset_key",
    "train_adapter",
]
