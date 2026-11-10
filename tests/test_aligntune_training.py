"""Unit and integration tests for Stage 4 Mandatory AlignTune LoRA Post-Training."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from transformers import EarlyStoppingCallback

from adversarial_physics.training.run_aligntune import (
    TaskAccuracyCallback,
    attach_early_stopping_and_accuracy_callbacks,
    build_aligntune_trainer_kwargs,
    build_arg_parser,
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
)


def test_yaml_configs_exist_and_match_stage4_design() -> None:
    """Verify `aligntune_training.yaml` and `aligntune_solver_curated.yaml` parameters."""
    cfg = load_yaml_config("configs/training/aligntune_training.yaml")
    assert cfg["base_model"]["repo_id"] == "CohereLabs/tiny-aya-global"
    assert cfg["base_model"]["load_in_8bit"] is False
    assert cfg["base_model"]["precision"] == "fp16"
    assert cfg["base_model"]["device_map"] == "balanced"
    assert (
        cfg["base_model"]["max_memory"][0] == "4.5GiB"
        or cfg["base_model"]["max_memory"]["0"] == "4.5GiB"
    )
    assert cfg["hardware"]["effective_batch_size"] == 16
    assert cfg["hardware"]["per_device_train_batch_size"] == 1
    assert cfg["hardware"]["gradient_accumulation_steps"] == 16
    assert cfg["training_hyperparameters"]["num_train_epochs"] == 10
    assert cfg["training_hyperparameters"]["backend"] == "trl"
    assert cfg["training_hyperparameters"]["gradient_accumulation_steps"] == 16
    assert cfg["training_hyperparameters"]["logging_dir"] == "runs/aligntune"
    assert cfg["training_hyperparameters"]["learning_rate_solver"] == 0.0002
    assert cfg["training_hyperparameters"]["learning_rate_verifier"] == 0.00015
    assert cfg["loss_and_collator"]["completion_only_loss"] is True
    assert cfg["loss_and_collator"]["label_smoothing"] == 0.0
    assert cfg["early_stopping"]["early_stopping_patience"] == 2
    assert cfg["early_stopping"]["early_stopping_threshold"] == 0.005
    assert cfg["early_stopping"]["load_best_model_at_end"] is True
    assert cfg["lora_configuration"]["r"] == 16
    assert cfg["lora_configuration"]["lora_alpha"] == 32
    assert cfg["lora_configuration"]["train_embeddings"] is False

    expected_datasets = {
        "solver_e1_manual_det": (1076, "vital-ar/physics-solver-e1-manual-rules"),
        "solver_e2_curatorkit_nollm": (728, "vital-ar/physics-solver-e2-curatorkit-nollm"),
        "solver_e3_canonical_llm": (681, "vital-ar/physics-solver-e3-canonical-llm"),
        "solver_e4_adversarial": (1444, "vital-ar/physics-solver-e4-adversarial"),
        "verifier_e4_curated": (1511, "vital-ar/physics-verifier-e4-curated"),
    }
    for ds_name, (count, hf_id) in expected_datasets.items():
        assert ds_name in cfg["datasets"]
        assert cfg["datasets"][ds_name]["expected_records"] == count
        assert cfg["datasets"][ds_name]["hf_repo_id"] == hf_id

    curated_cfg = load_yaml_config("configs/training/aligntune_solver_curated.yaml")
    assert curated_cfg["training"]["parameters"]["num_train_epochs"] == 10
    assert curated_cfg["training"]["parameters"]["effective_batch_size"] == 16
    assert curated_cfg["training"]["parameters"]["per_device_train_batch_size"] == 1
    assert curated_cfg["datasets"]["E4_verifier_curated"]["expected_records"] == 1511


def test_gradient_accumulation_calculation() -> None:
    """Verify clean gradient accumulation calculation and direct manual specification."""
    # Direct manual setting
    assert compute_gradient_accumulation_steps(gradient_accumulation_steps=16) == 16
    assert compute_gradient_accumulation_steps(gradient_accumulation_steps=8) == 8
    with pytest.raises(ValueError):
        compute_gradient_accumulation_steps(gradient_accumulation_steps=0)

    # Fallback ratio calculation
    assert compute_gradient_accumulation_steps(effective_batch_size=16, per_device_batch_size=1) == 16
    assert compute_gradient_accumulation_steps(effective_batch_size=16, per_device_batch_size=2) == 8
    assert compute_gradient_accumulation_steps(effective_batch_size=32, per_device_batch_size=4) == 8
    assert compute_gradient_accumulation_steps(effective_batch_size=4, per_device_batch_size=8) == 1

    with pytest.raises(ValueError):
        compute_gradient_accumulation_steps(effective_batch_size=0, per_device_batch_size=1)
    with pytest.raises(ValueError):
        compute_gradient_accumulation_steps(effective_batch_size=16, per_device_batch_size=-1)


def test_resolve_dataset_key_and_aliases() -> None:
    """Verify both short SFT names, HF Hub repo names, and Blueprint experiment IDs resolve cleanly."""
    cfg = load_yaml_config("configs/training/aligntune_training.yaml")
    assert resolve_dataset_key("solver_e1_manual_det", cfg) == "solver_e1_manual_det"
    assert resolve_dataset_key("E1_solver_manual_deterministic_only", cfg) == "solver_e1_manual_det"
    assert resolve_dataset_key("E4_solver_adversarial_curated", cfg) == "solver_e4_adversarial"
    assert resolve_dataset_key("E4_verifier_curated", cfg) == "verifier_e4_curated"
    assert resolve_dataset_key("physics-solver-e1-manual-rules", cfg) == "solver_e1_manual_det"
    assert resolve_dataset_key("vital-ar/physics-solver-e1-manual-rules", cfg) == "solver_e1_manual_det"
    assert resolve_dataset_key("vital-ar/physics-verifier-e4-curated", cfg) == "verifier_e4_curated"

    with pytest.raises(ValueError, match="not defined in config"):
        resolve_dataset_key("non_existent_dataset", cfg)


def test_ensure_dataset_hf_hub_fallback(tmp_path: Path) -> None:
    """Verify automatic HF Hub download fallback when local file is absent."""
    existing_file = tmp_path / "train.jsonl"
    existing_file.write_text('{"prompt": "p", "output": "o"}\n', encoding="utf-8")
    assert Path(ensure_dataset(str(existing_file), "vital-ar/physics-solver-e1-manual-rules")) == existing_file

    missing_file = tmp_path / "missing_dir" / "train.jsonl"
    fake_downloaded = tmp_path / "missing_dir" / "train.jsonl"
    with patch("huggingface_hub.hf_hub_download", return_value=str(fake_downloaded)) as mock_dl:
        resolved = ensure_dataset(
            str(missing_file),
            hf_repo_id="vital-ar/physics-solver-e1-manual-rules",
            split_name="train",
        )
        assert resolved == str(fake_downloaded)
        mock_dl.assert_called_once()


def test_task_accuracy_evaluation_solver_and_verifier() -> None:
    """Verify task accuracy evaluation for Solvers (`final_answer_accuracy`) and Verifier (`verifier_sentinel_accuracy`)."""
    # Solver numerical tolerance (+/-1.5%) & symbolic equivalence
    solver_pred_ok = "Step 1: Use F = ma.\nANSWER: 9.81 m/s^2"
    solver_ref = "Derivation...\nANSWER: 9.80 m/s^2"
    assert evaluate_task_accuracy(solver_pred_ok, solver_ref, is_verifier=False) == {
        "final_answer_accuracy": 1.0
    }

    solver_pred_wrong = "Step 1: Use F = ma.\nANSWER: 42.0 m/s^2"
    assert evaluate_task_accuracy(solver_pred_wrong, solver_ref, is_verifier=False) == {
        "final_answer_accuracy": 0.0
    }

    # Verifier sentinel accuracy
    ver_pred_ok = "Conservation of momentum is violated during the inelastic collision.\n[[CHALLENGE]]"
    ver_ref = "Apply conservation of linear momentum before energy.\n[[CHALLENGE]]"
    assert evaluate_task_accuracy(ver_pred_ok, ver_ref, is_verifier=True) == {
        "verifier_sentinel_accuracy": 1.0
    }

    ver_pred_wrong = "The derivation looks valid.\n[[APPROVE]]"
    assert evaluate_task_accuracy(ver_pred_wrong, ver_ref, is_verifier=True) == {
        "verifier_sentinel_accuracy": 0.0
    }

    # AlignTune custom_metrics callable signature: (prediction, reference, instruction)
    custom_fn = make_task_accuracy_metric_fn(is_verifier=False)
    assert custom_fn("ANSWER: 2*pi*sqrt(L/g)", "ANSWER: 2\\pi\\sqrt{\\frac{L}{g}}", "Find T") == {
        "final_answer_accuracy": 1.0
    }


def test_create_aligntune_sft_trainer_and_callbacks() -> None:
    """Verify native `aligntune.create_sft_trainer` instantiation, config fields, callbacks, and DataManager loading."""
    cfg = load_yaml_config("configs/training/aligntune_training.yaml")

    trainer = create_aligntune_sft_trainer(
        dataset_key="solver_e1_manual_det",
        config=cfg,
        effective_batch_size=16,
        per_device_batch_size=1,
        epochs=10,
    )

    assert trainer.config.model.name_or_path == "CohereLabs/tiny-aya-global"
    assert trainer.config.model.max_memory == {0: "4.5GiB", 1: "13GiB"}
    assert trainer.config.train.epochs == 10
    assert trainer.config.train.per_device_batch_size == 1
    assert trainer.config.train.gradient_accumulation_steps == 16
    assert trainer.config.train.learning_rate == 0.0002
    assert trainer.config.train.completion_only_loss is True
    assert trainer.config.train.label_smoothing_factor == 0.0
    assert trainer.config.train.early_stopping_patience == 2
    assert trainer.config.train.early_stopping_threshold == 0.005
    assert trainer.config.train.load_best_model_at_end is True
    assert trainer.config.model.peft.rank == 16
    assert trainer.config.model.peft.alpha == 32
    assert trainer.config.model.train_embeddings is False
    assert trainer.config.evaluation.compute_rouge is False
    assert trainer.config.evaluation.compute_bleu is False
    assert len(trainer.config.evaluation.custom_metrics) == 1

    # Verify HF callbacks include TaskAccuracyCallback, HuggingFaceCallbackAdapter, and EarlyStoppingCallback
    hf_callbacks = trainer.get_hf_callbacks()
    assert len(hf_callbacks) == 3
    assert any(isinstance(cb, EarlyStoppingCallback) for cb in hf_callbacks)

    # Verify AlignTune's DataManager loads Solver E1 (1076 train, 38 validation)
    trainer.setup_data()
    assert len(trainer.train_dataset) == 1076
    assert len(trainer.eval_dataset) == 38
    assert "prompt" in trainer.train_dataset.column_names
    assert "completion" in trainer.train_dataset.column_names

    # Verify Verifier E4 trainer and 10% validation split via AlignTune DataManager
    verifier_trainer = create_aligntune_sft_trainer(
        dataset_key="verifier_e4_curated",
        config=cfg,
    )
    assert verifier_trainer.config.train.learning_rate == 0.00015
    assert verifier_trainer.config.dataset.val_split_ratio == 0.10
    verifier_trainer.setup_data()
    assert len(verifier_trainer.train_dataset) + len(verifier_trainer.eval_dataset) == 1511
    assert len(verifier_trainer.train_dataset) == 1359
    assert len(verifier_trainer.eval_dataset) == 152


def test_cli_arg_parser_and_helpers() -> None:
    """Verify CLI parser flags and prompt/sentinel extraction helpers."""
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--dataset",
            "solver_e4_adversarial",
            "--effective-batch-size",
            "16",
            "--per-device-batch-size",
            "1",
            "--gradient-accumulation-steps",
            "16",
            "--epochs",
            "10",
            "--lr",
            "0.0002",
            "--lora-r",
            "16",
        ]
    )
    assert args.dataset == "solver_e4_adversarial"
    assert args.effective_batch_size == 16
    assert args.per_device_batch_size == 1
    assert args.gradient_accumulation_steps == 16
    assert args.epochs == 10
    assert args.lr == 0.0002
    assert args.lora_r == 16

    assert extract_answer_sentinel("Step 1...\nANSWER: 42 N") == "42 N"
    assert extract_verifier_sentinel("Valid.\n[[APPROVE]]") == "APPROVE"
    assert extract_verifier_sentinel("Error.\n[[CHALLENGE]]") == "CHALLENGE"
    assert "### Response:" in format_alpaca_prompt({"prompt": "Q", "target_output": "A"})
