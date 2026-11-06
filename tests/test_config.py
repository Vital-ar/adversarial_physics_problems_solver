from pathlib import Path

import pytest
import yaml

from adversarial_physics.utils.config import (
    ConfigError,
    load_ingestion_config,
    load_training_config,
)


def test_load_training_config_accepts_complete_config(tmp_path: Path, monkeypatch):
    data = {
        "experiment_id": "E2_solver_curated",
        "dataset": {"repo_id": "user/data", "revision": "abc123", "split": "train"},
        "base_model": {"repo_id": "CohereLabs/tiny-aya-global", "revision": "main"},
        "training": {"method": "sft", "adapter": "lora", "seed": 42, "parameters": {}},
        "evaluation": {"config_path": "configs/evaluation/auditkit_benchmark.yaml"},
        "output": {"repo_id": "user/model", "local_dir": "checkpoints/solver_curated_lora"},
    }
    config_path = tmp_path / "valid.yaml"
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    monkeypatch.setattr("adversarial_physics.utils.config.project_root", lambda: tmp_path)
    loaded = load_training_config(config_path)
    assert loaded["experiment_id"] == "E2_solver_curated"


def test_load_training_config_rejects_missing_dataset_revision(tmp_path: Path, monkeypatch):
    data = {
        "experiment_id": "E2_solver_curated",
        "dataset": {"repo_id": "user/data", "split": "train"},
        "base_model": {"repo_id": "CohereLabs/tiny-aya-global", "revision": "main"},
        "training": {"method": "sft", "adapter": "lora", "seed": 42},
        "evaluation": {"config_path": "auditkit_benchmark.yaml"},
        "output": {"repo_id": "user/model", "local_dir": "checkpoints/solver_curated_lora"},
    }
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    monkeypatch.setattr("adversarial_physics.utils.config.project_root", lambda: tmp_path)
    with pytest.raises(ConfigError, match="dataset.revision"):
        load_training_config(config_path)


def test_load_ingestion_config_accepts_valid_physgym_config(tmp_path: Path, monkeypatch):
    data = {
        "dataset_id": "physgym_seed_ingestion",
        "source": {
            "kind": "hub",
            "repo_id": "YimengChen/PhysGym",
            "revision": "8e122553897e77495dc2690f16d8055aea1655d7",
            "split": "train",
        },
        "splitting": {"seed": 42, "train_size": 21, "val_size": 38, "test_size": 38},
    }
    config_path = tmp_path / "ingestion.yaml"
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    monkeypatch.setattr("adversarial_physics.utils.config.project_root", lambda: tmp_path)
    loaded = load_ingestion_config(config_path)
    assert loaded["source"]["repo_id"] == "YimengChen/PhysGym"
    assert loaded["splitting"]["train_size"] == 21
