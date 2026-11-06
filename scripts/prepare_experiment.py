"""Resolve Hub revisions and create a provenance manifest before training.

Run from the repository root:
    python scripts/prepare_experiment.py --config configs/training/aligntune_solver_curated.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adversarial_physics.utils.config import (  # noqa: E402
    load_evaluation_config,
    load_training_config,
    resolve_project_path,
)
from adversarial_physics.utils.hf_hub import resolve_dataset_ref, resolve_model_ref  # noqa: E402
from adversarial_physics.utils.manifest import create_experiment_manifest  # noqa: E402
from adversarial_physics.utils.provenance import git_info  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the experiment YAML")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow uncommitted tracked changes (not recommended for recorded experiments).",
    )
    args = parser.parse_args()

    config_path = resolve_project_path(args.config, root=ROOT)
    config = load_training_config(config_path)
    evaluation_path = resolve_project_path(config["evaluation"]["config_path"], root=ROOT)
    load_evaluation_config(evaluation_path)

    git = git_info(cwd=ROOT)
    require_clean = config.get("reproducibility", {}).get("require_clean_git", True)
    if require_clean and git["working_tree_dirty"] and not args.allow_dirty:
        raise SystemExit(
            "Git working tree has changes. Commit the code/config first, or pass --allow-dirty "
            "and accept that the experiment is not fully tied to a clean commit."
        )
    if not git["is_git_repository"]:
        raise SystemExit("This project directory is not a Git repository. Run git init first.")

    dataset_cfg = config["dataset"]
    dataset_ref = resolve_dataset_ref(
        repo_id=dataset_cfg["repo_id"],
        revision=dataset_cfg["revision"],
        split=dataset_cfg["split"],
        config_name=dataset_cfg.get("config_name"),
        data_files=dataset_cfg.get("data_files"),
    )

    model_cfg = config["base_model"]
    model_ref = resolve_model_ref(
        repo_id=model_cfg["repo_id"], revision=model_cfg["revision"]
    )

    manifest_path = create_experiment_manifest(
        experiment_config_path=config_path,
        config=config,
        dataset_ref=dataset_ref.to_dict(),
        model_ref=model_ref.to_dict(),
        root=ROOT,
    )

    print("Experiment prepared successfully.")
    print(f"Experiment ID: {config['experiment_id']}")
    print(f"Dataset: {dataset_ref.repo_id}@{dataset_ref.resolved_revision} ({dataset_ref.split})")
    print(f"Base model: {model_ref.repo_id}@{model_ref.resolved_revision}")
    print(f"Manifest: {manifest_path.relative_to(ROOT)}")
    print("Next: load the dataset at the resolved revision, then start the AlignTune run.")


if __name__ == "__main__":
    main()
