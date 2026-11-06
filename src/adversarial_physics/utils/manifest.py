"""Create and update experiment manifests."""
from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from .config import project_root, resolve_project_path
from .provenance import git_info, runtime_info, sha256_file


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _to_yaml_safe(value: Any, location: str = "root") -> Any:
    """Convert common library-specific scalar/container types to plain YAML types.

    Some libraries expose values that behave like built-in scalars but are
    subclasses (for example, torch.torch_version.TorchVersion, a str subclass).
    PyYAML SafeDumper dispatches on the exact type and may reject those values.
    This normalizer also makes manifests portable across runtimes.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return str(value)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return _to_yaml_safe(value.value, location)
    if isinstance(value, Mapping):
        normalized = {}
        for key, item in value.items():
            safe_key = _to_yaml_safe(key, f"{location}.<key>")
            if not isinstance(safe_key, (str, int, float, bool)):
                raise TypeError(
                    f"Unsupported YAML mapping key at {location}: {type(key).__name__}"
                )
            normalized[safe_key] = _to_yaml_safe(item, f"{location}.{safe_key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            _to_yaml_safe(item, f"{location}[{index}]")
            for index, item in enumerate(value)
        ]

    # Handles NumPy scalar types and similar objects exposing item().
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            scalar = item_method()
        except (TypeError, ValueError):
            scalar = value
        if scalar is not value:
            return _to_yaml_safe(scalar, location)

    raise TypeError(
        f"Unsupported value for YAML manifest at {location}: "
        f"{type(value).__module__}.{type(value).__qualname__}. "
        "Convert it to a plain Python scalar/container before saving."
    )


def _write_yaml_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    safe_data = _to_yaml_safe(data)
    with temporary.open("w", encoding="utf-8") as file:
        yaml.safe_dump(safe_data, file, sort_keys=False, allow_unicode=True)
    temporary.replace(path)


def create_experiment_manifest(
    *,
    experiment_config_path: str | Path,
    config: dict[str, Any],
    dataset_ref: dict[str, Any],
    model_ref: dict[str, Any],
    root: str | Path | None = None,
) -> Path:
    """Create the initial manifest before training starts."""
    repo_root = Path(root) if root else project_root()
    experiment_id = str(config["experiment_id"])
    if not experiment_id.replace("_", "").replace("-", "").isalnum():
        raise ValueError("experiment_id may contain only letters, numbers, '_' and '-'")

    config_path = resolve_project_path(experiment_config_path, root=repo_root)
    config_relative = str(config_path.relative_to(repo_root))

    eval_config_path = resolve_project_path(
        config["evaluation"]["config_path"], root=repo_root
    )
    if not eval_config_path.is_file():
        raise FileNotFoundError(f"Evaluation configuration not found: {eval_config_path}")
    eval_config_hash = sha256_file(eval_config_path)

    curation_config_info = None
    curation_config_value = config["dataset"].get("curation_config")
    if curation_config_value:
        curation_config_path = resolve_project_path(curation_config_value, root=repo_root)
        if not curation_config_path.is_file():
            raise FileNotFoundError(f"Curation configuration not found: {curation_config_path}")
        curation_config_info = {
            "path": str(curation_config_path.relative_to(repo_root)),
            "sha256": sha256_file(curation_config_path),
        }

    manifest = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "status": "prepared",
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "code": {
            **git_info(cwd=repo_root),
            "experiment_config": config_relative,
            "experiment_config_sha256": sha256_file(config_path),
        },
        "dataset": {**dataset_ref, "curation_config": curation_config_info},
        "base_model": model_ref,
        "training": config["training"],
        "evaluation": {
            "config_path": config["evaluation"]["config_path"],
            "config_sha256": eval_config_hash,
            "metrics_file": f"experiments/{experiment_id}/metrics.json",
        },
        "output": {
            "repo_id": config["output"]["repo_id"],
            "local_dir": config["output"]["local_dir"],
            "hf_revision": None,
            "revision_label": config["output"].get("revision_label"),
        },
        "runtime": runtime_info(),
    }

    manifest_path = repo_root / "experiments" / experiment_id / "manifest.yaml"
    _write_yaml_atomic(manifest_path, manifest)
    return manifest_path


def update_experiment_manifest(
    manifest_path: str | Path,
    *,
    status: str | None = None,
    output_hf_revision: str | None = None,
    metrics_file: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Update a manifest after training/evaluation or artifact publication."""
    path = Path(manifest_path)
    with path.open("r", encoding="utf-8") as file:
        manifest = yaml.safe_load(file)
    if not isinstance(manifest, dict):
        raise ValueError(f"Invalid manifest: {path}")

    if status is not None:
        manifest["status"] = status
    if output_hf_revision is not None:
        manifest.setdefault("output", {})["hf_revision"] = output_hf_revision
    if metrics_file is not None:
        manifest.setdefault("evaluation", {})["metrics_file"] = metrics_file
    if extra:
        manifest.update(extra)
    manifest["updated_at_utc"] = utc_now()
    _write_yaml_atomic(path, manifest)
    return manifest
