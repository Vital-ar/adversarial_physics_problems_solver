"""PhysGym dataset ingestion and normalization (Stage 1.1)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from adversarial_physics.data.processing.normalization import normalize_physgym_records
from adversarial_physics.data.schemas.problem import CanonicalProblem
from adversarial_physics.utils.config import load_ingestion_config
from adversarial_physics.utils.hf_hub import (
    HubDatasetRef,
    load_dataset_ref,
    load_local_dataset,
    resolve_dataset_ref,
)

DEFAULT_PHYSGYM_REPO = "YimengChen/PhysGym"
DEFAULT_PHYSGYM_REVISION = "8e122553897e77495dc2690f16d8055aea1655d7"


def load_physgym_dataset(
    *,
    repo_id: str = DEFAULT_PHYSGYM_REPO,
    revision: str = DEFAULT_PHYSGYM_REVISION,
    split: str = "train",
    kind: str = "hub",
    local_path: str | None = None,
    local_format: str = "json",
    data_files: Any | None = None,
    config_name: str | None = None,
    canonicalize_domains: bool = False,
    domain_aliases: dict[str, str] | None = None,
    id_prefix: str = "physgym",
    resolve_hub_sha: bool = True,
    token: str | None = None,
) -> tuple[list[CanonicalProblem], dict[str, Any]]:
    """Load PhysGym from Hugging Face Hub or local files and normalize into CanonicalProblem records.

    Returns:
        Tuple of (normalized CanonicalProblem list, dataset provenance dictionary).
    """
    if kind == "local":
        if not local_path:
            raise ValueError("local_path is required when kind='local'")
        raw_ds = load_local_dataset(
            format=local_format,
            path=local_path,
            split=split,
            data_files=data_files,
        )
        provenance = {
            "kind": "local",
            "path": str(local_path),
            "format": local_format,
            "split": split,
            "num_rows": len(raw_ds),
        }
    elif kind == "hub":
        if resolve_hub_sha:
            ref = resolve_dataset_ref(
                repo_id=repo_id,
                revision=revision,
                split=split,
                config_name=config_name,
                data_files=data_files,
                token=token,
            )
        else:
            ref = HubDatasetRef(
                repo_id=repo_id,
                requested_revision=revision,
                resolved_revision=revision,
                split=split,
                config_name=config_name,
                data_files=data_files,
            )
        raw_ds = load_dataset_ref(ref, token=token)
        provenance = {
            "kind": "hub",
            **ref.to_dict(),
            "num_rows": len(raw_ds),
            "fingerprint": getattr(raw_ds, "_fingerprint", None),
        }
    else:
        raise ValueError(f"Unsupported dataset kind: {kind!r}. Must be 'hub' or 'local'.")

    problems = normalize_physgym_records(
        raw_ds,
        canonicalize_domains=canonicalize_domains,
        domain_aliases=domain_aliases,
        id_prefix=id_prefix,
    )
    return problems, provenance


def load_physgym_from_config(
    config_or_path: str | Path | dict[str, Any],
    *,
    canonicalize_domains: bool | None = None,
    resolve_hub_sha: bool = True,
    token: str | None = None,
) -> tuple[list[CanonicalProblem], dict[str, Any]]:
    """Load PhysGym using an ingestion YAML config path or dictionary."""
    if isinstance(config_or_path, (str, Path)):
        config = load_ingestion_config(config_or_path)
    else:
        config = config_or_path

    source = config["source"]
    norm_cfg = config.get("normalization", {})
    use_canonical = (
        canonicalize_domains
        if canonicalize_domains is not None
        else bool(norm_cfg.get("canonicalize_domains", False))
    )
    id_prefix = str(norm_cfg.get("id_prefix", "physgym"))
    domain_aliases = norm_cfg.get("domain_aliases")

    return load_physgym_dataset(
        repo_id=source.get("repo_id", DEFAULT_PHYSGYM_REPO),
        revision=source.get("revision", DEFAULT_PHYSGYM_REVISION),
        split=source.get("split", "train"),
        kind=source.get("kind", "hub"),
        local_path=source.get("path"),
        local_format=source.get("format", "json"),
        data_files=source.get("data_files"),
        config_name=source.get("config_name"),
        canonicalize_domains=use_canonical,
        domain_aliases=domain_aliases,
        id_prefix=id_prefix,
        resolve_hub_sha=resolve_hub_sha,
        token=token,
    )
