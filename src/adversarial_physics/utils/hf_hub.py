"""Hugging Face Hub revision resolution and dataset loading."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class HubDatasetRef:
    repo_id: str
    requested_revision: str
    resolved_revision: str
    split: str
    config_name: str | None = None
    data_files: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HubModelRef:
    repo_id: str
    requested_revision: str
    resolved_revision: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _token() -> str | None:
    # Prefer an environment secret; otherwise huggingface_hub may use a saved login.
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")


def resolve_dataset_ref(
    *,
    repo_id: str,
    revision: str,
    split: str,
    config_name: str | None = None,
    data_files: Any | None = None,
    token: str | None = None,
) -> HubDatasetRef:
    """Resolve a branch/tag such as 'main' to an immutable Hub commit SHA."""
    from huggingface_hub import HfApi

    api = HfApi(token=token or _token())
    info = api.repo_info(repo_id=repo_id, repo_type="dataset", revision=revision)
    if not getattr(info, "sha", None):
        raise RuntimeError(f"Hugging Face did not return a commit SHA for {repo_id}@{revision}")

    return HubDatasetRef(
        repo_id=repo_id,
        requested_revision=revision,
        resolved_revision=info.sha,
        split=split,
        config_name=config_name,
        data_files=data_files,
    )


def resolve_model_ref(
    *, repo_id: str, revision: str, token: str | None = None
) -> HubModelRef:
    """Resolve a model branch/tag to an immutable Hub commit SHA."""
    from huggingface_hub import HfApi

    api = HfApi(token=token or _token())
    info = api.repo_info(repo_id=repo_id, repo_type="model", revision=revision)
    if not getattr(info, "sha", None):
        raise RuntimeError(f"Hugging Face did not return a commit SHA for {repo_id}@{revision}")
    return HubModelRef(repo_id, revision, info.sha)


def load_dataset_ref(ref: HubDatasetRef, *, token: str | None = None):
    """Download a particular dataset split at the exact resolved commit."""
    from datasets import load_dataset

    kwargs: dict[str, Any] = {
        "path": ref.repo_id,
        "split": ref.split,
        "revision": ref.resolved_revision,
        "token": token or _token(),
    }
    if ref.config_name is not None:
        kwargs["name"] = ref.config_name
    if ref.data_files is not None:
        kwargs["data_files"] = ref.data_files
    return load_dataset(**kwargs)


def load_local_dataset(
    *, format: str, path: str, split: str, data_files: Any | None = None
):
    """Load a local CSV/JSON/Parquet-style dataset through Hugging Face Datasets."""
    from datasets import load_dataset

    kwargs: dict[str, Any] = {"path": format, "split": split}
    if data_files is not None:
        kwargs["data_files"] = data_files
    else:
        kwargs["data_files"] = path
    return load_dataset(**kwargs)


def summarize_dataset(dataset) -> dict[str, Any]:
    """Return lightweight metadata; does not copy dataset contents."""
    return {
        "num_rows": len(dataset),
        "columns": list(dataset.column_names),
        "features": {name: str(feature) for name, feature in dataset.features.items()},
        "fingerprint": getattr(dataset, "_fingerprint", None),
    }
