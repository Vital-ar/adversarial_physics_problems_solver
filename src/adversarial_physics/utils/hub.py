"""Compatibility alias for hf_hub utilities."""
from .hf_hub import (  # noqa: F401
    HubDatasetRef,
    HubModelRef,
    load_dataset_ref,
    load_local_dataset,
    resolve_dataset_ref,
    resolve_model_ref,
    summarize_dataset,
)
