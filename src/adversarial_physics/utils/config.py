"""YAML loading and validation for the Adversarial Physics Solver-Verifier project."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a project configuration is missing or malformed."""


def project_root() -> Path:
    """Return the repository root (the parent of ``src``)."""
    return Path(__file__).resolve().parents[3]


def resolve_project_path(path: str | Path, *, root: Path | None = None) -> Path:
    """Resolve a relative path against the repository root."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (root or project_root()) / candidate


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and require a mapping at its root."""
    config_path = resolve_project_path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        content = yaml.safe_load(file)

    if not isinstance(content, dict):
        raise ConfigError(f"Expected a YAML mapping at the root of {config_path}")
    return content


def require_mapping(data: dict[str, Any], key: str, *, context: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"{context}.{key} must be a YAML mapping")
    return value


def require_value(data: dict[str, Any], key: str, *, context: str) -> Any:
    if key not in data or data[key] is None or data[key] == "":
        raise ConfigError(f"Missing required value: {context}.{key}")
    return data[key]


def load_curation_config(path: str | Path) -> dict[str, Any]:
    config = load_yaml(path)
    require_value(config, "dataset_id", context="curation")
    source = require_mapping(config, "source", context="curation")
    output = require_mapping(config, "output", context="curation")

    kind = require_value(source, "kind", context="curation.source")
    if kind not in {"hub", "local"}:
        raise ConfigError("curation.source.kind must be either 'hub' or 'local'")

    if kind == "hub":
        require_value(source, "repo_id", context="curation.source")
        require_value(source, "revision", context="curation.source")
    else:
        require_value(source, "path", context="curation.source")
        require_value(source, "format", context="curation.source")

    require_value(source, "split", context="curation.source")
    require_value(output, "repo_id", context="curation.output")
    return config


def load_ingestion_config(path: str | Path) -> dict[str, Any]:
    """Load and validate Stage 1.1 ingestion and stratified split configuration."""
    config = load_yaml(path)
    source = require_mapping(config, "source", context="ingestion")
    kind = require_value(source, "kind", context="ingestion.source")
    if kind not in {"hub", "local"}:
        raise ConfigError("ingestion.source.kind must be either 'hub' or 'local'")

    if kind == "hub":
        require_value(source, "repo_id", context="ingestion.source")
        require_value(source, "revision", context="ingestion.source")
    else:
        require_value(source, "path", context="ingestion.source")
        require_value(source, "format", context="ingestion.source")

    require_value(source, "split", context="ingestion.source")
    if "splitting" in config and not isinstance(config["splitting"], dict):
        raise ConfigError("ingestion.splitting must be a YAML mapping")
    if "output" in config and not isinstance(config["output"], dict):
        raise ConfigError("ingestion.output must be a YAML mapping")
    return config


def load_book_ingestion_config(path: str | Path) -> dict[str, Any]:
    """Load and validate Stage 1.2 book ingestion and literature extraction configuration."""
    config = load_yaml(path)
    require_value(config, "dataset_id", context="book_ingestion")

    books = config.get("books")
    if not isinstance(books, list):
        raise ConfigError("book_ingestion.books must be a YAML list of book source mappings")
    book_ids: set[str] = set()
    solution_sections_by_id: dict[str, set[str]] = {}
    solution_references: list[tuple[str, str, str | None, bool]] = []

    for idx, entry in enumerate(books):
        if not isinstance(entry, dict):
            raise ConfigError(f"book_ingestion.books[{idx}] must be a YAML mapping")
        bid = str(entry.get("book_id", "")).strip()
        if bid:
            book_ids.add(bid)
        fmt = str(entry.get("format", "pdf")).lower()
        if fmt not in {"pdf", "docx", "txt", "md", "jsonl", "json"}:
            raise ConfigError(
                f"book_ingestion.books[{idx}].format must be one of pdf, docx, txt, md, jsonl"
            )
        role = str(entry.get("role", "standalone")).lower()
        if role not in {"problem_book", "solution_book", "standalone"}:
            raise ConfigError(
                f"book_ingestion.books[{idx}].role must be one of 'problem_book', 'solution_book', 'standalone'"
            )
        sections = entry.get("sections")
        if sections is not None:
            if not isinstance(sections, list) or not all(isinstance(s, str) for s in sections):
                raise ConfigError(f"book_ingestion.books[{idx}].sections must be a list of strings")
            if bid:
                solution_sections_by_id[bid] = {s.strip().lower() for s in sections if s.strip()}

        sol_ref = entry.get("solution_book_id")
        sol_sec = entry.get("solution_section")
        if sol_ref:
            solution_references.append(
                (
                    bid or f"book_{idx}",
                    str(sol_ref),
                    str(sol_sec).strip() if sol_sec else None,
                    bool(entry.get("enabled", True)),
                )
            )

    for pb_id, sol_id, sol_sec, is_enabled in solution_references:
        if is_enabled and sol_id not in book_ids:
            raise ConfigError(
                f"Enabled problem book {pb_id!r} references unknown solution_book_id {sol_id!r}"
            )
        known_secs = solution_sections_by_id.get(sol_id)
        if is_enabled and known_secs and sol_sec and sol_sec.lower() not in known_secs:
            raise ConfigError(
                f"Problem book {pb_id!r} specifies solution_section={sol_sec!r}, which is not "
                f"in {sol_id!r}.sections"
            )

    if "download" in config and not isinstance(config["download"], dict):
        raise ConfigError("book_ingestion.download must be a YAML mapping")
    if "pairing" in config and not isinstance(config["pairing"], dict):
        raise ConfigError("book_ingestion.pairing must be a YAML mapping")

    if "filtering" in config and not isinstance(config["filtering"], dict):
        raise ConfigError("book_ingestion.filtering must be a YAML mapping")
    if "extraction" in config and not isinstance(config["extraction"], dict):
        raise ConfigError("book_ingestion.extraction must be a YAML mapping")
    return config


def load_training_config(path: str | Path) -> dict[str, Any]:
    config = load_yaml(path)
    require_value(config, "experiment_id", context="training")

    dataset = require_mapping(config, "dataset", context="training")
    require_value(dataset, "repo_id", context="training.dataset")
    require_value(dataset, "revision", context="training.dataset")
    require_value(dataset, "split", context="training.dataset")

    base_model = require_mapping(config, "base_model", context="training")
    require_value(base_model, "repo_id", context="training.base_model")
    require_value(base_model, "revision", context="training.base_model")

    training = require_mapping(config, "training", context="training")
    require_value(training, "method", context="training.training")
    require_value(training, "adapter", context="training.training")
    require_value(training, "seed", context="training.training")
    if not isinstance(training.get("parameters", {}), dict):
        raise ConfigError("training.parameters must be a YAML mapping")

    evaluation = require_mapping(config, "evaluation", context="training")
    require_value(evaluation, "config_path", context="training.evaluation")

    output = require_mapping(config, "output", context="training")
    require_value(output, "repo_id", context="training.output")
    require_value(output, "local_dir", context="training.output")

    return config


def load_evaluation_config(path: str | Path) -> dict[str, Any]:
    config = load_yaml(path)
    data = require_mapping(config, "data", context="evaluation")
    require_value(data, "validation_split", context="evaluation.data")
    require_value(data, "test_split", context="evaluation.data")

    selection = require_mapping(config, "selection", context="evaluation")
    require_value(selection, "primary_metric", context="evaluation.selection")
    direction = require_value(selection, "direction", context="evaluation.selection")
    if direction not in {"maximize", "minimize"}:
        raise ConfigError("evaluation.selection.direction must be 'maximize' or 'minimize'")

    metrics = config.get("metrics")
    if not isinstance(metrics, list) or not metrics or not all(isinstance(x, str) for x in metrics):
        raise ConfigError("evaluation.metrics must be a non-empty list of metric names")
    return config
