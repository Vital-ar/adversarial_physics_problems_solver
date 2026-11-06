"""Unit tests for Stage 1.1 domain-stratified splitting and PhysGym normalization."""
from pathlib import Path

import pytest

from adversarial_physics.data.processing.normalization import normalize_physgym_record
from adversarial_physics.data.processing.splitting import (
    save_split_artifacts,
    stratified_split_problems,
)
from adversarial_physics.data.schemas.problem import CanonicalProblem


def _make_mock_physgym_problems() -> list[CanonicalProblem]:
    """Build 97 synthetic CanonicalProblem records matching PhysGym's exact domain counts."""
    domain_distribution = {
        "MECHANICS": 39,
        "ELECTROMAGNETISM": 31,
        "THERMODYNAMICS": 10,
        "OPTICS": 7,
        "MODERN_PHYSICS": 7,
        "RELATIVITY": 3,
    }
    problems: list[CanonicalProblem] = []
    idx = 1
    for domain, count in domain_distribution.items():
        for _ in range(count):
            problems.append(
                CanonicalProblem(
                    problem_id=f"physgym_{idx:04d}",
                    domain=domain,
                    problem_statement=f"Problem {idx} in {domain}",
                    reference_solution=f"Solution {idx}",
                )
            )
            idx += 1
    return problems


def test_stratified_split_default_21_38_38_preserves_all_domains():
    problems = _make_mock_physgym_problems()
    result = stratified_split_problems(problems, seed=42)

    assert len(result.train) == 21
    assert len(result.val) == 38
    assert len(result.test) == 38

    train_ids = set(result.train_ids)
    val_ids = set(result.val_ids)
    test_ids = set(result.test_ids)
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert len(train_ids | val_ids | test_ids) == 97

    for domain, counts in result.domain_counts.items():
        assert counts["train"] >= 1, f"Domain {domain} missing from train split"
        assert counts["val"] >= 1, f"Domain {domain} missing from val split"
        assert counts["test"] >= 1, f"Domain {domain} missing from test split"
        assert counts["train"] + counts["val"] + counts["test"] == counts["total"]


def test_stratified_split_custom_ratios():
    problems = _make_mock_physgym_problems()
    result = stratified_split_problems(
        problems,
        train_ratio=0.50,
        val_ratio=0.25,
        test_ratio=0.25,
        seed=123,
    )
    assert len(result.train) + len(result.val) + len(result.test) == 97
    assert len(result.train) == 49
    assert len(result.val) == 24
    assert len(result.test) == 24

    for domain, counts in result.domain_counts.items():
        assert counts["train"] >= 1
        assert counts["val"] >= 1
        assert counts["test"] >= 1


def test_stratified_split_rejects_mixed_ratios_and_sizes():
    problems = _make_mock_physgym_problems()
    with pytest.raises(ValueError, match="not both"):
        stratified_split_problems(
            problems,
            train_ratio=0.2,
            val_ratio=0.4,
            test_ratio=0.4,
            train_size=21,
            val_size=38,
            test_size=38,
        )


def test_normalize_physgym_record_maps_domains_and_fields():
    raw = {
        "id": 495,
        "tag": "ELECTRICITY",
        "content": "Find the electric field.",
        "solution": "By Gauss's law...",
    }
    norm_raw = normalize_physgym_record(raw, canonicalize_domains=False)
    assert norm_raw.problem_id == "physgym_0495"
    assert norm_raw.domain == "ELECTRICITY"

    norm_canonical = normalize_physgym_record(raw, canonicalize_domains=True)
    assert norm_canonical.domain == "ELECTROMAGNETISM"
    assert set(norm_canonical.to_dict().keys()) == {
        "problem_id",
        "domain",
        "problem_statement",
        "reference_solution",
        "source",
        "original_id",
    }


def test_save_split_artifacts_writes_manifest_and_jsonl(tmp_path: Path):
    problems = _make_mock_physgym_problems()
    result = stratified_split_problems(problems, seed=42)
    paths = save_split_artifacts(
        result,
        split_manifest_path=tmp_path / "split.yaml",
        output_dir=tmp_path / "splits",
        root=tmp_path,
    )
    assert paths["manifest"].is_file()
    assert paths["train"].is_file()
    assert paths["val"].is_file()
    assert paths["test"].is_file()
