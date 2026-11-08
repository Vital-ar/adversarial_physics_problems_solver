"""Stratified domain-aware splitting for PhysGym and canonical physics problems (Stage 1.1)."""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from adversarial_physics.data.schemas.problem import CanonicalProblem, save_problems_jsonl
from adversarial_physics.utils.config import project_root, resolve_project_path


@dataclass(frozen=True)
class SplitResult:
    """Container for stratified train, validation, and test splits and domain statistics."""

    train: list[CanonicalProblem]
    val: list[CanonicalProblem]
    test: list[CanonicalProblem]
    seed: int
    target_counts: dict[str, int]
    domain_counts: dict[str, dict[str, int]]

    @property
    def train_ids(self) -> list[str]:
        return [p.problem_id for p in self.train]

    @property
    def val_ids(self) -> list[str]:
        return [p.problem_id for p in self.val]

    @property
    def test_ids(self) -> list[str]:
        return [p.problem_id for p in self.test]

    def to_manifest_dict(
        self,
        *,
        source_provenance: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Serialize the split result into a deterministic YAML-ready manifest dictionary."""
        total = len(self.train) + len(self.val) + len(self.test)
        return {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seed": self.seed,
            "total_problems": total,
            "split_counts": {
                "train": len(self.train),
                "val": len(self.val),
                "test": len(self.test),
            },
            "split_ratios": {
                "train": round(len(self.train) / total, 4) if total else 0.0,
                "val": round(len(self.val) / total, 4) if total else 0.0,
                "test": round(len(self.test) / total, 4) if total else 0.0,
            },
            "domain_distribution": self.domain_counts,
            "source_provenance": source_provenance or {},
            "splits": {
                "train_ids": self.train_ids,
                "val_ids": self.val_ids,
                "test_ids": self.test_ids,
            },
        }


def resolve_target_split_counts(
    total_count: int,
    *,
    train_ratio: float | None = None,
    val_ratio: float | None = None,
    test_ratio: float | None = None,
    train_size: int | None = None,
    val_size: int | None = None,
    test_size: int | None = None,
) -> tuple[int, int, int]:
    """Resolve exact integer (train, val, test) counts from either explicit sizes or ratios.

    Defaults to the blueprint's 21 / 38 / 38 split when total_count == 97 and no overrides
    are provided, or approximately (0.2165, 0.3918, 0.3918) for arbitrary dataset sizes.
    """
    if total_count <= 0:
        raise ValueError(f"total_count must be positive, got {total_count}")

    sizes_given = [s is not None for s in (train_size, val_size, test_size)]
    ratios_given = [r is not None for r in (train_ratio, val_ratio, test_ratio)]

    if any(sizes_given) and any(ratios_given):
        raise ValueError(
            "Specify either explicit split sizes (train_size, val_size, test_size) "
            "or split ratios (train_ratio, val_ratio, test_ratio), not both."
        )

    if any(sizes_given):
        if not all(sizes_given):
            raise ValueError(
                "When specifying explicit split sizes, all three of "
                "train_size, val_size, and test_size must be provided."
            )
        t_train, t_val, t_test = int(train_size), int(val_size), int(test_size)  # type: ignore[arg-type]
        if t_train < 0 or t_val < 0 or t_test < 0:
            raise ValueError("Split sizes must be non-negative integers.")
        if t_train + t_val + t_test != total_count:
            raise ValueError(
                f"Explicit split sizes ({t_train} + {t_val} + {t_test} = "
                f"{t_train + t_val + t_test}) do not sum to total_count ({total_count})."
            )
        return t_train, t_val, t_test

    if any(ratios_given):
        if not all(ratios_given):
            raise ValueError(
                "When specifying split ratios, all three of "
                "train_ratio, val_ratio, and test_ratio must be provided."
            )
        r_train, r_val, r_test = float(train_ratio), float(val_ratio), float(test_ratio)  # type: ignore[arg-type]
        if r_train < 0 or r_val < 0 or r_test < 0:
            raise ValueError("Split ratios must be non-negative.")
        ratio_sum = r_train + r_val + r_test
        if ratio_sum <= 0:
            raise ValueError("Sum of split ratios must be positive.")
        # Normalize if user passed percentages (e.g. 21.65, 39.18, 39.18) or unnormalized weights
        r_train, r_val, r_test = r_train / ratio_sum, r_val / ratio_sum, r_test / ratio_sum
    else:
        # Default blueprint ratios (21 / 38 / 38 for 97 problems)
        if total_count == 97:
            return 21, 38, 38
        r_train, r_val, r_test = 21 / 97.0, 38 / 97.0, 38 / 97.0

    raw = [r_train * total_count, r_val * total_count, r_test * total_count]
    floors = [int(x) for x in raw]
    remainder = total_count - sum(floors)
    # Largest remainder method; tie-break favors test, then val, then train
    fractions = [(raw[i] - floors[i], i) for i in range(3)]
    fractions.sort(key=lambda item: (item[0], item[1]), reverse=True)
    for idx in range(remainder):
        floors[fractions[idx][1]] += 1

    return floors[0], floors[1], floors[2]


def compute_domain_allocations(
    domain_sizes: dict[str, int],
    target_train: int,
    target_val: int,
    target_test: int,
) -> dict[str, tuple[int, int, int]]:
    """Compute exact per-domain (train, val, test) counts that sum to both domain totals and global split totals.

    Ensures every domain is proportionally represented across train, val, and test splits,
    and guarantees at least 1 problem in each non-empty split whenever a domain has >= 3 problems.
    """
    total = sum(domain_sizes.values())
    if total != target_train + target_val + target_test:
        raise ValueError(
            f"Sum of domain sizes ({total}) does not match target split sum "
            f"({target_train + target_val + target_test})."
        )

    targets = [target_train, target_val, target_test]
    domains = sorted(domain_sizes.keys())

    # Step 1: Compute exact fractional ideals per domain and split:
    # ideal[d][s] = domain_sizes[d] * targets[s] / total
    ideals: dict[str, list[float]] = {}
    allocs: dict[str, list[int]] = {}
    for d in domains:
        n_d = domain_sizes[d]
        ideals[d] = [(n_d * targets[s]) / total if total > 0 else 0.0 for s in range(3)]
        # Initial per-domain largest remainder allocation summing to n_d
        floors = [int(ideals[d][s]) for s in range(3)]
        rem = n_d - sum(floors)
        order = sorted(
            range(3),
            key=lambda s: (ideals[d][s] - floors[s], targets[s], s),
            reverse=True,
        )
        for i in range(rem):
            floors[order[i]] += 1

        # Guarantee at least 1 item in every active split if n_d >= number of active splits
        active_splits = [s for s in range(3) if targets[s] > 0]
        if n_d >= len(active_splits):
            for s in active_splits:
                if floors[s] == 0:
                    # Take 1 from the split in this domain with the largest count > 1
                    donor = max(active_splits, key=lambda k: (floors[k], -ideals[d][k]))
                    if floors[donor] > 1:
                        floors[donor] -= 1
                        floors[s] += 1

        allocs[d] = floors

    # Step 2: Reconcile column sums so sum_d allocs[d][s] == targets[s] for each split s
    current_sums = [sum(allocs[d][s] for d in domains) for s in range(3)]

    while current_sums != targets:
        over_splits = [s for s in range(3) if current_sums[s] > targets[s]]
        under_splits = [s for s in range(3) if current_sums[s] < targets[s]]
        if not over_splits or not under_splits:
            break

        best_swap: tuple[float, str, int, int] | None = None
        for s_over in over_splits:
            for s_under in under_splits:
                for d in domains:
                    n_d = domain_sizes[d]
                    active_splits = [s for s in range(3) if targets[s] > 0]
                    min_allowed = 1 if (n_d >= len(active_splits) and targets[s_over] > 0) else 0
                    if allocs[d][s_over] <= min_allowed:
                        continue
                    # Score how beneficial moving 1 item from s_over to s_under in domain d is:
                    # (allocs[d][s_over] - ideals[d][s_over]) + (ideals[d][s_under] - allocs[d][s_under])
                    gain = (allocs[d][s_over] - ideals[d][s_over]) + (
                        ideals[d][s_under] - allocs[d][s_under]
                    )
                    candidate = (gain, d, s_over, s_under)
                    if best_swap is None or candidate > best_swap:
                        best_swap = candidate

        if best_swap is None:
            # Fallback if min_allowed constraint was too strict
            for s_over in over_splits:
                for s_under in under_splits:
                    for d in domains:
                        if allocs[d][s_over] > 0:
                            best_swap = (0.0, d, s_over, s_under)
                            break
                    if best_swap:
                        break
                if best_swap:
                    break

        if best_swap is None:
            raise RuntimeError("Failed to reconcile stratified domain allocations.")

        _, d_chosen, s_over, s_under = best_swap
        allocs[d_chosen][s_over] -= 1
        allocs[d_chosen][s_under] += 1
        current_sums[s_over] -= 1
        current_sums[s_under] += 1

    return {d: (allocs[d][0], allocs[d][1], allocs[d][2]) for d in domains}


def stratified_split_problems(
    problems: list[CanonicalProblem],
    *,
    train_ratio: float | None = None,
    val_ratio: float | None = None,
    test_ratio: float | None = None,
    train_size: int | None = None,
    val_size: int | None = None,
    test_size: int | None = None,
    seed: int = 42,
    domain_field: str = "domain",
    sort_within_splits: bool = True,
) -> SplitResult:
    """Perform a deterministic stratified split across physics domains.

    Args:
        problems: List of CanonicalProblem instances to split.
        train_ratio: Optional fraction of problems for the train split.
        val_ratio: Optional fraction of problems for the validation split.
        test_ratio: Optional fraction of problems for the test split.
        train_size: Optional exact count for the train split (e.g., 21).
        val_size: Optional exact count for the validation split (e.g., 38).
        test_size: Optional exact count for the test split (e.g., 38).
        seed: Random seed for deterministic shuffling within each domain.
        domain_field: Attribute on CanonicalProblem used for stratification (default: 'domain').
        sort_within_splits: Whether to sort output splits by problem_id after splitting.

    Returns:
        SplitResult containing train, val, test problem lists and domain distribution stats.
    """
    if not problems:
        raise ValueError("Cannot split an empty list of problems.")

    # Verify unique problem IDs
    seen_ids: set[str] = set()
    for p in problems:
        if p.problem_id in seen_ids:
            raise ValueError(f"Duplicate problem_id detected before splitting: {p.problem_id}")
        seen_ids.add(p.problem_id)

    target_train, target_val, target_test = resolve_target_split_counts(
        len(problems),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
    )

    by_domain: dict[str, list[CanonicalProblem]] = defaultdict(list)
    for problem in sorted(problems, key=lambda item: item.problem_id):
        domain_val = str(getattr(problem, domain_field, problem.domain))
        by_domain[domain_val].append(problem)

    domain_sizes = {d: len(items) for d, items in by_domain.items()}
    allocations = compute_domain_allocations(
        domain_sizes,
        target_train=target_train,
        target_val=target_val,
        target_test=target_test,
    )

    train_split: list[CanonicalProblem] = []
    val_split: list[CanonicalProblem] = []
    test_split: list[CanonicalProblem] = []
    domain_counts: dict[str, dict[str, int]] = {}

    for idx, domain in enumerate(sorted(by_domain.keys())):
        domain_items = list(by_domain[domain])
        # Derive a deterministic per-domain RNG from the master seed
        domain_rng = random.Random(f"{seed}:{domain}:{idx}")
        domain_rng.shuffle(domain_items)

        n_train, n_val, n_test = allocations[domain]
        train_part = domain_items[:n_train]
        val_part = domain_items[n_train : n_train + n_val]
        test_part = domain_items[n_train + n_val : n_train + n_val + n_test]

        train_split.extend(train_part)
        val_split.extend(val_part)
        test_split.extend(test_part)

        domain_counts[domain] = {
            "total": len(domain_items),
            "train": len(train_part),
            "val": len(val_part),
            "test": len(test_part),
        }

    if sort_within_splits:
        train_split.sort(key=lambda p: p.problem_id)
        val_split.sort(key=lambda p: p.problem_id)
        test_split.sort(key=lambda p: p.problem_id)

    return SplitResult(
        train=train_split,
        val=val_split,
        test=test_split,
        seed=seed,
        target_counts={
            "train": target_train,
            "val": target_val,
            "test": target_test,
        },
        domain_counts=domain_counts,
    )


def save_split_artifacts(
    split_result: SplitResult,
    *,
    split_manifest_path: str | Path = "configs/data/split_21_38_38.yaml",
    output_dir: str | Path | None = "data/splits",
    source_provenance: dict[str, Any] | None = None,
    root: Path | None = None,
) -> dict[str, Path]:
    """Save the locked split manifest YAML and optionally the train/val/test JSONL files."""
    repo_root = root or project_root()
    manifest_file = resolve_project_path(split_manifest_path, root=repo_root)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)

    manifest_data = split_result.to_manifest_dict(source_provenance=source_provenance)
    with manifest_file.open("w", encoding="utf-8") as file:
        yaml.safe_dump(manifest_data, file, sort_keys=False, allow_unicode=True)

    saved_paths: dict[str, Path] = {"manifest": manifest_file}

    if output_dir is not None:
        splits_dir = resolve_project_path(output_dir, root=repo_root)
        splits_dir.mkdir(parents=True, exist_ok=True)
        saved_paths["train"] = save_problems_jsonl(
            split_result.train, splits_dir / "train_seeds.jsonl"
        )
        saved_paths["val"] = save_problems_jsonl(
            split_result.val, splits_dir / "validation.jsonl"
        )
        saved_paths["test"] = save_problems_jsonl(
            split_result.test, splits_dir / "test.jsonl"
        )

    return saved_paths


def load_split_manifest(path: str | Path = "configs/data/split_21_38_38.yaml") -> dict[str, Any]:
    """Load a previously saved split manifest YAML."""
    manifest_path = resolve_project_path(path)
    with manifest_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in split manifest: {manifest_path}")
    return data
