"""Canonical problem schema for PhysGym and literature-extracted physics problems."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CanonicalProblem:
    """Unified schema for all physics problems across train, validation, and test splits."""

    problem_id: str
    domain: str
    problem_statement: str
    reference_solution: str
    source: str = "physgym"
    original_id: int | str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CanonicalProblem:
        return cls(
            problem_id=str(data["problem_id"]),
            domain=str(data["domain"]),
            problem_statement=str(data["problem_statement"]),
            reference_solution=str(data["reference_solution"]),
            source=str(data.get("source") or "physgym"),
            original_id=data.get("original_id"),
        )


def save_problems_jsonl(problems: list[CanonicalProblem], path: str | Path) -> Path:
    """Write a list of CanonicalProblem records to a JSONL file."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as file:
        for problem in problems:
            file.write(json.dumps(problem.to_dict(), ensure_ascii=False) + "\n")
    return out_path


def load_problems_jsonl(path: str | Path) -> list[CanonicalProblem]:
    """Load CanonicalProblem records from a JSONL file."""
    in_path = Path(path)
    problems: list[CanonicalProblem] = []
    with in_path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if stripped:
                problems.append(CanonicalProblem.from_dict(json.loads(stripped)))
    return problems
