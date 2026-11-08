"""Trajectory schemas for Solver (Model A) and Verifier (Model B) SFT datasets."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class CandidateSolution:
    """A candidate derivation (rephrased correct, synthetic error, or real Model A output)."""

    problem_id: str
    candidate_id: str
    solution_text: str
    is_correct: bool
    error_type: str | None = None
    paired_directive: str | None = None
    source: str = "aya_expanse_synthetic"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SolverTrajectory:
    """Dataset 1 (Solver LoRA) training example with 0, 1, or 2 accumulated directives."""

    trajectory_id: str
    problem_id: str
    domain: str
    turn: int
    accumulated_directives: list[str] = field(default_factory=list)
    prompt: str = ""
    target_output: str = ""
    source_candidate_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VerifierTrajectory:
    """Dataset 2 (Verifier LoRA) training example with feedback-first sentinel output."""

    trajectory_id: str
    problem_id: str
    domain: str
    proposed_solution: str
    directive_text: str
    verdict: str  # "[[APPROVE]]" or "[[CHALLENGE]]"
    error_type: str | None = None
    part: str = "part_2_synthetic_challenged"
    prompt: str = ""
    target_output: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
