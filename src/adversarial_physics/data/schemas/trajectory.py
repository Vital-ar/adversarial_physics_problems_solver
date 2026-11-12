"""Trajectory schemas and prompt templates for Solver (Model A) and Verifier (Model B) SFT datasets."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Controlled 5-category physics error taxonomy for Stage 2
ERROR_TAXONOMY: tuple[str, ...] = (
    "incorrect_physical_model",
    "incorrect_formula",
    "algebraic_error",
    "dimensional_unit_error",
    "arithmetic_sign_error",
)

ERROR_TAXONOMY_DESCRIPTIONS: dict[str, str] = {
    "incorrect_physical_model": (
        "Omit a required physical effect or conservation term (e.g., omitting rotational kinetic energy "
        "1/2 I*omega^2 for a rolling body, ignoring relativistic gamma factor, or neglecting internal resistance)."
    ),
    "incorrect_formula": (
        "Apply a related but inapplicable physical equation (e.g., applying constant-acceleration kinematics "
        "to a variable force, using mirror/lens sign convention for the wrong optical element, or confusing "
        "isothermal work with adiabatic work)."
    ),
    "algebraic_error": (
        "Perform a realistic mid-derivation algebraic mistake when isolating the target variable (e.g., "
        "multiplying instead of dividing by a factor, dropping a square root or exponent, or mis-combining fractions)."
    ),
    "dimensional_unit_error": (
        "Introduce a dimensional or unit conversion error (e.g., failing to convert degrees/radians, "
        "cm/mm to m, grams to kg, or dropping a physical constant factor with dimensions)."
    ),
    "arithmetic_sign_error": (
        "Flip a critical sign or numerical factor during substitution (e.g., wrong sign in potential energy, "
        "work-energy theorem, Lorentz force, or lens equation, leading to a demonstrably wrong final ANSWER:)."
    ),
}

# Underrepresented domains in canonical_train.jsonl (24 + 36 + 22 = 82 problems) that receive Alternative Solution 2
MINORITY_SOLUTION2_DOMAINS: tuple[str, ...] = (
    "OPTICS",
    "RELATIVITY",
    "THERMODYNAMICS",
)

VERDICT_APPROVE = "[[APPROVE]]"
VERDICT_CHALLENGE = "[[CHALLENGE]]"


def format_solver_prompt(problem_statement: str, accumulated_directives: list[str] | None = None) -> str:
    """Format the Model A (Solver) input prompt for Turn 0 (0 hints) or Hinted Revision (1 or 2 hints)."""
    base = (
        "Solve the following physics problem step by step (in at most 10 logical steps). "
        "State the governing physical equations, perform algebraic substitutions clearly, include SI units, "
        "and end with the final answer on a new line prefixed with ANSWER:.\n\n"
        f"PROBLEM:\n{problem_statement.strip()}"
    )
    hints = [h.strip() for h in (accumulated_directives or []) if h and h.strip()]
    if not hints:
        return base
    bullet_lines = "\n".join(f"- {h}" for h in hints)
    return f"{base}\n\nCRITICAL PHYSICAL CONSTRAINTS TO FOLLOW:\n{bullet_lines}"


def format_verifier_prompt(problem_statement: str, proposed_solution: str) -> str:
    """Format the Model B (Verifier) input prompt inspecting `[problem_statement + proposed_solution]`."""
    return (
        "Review the proposed solution to the physics problem below. First, write a concise, self-contained "
        "physical or mathematical directive that either confirms why the derivation is valid or states the "
        "exact rule/constraint needed to fix the error without referencing line or step numbers. Then, on a "
        "new final line, output [[APPROVE]] if the solution and final answer are completely correct, or "
        "[[CHALLENGE]] if any physical, algebraic, or unit error exists.\n\n"
        f"PROBLEM:\n{problem_statement.strip()}\n\n"
        f"PROPOSED SOLUTION:\n{proposed_solution.strip()}"
    )


@dataclass(frozen=True)
class CandidateSolution:
    """A candidate derivation (Alternative Solution 2, synthetic error, or real Model A output)."""

    problem_id: str
    candidate_id: str
    solution_text: str
    is_correct: bool
    domain: str = "MECHANICS"
    error_type: str | None = None
    paired_directive: str | None = None
    source: str = "aya_expanse_8b_synthetic"
    original_id: Any = None
    latency_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CandidateSolution:
        return cls(
            problem_id=str(data["problem_id"]),
            candidate_id=str(data["candidate_id"]),
            solution_text=str(data["solution_text"]),
            is_correct=bool(data["is_correct"]),
            domain=str(data.get("domain") or "MECHANICS"),
            error_type=data.get("error_type"),
            paired_directive=data.get("paired_directive"),
            source=str(data.get("source") or "aya_expanse_8b_synthetic"),
            original_id=data.get("original_id"),
            latency_seconds=float(data.get("latency_seconds") or 0.0),
        )


@dataclass(frozen=True)
class SolverTrajectory:
    """Solver LoRA (Model A) SFT training example with 0, 1, or 2 accumulated self-contained directives."""

    trajectory_id: str
    problem_id: str
    domain: str
    turn: int
    accumulated_directives: list[str] = field(default_factory=list)
    prompt: str = ""
    target_output: str = ""
    source_candidate_ids: list[str] = field(default_factory=list)
    dataset_tier: str = "E4_solver_adversarial"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Include standard AlignTune / CuratorKIT SFT fields (`instruction`, `input`, `output`)
        d["instruction"] = self.prompt
        d["input"] = ""
        d["output"] = self.target_output
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SolverTrajectory:
        return cls(
            trajectory_id=str(data["trajectory_id"]),
            problem_id=str(data["problem_id"]),
            domain=str(data["domain"]),
            turn=int(data.get("turn", 0)),
            accumulated_directives=list(data.get("accumulated_directives") or []),
            prompt=str(data.get("prompt") or data.get("instruction") or ""),
            target_output=str(data.get("target_output") or data.get("output") or ""),
            source_candidate_ids=list(data.get("source_candidate_ids") or []),
            dataset_tier=str(data.get("dataset_tier") or "E4_solver_adversarial"),
        )


@dataclass(frozen=True)
class VerifierTrajectory:
    """Verifier LoRA (Model B) SFT training example with feedback-first sentinel output."""

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
        d = asdict(self)
        d["instruction"] = self.prompt
        d["input"] = ""
        d["output"] = self.target_output
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VerifierTrajectory:
        return cls(
            trajectory_id=str(data["trajectory_id"]),
            problem_id=str(data["problem_id"]),
            domain=str(data["domain"]),
            proposed_solution=str(data["proposed_solution"]),
            directive_text=str(data["directive_text"]),
            verdict=str(data["verdict"]),
            error_type=data.get("error_type"),
            part=str(data.get("part") or "part_2_synthetic_challenged"),
            prompt=str(data.get("prompt") or data.get("instruction") or ""),
            target_output=str(data.get("target_output") or data.get("output") or ""),
        )


def save_candidates_jsonl(candidates: list[CandidateSolution], path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    return out_path


def load_candidates_jsonl(path: str | Path) -> list[CandidateSolution]:
    in_path = Path(path)
    if not in_path.is_file():
        return []
    rows: list[CandidateSolution] = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(CandidateSolution.from_dict(json.loads(line)))
    return rows


def load_candidate_checkpoint_map(ckpt_file: Path) -> dict[str, CandidateSolution]:
    """Load resumable JSONL checkpoint records keyed by ``problem_id``."""
    cached: dict[str, CandidateSolution] = {}
    if not ckpt_file.is_file():
        return cached
    with ckpt_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                cand = CandidateSolution.from_dict(json.loads(line))
                cached[cand.problem_id] = cand
            except Exception:
                continue
    return cached


def normalize_solution_fences_and_answer(raw_text: str, *, strip_fences: bool = True) -> str:
    """Strip optional markdown code fences and normalize bold ``**ANSWER:**`` headers."""
    text = (raw_text or "").strip()
    if strip_fences:
        text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text).strip()
    return re.sub(r"\*\*ANSWER\s*:\*\*\s*\n*\s*", "ANSWER: ", text, flags=re.IGNORECASE)
