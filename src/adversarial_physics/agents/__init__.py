"""Dual-LoRA model manager and Solver/Verifier orchestration re-exports (Stage 3/5)."""
from __future__ import annotations

from adversarial_physics.orchestration.loop import (
    DualLoRAEngine,
    ParsedVerifierOutput,
    ProblemLoopResult,
    TurnTrace,
    parse_verifier_response,
    run_adversarial_validation_loop,
    run_single_problem_loop,
)

__all__ = [
    "DualLoRAEngine",
    "ParsedVerifierOutput",
    "ProblemLoopResult",
    "TurnTrace",
    "parse_verifier_response",
    "run_adversarial_validation_loop",
    "run_single_problem_loop",
]
