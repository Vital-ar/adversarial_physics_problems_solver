"""Data schemas for canonical physics problems and synthetic trajectories."""
from .problem import CanonicalProblem
from .trajectory import CandidateSolution, SolverTrajectory, VerifierTrajectory

__all__ = [
    "CanonicalProblem",
    "CandidateSolution",
    "SolverTrajectory",
    "VerifierTrajectory",
]
