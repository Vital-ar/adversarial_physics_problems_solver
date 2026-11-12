"""AuditKIT & AlignTune benchmark evaluation, symbolic/numerical physics scorers, and Verifier metrics (Stage 6)."""

from adversarial_physics.evaluation.auditkit_bridge import (
    is_auditkit_installed,
    run_physics_eval_audit,
)

__all__ = [
    "is_auditkit_installed",
    "run_physics_eval_audit",
]
