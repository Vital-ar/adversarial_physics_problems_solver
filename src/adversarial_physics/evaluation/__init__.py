"""AuditKIT & AlignTune benchmark evaluation, symbolic/numerical physics scorers, and Verifier metrics (Stage 6)."""

from adversarial_physics.evaluation.auditkit_bridge import (
    is_auditkit_installed,
    run_physics_eval_audit,
)
from adversarial_physics.evaluation.auditkit_native import (
    compare_version_runs_with_auditkit,
    evaluate_loop_traces_with_auditkit,
    evaluate_predictions_with_auditkit,
    run_full_auditkit_suite,
)

__all__ = [
    "compare_version_runs_with_auditkit",
    "evaluate_loop_traces_with_auditkit",
    "evaluate_predictions_with_auditkit",
    "is_auditkit_installed",
    "run_full_auditkit_suite",
    "run_physics_eval_audit",
]
