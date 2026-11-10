"""Stage 4 CLI Entrypoint: Mandatory AlignTune LoRA Post-Training (`05_train_aligntune.py`).

Usage Examples (Kaggle 2x T4 GPUs or Local GPU):
    python scripts/05_train_aligntune.py --dataset solver_e1_manual_det
    python scripts/05_train_aligntune.py --dataset solver_e2_curatorkit_nollm
    python scripts/05_train_aligntune.py --dataset solver_e3_canonical_llm
    python scripts/05_train_aligntune.py --dataset solver_e4_adversarial
    python scripts/05_train_aligntune.py --dataset verifier_e4_curated

Custom Hyperparameter Sweep Example:
    python scripts/05_train_aligntune.py \\
        --dataset solver_e4_adversarial \\
        --config configs/training/aligntune_training.yaml \\
        --effective-batch-size 16 \\
        --per-device-batch-size 1 \\
        --epochs 10 \\
        --lr 0.0002 \\
        --lora-r 16
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure `src/` is on sys.path when invoked directly as `python scripts/05_train_aligntune.py`
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from adversarial_physics.training.run_aligntune import main  # noqa: E402

if __name__ == "__main__":
    main()
