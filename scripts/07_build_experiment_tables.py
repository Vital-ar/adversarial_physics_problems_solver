"""CLI Builder for Multi-Version (`v1`, `v2`, `v3`) Experiment Comparison Tables & Statistical Reports.

Usage:
    python scripts/07_build_experiment_tables.py
    python scripts/07_build_experiment_tables.py --root-dir . --output-dir results/comparison_tables
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from adversarial_physics.evaluation.comparison_tables import build_all_version_tables  # noqa: E402
from adversarial_physics.utils.logging import get_logger  # noqa: E402

logger = get_logger("07_build_experiment_tables")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build unified Markdown, JSON, and CSV comparison tables for v1, v2, and v3 experiments."
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        default=".",
        help="Repository root directory containing results/ and models/adapters/.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/comparison_tables",
        help="Directory where experiment_comparison_v1_v2_v3.{md,json,csv} are written.",
    )
    args = parser.parse_args(argv)

    res = build_all_version_tables(root_dir=args.root_dir, output_dir=args.output_dir)
    logger.info(f"Markdown report written to : {res['markdown_path']}")
    logger.info(f"JSON report written to     : {res['json_path']}")
    logger.info(f"CSV table written to       : {res['csv_path']}")


if __name__ == "__main__":
    main()
