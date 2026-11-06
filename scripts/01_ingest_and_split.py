"""Stage 1.1: Ingest the PhysGym dataset and perform a domain-stratified train/val/test split.

Can be invoked programmatically via `run_ingestion_and_split(...)` or via the CLI:
    python scripts/01_ingest_and_split.py --config configs/data/ingestion.yaml
    python scripts/01_ingest_and_split.py --train-ratio 0.22 --val-ratio 0.39 --test-ratio 0.39 --seed 42
    python scripts/01_ingest_and_split.py --train-size 21 --val-size 38 --test-size 38 --canonicalize-domains
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from adversarial_physics.data.ingestion.physgym_loader import (  # noqa: E402
    DEFAULT_PHYSGYM_REPO,
    DEFAULT_PHYSGYM_REVISION,
    load_physgym_dataset,
)
from adversarial_physics.data.processing.splitting import (  # noqa: E402
    SplitResult,
    save_split_artifacts,
    stratified_split_problems,
)
from adversarial_physics.utils.config import load_ingestion_config  # noqa: E402
from adversarial_physics.utils.logging import get_logger  # noqa: E402

logger = get_logger("stage_1_1_ingest_split")


def run_ingestion_and_split(
    *,
    config_path: str | Path | None = None,
    repo_id: str | None = None,
    revision: str | None = None,
    source_split: str | None = None,
    kind: str | None = None,
    local_path: str | None = None,
    local_format: str | None = None,
    train_ratio: float | None = None,
    val_ratio: float | None = None,
    test_ratio: float | None = None,
    train_size: int | None = None,
    val_size: int | None = None,
    test_size: int | None = None,
    seed: int | None = None,
    canonicalize_domains: bool | None = None,
    id_prefix: str | None = None,
    domain_field: str = "domain",
    split_manifest_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    save_jsonl: bool = True,
    resolve_hub_sha: bool = True,
    token: str | None = None,
) -> tuple[SplitResult, dict[str, Path]]:
    """Ingest PhysGym and execute a domain-stratified split with full CLI/config override support.

    Explicit function arguments override values in `config_path` when both are provided.
    """
    cfg: dict[str, Any] = {}
    if config_path is not None:
        cfg = load_ingestion_config(config_path)

    source_cfg = cfg.get("source", {})
    norm_cfg = cfg.get("normalization", {})
    split_cfg = cfg.get("splitting", {})
    out_cfg = cfg.get("output", {})

    effective_kind = kind if kind is not None else source_cfg.get("kind", "hub")
    effective_repo = repo_id if repo_id is not None else source_cfg.get("repo_id", DEFAULT_PHYSGYM_REPO)
    effective_rev = revision if revision is not None else source_cfg.get("revision", DEFAULT_PHYSGYM_REVISION)
    effective_source_split = source_split if source_split is not None else source_cfg.get("split", "train")
    effective_local_path = local_path if local_path is not None else source_cfg.get("path")
    effective_local_format = local_format if local_format is not None else source_cfg.get("format", "json")

    effective_canonicalize = (
        canonicalize_domains
        if canonicalize_domains is not None
        else bool(norm_cfg.get("canonicalize_domains", False))
    )
    effective_id_prefix = id_prefix if id_prefix is not None else str(norm_cfg.get("id_prefix", "physgym"))
    domain_aliases = norm_cfg.get("domain_aliases")

    effective_seed = seed if seed is not None else int(split_cfg.get("seed", 42))

    # Determine whether explicit sizes or ratios were passed on CLI/function call vs config
    cli_sizes = any(x is not None for x in (train_size, val_size, test_size))
    cli_ratios = any(x is not None for x in (train_ratio, val_ratio, test_ratio))

    if cli_sizes:
        eff_train_size, eff_val_size, eff_test_size = train_size, val_size, test_size
        eff_train_ratio = eff_val_ratio = eff_test_ratio = None
    elif cli_ratios:
        eff_train_ratio, eff_val_ratio, eff_test_ratio = train_ratio, val_ratio, test_ratio
        eff_train_size = eff_val_size = eff_test_size = None
    else:
        # Fall back to config file settings
        eff_train_size = split_cfg.get("train_size")
        eff_val_size = split_cfg.get("val_size")
        eff_test_size = split_cfg.get("test_size")
        eff_train_ratio = split_cfg.get("train_ratio")
        eff_val_ratio = split_cfg.get("val_ratio")
        eff_test_ratio = split_cfg.get("test_ratio")

    effective_manifest_path = (
        split_manifest_path
        if split_manifest_path is not None
        else out_cfg.get("split_manifest_path", "configs/data/split_21_38_38.yaml")
    )
    effective_output_dir = (
        output_dir
        if output_dir is not None
        else out_cfg.get("splits_dir", "data/splits")
    )

    logger.info(
        "Loading PhysGym dataset (%s) ...",
        effective_repo if effective_kind == "hub" else effective_local_path,
    )
    problems, provenance = load_physgym_dataset(
        repo_id=effective_repo,
        revision=effective_rev,
        split=effective_source_split,
        kind=effective_kind,
        local_path=effective_local_path,
        local_format=effective_local_format,
        data_files=source_cfg.get("data_files"),
        config_name=source_cfg.get("config_name"),
        canonicalize_domains=effective_canonicalize,
        domain_aliases=domain_aliases,
        id_prefix=effective_id_prefix,
        resolve_hub_sha=resolve_hub_sha,
        token=token,
    )

    logger.info("Loaded %d problems. Performing domain-stratified split (seed=%d)...", len(problems), effective_seed)
    split_result = stratified_split_problems(
        problems,
        train_ratio=eff_train_ratio,
        val_ratio=eff_val_ratio,
        test_ratio=eff_test_ratio,
        train_size=eff_train_size,
        val_size=eff_val_size,
        test_size=eff_test_size,
        seed=effective_seed,
        domain_field=domain_field,
    )

    saved_paths = save_split_artifacts(
        split_result,
        split_manifest_path=effective_manifest_path,
        output_dir=effective_output_dir if save_jsonl else None,
        source_provenance=provenance,
    )

    return split_result, saved_paths


def print_split_summary(split_result: SplitResult, saved_paths: dict[str, Path]) -> None:
    """Print a formatted domain-by-domain table of the stratified split."""
    total = len(split_result.train) + len(split_result.val) + len(split_result.test)
    print("\n" + "=" * 72)
    print(f"STAGE 1.1 STRATIFIED SPLIT SUMMARY (Total: {total} problems, Seed: {split_result.seed})")
    print("=" * 72)
    print(f"{'DOMAIN':<22} | {'TOTAL':>7} | {'TRAIN':>7} | {'VAL':>7} | {'TEST':>7}")
    print("-" * 72)
    for domain, counts in sorted(split_result.domain_counts.items()):
        print(
            f"{domain:<22} | {counts['total']:>7} | {counts['train']:>7} | "
            f"{counts['val']:>7} | {counts['test']:>7}"
        )
    print("-" * 72)
    print(
        f"{'TOTAL':<22} | {total:>7} | {len(split_result.train):>7} | "
        f"{len(split_result.val):>7} | {len(split_result.test):>7}"
    )
    print("=" * 72)
    for label, path in saved_paths.items():
        print(f"Saved {label:<10}: {path}")
    print()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data/ingestion.yaml",
        help="Path to ingestion YAML config (default: configs/data/ingestion.yaml). Pass '' to ignore config.",
    )
    parser.add_argument("--repo-id", type=str, default=None, help="Hugging Face dataset repo ID (e.g. YimengChen/PhysGym)")
    parser.add_argument("--revision", type=str, default=None, help="Hugging Face dataset commit SHA or branch")
    parser.add_argument("--source-split", type=str, default=None, help="Source split name on HF Hub (default: train)")
    parser.add_argument("--kind", choices=["hub", "local"], default=None, help="Dataset source kind: 'hub' or 'local'")
    parser.add_argument("--local-path", type=str, default=None, help="Path to local dataset file if --kind local")
    parser.add_argument("--local-format", type=str, default=None, help="Format of local dataset (json, parquet, csv)")

    # Split ratios vs exact sizes
    parser.add_argument("--train-ratio", type=float, default=None, help="Fraction of problems for train split (e.g. 0.2165)")
    parser.add_argument("--val-ratio", type=float, default=None, help="Fraction of problems for validation split (e.g. 0.3918)")
    parser.add_argument("--test-ratio", type=float, default=None, help="Fraction of problems for test split (e.g. 0.3918)")

    parser.add_argument("--train-size", type=int, default=None, help="Exact number of problems for train split (default: 21)")
    parser.add_argument("--val-size", type=int, default=None, help="Exact number of problems for validation split (default: 38)")
    parser.add_argument("--test-size", type=int, default=None, help="Exact number of problems for test split (default: 38)")

    parser.add_argument("--seed", type=int, default=None, help="Random seed for stratified splitting (default: 42)")
    parser.add_argument(
        "--canonicalize-domains",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Map raw PhysGym tags (ELECTRICITY, MODERN, ADVANCED) to canonical domain names",
    )
    parser.add_argument("--id-prefix", type=str, default=None, help="Prefix for normalized problem_id (default: physgym)")
    parser.add_argument(
        "--split-manifest",
        type=str,
        default=None,
        help="Output YAML path for locked split IDs (default: configs/data/split_21_38_38.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write train_seeds.jsonl, validation.jsonl, and test.jsonl (default: data/splits)",
    )
    parser.add_argument(
        "--no-jsonl",
        action="store_true",
        help="Only write the split YAML manifest without exporting JSONL split files",
    )
    parser.add_argument(
        "--no-resolve-sha",
        action="store_true",
        help="Skip live HF Hub API SHA resolution and use the revision string directly (offline cache mode)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    config_path: str | Path | None = args.config if args.config else None
    if config_path is not None and not (ROOT / config_path).is_file() and not Path(config_path).is_file():
        config_path = None

    split_result, saved_paths = run_ingestion_and_split(
        config_path=config_path,
        repo_id=args.repo_id,
        revision=args.revision,
        source_split=args.source_split,
        kind=args.kind,
        local_path=args.local_path,
        local_format=args.local_format,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
        canonicalize_domains=args.canonicalize_domains,
        id_prefix=args.id_prefix,
        split_manifest_path=args.split_manifest,
        output_dir=args.output_dir,
        save_jsonl=not args.no_jsonl,
        resolve_hub_sha=not args.no_resolve_sha,
    )
    print_split_summary(split_result, saved_paths)


if __name__ == "__main__":
    main()
