"""Stage 1.2 CLI: Download textbooks & solution manuals, split into chapter exercise PDFs, pair problems with official solutions, and extract problems via Aya-Expanse."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adversarial_physics.data.ingestion.book_downloader import DEFAULT_DOWNLOAD_DIR, download_books  # noqa: E402
from adversarial_physics.data.ingestion.chapter_domain_extractor import (  # noqa: E402
    DEFAULT_CHAPTER_DOMAIN_SUMMARY_PATH,
    DEFAULT_INITIAL_BOOK_DATASET_PATH,
    build_initial_book_dataset,
)
from adversarial_physics.data.ingestion.literature_extractor import (  # noqa: E402
    build_paired_problem_chunks,
    run_literature_extraction_from_config,
)
from adversarial_physics.data.ingestion.pdf_chapter_splitter import (  # noqa: E402
    DEFAULT_CHAPTERS_DIR,
    split_books_into_chapter_pdfs,
)
from adversarial_physics.data.processing.canonical_pool_builder import (  # noqa: E402
    build_stage1_canonical_pool,
)
from adversarial_physics.utils.config import load_book_ingestion_config  # noqa: E402
from adversarial_physics.utils.logging import get_logger  # noqa: E402

logger = get_logger("stage_1_2_cli")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 1.2: Download physics books, split chapter exercise PDFs, map problems to official solutions, and extract CanonicalProblem records with Aya-Expanse."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data/book_ingestion.yaml",
        help="Path to book ingestion YAML config (default: configs/data/book_ingestion.yaml).",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "kaggle_gpu", "hf_gpu", "local_cpu"],
        default="auto",
        help="Execution backend for Aya-Expanse inference (default: auto).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override Aya-Expanse model repo ID (e.g. CohereForAI/aya-expanse-8b or CohereForAI/aya-expanse-32b).",
    )
    parser.add_argument(
        "--book-id",
        dest="book_ids",
        action="append",
        default=None,
        help="Restrict processing to specific book_id(s) (repeatable; paired solution book is included automatically).",
    )
    parser.add_argument(
        "--book-path",
        dest="book_paths",
        action="append",
        default=None,
        help="Override config books with one or more local PDF/DOCX/TXT/JSONL file paths.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download missing books that have a 'url' configured before running.",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download configured books to data/books/ and exit.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download files even if they already exist locally.",
    )
    parser.add_argument(
        "--split-chapters",
        action="store_true",
        help="Split downloaded books into data/chapters/<book_id>/<XX>_<chapter_title>.pdf and <book_id>_solutions.pdf.",
    )
    parser.add_argument(
        "--split-chapters-only",
        action="store_true",
        help="Only generate data/chapters/<book_id>/ PDFs and exit.",
    )
    parser.add_argument(
        "--build-initial-dataset",
        action="store_true",
        help="Extract problems and paired solutions from non-SIDE chapters in data/chapters/ into data/literature/initial_book_dataset.jsonl and exit.",
    )
    parser.add_argument(
        "--build-canonical-pool",
        action="store_true",
        help="Run Stage 1.3 & 1.4 filtering, 0.85 anti-leakage deduplication, and domain-balanced selection to produce data/processed/canonical_train_250.jsonl.",
    )
    parser.add_argument(
        "--use-llm-refinement",
        action="store_true",
        help="When used with --build-canonical-pool, run resumable CohereForAI/aya-expanse-8b normalization over the selected 250 canonical problems.",
    )
    parser.add_argument(
        "--link-only",
        action="store_true",
        help="Segment problem books, map problems to their solutions in the solution manual, print the linking report, and exit without calling Aya-Expanse.",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=None,
        help="Target number of verified CanonicalProblem records to extract (default from config: 229).",
    )
    parser.add_argument(
        "--min-steps",
        type=int,
        default=None,
        help="Minimum derivation steps filter (default from config: 1).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum derivation steps filter (default from config: 5).",
    )
    parser.add_argument(
        "--allow-unmatched",
        action="store_true",
        help="Also extract problems that do not have a matching entry in the solution manual.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run pipeline without invoking the live Aya-Expanse model.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = load_book_ingestion_config(args.config)
    sources = list(cfg.get("books", []))
    download_cfg = cfg.get("download", {}) or {}
    chapters_cfg = cfg.get("chapters", {}) or {}
    download_dir = str(download_cfg.get("dir") or DEFAULT_DOWNLOAD_DIR)
    chapters_dir = str(chapters_cfg.get("dir") or DEFAULT_CHAPTERS_DIR)

    if args.book_ids:
        wanted = set(args.book_ids)
        wanted |= {
            str(e.get("solution_book_id"))
            for e in sources
            if e.get("book_id") in wanted and e.get("solution_book_id")
        }
        sources = [{**e, "enabled": True} for e in sources if e.get("book_id") in wanted]

    if args.download_only:
        results = download_books(
            sources,
            download_dir=download_dir,
            force=args.force_download,
            timeout=float(download_cfg.get("timeout_s", 120)),
        )
        print(json.dumps([r.to_dict() for r in results], indent=2))
        return 0

    if args.split_chapters or args.split_chapters_only:
        if args.download:
            download_books(
                sources,
                download_dir=download_dir,
                force=args.force_download,
                timeout=float(download_cfg.get("timeout_s", 120)),
            )
        split_res = split_books_into_chapter_pdfs(
            args.config,
            chapters_dir=chapters_dir,
            download_dir=download_dir,
            book_ids=args.book_ids,
            include_disabled=True,
        )
        if args.split_chapters_only:
            print(json.dumps([r.to_dict() for r in split_res], indent=2))
            return 0

    if args.build_initial_dataset:
        out_path = str(chapters_cfg.get("initial_book_dataset_path") or DEFAULT_INITIAL_BOOK_DATASET_PATH)
        sum_path = str(chapters_cfg.get("initial_book_summary_path") or DEFAULT_CHAPTER_DOMAIN_SUMMARY_PATH)
        ch_domains = chapters_cfg.get("chapter_domains")
        _, ds_summary = build_initial_book_dataset(
            chapters_dir=chapters_dir,
            output_path=out_path,
            summary_path=sum_path,
            chapter_domain_map=ch_domains if isinstance(ch_domains, dict) else None,
            require_matched_solution=not args.allow_unmatched,
        )
        print(json.dumps(ds_summary.to_dict(), indent=2))
        return 0

    if args.build_canonical_pool:
        _, pool_summary = build_stage1_canonical_pool(
            target_total=args.target_count,
            min_logical_steps=int(args.min_steps or 1),
            max_logical_steps=int(args.max_steps or 10),
            use_llm_refinement=args.use_llm_refinement,
            llm_backend=args.backend,  # type: ignore[arg-type]
            llm_model_name=args.model or "CohereForAI/aya-expanse-8b",
        )
        print(json.dumps(pool_summary.to_dict(), indent=2))
        return 0

    if args.link_only:
        if args.download:
            download_books(
                sources,
                download_dir=download_dir,
                force=args.force_download,
                timeout=float(download_cfg.get("timeout_s", 120)),
            )
        chunks, reports, _, _ = build_paired_problem_chunks(
            sources,
            download_dir=download_dir,
            require_official_solution=not args.allow_unmatched,
        )
        print(
            json.dumps(
                {
                    "paired_chunks_ready": len(chunks),
                    "linking_reports": [r.to_dict() for r in reports],
                },
                indent=2,
            )
        )
        return 0

    accepted, rejected, summary = run_literature_extraction_from_config(
        args.config,
        book_paths=args.book_paths,
        book_ids=args.book_ids,
        target_count=args.target_count,
        min_steps=args.min_steps,
        max_steps=args.max_steps,
        require_official_solution=(False if args.allow_unmatched else None),
        download=True if args.download else None,
        backend=args.backend,  # type: ignore[arg-type]
        model_name=args.model,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary.to_dict(), indent=2))
    logger.info("Done: %d accepted, %d rejected.", len(accepted), len(rejected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
