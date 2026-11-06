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
from adversarial_physics.data.ingestion.literature_extractor import (  # noqa: E402
    build_paired_problem_chunks,
    run_literature_extraction_from_config,
)
from adversarial_physics.data.ingestion.pdf_chapter_splitter import (  # noqa: E402
    DEFAULT_CHAPTERS_DIR,
    split_books_into_chapter_pdfs,
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
        choices=["auto", "kaggle_gpu", "hf_zerogpu", "local_cpu"],
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
