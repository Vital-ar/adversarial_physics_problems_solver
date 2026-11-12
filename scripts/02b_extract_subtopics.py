"""Stage 1.2b CLI: Extract & CuratorKIT-Deduplicate Textbook Subtopic Chunks (`02b_extract_subtopics.py`).

Builds `data/literature/subtopic_chunks.jsonl` for zero-LLM subtopic grounding during E2
synthetic physics problem generation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adversarial_physics.data.ingestion.book_downloader import DEFAULT_DOWNLOAD_DIR  # noqa: E402
from adversarial_physics.data.ingestion.subtopic_extractor import (  # noqa: E402
    DEFAULT_SUBTOPIC_CHUNKS_PATH,
    DEFAULT_SUBTOPIC_SUMMARY_PATH,
    build_subtopic_corpus,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book-config", type=str, default="configs/data/book_ingestion.yaml")
    parser.add_argument("--download-dir", type=str, default=DEFAULT_DOWNLOAD_DIR)
    parser.add_argument("--output", type=str, default=DEFAULT_SUBTOPIC_CHUNKS_PATH)
    parser.add_argument("--summary-out", type=str, default=DEFAULT_SUBTOPIC_SUMMARY_PATH)
    parser.add_argument("--max-chunk-tokens", type=int, default=380)
    parser.add_argument("--minhash-threshold", type=float, default=0.78)
    args = parser.parse_args(argv)

    _, summary = build_subtopic_corpus(
        book_config_path=args.book_config,
        download_dir=args.download_dir,
        output_path=args.output,
        summary_path=args.summary_out,
        max_chunk_tokens=args.max_chunk_tokens,
        minhash_threshold=args.minhash_threshold,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
