"""Download and inspect a pinned Hugging Face dataset split."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adversarial_physics.utils.config import load_yaml  # noqa: E402
from adversarial_physics.utils.hf_hub import (  # noqa: E402
    load_dataset_ref,
    load_local_dataset,
    resolve_dataset_ref,
    summarize_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/data/ingestion.yaml",
        help="Ingestion YAML, curation YAML, or dataset-reference YAML",
    )
    parser.add_argument("--show-rows", type=int, default=2)
    args = parser.parse_args()

    config = load_yaml(args.config)
    source = config.get("source", config.get("dataset", {}))
    if source.get("kind", "hub") == "local":
        dataset = load_local_dataset(
            format=source["format"],
            path=source["path"],
            split=source["split"],
            data_files=source.get("data_files"),
        )
        print("Loaded local dataset:", source["path"])
    else:
        ref = resolve_dataset_ref(
            repo_id=source["repo_id"],
            revision=source["revision"],
            split=source["split"],
            config_name=source.get("config_name"),
            data_files=source.get("data_files"),
        )
        dataset = load_dataset_ref(ref)
        print("Resolved dataset:", ref.to_dict())
    print("Dataset summary:", summarize_dataset(dataset))
    for index in range(min(args.show_rows, len(dataset))):
        print(f"\nExample {index}:\n{dataset[index]}")


if __name__ == "__main__":
    main()
