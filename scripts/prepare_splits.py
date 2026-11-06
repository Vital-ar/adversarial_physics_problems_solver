"""CLI alias for scripts/01_ingest_and_split.py (Stage 1.1)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parent / "01_ingest_and_split.py"
spec = importlib.util.spec_from_file_location("stage_01_ingest_and_split", SCRIPT_PATH)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules["stage_01_ingest_and_split"] = module
spec.loader.exec_module(module)

run_ingestion_and_split = module.run_ingestion_and_split
main = module.main

if __name__ == "__main__":
    main()
