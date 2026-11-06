# Adversarial Physics Solver-Verifier

**Competition:** Lexsi AI Safety & Alignment Hackathon — Track 1, Problem Statement 8 (Data-Scarce Domains)  
**Base Model:** `CohereLabs/tiny-aya-global` (Shared Base with Dual LoRA Adapters)  
**Core Stack:** `CuratorKIT` (Data Generation & Curation) · `AlignTune` (LoRA Post-Training) · `AuditKIT` (Evaluation & Benchmarking)

---

## 1. Architecture Overview

The **Adversarial Physics Solver-Verifier** is a two-agent collaborative reasoning pipeline built on a single small open-weight language model (`CohereLabs/tiny-aya-global`) equipped with two specialized LoRA adapters:

* **Model A — Solver LoRA:** Generates a step-by-step physical derivation and final answer (`ANSWER: <value>`). When challenged, it re-solves the problem from a clean slate guided by compact, self-contained physical directives accumulated from the Verifier.
* **Model B — Verifier LoRA:** Independently inspects the original problem and Model A's proposed derivation, outputting a concise self-contained directive or confirmation followed by a deterministic text sentinel (`[[APPROVE]]` or `[[CHALLENGE]]`).

---

## 2. Environment Setup

Python 3.11+ is recommended.

### Windows Local Environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements/local.txt
python -m pip install -e .
```

### Kaggle / Linux GPU Environment

```bash
python -m pip install -r requirements/kaggle.txt
```

Keep `HF_TOKEN` and `GEMINI_API_KEY` in environment variables or `.env` (never commit secrets).

---

## 3. Stage 1.1: PhysGym Ingestion & Domain-Stratified Splitting

We start with the **PhysGym** dataset (`YimengChen/PhysGym`, pinned to commit `8e122553897e77495dc2690f16d8055aea1655d7`), containing **97 university-level physics problems** across **6 domains** (`MECHANICS`, `ELECTROMAGNETISM`, `THERMODYNAMICS`, `OPTICS`, `MODERN_PHYSICS`, `RELATIVITY`).

### Run via CLI (`scripts/01_ingest_and_split.py` or `scripts/prepare_splits.py`)

1. **Default 21 Train / 38 Val / 38 Test Split (from `configs/data/ingestion.yaml`):**
   ```powershell
   python scripts/01_ingest_and_split.py --config configs/data/ingestion.yaml
   ```

2. **Custom Split Ratios via CLI Arguments:**
   ```powershell
   python scripts/01_ingest_and_split.py --train-ratio 0.22 --val-ratio 0.39 --test-ratio 0.39 --seed 42
   ```

3. **Custom Exact Split Sizes & Domain Tag Options:**
   ```powershell
   python scripts/01_ingest_and_split.py --train-size 21 --val-size 38 --test-size 38 --seed 42 --no-canonicalize-domains
   ```

### Run via Python Function Call

```python
from scripts.prepare_splits import run_ingestion_and_split

split_result, saved_paths = run_ingestion_and_split(
    config_path="configs/data/ingestion.yaml",
    train_ratio=0.2165,
    val_ratio=0.3918,
    test_ratio=0.3918,
    seed=42,
    canonicalize_domains=True,
)
```

This writes:
* `configs/data/split_21_38_38.yaml` — Locked split manifest with domain distribution, provenance, and `train_ids`, `val_ids`, `test_ids`.
* `data/splits/train_seeds.jsonl`, `data/splits/validation.jsonl`, `data/splits/test.jsonl` — Normalized `CanonicalProblem` JSONL files.

---

## 4. Stage 1.2: Literature & Textbook Ingestion with Paired Solution Manuals

Stage 1.2 ingests open-source physics textbooks (e.g., Benjamin Crowell's *Light and Matter* series) and maps end-of-chapter problems to their official worked solutions in the all-in-one solution manual (`solns-all.pdf`) using the composite key `(solution_book_id, solution_section, chapter, problem_number)`.

1. **Download Configured Textbooks & Solution Manual (`configs/data/book_ingestion.yaml`):**
   ```powershell
   python scripts/02_extract_literature.py --download-only
   ```

2. **Segment & Link Problems to Official Solutions (No LLM Calls):**
   ```powershell
   python scripts/02_extract_literature.py --link-only
   ```

3. **Run Full Extraction + Offline Python/SymPy Verification:**
   ```powershell
   python scripts/02_extract_literature.py --config configs/data/book_ingestion.yaml
   ```

---

## 5. Inspecting Datasets & Preparing Experiment Manifests

### Inspect PhysGym or Curated Splits

```powershell
python scripts/inspect_dataset.py --config configs/data/ingestion.yaml --show-rows 2
```

### Prepare an Experiment Provenance Manifest

```powershell
python scripts/prepare_experiment.py --config configs/training/aligntune_solver_curated.yaml
```

---

## 6. Running Unit Tests

```powershell
python -m pytest -q
```
