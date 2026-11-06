# Adversarial Physics Solver-Verifier

**Competition:** Lexsi AI Safety & Alignment Hackathon — Track 1, Problem Statement 8 (Data-Scarce Domains)  
**Author:** [`vital-ar`](https://github.com/vital-ar) · **Hugging Face:** [`vital-ar`](https://huggingface.co/vital-ar)  
**Base Model:** `CohereLabs/tiny-aya-global` (Shared Base with Dual LoRA Adapters)  
**Teacher / Curation Models:** `CohereForAI/aya-expanse-8b` & `CohereForAI/aya-expanse-32b`  
**Core Stack:** `CuratorKIT` (Data Generation & Curation) · `AlignTune` (LoRA Post-Training) · `AuditKIT` (Evaluation & Benchmarking)

---

## 1. Architecture & Model Selection

The **Adversarial Physics Solver-Verifier** is a two-agent collaborative reasoning pipeline built on `CohereLabs/tiny-aya-global` equipped with two specialized LoRA adapters:

* **Model A — Solver LoRA:** Generates a step-by-step physical derivation (`ANSWER: <value>`). When challenged, it re-solves the problem from a clean slate guided by compact, self-contained physical directives accumulated from the Verifier.
* **Model B — Verifier LoRA:** Independently inspects the original problem and Model A's proposed derivation, outputting a concise self-contained directive or confirmation followed by a deterministic text sentinel (`[[APPROVE]]` or `[[CHALLENGE]]`).

### Aya-Expanse 8B vs 32B Selection Policy (`src/adversarial_physics/utils/runtime.py`)
* **`CohereForAI/aya-expanse-8b`** is used for **Stage 1.2 literature extraction, multi-part splitting, domain classification, rephrasing, and Stage 2.4 CuratorKIT quality gating**. Because problem statements and official solutions are already deterministically paired by chapter and problem number, the 8B model is fast, accurate, and fits on a single 16 GB Kaggle T4 GPU or local CPU.
* **`CohereForAI/aya-expanse-32b`** is used for **Stage 2 adversarial flaw synthesis and self-contained physical directive generation**, where injecting subtle physical/algebraic errors without leaking the answer benefits from 32B reasoning (with automatic fallback to `CohereForAI/aya-expanse-8b` on `local_cpu` or single-GPU `<15 GB` VRAM).

### Multi-Backend Execution (`kaggle_gpu` · `hf_zerogpu` · `local_cpu`)
Every inference, synthesis, curation, and training pipeline supports `--backend auto|kaggle_gpu|hf_zerogpu|local_cpu`:
* **`kaggle_gpu`**: CUDA execution with `bfloat16`/`float16` and 4-bit NF4 `bitsandbytes` quantization.
* **`hf_zerogpu`**: Hugging Face Spaces ZeroGPU (`@spaces.GPU` decorator).
* **`local_cpu`**: Local CPU execution (`torch.float32`, `low_cpu_mem_usage=True`).

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

### Kaggle / Linux GPU / ZeroGPU Environment

```bash
python -m pip install -r requirements/kaggle.txt
```

Keep `HF_TOKEN` in environment variables or `.env` (never commit secrets).

---

## 3. Stage 1.1: PhysGym Ingestion & Domain-Stratified Splitting

We start with the **PhysGym** dataset (`YimengChen/PhysGym`, pinned to commit `8e122553897e77495dc2690f16d8055aea1655d7`), containing **97 university-level physics problems** across **6 domains** (`MECHANICS`, `ELECTROMAGNETISM`, `THERMODYNAMICS`, `OPTICS`, `MODERN_PHYSICS`, `RELATIVITY`), normalized into the clean `CanonicalProblem` schema (`problem_id`, `domain`, `problem_statement`, `reference_solution`, `source`, `original_id`).

### Run via CLI (`scripts/01_ingest_and_split.py`)

```powershell
# Default 21 Train / 38 Val / 38 Test Split
python scripts/01_ingest_and_split.py --config configs/data/ingestion.yaml

# Custom Split Ratios
python scripts/01_ingest_and_split.py --train-ratio 0.22 --val-ratio 0.39 --test-ratio 0.39 --seed 42
```

---

## 4. Stage 1.2: Textbook Ingestion, Per-Chapter Exercise PDFs & Solution Mapping

Stage 1.2 ingests Benjamin Crowell's open-source physics textbooks (`lightandmatter.com`), splits each downloaded book into per-chapter exercise PDFs and its corresponding slice of the all-in-one solution manual (`solns-all.pdf`) under `data/chapters/<book_id>/`, and maps problems to their official solutions using `(solution_book_id, solution_section, chapter, problem_number)`.

1. **Download Configured Textbooks & Solution Manual (`data/books/`):**
   ```powershell
   python scripts/02_extract_literature.py --download-only
   ```

2. **Split Downloaded Books into Per-Chapter Exercise PDFs & Book Solution PDFs (`data/chapters/<book_id>/`):**
   ```powershell
   python scripts/02_extract_literature.py --split-chapters-only
   ```

3. **Segment & Link Problems to Official Solutions (No LLM Calls):**
   ```powershell
   python scripts/02_extract_literature.py --link-only
   ```

4. **Run Extraction with `CohereForAI/aya-expanse-8b` (`kaggle_gpu`, `hf_zerogpu`, or `local_cpu`):**
   ```powershell
   python scripts/02_extract_literature.py --config configs/data/book_ingestion.yaml --backend auto
   ```

---

## 5. Inspecting Datasets & Preparing Experiment Manifests

```powershell
python scripts/inspect_dataset.py --config configs/data/ingestion.yaml --show-rows 2
python scripts/prepare_experiment.py --config configs/training/aligntune_solver_curated.yaml
```
