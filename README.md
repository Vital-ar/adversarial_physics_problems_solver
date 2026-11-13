# Adversarial Physics Solver-Verifier

**Competition:** Lexsi AI Safety & Alignment Hackathon — Track 1, Problem Statement 8 (Data-Scarce Domains)  
**Author:** [`vital-ar`](https://github.com/vital-ar) · **Hugging Face:** [`vital-ar`](https://huggingface.co/vital-ar)  
**Base Model:** `CohereLabs/tiny-aya-global` (Shared Base with Dual LoRA Adapters)  
**Teacher / Curation Model:** `CohereForAI/aya-expanse-8b` (Unified Teacher; optional `CohereForAI/aya-expanse-32b` override)  
**Core Stack:** `CuratorKIT` (Data Generation & Curation) · `AlignTune` (LoRA Post-Training) · `AuditKIT` (Evaluation & Benchmarking)

---

## 1. Architecture & Model Selection

The **Adversarial Physics Solver-Verifier** is a two-agent collaborative reasoning pipeline built on `CohereLabs/tiny-aya-global` equipped with two specialized LoRA adapters:

* **Model A — Solver LoRA:** Generates a step-by-step physical derivation (`ANSWER: <value>`). When challenged, it re-solves the problem from a clean slate guided by compact, self-contained physical directives accumulated from the Verifier.
* **Model B — Verifier LoRA:** Independently inspects the original problem and Model A's proposed derivation, outputting a concise self-contained directive or confirmation followed by a deterministic text sentinel (`[[APPROVE]]` or `[[CHALLENGE]]`).

### Teacher Model & Multi-Backend Execution (`src/adversarial_physics/utils/runtime.py`)
* **Unified Teacher (`CohereForAI/aya-expanse-8b`):** `TASK_MODEL_RECOMMENDATIONS` configures `CohereForAI/aya-expanse-8b` across extraction, curation/refinement, adversarial flaw synthesis, and directive generation (with `CohereForAI/aya-expanse-32b` supported as an explicit high-VRAM override).
* **Backends (`--backend auto|kaggle_gpu|hf_gpu|local_cpu`, with `hf_zerogpu` accepted as an alias for `hf_gpu`):**
  * **`kaggle_gpu`**: Local CUDA execution (`bfloat16`/`float16`, plus 4-bit NF4 `bitsandbytes` quantization when VRAM `< 24 GB`).
  * **`hf_gpu`**: Remote execution via `huggingface_hub.InferenceClient` (Inference Providers or `HF_ENDPOINT_URL` on `l4x1`/`t4-small`), or local CUDA inside an HF Space/Job.
  * **`local_cpu`**: Local CPU fallback (`torch.float32`, `low_cpu_mem_usage=True`).

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

3. **Build Initial Book Dataset by Chapter Domain (`data/literature/initial_book_dataset.jsonl`):**
   ```powershell
   python scripts/02_extract_literature.py --build-initial-dataset
   ```

4. **Extract Chapter Subtopic Chunks (`data/literature/subtopic_chunks.jsonl` — `1,491` chunks):**
   ```powershell
   python scripts/02b_extract_subtopics.py
   ```

5. **Build Deterministic Stage 1 Pools (`01a_manual_deterministic_only_pool.jsonl` & `01b_deterministic_curated_pool.jsonl`):**
   ```powershell
   python scripts/03_build_canonical_pool.py
   ```

6. **Segment & Link Problems / Full Extraction (`configs/data/book_ingestion.yaml`):**
   ```powershell
   python scripts/02_extract_literature.py --link-only
   python scripts/02_extract_literature.py --config configs/data/book_ingestion.yaml --backend auto
   ```

---

## 5. Stage 2–4: v2 Synthetic Curation, AlignTune Training & AuditKIT Evaluation

1. **Generate Grounded Synthetic E2/E3 Pools & Adversarial Solver/Verifier Trajectories (`03b` → `04d`):**
   Run the full `*_v2*` Phase B commands documented in [`docs/PROJECT_BLUEPRINT.md`](docs/PROJECT_BLUEPRINT.md#6-execution-roadmap--verification-commands), then upload `data/sft_v2/*` to the Hugging Face Hub `-v2` dataset repos configured in `configs/training/aligntune_training_v2.yaml` (since `/data/` is gitignored):
   ```powershell
   hf upload vital-ar/physics-solver-e1-manual-rules-v2     data/sft_v2/solver_e1_manual_det     . --repo-type dataset
   hf upload vital-ar/physics-solver-e2-curatorkit-nollm-v2 data/sft_v2/solver_e2_curatorkit_nollm . --repo-type dataset
   hf upload vital-ar/physics-solver-e3-canonical-llm-v2    data/sft_v2/solver_e3_canonical_llm    . --repo-type dataset
   hf upload vital-ar/physics-solver-e4-adversarial-v2      data/sft_v2/solver_e4_adversarial      . --repo-type dataset
   hf upload vital-ar/physics-verifier-e4-curated-v2        data/sft_v2/verifier_e4_curated        . --repo-type dataset
   ```

2. **Train LoRA Adapters (`05_train_aligntune.py`, pull datasets automatically from HF Hub on Kaggle):**
   ```powershell
   python scripts/05_train_aligntune.py --dataset solver_e2_curatorkit_nollm --config configs/training/aligntune_training_v2.yaml --output-dir models/adapters/solver_e2_curatorkit_nollm_v2_lora --push-to-hub
   python scripts/05_train_aligntune.py --dataset solver_e3_canonical_llm --config configs/training/aligntune_training_v2.yaml --output-dir models/adapters/solver_e3_canonical_llm_v2_lora --push-to-hub
   python scripts/05_train_aligntune.py --dataset solver_e4_adversarial --config configs/training/aligntune_training_v2.yaml --output-dir models/adapters/solver_e4_adversarial_v2_lora --push-to-hub
   python scripts/05_train_aligntune.py --dataset verifier_e4_curated --config configs/training/aligntune_training_v2.yaml --output-dir models/adapters/verifier_e4_curated_v2_lora --push-to-hub
   ```

3. **Evaluate & Validate Iterative Solver–Verifier Loop (`06_eval_solver.py` & `validate_e4_loop.py`):**
   ```powershell
   python scripts/06_eval_solver.py --experiment E0_base_tiny_aya_v2 --base-model CohereLabs/tiny-aya-global --chat-template --output-dir results/eval_e0_v2
   python scripts/06_eval_solver.py --experiment E4a_solver_adversarial_v2 --base-model CohereLabs/tiny-aya-global --adapter-dir vital-ar/adversarial-physics-solver-e4-adversarial-v2-lora --output-dir results/eval_e4a_v2
   python scripts/validate_e4_loop.py --base-model CohereLabs/tiny-aya-global --solver-adapter vital-ar/adversarial-physics-solver-e4-adversarial-v2-lora --verifier-adapter vital-ar/adversarial-physics-verifier-e4-curated-v2-lora --validation-path data/splits/test.jsonl --output-dir outputs/validation/e4_loop_test_v2
   ```

4. **Inspect Datasets & Manifests:**
   ```powershell
   python scripts/inspect_dataset.py --config configs/data/ingestion.yaml --show-rows 2
   python scripts/prepare_experiment.py --config configs/training/aligntune_training_v2.yaml
   ```
