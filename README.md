# Adversarial Physics Solver–Verifier

**Competition:** Lexsi AI Safety & Alignment Hackathon — Track 1, Problem Statement 8 (*Data-Scarce Domains*)  
**Author:** [`vital-ar`](https://github.com/vital-ar) · **Full Write-Up:** [`docs/HACKATHON_WRITEUP.md`](docs/HACKATHON_WRITEUP.md)  
**Base Student Model:** [`CohereLabs/tiny-aya-global`](https://huggingface.co/CohereLabs/tiny-aya-global) (`3.35B`, `cc-by-nc-4.0`)  
**Teacher / Curator Model:** [`CohereLabs/aya-expanse-8b`](https://huggingface.co/CohereLabs/aya-expanse-8b) (alias `CohereForAI/aya-expanse-8b`, `8.03B`, `cc-by-nc-4.0`)  
**Core Stack:** `CuratorKIT` (`0.1.22`) · `AlignTune` (`0.1.32`) · `AuditKIT` (`1.1.12`)

## Public Hugging Face Collections

* **Models Collection (17 public repos — base, teacher & 15 `v1/v2/v3` LoRA adapters):**  
  [Adversarial Physics Solver-Verifier — Models (Track 1)](https://huggingface.co/collections/vital-ar/adversarial-physics-solver-verifier-models-track-1-6af21ed6fe3d3d356ac11bc9)
* **Datasets & Traces Collection (17 public repos — 15 SFT tiers, `PhysGym`, & full eval/loop traces):**  
  [Adversarial Physics Solver-Verifier — Datasets & Traces](https://huggingface.co/collections/vital-ar/adversarial-physics-solver-verifier-datasets-and-traces-6af9bd217256661d47a74225)
* **Benchmarks, Traces & Calibration Reports Repo:**  
  [`vital-ar/adversarial-physics-eval-traces-benchmarks`](https://huggingface.co/datasets/vital-ar/adversarial-physics-eval-traces-benchmarks)

---

## 1. System Overview & Pipeline Architecture

The **Adversarial Physics Solver–Verifier** equips a single shared `CohereLabs/tiny-aya-global` (`3.35B`, FP16) backbone with two hot-swapped `AlignTune` LoRA adapters:

* **Model A — Solver (`r=16, alpha=24`):** Generates a 1–10 step symbolic physics derivation ending in `ANSWER: <value>`. When challenged, it re-solves **from a clean slate** conditioned only on self-contained physical constraints from the Verifier.
* **Model B — Verifier (`r=8, alpha=16`):** Audits Model A's normalized derivation without seeing ground-truth answers, emits a non-leaking physical directive, and concludes with `[[APPROVE]]` or `[[CHALLENGE]]`.

```mermaid
flowchart LR
    PG["YimengChen/PhysGym (97)<br/>21 Seeds | 38 Val | 38 Test"] --> SPL["Locked Benchmarks<br/>60 Val / 60 Test (v2 & v3)"]
    BK["5 Crowell Textbooks<br/>(986 Book Problems + 1,491 Chunks)"] --> CK["CuratorKIT Stage 1 & 2<br/>Anti-Leak + MinHash + Aya-8B Gates"]
    CK --> SFT["v1 -> v2 -> v3 Ablation Ladder<br/>E1 -> E2 -> E3 -> E4a / E4b"]
    SPL -.->|"0 overlaps, max Jaccard 0.5556"| SFT
    SFT --> AT["AlignTune 0.1.32 (2x T4 FP16)<br/>Solver r=16 | Verifier r=8"]
    AT --> LOOP["Dual-LoRA Adversarial Loop<br/>max_iterations=3 (up to 2 revisions)"]
    SPL --> LOOP
    LOOP --> AK["AuditKIT 1.1.12 + SymPy Oracle<br/>Tables, CIs & Cohen's kappa Audit"]
```

---

## 2. Headline Results (`v3` Locked `n = 60` Validation & Test)

Zero-shot `E0` (`tiny-aya-global`) suffers **31.67–36.67%** token repetition collapse and **0.00% (0/60)** Validation accuracy. Across progressive `CuratorKIT` + `AlignTune` tiers:

| Experiment | Val Acc (`n=60`) | Val Curated (`n=22`) | *p* vs `E0` (Val) | Test Acc (`n=60`) | Test Curated (`n=22`) | Sentinel Rate (Val / Test) | Eq Token F1 (Val / Test) | Degenerate Rate ↓ (Val / Test) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`E0_v3` (Zero-Shot Base)** | `0.00%` (0/60) | `0.00%` | — | `1.67%` (1/60) | `4.55%` | `95.0%` / `93.3%` | `0.2909` / `0.2854` | `31.7%` / `36.7%` |
| **`E1_v3` (Manual Rules)** | `5.00%` (3/60) | `13.64%` | `0.0890` | `1.67%` (1/60) | `4.55%` | `43.3%` / `46.7%` | `0.1773` / `0.1527` | `40.0%` / `33.3%` |
| **`E2_v3` (`CuratorKIT` No-LLM)** | `6.67%` (4/60) | `18.18%` | **`0.0290`** | **`5.00%` (3/60)** | **`13.64%`** | `90.0%` / `95.0%` | `0.2987` / `0.3064` | `10.0%` / `11.7%` |
| **`E3_v3` (`CuratorKIT` + `Aya-8B`)** | **`8.33%` (5/60)** | **`22.73%`** | **`0.0050`** | `3.33%` (2/60) | `9.09%` | `90.0%` / `90.0%` | **`0.3862`** / `0.3723` | `1.7%` / `1.7%` |
| **`E4a_v3` (Turn-0 Adv Solver)** | `5.00%` (3/60) | `13.64%` | `0.0990` | `3.33%` (2/60) | `9.09%` | `91.7%` / `95.0%` | `0.3788` / `0.3830` | `1.7%` / `1.7%` |
| **`E4b_v3` (3-Iter Loop `E4a+E4b`)** | `6.67%` (4/60) | `18.18%` | **`0.0330`** | `3.33%` (2/60) | `9.09%` | **`100.0%` / `100.0%`** | `0.3830` / **`0.3911`** | **`0.0%` / `0.0%`** |

* **Verifier Retraining Calibration:** Fixing a training-vs-inference prefix mismatch and retraining Epoch-7 rank-8 `E4b_v3` reduced false approvals from **`92.50%` → `12.98%` (Val) / `22.50%` (Test)** and lifted Challenge F1 from **`13.95%` → `91.94%` (Val) / `86.11%` (Test)** (`100.0%` Val / `97.9%` Test directive no-leak rate).
* **SymPy-vs-`AlignTune` Grader Agreement (`n=600` pooled `v3`):** **`98.83%`** raw agreement, **Cohen's $\kappa$ = `0.8234`**.
* **Total Tracked Training Compute:** **`19.47` GPU-hours** (`70,096.72s`) across 15 LoRA runs (`v1`, `v2`, `v3`) on free Kaggle `2x T4`. Full reports: [`experiment_comparison_v1_v2_v3.md`](results/comparison_tables/experiment_comparison_v1_v2_v3.md) and [`leak_and_calibration_summary.json`](results/audit_reports/leak_and_calibration_summary.json).

---

## 3. Repository Layout

* [`configs/`](configs) — YAML configurations for ingestion (`ingestion.yaml`, `book_ingestion.yaml`, `split_21_38_38.yaml`), `CuratorKIT` gating (`curatorkit_pipeline.yaml`), `AlignTune` training (`aligntune_training{,_v2,_v3}.yaml`), `AuditKIT` evaluation (`auditkit_benchmark.yaml`), and `v3` dual-LoRA loop inference (`orchestrator.yaml`).
* [`src/adversarial_physics/`](src/adversarial_physics) — Modular Python package (`data/ingestion`, `data/curation`, `data/generation`, `data/processing`, `training`, `orchestration`, `evaluation`, `agents`, `utils`).
* [`scripts/`](scripts) — End-to-end CLI runners (`01_ingest_and_split.py` → `04f_build_and_upload_v3.py`, `05_train_aligntune.py`, `06_eval_solver.py`, `validate_e4_loop.py`, `07_build_experiment_tables.py`, `09_leak_and_calibration_audit.py`).
* [`notebooks/`](notebooks) — Kaggle `2x T4` training & loop evaluation notebooks (`training-v3-e4s-e2.ipynb`, `training-v3-e3-e4v-e1.ipynb`, `retrain-v3-verifier-and-eval.ipynb`, `eval-v3-loop.ipynb`).
* [`data/splits/`](data/splits) — Locked `PhysGym` (`21/38/38`) and expanded `v2/v3` (`60/60`) splits.
* [`results/`](results) — Raw per-problem JSON/JSONL traces, unified comparison tables, prefix-leak ablation, and audit reports.
* [`docs/`](docs) — [`HACKATHON_WRITEUP.md`](docs/HACKATHON_WRITEUP.md) and [`PROJECT_BLUEPRINT.md`](docs/PROJECT_BLUEPRINT.md).
* [`tests/`](tests) — `64` CPU unit & audit tests covering curation gates, anti-leak checks, prompt normalization, loop orchestration, and Cohen's $\kappa$ audits.

---

## 4. Environment Setup & Reproducibility

Python `3.11+` is recommended (`HF_TOKEN` only needed when pushing new models/datasets or accessing gated resources).

### Local CPU Installation & Offline Verification (`Windows` / `Linux`)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1        # Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/local.txt
python -m pip install -e .

# Run full test suite (64 tests), regenerate v1/v2/v3 comparison tables, and run leak & Cohen's kappa audit:
python -m pytest -q
python scripts/07_build_experiment_tables.py
python scripts/09_leak_and_calibration_audit.py
```

### Kaggle `2x T4` GPU Reproduction (`v3` Pipeline)

```bash
python -m pip install -r requirements/kaggle.txt && python -m pip install -e .

# 1. Stage 1 & 2 Data Splits & SFT Export (rebuild locally from curated intermediates; --no-upload skips HF push)
python scripts/01_ingest_and_split.py --config configs/data/ingestion.yaml
python scripts/04f_build_and_upload_v3.py --no-upload

# 2. Train v3 LoRA Adapters via AlignTune (pulls datasets automatically from HF Hub if absent locally)
python scripts/05_train_aligntune.py --dataset solver_e1_manual_det     --config configs/training/aligntune_training_v3.yaml --output-dir models/adapters/solver_e1_manual_det_v3_lora
python scripts/05_train_aligntune.py --dataset solver_e2_curatorkit_nollm --config configs/training/aligntune_training_v3.yaml --output-dir models/adapters/solver_e2_curatorkit_nollm_v3_lora
python scripts/05_train_aligntune.py --dataset solver_e3_canonical_llm  --config configs/training/aligntune_training_v3.yaml --output-dir models/adapters/solver_e3_canonical_llm_v3_lora
python scripts/05_train_aligntune.py --dataset solver_e4_adversarial    --config configs/training/aligntune_training_v3.yaml --output-dir models/adapters/solver_e4_adversarial_v3_lora
python scripts/05_train_aligntune.py --dataset verifier_e4_curated      --config configs/training/aligntune_training_v3.yaml --output-dir models/adapters/verifier_e4_curated_v3_lora

# 3. Single-Turn Solver Evaluation (E0..E4a) on Locked 60-Problem Validation & Test Splits
python scripts/06_eval_solver.py --experiment E3_v3 --base-model CohereLabs/tiny-aya-global --adapter-dir vital-ar/adversarial-physics-solver-e3-canonical-llm-v3-lora --validation-path data/splits/validation_v2.jsonl --test-path data/splits/test_v2.jsonl --prompt-mode raw --output-dir results/eval_e3_v3

# 4. Multi-Turn Adversarial Loop Evaluation (uses v3 defaults + verifier_max_new_tokens=384 from configs/inference/orchestrator.yaml)
python scripts/validate_e4_loop.py --validation-path data/splits/validation_v2.jsonl --max-iterations 3 --output-dir results/eval_e4_loop_v3_val
python scripts/validate_e4_loop.py --validation-path data/splits/test_v2.jsonl       --max-iterations 3 --output-dir results/eval_e4_loop_v3_test
```

For planned extensions (`E3_v3 Solver + E4b_v3 Verifier` cross-pairing, `max_iterations > 3` on PhysGym problems, and DPO/GRPO Verifier alignment), see Section 9 of [`docs/HACKATHON_WRITEUP.md`](docs/HACKATHON_WRITEUP.md).
