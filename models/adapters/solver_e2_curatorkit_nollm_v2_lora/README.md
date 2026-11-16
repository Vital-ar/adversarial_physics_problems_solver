---
base_model: CohereLabs/tiny-aya-global
library_name: peft
license: cc-by-nc-4.0
language:
- en
pipeline_tag: text-generation
datasets:
- vital-ar/physics-solver-e2-curatorkit-nollm-v2
- YimengChen/PhysGym
tags:
- base_model:adapter:CohereLabs/tiny-aya-global
- lora
- sft
- trl
- aligntune
- curatorkit
- auditkit
- physics
---

# E2 Solver LoRA — CuratorKIT Filter-Before-Generate + SymPy Synthesis

Part of [Adversarial Physics Solver–Verifier](https://github.com/vital-ar/adversarial_physics_problems_solver) (**Track 1: Data-Scarce University Physics**). Single-turn university physics Solver (`CohereLabs/tiny-aya-global`, Model A) testing **filter-before-generation**: `CuratorKIT` (`TextCleaner`, `SchemaGate`, `DiversityGate`, `MinHashDeduplicator`, `EmbeddingDeduplicator < 0.80`) filters raw textbooks *before* `PDFChunker` + `PostHocRetriever` condition `CohereForAI/aya-expanse-8b` to synthesize `SymPy`-verified problems (no Part-2 LLM judge gates), showing consistent clean gains over raw `E1`.

## Training Data & Provenance

- **Dataset Repo:** [`vital-ar/physics-solver-e2-curatorkit-nollm-v2`](https://huggingface.co/datasets/vital-ar/physics-solver-e2-curatorkit-nollm-v2) (**`865`** training examples; strict `< 0.80` MiniLM separation vs. Clean held-out splits)
- **Sources & Origin:** Crowell Open Textbooks (`CC BY-SA 4.0`) + `YimengChen/PhysGym` seeds (`CC0-1.0`) — `book_initial`: 439, `literature_curated`: 388, `physgym`: 1, `physgym_seed`: 5, `synthetic_seed_subtopic`: 32
- **Domains & Role Split:** `ELECTROMAGNETISM` 174, `MECHANICS` 535, `MODERN_PHYSICS` 69, `OPTICS` 35, `RELATIVITY` 31, `THERMODYNAMICS` 21. Data filtering/generation used `CohereForAI/aya-expanse-8b` exclusively; training & failure harvesting used `CohereLabs/tiny-aya-global` exclusively.

## Training Configuration

- **Version Strategy (v2):** v2 run: lowered Solver LR (`1.0e-4`, `5` epochs), expanded held-out `60`-problem splits, normalized `E1`, added `v2` deduplicated exports.
- **Hyperparameters (`AlignTune` `aligntune==0.1.32`):** Base `CohereLabs/tiny-aya-global` (`3.35B` FP16), LoRA `r=16 / alpha=32`, LR `0.0001`, epochs `5.0/5`, steps `275`, final train loss `1.1818`, selected checkpoint `checkpoint-165`, `completion_only_loss=true`, `seed=42`.

## Evaluation (Clean `cosine < 0.80` Splits)

| Clean Split | Overall Acc | Textbook Acc | Sentinel % | Degenerate % | `AuditKIT` SHA-256 |
|---|---|---|---|---|---|
| `validation` (`n=44`) | **2.27% (1/44)** | **16.67% (1/6)** | 88.64% | 4.55% | `a26c87bfc992f97b` |
| `test` (`n=49`) | **0.00% (0/49)** | **0.00% (0/12)** | 89.80% | 8.16% | `b279eb7b58ab69bb` |
- Hard held-out `YimengChen/PhysGym` (`38` problems) stays `0/38` across all runs (see main repo for root-cause analysis and Table 1 pre-filter comparison).

## Usage (`AlignTune` Raw Prompt Format)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_id = "CohereLabs/tiny-aya-global"
adapter_id = "vital-ar/adversarial-physics-solver-e2-curatorkit-nollm-v2-lora"
tok = AutoTokenizer.from_pretrained(base_id)
base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(base, adapter_id).eval()
prompt = (
    "Solve the following physics problem step by step (in at most 10 logical steps). "
    "State the governing physical equations, perform algebraic substitutions clearly, include "
    "SI units, and end with the final answer on a new line prefixed with ANSWER:.\n\n"
    "PROBLEM:\nA 2 kg block starts from rest on a 5 m frictionless ramp. Find its speed "
    "at the bottom.\n\n"
)
inputs = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
print(tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

## Links & Attribution

- **Author:** `@vital-ar` | **Repo:** [GitHub](https://github.com/vital-ar/adversarial_physics_problems_solver) | **Collections:** [Models (15)](https://huggingface.co/collections/vital-ar/adversarial-physics-solver-verifier-models-track-1-6af21ed6fe3d3d356ac11bc9) · [Datasets & Traces](https://huggingface.co/collections/vital-ar/adversarial-physics-solver-verifier-datasets-and-traces-6af9bd217256661d47a74225)
