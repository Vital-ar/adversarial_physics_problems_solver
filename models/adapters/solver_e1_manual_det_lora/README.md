---
base_model: CohereLabs/tiny-aya-global
library_name: peft
license: cc-by-nc-4.0
language:
- en
pipeline_tag: text-generation
datasets:
- vital-ar/physics-solver-e1-manual-rules
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

# E1 Solver LoRA — Raw Textbook Extraction Baseline

Part of [Adversarial Physics Solver–Verifier](https://github.com/vital-ar/adversarial_physics_problems_solver) (**Track 1: Data-Scarce University Physics**). Single-turn university physics Solver baseline (`CohereLabs/tiny-aya-global`, Model A) trained strictly on raw extracted Crowell textbook exercises and `YimengChen/PhysGym` train seeds filtered with manual deterministic Python rules only (zero `CuratorKIT` LLM judging and zero synthetic augmentation). Exists to isolate the raw textbook baseline inside the `E1→E4b` ablation ladder.

## Training Data & Provenance

- **Dataset Repo:** [`vital-ar/physics-solver-e1-manual-rules`](https://huggingface.co/datasets/vital-ar/physics-solver-e1-manual-rules) (**`1076`** training examples; strict `< 0.80` MiniLM separation vs. Clean held-out splits)
- **Sources & Origin:** Crowell Open Textbooks (`CC BY-SA 4.0`) + `YimengChen/PhysGym` seeds (`CC0-1.0`) — `book_extracted`: 1064, `physgym`: 12
- **Domains & Role Split:** `ELECTROMAGNETISM` 208, `MECHANICS` 628, `MODERN_PHYSICS` 100, `OPTICS` 62, `RELATIVITY` 45, `THERMODYNAMICS` 33. Data filtering/generation used `CohereForAI/aya-expanse-8b` exclusively; training & failure harvesting used `CohereLabs/tiny-aya-global` exclusively.

## Training Configuration

- **Version Strategy (v1):** Initial v1 run (LR `2.0e-4` Solvers / `1.5e-4` Verifier, LoRA `16/32`, `max_epochs=10`, raw unnormalized steps).
- **Hyperparameters (`AlignTune` `aligntune==0.1.32`):** Base `CohereLabs/tiny-aya-global` (`3.35B` FP16), LoRA `r=16 / alpha=32`, LR `0.0002`, epochs `5.0/10`, steps `340`, final train loss `1.1443`, selected checkpoint `final`, `completion_only_loss=true`, `seed=42`.

## Evaluation (Clean `cosine < 0.80` Splits)

| Clean Split | Overall Acc | Textbook Acc | Sentinel % | Degenerate % | `AuditKIT` SHA-256 |
|---|---|---|---|---|---|
| `validation` (`n=38`) | **0.00% (0/38)** | **N/A (PhysGym only)** | 50.00% | 39.47% | `7b8ddac44a121047` |
| `test` (`n=37`) | **0.00% (0/37)** | **N/A (PhysGym only)** | 56.76% | 32.43% | `8f315d25c67b9615` |
- Hard held-out `YimengChen/PhysGym` (`38` problems) stays `0/38` across all runs (see main repo for root-cause analysis and Table 1 pre-filter comparison).

## Usage (`AlignTune` Raw Prompt Format)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_id = "CohereLabs/tiny-aya-global"
adapter_id = "vital-ar/adversarial-physics-solver-e1-manual-det-lora"
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
