---
base_model: CohereLabs/tiny-aya-global
library_name: peft
license: cc-by-nc-4.0
language:
- en
pipeline_tag: text-generation
datasets:
- vital-ar/physics-solver-e4-adversarial-v3
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

# E4a Solver LoRA — Hint-Conditioned Multi-Turn Physics Solver (Final v3)

Part of [Adversarial Physics Solver–Verifier](https://github.com/vital-ar/adversarial_physics_problems_solver) (**Track 1: Data-Scarce University Physics**). Multi-turn hint-conditioned Solver (`CohereLabs/tiny-aya-global`, Model A) designed to solve Turn-0 physics problems and revise derivations when given self-contained `[[CHALLENGE]]` directives from the `E4b` Verifier without repeating mistakes. Extends `E3` Turn-0 derivations with harvested valid `tiny-aya-global` attempts, `aya-expanse-8b` Alternative Solution 2s, and Turn-1/Turn-2 corrective hint trajectories.

## Training Data & Provenance

- **Dataset Repo:** [`vital-ar/physics-solver-e4-adversarial-v3`](https://huggingface.co/datasets/vital-ar/physics-solver-e4-adversarial-v3) (**`758`** training examples; strict `< 0.80` MiniLM separation vs. Clean held-out splits)
- **Turn Mix & Sources:** Turn 0: `623`, Turn 1 (1 hint): `100`, Turn 2 (2 hints): `35` | Sources: `literature_curated`: 720, `physgym_seed`: 7, `synthetic_seed_subtopic`: 31
- **Domains & Role Split:** `ELECTROMAGNETISM` 138, `MECHANICS` 421, `MODERN_PHYSICS` 69, `OPTICS` 37, `RELATIVITY` 59, `THERMODYNAMICS` 34. Data filtering/generation used `CohereForAI/aya-expanse-8b` exclusively; training & failure harvesting used `CohereLabs/tiny-aya-global` exclusively.
- **v3 Data Curation:** Downgraded fallback 2nd hints to Turn-1, capped completion multiplicity `<= 2`, `>= 400`-char clean steps.

## Training Configuration

- **Version Strategy (v3):** Final regularized v3 run: `r=16/alpha=24` Solvers (`dropout=0.08`), `r=8/alpha=16` Verifier (`lr=1.0e-4`, `wd=0.03`, `10%` val split), short-key/formula-leak pruning, `**Derivation:**\n` prefix parity, and `< 0.80` clean audit.
- **Hyperparameters (`AlignTune` `aligntune==0.1.32`):** Base `CohereLabs/tiny-aya-global` (`3.35B` FP16), LoRA `r=16 / alpha=24`, LR `0.0001`, epochs `5.0/7`, steps `240`, final train loss `0.788`, selected checkpoint `checkpoint-96`, `completion_only_loss=true`, `seed=42`.

## Evaluation (Clean `cosine < 0.80` Splits)

| Clean Split | Overall Acc | Textbook Acc | Sentinel % | Degenerate % | `AuditKIT` SHA-256 |
|---|---|---|---|---|---|
| `validation` (`n=48`) | **2.08% (1/48)** | **10.00% (1/10)** | 89.58% | 2.08% | `54cbe44db34f8fde` |
| `test` (`n=55`) | **3.64% (2/55)** | **11.76% (2/17)** | 94.55% | 1.82% | `29b4374863bc2dfb` |
- Hard held-out `YimengChen/PhysGym` (`38` problems) stays `0/38` across all runs (see main repo for root-cause analysis and Table 1 pre-filter comparison).

## Usage (`AlignTune` Raw Prompt Format)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_id = "CohereLabs/tiny-aya-global"
adapter_id = "vital-ar/adversarial-physics-solver-e4-adversarial-v3-lora"
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
