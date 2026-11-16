---
base_model: CohereLabs/tiny-aya-global
library_name: peft
license: cc-by-nc-4.0
language:
- en
pipeline_tag: text-generation
datasets:
- vital-ar/physics-verifier-e4-curated
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

# E4b Verifier LoRA — Adversarial Physics Derivation Evaluator

Part of [Adversarial Physics Solver–Verifier](https://github.com/vital-ar/adversarial_physics_problems_solver) (**Track 1: Data-Scarce University Physics**). Adversarial Physics Evaluator/Verifier (`CohereLabs/tiny-aya-global`, Model B) paired with the `E4a` Solver in a single-backbone dual-LoRA revision loop. Inspects `[problem, proposed_solution]` pairs, emits a concise, self-contained physical directive (zero step/line references and zero final-answer leakage), and outputs `[[APPROVE]]` or `[[CHALLENGE]]` to force the Solver to correct mistakes across up to 3 iterations.

## Training Data & Provenance

- **Dataset Repo:** [`vital-ar/physics-verifier-e4-curated`](https://huggingface.co/datasets/vital-ar/physics-verifier-e4-curated) (**`1511`** training examples; strict `< 0.80` MiniLM separation vs. Clean held-out splits)
- **Class Balance:** `180` `[[APPROVE]]` (`11.9%`, across all 6 physics domains) + `1331` `[[CHALLENGE]]` (`681` `aya-expanse-8b` 5-taxonomy synthetic errors + `650` real harvested `tiny-aya-global` failures)
- **Domains & Role Split:** `ELECTROMAGNETISM` 324, `MECHANICS` 782, `MODERN_PHYSICS` 153, `OPTICS` 78, `RELATIVITY` 100, `THERMODYNAMICS` 74. Data filtering/generation used `CohereForAI/aya-expanse-8b` exclusively; training & failure harvesting used `CohereLabs/tiny-aya-global` exclusively.

## Training Configuration

- **Version Strategy (v1):** Initial v1 run (LR `2.0e-4` Solvers / `1.5e-4` Verifier, LoRA `16/32`, `max_epochs=10`, raw unnormalized steps).
- **Hyperparameters (`AlignTune` `aligntune==0.1.32`):** Base `CohereLabs/tiny-aya-global` (`3.35B` FP16), LoRA `r=16 / alpha=32`, LR `0.00015`, epochs `5.0/10`, steps `425`, final train loss `0.6626`, selected checkpoint `final`, `completion_only_loss=true`, `seed=42`.

## Evaluation (Clean `cosine < 0.80` Splits)

- Not evaluated separately in `v1` (superseded by `v2` and `v3`).

## Usage (`AlignTune` Raw Prompt Format)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_id = "CohereLabs/tiny-aya-global"
adapter_id = "vital-ar/adversarial-physics-verifier-e4-curated-lora"
tok = AutoTokenizer.from_pretrained(base_id)
base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(base, adapter_id).eval()
prompt = (
    "Review the proposed solution to the physics problem below. First, write a concise, "
    "self-contained physical or mathematical directive that either confirms why the derivation "
    "is valid or states the exact rule/constraint needed to fix the error without referencing "
    "line or step numbers. Then, on a new final line, output [[APPROVE]] if the solution and "
    "final answer are completely correct, or [[CHALLENGE]] if any physical, algebraic, or "
    "unit error exists.\n\nPROBLEM:\nA 2 kg block starts from rest on a 5 m frictionless "
    "ramp. Find speed at bottom.\n\nPROPOSED SOLUTION:\n**Derivation:**\n"
    "Step 1: v = sqrt(2*g*h) = sqrt(2*9.8*5) = 9.9 m/s\nANSWER: 9.9 m/s\n\n"
)
inputs = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=384, do_sample=False, stop_strings=["[[APPROVE]]", "[[CHALLENGE]]"], tokenizer=tok)
print(tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

## Links & Attribution

- **Author:** `@vital-ar` | **Repo:** [GitHub](https://github.com/vital-ar/adversarial_physics_problems_solver) | **Collections:** [Models (15)](https://huggingface.co/collections/vital-ar/adversarial-physics-solver-verifier-models-track-1-6af21ed6fe3d3d356ac11bc9) · [Datasets & Traces](https://huggingface.co/collections/vital-ar/adversarial-physics-solver-verifier-datasets-and-traces-6af9bd217256661d47a74225)
