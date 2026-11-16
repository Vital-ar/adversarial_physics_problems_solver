---
base_model: CohereLabs/tiny-aya-global
library_name: peft
license: cc-by-nc-4.0
language:
- en
pipeline_tag: text-generation
datasets:
- vital-ar/physics-verifier-e4-curated-v2
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

- **Dataset Repo:** [`vital-ar/physics-verifier-e4-curated-v2`](https://huggingface.co/datasets/vital-ar/physics-verifier-e4-curated-v2) (**`848`** training examples; strict `< 0.80` MiniLM separation vs. Clean held-out splits)
- **Class Balance:** `128` `[[APPROVE]]` (`15.1%`, across all 6 physics domains) + `720` `[[CHALLENGE]]` (`181` `aya-expanse-8b` 5-taxonomy synthetic errors + `539` real harvested `tiny-aya-global` failures)
- **Domains & Role Split:** `ELECTROMAGNETISM` 166, `MECHANICS` 415, `MODERN_PHYSICS` 82, `OPTICS` 56, `RELATIVITY` 81, `THERMODYNAMICS` 48. Data filtering/generation used `CohereForAI/aya-expanse-8b` exclusively; training & failure harvesting used `CohereLabs/tiny-aya-global` exclusively.

## Training Configuration

- **Version Strategy (v2):** v2 run: lowered Solver LR (`1.0e-4`, `5` epochs), expanded held-out `60`-problem splits, normalized `E1`, added `v2` deduplicated exports.
- **Hyperparameters (`AlignTune` `aligntune==0.1.32`):** Base `CohereLabs/tiny-aya-global` (`3.35B` FP16), LoRA `r=16 / alpha=32`, LR `0.00015`, epochs `5.0/5`, steps `240`, final train loss `0.7214`, selected checkpoint `final`, `completion_only_loss=true`, `seed=42`.

## Evaluation (Clean `cosine < 0.80` Splits)

*Paired Dual-LoRA Loop (`E4a` Solver + this `E4b` Verifier):*

| Clean Split | Overall Acc | Textbook Acc | Sentinel % | Degenerate % | `AuditKIT` SHA-256 |
|---|---|---|---|---|---|
| `val` (`n=37` Clean PhysGym) | **0.00% (0/37)** | **N/A (PhysGym only)** | 97.30% | 2.70% | `—` |
- **Standalone Verifier & Loop Diagnostics:** `val (Clean PhysGym)`: Challenge F1 **100.00%**, False Approval **0.00%**, `AuditKIT AgentLoopDetection` loop-free `1.00`.
- Hard held-out `YimengChen/PhysGym` (`38` problems) stays `0/38` across all runs (see main repo for root-cause analysis and Table 1 pre-filter comparison).

## Usage (`AlignTune` Raw Prompt Format)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_id = "CohereLabs/tiny-aya-global"
adapter_id = "vital-ar/adversarial-physics-verifier-e4-curated-v2-lora"
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
