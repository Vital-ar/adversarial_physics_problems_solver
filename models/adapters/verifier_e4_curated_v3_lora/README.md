---
base_model: CohereLabs/tiny-aya-global
library_name: peft
license: cc-by-nc-4.0
language:
- en
pipeline_tag: text-generation
datasets:
- vital-ar/physics-verifier-e4-curated-v3
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

# E4b Verifier LoRA — Adversarial Physics Derivation Evaluator (Final v3)

Part of [Adversarial Physics Solver–Verifier](https://github.com/vital-ar/adversarial_physics_problems_solver) (**Track 1: Data-Scarce University Physics**). Adversarial Physics Evaluator/Verifier (`CohereLabs/tiny-aya-global`, Model B) paired with the `E4a` Solver in a single-backbone dual-LoRA revision loop. Inspects `[problem, proposed_solution]` pairs, emits a concise, self-contained physical directive (zero step/line references and zero final-answer leakage), and outputs `[[APPROVE]]` or `[[CHALLENGE]]` to force the Solver to correct mistakes across up to 3 iterations.

## Training Data & Provenance

- **Dataset Repo:** [`vital-ar/physics-verifier-e4-curated-v3`](https://huggingface.co/datasets/vital-ar/physics-verifier-e4-curated-v3) (**`720`** training examples; strict `< 0.80` MiniLM separation vs. Clean held-out splits)
- **Class Balance:** `108` `[[APPROVE]]` (`15.0%`, domain-balanced `18/domain`) + `612` `[[CHALLENGE]]` (`558` `aya-expanse-8b` 5-taxonomy synthetic errors + `54` real harvested `tiny-aya-global` failures)
- **Domains & Role Split:** `ELECTROMAGNETISM` 138, `MECHANICS` 373, `MODERN_PHYSICS` 60, `OPTICS` 52, `RELATIVITY` 63, `THERMODYNAMICS` 34. Data filtering/generation used `CohereForAI/aya-expanse-8b` exclusively; training & failure harvesting used `CohereLabs/tiny-aya-global` exclusively.
- **v3 Data Curation:** Purged off-by-one approvals, stripped hallucinated formulas (`< 50%` overlap), normalized `**Derivation:**\n` headers & openings.

## Training Configuration

- **Version Strategy (v3):** Final regularized v3 run: `r=16/alpha=24` Solvers (`dropout=0.08`), `r=8/alpha=16` Verifier (`lr=1.0e-4`, `wd=0.03`, `10%` val split), short-key/formula-leak pruning, `**Derivation:**\n` prefix parity, and `< 0.80` clean audit.
- **Hyperparameters (`AlignTune` `aligntune==0.1.32`):** Base `CohereLabs/tiny-aya-global` (`3.35B` FP16), LoRA `r=8 / alpha=16`, LR `0.0001`, epochs `7.0/7`, steps `287`, final train loss `0.8361`, selected checkpoint `final`, `completion_only_loss=true`, `seed=42`.

## Evaluation (Clean `cosine < 0.80` Splits)

*Paired Dual-LoRA Loop (`E4a` Solver + this `E4b` Verifier):*

| Clean Split | Overall Acc | Textbook Acc | Sentinel % | Degenerate % | `AuditKIT` SHA-256 |
|---|---|---|---|---|---|
| `validation` (`n=48`) | **4.17% (2/48)** | **20.00% (2/10)** | 100.00% | 0.00% | `781e797efc3463b9` |
| `test` (`n=55`) | **3.64% (2/55)** | **11.76% (2/17)** | 100.00% | 0.00% | `f6e22dd6625778c7` |
- **Standalone Verifier & Loop Diagnostics:** `validation`: Challenge F1 **92.61%**, False Approval **12.96%**, `AuditKIT AgentLoopDetection` loop-free `1.00`; `test`: Challenge F1 **84.97%**, False Approval **24.07%**, `AuditKIT AgentLoopDetection` loop-free `1.00`.
- Hard held-out `YimengChen/PhysGym` (`38` problems) stays `0/38` across all runs (see main repo for root-cause analysis and Table 1 pre-filter comparison).

## Usage (`AlignTune` Raw Prompt Format)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_id = "CohereLabs/tiny-aya-global"
adapter_id = "vital-ar/adversarial-physics-verifier-e4-curated-v3-lora"
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
