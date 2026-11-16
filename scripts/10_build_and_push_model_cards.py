"""Build concise, verified Hugging Face model cards for all 15 LoRA adapters (`v1`, `v2`, `v3`) and push them cleanly.

Usage:
    python scripts/10_build_and_push_model_cards.py           # write local adapter README.md files only
    python scripts/10_build_and_push_model_cards.py --push    # write locally + upload README.md to all 15 HF repos
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any


HF_NAMESPACE = "vital-ar"
GITHUB_URL = "https://github.com/vital-ar/adversarial_physics_problems_solver"
MODELS_COLLECTION_URL = (
    "https://huggingface.co/collections/vital-ar/"
    "adversarial-physics-solver-verifier-models-track-1-6af21ed6fe3d3d356ac11bc9"
)
DATASETS_COLLECTION_URL = (
    "https://huggingface.co/collections/vital-ar/"
    "adversarial-physics-solver-verifier-datasets-and-traces-6af9bd217256661d47a74225"
)

ADAPTER_SPECS: list[dict[str, str]] = [
    # v1
    {
        "exp_id": "E1",
        "version": "v1",
        "adapter_dir": "solver_e1_manual_det_lora",
        "hf_repo": f"{HF_NAMESPACE}/adversarial-physics-solver-e1-manual-det-lora",
        "dataset_key": "solver_e1_manual_det",
        "dataset_repo": f"{HF_NAMESPACE}/physics-solver-e1-manual-rules",
        "role_title": "E1 Solver LoRA — Raw Textbook Extraction Baseline",
    },
    {
        "exp_id": "E2",
        "version": "v1",
        "adapter_dir": "solver_e2_curatorkit_nollm_lora",
        "hf_repo": f"{HF_NAMESPACE}/adversarial-physics-solver-e2-curatorkit-nollm-lora",
        "dataset_key": "solver_e2_curatorkit_nollm",
        "dataset_repo": f"{HF_NAMESPACE}/physics-solver-e2-curatorkit-nollm",
        "role_title": "E2 Solver LoRA — CuratorKIT Filter-Before-Generate + SymPy Synthesis",
    },
    {
        "exp_id": "E3",
        "version": "v1",
        "adapter_dir": "solver_e3_canonical_llm_lora",
        "hf_repo": f"{HF_NAMESPACE}/adversarial-physics-solver-e3-canonical-llm-lora",
        "dataset_key": "solver_e3_canonical_llm",
        "dataset_repo": f"{HF_NAMESPACE}/physics-solver-e3-canonical-llm",
        "role_title": "E3 Solver LoRA — Full CuratorKIT LLM Gating & Canonical Steps",
    },
    {
        "exp_id": "E4a",
        "version": "v1",
        "adapter_dir": "solver_e4_adversarial_lora",
        "hf_repo": f"{HF_NAMESPACE}/adversarial-physics-solver-e4-adversarial-lora",
        "dataset_key": "solver_e4_adversarial",
        "dataset_repo": f"{HF_NAMESPACE}/physics-solver-e4-adversarial",
        "role_title": "E4a Solver LoRA — Hint-Conditioned Multi-Turn Physics Solver",
    },
    {
        "exp_id": "E4b",
        "version": "v1",
        "adapter_dir": "verifier_e4_curated_lora",
        "hf_repo": f"{HF_NAMESPACE}/adversarial-physics-verifier-e4-curated-lora",
        "dataset_key": "verifier_e4_curated",
        "dataset_repo": f"{HF_NAMESPACE}/physics-verifier-e4-curated",
        "role_title": "E4b Verifier LoRA — Adversarial Physics Derivation Evaluator",
    },
    # v2
    {
        "exp_id": "E1",
        "version": "v2",
        "adapter_dir": "solver_e1_manual_det_v2_lora",
        "hf_repo": f"{HF_NAMESPACE}/adversarial-physics-solver-e1-manual-det-v2-lora",
        "dataset_key": "solver_e1_manual_det",
        "dataset_repo": f"{HF_NAMESPACE}/physics-solver-e1-manual-rules-v2",
        "role_title": "E1 Solver LoRA — Raw Textbook Extraction Baseline",
    },
    {
        "exp_id": "E2",
        "version": "v2",
        "adapter_dir": "solver_e2_curatorkit_nollm_v2_lora",
        "hf_repo": f"{HF_NAMESPACE}/adversarial-physics-solver-e2-curatorkit-nollm-v2-lora",
        "dataset_key": "solver_e2_curatorkit_nollm",
        "dataset_repo": f"{HF_NAMESPACE}/physics-solver-e2-curatorkit-nollm-v2",
        "role_title": "E2 Solver LoRA — CuratorKIT Filter-Before-Generate + SymPy Synthesis",
    },
    {
        "exp_id": "E3",
        "version": "v2",
        "adapter_dir": "solver_e3_canonical_llm_v2_lora",
        "hf_repo": f"{HF_NAMESPACE}/adversarial-physics-solver-e3-canonical-llm-v2-lora",
        "dataset_key": "solver_e3_canonical_llm",
        "dataset_repo": f"{HF_NAMESPACE}/physics-solver-e3-canonical-llm-v2",
        "role_title": "E3 Solver LoRA — Full CuratorKIT LLM Gating & Canonical Steps",
    },
    {
        "exp_id": "E4a",
        "version": "v2",
        "adapter_dir": "solver_e4_adversarial_v2_lora",
        "hf_repo": f"{HF_NAMESPACE}/adversarial-physics-solver-e4-adversarial-v2-lora",
        "dataset_key": "solver_e4_adversarial",
        "dataset_repo": f"{HF_NAMESPACE}/physics-solver-e4-adversarial-v2",
        "role_title": "E4a Solver LoRA — Hint-Conditioned Multi-Turn Physics Solver",
    },
    {
        "exp_id": "E4b",
        "version": "v2",
        "adapter_dir": "verifier_e4_curated_v2_lora",
        "hf_repo": f"{HF_NAMESPACE}/adversarial-physics-verifier-e4-curated-v2-lora",
        "dataset_key": "verifier_e4_curated",
        "dataset_repo": f"{HF_NAMESPACE}/physics-verifier-e4-curated-v2",
        "role_title": "E4b Verifier LoRA — Adversarial Physics Derivation Evaluator",
    },
    # v3
    {
        "exp_id": "E1",
        "version": "v3",
        "adapter_dir": "solver_e1_manual_det_v3_lora",
        "hf_repo": f"{HF_NAMESPACE}/adversarial-physics-solver-e1-manual-det-v3-lora",
        "dataset_key": "solver_e1_manual_det",
        "dataset_repo": f"{HF_NAMESPACE}/physics-solver-e1-manual-rules-v3",
        "role_title": "E1 Solver LoRA — Raw Textbook Extraction Baseline (Final v3)",
    },
    {
        "exp_id": "E2",
        "version": "v3",
        "adapter_dir": "solver_e2_curatorkit_nollm_v3_lora",
        "hf_repo": f"{HF_NAMESPACE}/adversarial-physics-solver-e2-curatorkit-nollm-v3-lora",
        "dataset_key": "solver_e2_curatorkit_nollm",
        "dataset_repo": f"{HF_NAMESPACE}/physics-solver-e2-curatorkit-nollm-v3",
        "role_title": "E2 Solver LoRA — CuratorKIT Filter-Before-Generate + SymPy Synthesis (Final v3)",
    },
    {
        "exp_id": "E3",
        "version": "v3",
        "adapter_dir": "solver_e3_canonical_llm_v3_lora",
        "hf_repo": f"{HF_NAMESPACE}/adversarial-physics-solver-e3-canonical-llm-v3-lora",
        "dataset_key": "solver_e3_canonical_llm",
        "dataset_repo": f"{HF_NAMESPACE}/physics-solver-e3-canonical-llm-v3",
        "role_title": "E3 Solver LoRA — Full CuratorKIT LLM Gating & Canonical Steps (Final v3)",
    },
    {
        "exp_id": "E4a",
        "version": "v3",
        "adapter_dir": "solver_e4_adversarial_v3_lora",
        "hf_repo": f"{HF_NAMESPACE}/adversarial-physics-solver-e4-adversarial-v3-lora",
        "dataset_key": "solver_e4_adversarial",
        "dataset_repo": f"{HF_NAMESPACE}/physics-solver-e4-adversarial-v3",
        "role_title": "E4a Solver LoRA — Hint-Conditioned Multi-Turn Physics Solver (Final v3)",
    },
    {
        "exp_id": "E4b",
        "version": "v3",
        "adapter_dir": "verifier_e4_curated_v3_lora",
        "hf_repo": f"{HF_NAMESPACE}/adversarial-physics-verifier-e4-curated-v3-lora",
        "dataset_key": "verifier_e4_curated",
        "dataset_repo": f"{HF_NAMESPACE}/physics-verifier-e4-curated-v3",
        "role_title": "E4b Verifier LoRA — Adversarial Physics Derivation Evaluator (Final v3)",
    },
]

_VERSION_DATA_DIR = {"v1": "data/sft", "v2": "data/sft_v2", "v3": "data/sft_v3"}

_PURPOSE_BY_EXP = {
    "E1": (
        "Single-turn university physics Solver baseline (`CohereLabs/tiny-aya-global`, Model A) trained strictly on raw extracted "
        "Crowell textbook exercises and `YimengChen/PhysGym` train seeds filtered with manual deterministic Python rules only "
        "(zero `CuratorKIT` LLM judging and zero synthetic augmentation). Exists to isolate the raw textbook baseline inside the `E1→E4b` ablation ladder."
    ),
    "E2": (
        "Single-turn university physics Solver (`CohereLabs/tiny-aya-global`, Model A) testing **filter-before-generation**: "
        "`CuratorKIT` (`TextCleaner`, `SchemaGate`, `DiversityGate`, `MinHashDeduplicator`, `EmbeddingDeduplicator < 0.80`) "
        "filters raw textbooks *before* `PDFChunker` + `PostHocRetriever` condition `CohereForAI/aya-expanse-8b` to synthesize "
        "`SymPy`-verified problems (no Part-2 LLM judge gates), showing consistent clean gains over raw `E1`."
    ),
    "E3": (
        "Single-turn university physics Solver (`CohereLabs/tiny-aya-global`, Model A) trained on `CuratorKIT` Part-2 LLM-gated "
        "(`SchemaGate`, 6-dim `RewardGate >= 0.82`, `HallucinationGate >= 0.85`, `DiversityGate >= 0.82` driven by `aya-expanse-8b`, "
        "plus `RewardRefiner` recovery) and step-canonicalized derivations (`Step 1..10`). Achieves the strongest single-turn clean accuracy and equation fidelity in `v2/v3`."
    ),
    "E4a": (
        "Multi-turn hint-conditioned Solver (`CohereLabs/tiny-aya-global`, Model A) designed to solve Turn-0 physics problems and revise derivations "
        "when given self-contained `[[CHALLENGE]]` directives from the `E4b` Verifier without repeating mistakes. Extends `E3` Turn-0 derivations "
        "with harvested valid `tiny-aya-global` attempts, `aya-expanse-8b` Alternative Solution 2s, and Turn-1/Turn-2 corrective hint trajectories."
    ),
    "E4b": (
        "Adversarial Physics Evaluator/Verifier (`CohereLabs/tiny-aya-global`, Model B) paired with the `E4a` Solver in a single-backbone dual-LoRA revision loop. "
        "Inspects `[problem, proposed_solution]` pairs, emits a concise, self-contained physical directive (zero step/line references and zero final-answer leakage), "
        "and outputs `[[APPROVE]]` or `[[CHALLENGE]]` to force the Solver to correct mistakes across up to 3 iterations."
    ),
}

_VERSION_DELTA = {
    "v1": "Initial v1 run (LR `2.0e-4` Solvers / `1.5e-4` Verifier, LoRA `16/32`, `max_epochs=10`, raw unnormalized steps).",
    "v2": "v2 run: lowered Solver LR (`1.0e-4`, `5` epochs), expanded held-out `60`-problem splits, normalized `E1`, added `v2` deduplicated exports.",
    "v3": "Final regularized v3 run: `r=16/alpha=24` Solvers (`dropout=0.08`), `r=8/alpha=16` Verifier (`lr=1.0e-4`, `wd=0.03`, `10%` val split), short-key/formula-leak pruning, `**Derivation:**\\n` prefix parity, and `< 0.80` clean audit.",
}


def _summarize_dataset_composition(root: Path, version: str, dataset_key: str) -> dict[str, Any]:
    """Summarize exact row count, sources, turns/verdicts, and domain counts from local train.jsonl."""
    tf = root / _VERSION_DATA_DIR[version] / dataset_key / "train.jsonl"
    rows = [json.loads(l) for l in tf.read_text(encoding="utf-8-sig").splitlines() if l.strip()]

    domains = collections.Counter(str(r.get("domain") or "UNKNOWN") for r in rows)
    prefixes = collections.Counter(
        "physgym"
        if str(r.get("problem_id", "")).startswith("physgym")
        else (
            "synthetic_seed"
            if str(r.get("problem_id", "")).startswith("syn_seed")
            else ("literature_curated" if str(r.get("problem_id", "")).startswith("lit_") else "book_extracted")
        )
        for r in rows
    )
    turns = collections.Counter(int(r["turn"]) for r in rows if "turn" in r) if rows and "turn" in rows[0] else None
    verdicts = collections.Counter(str(r.get("verdict") or "") for r in rows if r.get("verdict"))
    err_sources = collections.Counter(
        str(r.get("source") or r.get("error_type") or "")
        for r in rows
        if r.get("verdict") == "[[CHALLENGE]]"
    )
    explicit_sources = collections.Counter(str(r["source"]) for r in rows if r.get("source"))

    return {
        "n_rows": len(rows),
        "domains": dict(sorted(domains.items())),
        "id_prefixes": dict(sorted(prefixes.items())),
        "explicit_sources": dict(sorted(explicit_sources.items())),
        "turns": dict(sorted(turns.items())) if turns else None,
        "verdicts": dict(sorted(verdicts.items())) if verdicts else None,
        "challenge_sources": dict(sorted(err_sources.items())) if err_sources else None,
    }


def _format_data_bullets(spec: dict[str, str], ds: dict[str, Any]) -> list[str]:
    """Render concise, complete training data breakdown lines for the adapter card."""
    exp = spec["exp_id"]
    ver = spec["version"]
    lines = [
        f"- **Dataset Repo:** [`{spec['dataset_repo']}`](https://huggingface.co/datasets/{spec['dataset_repo']}) (**`{ds['n_rows']}`** training examples; strict `< 0.80` MiniLM separation vs. Clean held-out splits)",
    ]
    if exp == "E4b" and ds["verdicts"]:
        app_n = ds["verdicts"].get("[[APPROVE]]", 0)
        chal_n = ds["verdicts"].get("[[CHALLENGE]]", 0)
        cs = ds["challenge_sources"] or {}
        real_n = cs.get("real_model_a_failure", 0)
        synth_n = chal_n - real_n
        extra_bal = "domain-balanced `18/domain`" if ver == "v3" else "across all 6 physics domains"
        lines.append(
            f"- **Class Balance:** `{app_n}` `[[APPROVE]]` (`{app_n / max(1, ds['n_rows']) * 100:.1f}%`, {extra_bal}) + "
            f"`{chal_n}` `[[CHALLENGE]]` (`{synth_n}` `aya-expanse-8b` 5-taxonomy synthetic errors + `{real_n}` real harvested `tiny-aya-global` failures)"
        )
    elif exp == "E4a" and ds["turns"]:
        t0 = ds["turns"].get(0, 0)
        t1 = ds["turns"].get(1, 0)
        t2 = ds["turns"].get(2, 0)
        src_desc = ", ".join(f"`{k}`: {v}" for k, v in (ds["explicit_sources"] or ds["id_prefixes"]).items())
        lines.append(
            f"- **Turn Mix & Sources:** Turn 0: `{t0}`, Turn 1 (1 hint): `{t1}`, Turn 2 (2 hints): `{t2}` | Sources: {src_desc}"
        )
    else:
        src_map = ds["explicit_sources"] if ds["explicit_sources"] else ds["id_prefixes"]
        src_desc = ", ".join(f"`{k}`: {v}" for k, v in src_map.items())
        lines.append(f"- **Sources & Origin:** Crowell Open Textbooks (`CC BY-SA 4.0`) + `YimengChen/PhysGym` seeds (`CC0-1.0`) — {src_desc}")

    dom_desc = ", ".join(f"`{k}` {v}" for k, v in ds["domains"].items())
    lines.append(f"- **Domains & Role Split:** {dom_desc}. Data filtering/generation used `CohereForAI/aya-expanse-8b` exclusively; training & failure harvesting used `CohereLabs/tiny-aya-global` exclusively.")
    if ver == "v3" and exp in {"E1", "E2", "E3", "E4a", "E4b"}:
        extra = {
            "E1": "Pruned `< 220`-char answer keys + `CuratorKIT` `ExactDeduplicator` + sentinel gate.",
            "E2": "`CuratorKIT` Part-1 filter before `aya-expanse-8b` synthesis + `>= 260`-char `Step 1..` normalization + dedup.",
            "E3": "Part-2 `CuratorKIT` LLM gates + `RewardRefiner` + `_ensure_structured_steps` (`>= 400` chars).",
            "E4a": "Downgraded fallback 2nd hints to Turn-1, capped completion multiplicity `<= 2`, `>= 400`-char clean steps.",
            "E4b": "Purged off-by-one approvals, stripped hallucinated formulas (`< 50%` overlap), normalized `**Derivation:**\\n` headers & openings.",
        }[exp]
        lines.append(f"- **v3 Data Curation:** {extra}")
    return lines


def _format_eval_section(
    spec: dict[str, str],
    comp: dict[str, Any],
    ak_nat: dict[str, Any],
) -> list[str]:
    """Render Clean (< 0.80) Validation/Test evaluation summary for this adapter."""
    exp = spec["exp_id"]
    ver = spec["version"]

    lines = ["## Evaluation (Clean `cosine < 0.80` Splits)", ""]
    if (ver == "v1" and exp in {"E3", "E4b"}):
        lines.append("- Not evaluated separately in `v1` (superseded by `v2` and `v3`).")
        return lines

    rows: list[str] = []
    ak_views = ak_nat.get("views", {}).get("clean_080", {}).get(ver, {})
    loop_audits = ak_nat.get("loop_agent_eval_and_judge_calibration", {})

    for split in ("validation", "test"):
        m = comp.get("versions_clean_080", {}).get(ver, {}).get(split, {}).get(exp)
        if not m:
            continue
        split_lbl = "`val` (`n=37` Clean PhysGym)" if (ver == "v2" and exp == "E4b" and split == "test") else f"`{split}` (`n={m['total']}`)"
        overall = f"{m['accuracy'] * 100:.2f}% ({m['correct']}/{m['total']})"
        cur_tot = int(m.get("curated_total", 0))
        textbook = f"{m['curated_accuracy'] * 100:.2f}% ({m['curated_correct']}/{cur_tot})" if cur_tot > 0 else "N/A (PhysGym only)"
        degen = f"{m['degenerate_rate'] * 100:.2f}%"
        sent = f"{m['sentinel_rate'] * 100:.2f}%"
        fp = (ak_views.get(split, {}).get("runs", {}).get(exp, {}) or {}).get("fingerprint", "—")
        rows.append(f"| {split_lbl} | **{overall}** | **{textbook}** | {sent} | {degen} | `{fp}` |")

    if rows:
        if exp == "E4b":
            lines.append("*Paired Dual-LoRA Loop (`E4a` Solver + this `E4b` Verifier):*")
            lines.append("")
        lines.append("| Clean Split | Overall Acc | Textbook Acc | Sentinel % | Degenerate % | `AuditKIT` SHA-256 |")
        lines.append("|---|---|---|---|---|---|")
        lines.extend(rows)

    if exp == "E4b":
        key_map = {
            ("v2", "test"): ("val (Clean PhysGym)", "E4b_v2_test_clean_080"),
            ("v3", "validation"): ("validation", "E4b_v3_val_clean_080"),
            ("v3", "test"): ("test", "E4b_v3_test_clean_080"),
        }
        notes = []
        for split in ("validation", "test"):
            m = comp.get("versions_clean_080", {}).get(ver, {}).get(split, {}).get("E4b")
            lbl_s, ak_key = key_map.get((ver, split), (split, ""))
            ld = loop_audits.get(ak_key, {})
            if not m:
                continue
            ver_m = (m.get("loop_diagnostics") or {}).get("verifier") or {}
            if ver_m:
                cf1 = ver_m.get("verifier_challenge_f1")
                fa = ver_m.get("verifier_false_approval_rate")
                lf = ld.get("loop_free_rate")
                cf1_s = f"{cf1 * 100:.2f}%" if cf1 is not None else "—"
                fa_s = f"{fa * 100:.2f}%" if fa is not None else "—"
                lf_s = f"{lf:.2f}" if lf is not None else "—"
                notes.append(f"`{lbl_s}`: Challenge F1 **{cf1_s}**, False Approval **{fa_s}**, `AuditKIT AgentLoopDetection` loop-free `{lf_s}`")
        if notes:
            lines.append("- **Standalone Verifier & Loop Diagnostics:** " + "; ".join(notes) + ".")

    lines.append("- Hard held-out `YimengChen/PhysGym` (`38` problems) stays `0/38` across all runs (see main repo for root-cause analysis and Table 1 pre-filter comparison).")
    return lines


def _format_usage_snippet(spec: dict[str, str]) -> list[str]:
    is_verifier = spec["exp_id"] == "E4b"
    repo_id = spec["hf_repo"]
    if is_verifier:
        prompt_example = (
            'prompt = (\n'
            '    "Review the proposed solution to the physics problem below. First, write a concise, "\n'
            '    "self-contained physical or mathematical directive that either confirms why the derivation "\n'
            '    "is valid or states the exact rule/constraint needed to fix the error without referencing "\n'
            '    "line or step numbers. Then, on a new final line, output [[APPROVE]] if the solution and "\n'
            '    "final answer are completely correct, or [[CHALLENGE]] if any physical, algebraic, or "\n'
            '    "unit error exists.\\n\\nPROBLEM:\\nA 2 kg block starts from rest on a 5 m frictionless "\n'
            '    "ramp. Find speed at bottom.\\n\\nPROPOSED SOLUTION:\\n**Derivation:**\\n"\n'
            '    "Step 1: v = sqrt(2*g*h) = sqrt(2*9.8*5) = 9.9 m/s\\nANSWER: 9.9 m/s\\n\\n"\n'
            ')'
        )
    else:
        prompt_example = (
            'prompt = (\n'
            '    "Solve the following physics problem step by step (in at most 10 logical steps). "\n'
            '    "State the governing physical equations, perform algebraic substitutions clearly, include "\n'
            '    "SI units, and end with the final answer on a new line prefixed with ANSWER:.\\n\\n"\n'
            '    "PROBLEM:\\nA 2 kg block starts from rest on a 5 m frictionless ramp. Find its speed "\n'
            '    "at the bottom.\\n\\n"\n'
            ')'
        )

    gen_kwargs = (
        'max_new_tokens=384, do_sample=False, stop_strings=["[[APPROVE]]", "[[CHALLENGE]]"], tokenizer=tok'
        if is_verifier
        else "max_new_tokens=1024, do_sample=False"
    )

    return [
        "## Usage (`AlignTune` Raw Prompt Format)",
        "",
        "```python",
        "import torch",
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
        "from peft import PeftModel",
        "",
        'base_id = "CohereLabs/tiny-aya-global"',
        f'adapter_id = "{repo_id}"',
        "tok = AutoTokenizer.from_pretrained(base_id)",
        'base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.float16, device_map="auto")',
        "model = PeftModel.from_pretrained(base, adapter_id).eval()",
        prompt_example,
        'inputs = tok(prompt, return_tensors="pt").to(model.device)',
        f"out = model.generate(**inputs, {gen_kwargs})",
        "print(tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))",
        "```",
    ]


def build_model_card(
    spec: dict[str, str],
    root: str | Path = ".",
    *,
    comp_data: dict[str, Any] | None = None,
    ak_nat_data: dict[str, Any] | None = None,
) -> str:
    """Build a short, self-contained, factually verified Hugging Face model card string."""
    root_p = Path(root)
    comp = comp_data or json.loads(
        (root_p / "results/comparison_tables/experiment_comparison_v1_v2_v3.json").read_text(encoding="utf-8")
    )
    ak_nat = ak_nat_data or json.loads(
        (root_p / "results/audit_reports/auditkit_native_summary.json").read_text(encoding="utf-8")
    )
    td_row = next(
        (r for r in comp.get("training_dynamics", []) if r.get("adapter_name") == spec["adapter_dir"]),
        {},
    )
    ds = _summarize_dataset_composition(root_p, spec["version"], spec["dataset_key"])

    lines: list[str] = [
        "---",
        "base_model: CohereLabs/tiny-aya-global",
        "library_name: peft",
        "license: cc-by-nc-4.0",
        "language:",
        "- en",
        "pipeline_tag: text-generation",
        "datasets:",
        f"- {spec['dataset_repo']}",
        "- YimengChen/PhysGym",
        "tags:",
        "- base_model:adapter:CohereLabs/tiny-aya-global",
        "- lora",
        "- sft",
        "- trl",
        "- aligntune",
        "- curatorkit",
        "- auditkit",
        "- physics",
        "---",
        "",
        f"# {spec['role_title']}",
        "",
        f"Part of [Adversarial Physics Solver–Verifier]({GITHUB_URL}) (**Track 1: Data-Scarce University Physics**). "
        + _PURPOSE_BY_EXP[spec["exp_id"]],
        "",
        "## Training Data & Provenance",
        "",
        *_format_data_bullets(spec, ds),
        "",
        "## Training Configuration",
        "",
        f"- **Version Strategy ({spec['version']}):** {_VERSION_DELTA[spec['version']]}",
        f"- **Hyperparameters (`AlignTune` `aligntune==0.1.32`):** Base `CohereLabs/tiny-aya-global` (`3.35B` FP16), "
        f"LoRA `r={td_row.get('lora_r', 16)} / alpha={td_row.get('lora_alpha', 32)}`, LR `{td_row.get('learning_rate')}`, "
        f"epochs `{td_row.get('epochs_completed')}/{td_row.get('max_epochs_configured')}`, "
        f"steps `{td_row.get('total_steps')}`, final train loss `{td_row.get('final_train_loss')}`, "
        f"selected checkpoint `{td_row.get('selected_checkpoint', 'final')}`, `completion_only_loss=true`, `seed=42`.",
        "",
        *_format_eval_section(spec, comp, ak_nat),
        "",
        *_format_usage_snippet(spec),
        "",
        "## Links & Attribution",
        "",
        f"- **Author:** `@vital-ar` | **Repo:** [GitHub]({GITHUB_URL}) | **Collections:** [Models (15)]({MODELS_COLLECTION_URL}) · [Datasets & Traces]({DATASETS_COLLECTION_URL})",
        "",
    ]
    return "\n".join(lines)


def build_all_and_optionally_push(root: str | Path = ".", *, push: bool = False) -> list[dict[str, str]]:
    """Build all 15 adapter cards locally and optionally upload README.md to HF Hub."""
    root_p = Path(root)
    comp = json.loads((root_p / "results/comparison_tables/experiment_comparison_v1_v2_v3.json").read_text(encoding="utf-8"))
    ak_nat = json.loads((root_p / "results/audit_reports/auditkit_native_summary.json").read_text(encoding="utf-8"))

    api = None
    if push:
        from huggingface_hub import HfApi

        api = HfApi()

    results: list[dict[str, str]] = []
    for spec in ADAPTER_SPECS:
        card = build_model_card(spec, root_p, comp_data=comp, ak_nat_data=ak_nat)
        adapter_root = root_p / "models/adapters" / spec["adapter_dir"]
        written_paths: list[str] = []
        for target_dir in (adapter_root, adapter_root / "final_adapter"):
            if target_dir.is_dir():
                out_file = target_dir / "README.md"
                out_file.write_text(card, encoding="utf-8")
                written_paths.append(str(out_file))

        if push and api is not None:
            tmp_file = adapter_root / "README.md"
            api.upload_file(
                path_or_fileobj=str(tmp_file),
                path_in_repo="README.md",
                repo_id=spec["hf_repo"],
                repo_type="model",
                commit_message=f"Add concise verified model card for {spec['exp_id']} ({spec['version']})",
            )

        results.append(
            {
                "hf_repo": spec["hf_repo"],
                "lines": str(len(card.splitlines())),
                "local_files": ", ".join(written_paths),
                "pushed": str(push),
            }
        )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=str, default=".")
    parser.add_argument("--push", action="store_true", help="Upload README.md to all 15 HF Hub model repos.")
    args = parser.parse_args(argv)

    rows = build_all_and_optionally_push(args.root_dir, push=args.push)
    for r in rows:
        print(f"{r['hf_repo']}: {r['lines']} lines | pushed={r['pushed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
