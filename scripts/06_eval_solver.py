"""Stage 6 CLI: Fast Batched Solver Evaluation (`06_eval_solver.py`) for E0, E1, E2, E3, and E4A.

Evaluates `CohereLabs/tiny-aya-global` (base E0 or with a LoRA adapter) on `validation.jsonl`
and/or `test.jsonl` using batched greedy decoding (`batch_size=8` by default) and `score_solver_output`
(`±1.5%` numeric, SymPy symbolic, exact normalized, and SI unit checks).

Examples:
    # Run E0 baseline concurrently on Kaggle / local GPU:
    python scripts/06_eval_solver.py --experiment E0 --splits validation test

    # Score the existing E1 adapter in raw prompt mode:
    python scripts/06_eval_solver.py --experiment E1 --adapter-dir models/adapters/solver_e1_manual_det_lora/final_adapter --raw-prompt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adversarial_physics.evaluation.auditkit_bridge import run_physics_eval_audit  # noqa: E402
from adversarial_physics.training.run_aligntune import batched_evaluate_solver_split  # noqa: E402
from adversarial_physics.utils.runtime import TINY_AYA_GLOBAL  # noqa: E402


def _load_eval_model_and_tokenizer(
    base_model: str,
    adapter_dir: str | None = None,
    hf_token: str | None = None,
) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = hf_token or os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(base_model, token=token)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    kwargs: dict[str, Any] = {"token": token, "torch_dtype": dtype}
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
    if adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
    model.eval()
    return model, tokenizer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=str, default="E0")
    parser.add_argument("--base-model", type=str, default=TINY_AYA_GLOBAL)
    parser.add_argument("--adapter-dir", type=str, default=None)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["validation", "test"],
        choices=["validation", "test"],
    )
    parser.add_argument("--validation-path", type=str, default="data/splits/validation.jsonl")
    parser.add_argument("--test-path", type=str, default="data/splits/test.jsonl")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--prompt-mode",
        choices=["raw", "chat_template", "alpaca"],
        default="raw",
        help="Prompt formatting mode. Default 'raw' matches Stage 4 AlignTune instruction_following training.",
    )
    parser.add_argument(
        "--raw-prompt",
        action="store_true",
        help="Alias for `--prompt-mode raw` (preserved for backwards compatibility).",
    )
    parser.add_argument(
        "--chat-template",
        action="store_true",
        help="Shorthand for `--prompt-mode chat_template` (e.g., zero-shot E0 instruct baseline).",
    )
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--hf-token", type=str, default=None)
    args = parser.parse_args(argv)

    prompt_mode = "raw" if args.raw_prompt else ("chat_template" if args.chat_template else args.prompt_mode)

    model, tokenizer = _load_eval_model_and_tokenizer(
        base_model=args.base_model,
        adapter_dir=args.adapter_dir,
        hf_token=args.hf_token,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "experiment": args.experiment,
        "base_model": args.base_model,
        "adapter_dir": args.adapter_dir,
        "prompt_mode": prompt_mode,
        "use_chat_template": prompt_mode == "chat_template",
        "splits": {},
    }

    split_paths = {"validation": args.validation_path, "test": args.test_path}
    for sp in args.splits:
        res = batched_evaluate_solver_split(
            model=model,
            tokenizer=tokenizer,
            split_path=split_paths[sp],
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            max_samples=args.max_samples,
            prompt_mode=prompt_mode,
        )
        out_file = out_dir / f"{args.experiment}_{sp}_eval.json"
        res["lexsi_audit"] = run_physics_eval_audit(res, experiment_name=f"{args.experiment}_{sp}")
        out_file.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        summary["splits"][sp] = {
            "total": res["total"],
            "correct": res["correct"],
            "accuracy": res["accuracy"],
            "sentinel_rate": res["sentinel_rate"],
            "domain_breakdown": res["domain_breakdown"],
            "lexsi_audit": res["lexsi_audit"],
            "report_file": str(out_file),
        }

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
