"""Multi-backend runtime execution and Aya-Expanse (8B / 32B) model selector.

Supports three execution environments across all inference, synthesis, curation, and training pipelines:
  1. ``kaggle_gpu``  — Kaggle T4 / P100 / 2xT4 GPU environment (CUDA, fp16/bf16 or 4-bit bitsandbytes).
  2. ``hf_zerogpu``  — Hugging Face Spaces ZeroGPU (``@spaces.GPU`` dynamic CUDA allocation).
  3. ``local_cpu``   — Local CPU execution (``torch.float32`` / ``bfloat16``, low_cpu_mem_usage, or HF Inference API fallback).

Model selection policy for ``CohereForAI/aya-expanse-8b`` vs ``CohereForAI/aya-expanse-32b``:
  * **8B (`CohereForAI/aya-expanse-8b`)** is used for structured extraction, single-target splitting,
    domain classification, rephrasing, and directive leak/quality gating (fast, fits easily on a single
    16 GB Kaggle T4 in fp16 or 4-bit, and runs on CPU / ZeroGPU).
  * **32B (`CohereForAI/aya-expanse-32b`)** is recommended for complex multi-step adversarial flaw synthesis
    and hard physical derivation generation when running on dual-T4 / A100 GPUs (with 4-bit quantization)
    or Hugging Face ZeroGPU / Inference API; automatically falls back to 8B on ``local_cpu`` or single-GPU
    memory-constrained runtimes unless explicitly overridden.
"""
from __future__ import annotations

import functools
import os
from dataclasses import asdict, dataclass
from typing import Any, Callable, Literal

from adversarial_physics.utils.logging import get_logger

logger = get_logger("runtime_backend")

BackendName = Literal["auto", "kaggle_gpu", "hf_zerogpu", "local_cpu"]
AyaTaskName = Literal[
    "literature_extraction",
    "domain_classification",
    "curation_filtering",
    "rephrase_derivation",
    "adversarial_flaw_synthesis",
    "directive_generation",
]

AYA_EXPANSE_8B = "CohereForAI/aya-expanse-8b"
AYA_EXPANSE_32B = "CohereForAI/aya-expanse-32b"
TINY_AYA_GLOBAL = "CohereLabs/tiny-aya-global"

# Task -> (recommended_model, rationale, requires_heavy_reasoning)
TASK_MODEL_RECOMMENDATIONS: dict[str, tuple[str, str, bool]] = {
    "literature_extraction": (
        AYA_EXPANSE_8B,
        "8B is sufficient: problem statement + official solution are already paired deterministically; "
        "the model only splits multi-part questions and formats clean derivations.",
        False,
    ),
    "domain_classification": (
        AYA_EXPANSE_8B,
        "8B is sufficient: 6-way physics domain classification and step counting.",
        False,
    ),
    "curation_filtering": (
        AYA_EXPANSE_8B,
        "8B is sufficient: CuratorKIT quality gating, self-contained directive verification, and leak checks.",
        False,
    ),
    "rephrase_derivation": (
        AYA_EXPANSE_8B,
        "8B is sufficient: faithful step-by-step rephrasing of an existing ground-truth derivation.",
        False,
    ),
    "adversarial_flaw_synthesis": (
        AYA_EXPANSE_32B,
        "32B recommended: injecting subtle, realistic physical/algebraic flaws and writing paired "
        "self-contained physical directives without leaking the numerical answer benefits from 32B reasoning "
        "(falls back to 8B on local_cpu or single 16GB GPU without 4-bit).",
        True,
    ),
    "directive_generation": (
        AYA_EXPANSE_32B,
        "32B recommended: synthesizing precise, self-contained Verifier directives for challenged derivations.",
        True,
    ),
}


@dataclass(frozen=True)
class RuntimeConfig:
    """Resolved execution backend and device/quantization settings."""

    backend: Literal["kaggle_gpu", "hf_zerogpu", "local_cpu"]
    device: str
    torch_dtype: str
    load_in_4bit: bool
    device_map: str | None
    is_kaggle: bool
    is_zerogpu: bool
    cuda_available: bool
    vram_gb: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_runtime_backend(preferred: BackendName = "auto") -> RuntimeConfig:
    """Detect or configure the active runtime backend (`kaggle_gpu`, `hf_zerogpu`, or `local_cpu`)."""
    is_kaggle = bool(
        os.getenv("KAGGLE_KERNEL_RUN_TYPE")
        or os.getenv("KAGGLE_URL_BASE")
        or os.path.exists("/kaggle/working")
    )
    is_zerogpu = bool(
        os.getenv("SPACES_ZERO_GPU")
        or os.getenv("ZERO_GPU")
        or os.getenv("SPACE_ID")
    )

    cuda_available = False
    vram_gb = 0.0
    bf16_supported = False
    try:
        import torch  # type: ignore[import-not-found]

        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            props = torch.cuda.get_device_properties(0)
            vram_gb = round(props.total_memory / (1024**3), 2)
            bf16_supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    except Exception:
        pass

    if preferred == "auto":
        if is_zerogpu:
            resolved: Literal["kaggle_gpu", "hf_zerogpu", "local_cpu"] = "hf_zerogpu"
        elif is_kaggle or cuda_available:
            resolved = "kaggle_gpu"
        else:
            resolved = "local_cpu"
    else:
        resolved = preferred

    if resolved == "local_cpu" or (resolved == "kaggle_gpu" and not cuda_available and not is_zerogpu):
        return RuntimeConfig(
            backend="local_cpu",
            device="cpu",
            torch_dtype="float32",
            load_in_4bit=False,
            device_map="cpu",
            is_kaggle=is_kaggle,
            is_zerogpu=is_zerogpu,
            cuda_available=cuda_available,
            vram_gb=vram_gb,
        )

    if resolved == "hf_zerogpu":
        return RuntimeConfig(
            backend="hf_zerogpu",
            device="cuda",
            torch_dtype="bfloat16" if bf16_supported else "float16",
            load_in_4bit=False,
            device_map="auto",
            is_kaggle=is_kaggle,
            is_zerogpu=True,
            cuda_available=cuda_available,
            vram_gb=vram_gb,
        )

    # kaggle_gpu (or generic CUDA GPU)
    return RuntimeConfig(
        backend="kaggle_gpu",
        device="cuda",
        torch_dtype="bfloat16" if bf16_supported else "float16",
        load_in_4bit=vram_gb < 24.0,
        device_map="auto",
        is_kaggle=is_kaggle,
        is_zerogpu=is_zerogpu,
        cuda_available=cuda_available,
        vram_gb=vram_gb,
    )


def select_aya_expanse_model(
    task: AyaTaskName | str = "literature_extraction",
    *,
    runtime: RuntimeConfig | None = None,
    override_model: str | None = None,
    allow_32b_on_cpu: bool = False,
) -> tuple[str, str]:
    """Select between ``CohereForAI/aya-expanse-8b`` and ``CohereForAI/aya-expanse-32b`` for a task.

    Returns:
        ``(model_repo_id, reason)``
    """
    if override_model:
        return override_model, f"Explicitly configured model override: {override_model}"

    rt = runtime or detect_runtime_backend("auto")
    recommended, rationale, is_heavy = TASK_MODEL_RECOMMENDATIONS.get(
        task,
        (AYA_EXPANSE_8B, "Defaulting to aya-expanse-8b for general data processing.", False),
    )

    if recommended == AYA_EXPANSE_32B and rt.backend == "local_cpu" and not allow_32b_on_cpu:
        return (
            AYA_EXPANSE_8B,
            f"Task {task!r} prefers {AYA_EXPANSE_32B}, but runtime is 'local_cpu'; "
            f"using {AYA_EXPANSE_8B} so local CPU inference remains feasible.",
        )

    if recommended == AYA_EXPANSE_32B and rt.backend == "kaggle_gpu" and 0 < rt.vram_gb < 15.0:
        return (
            AYA_EXPANSE_8B,
            f"Task {task!r} prefers {AYA_EXPANSE_32B}, but detected GPU VRAM ({rt.vram_gb:.1f} GB) < 15 GB; "
            f"using {AYA_EXPANSE_8B}.",
        )

    return recommended, rationale


def zerogpu_decorator(duration: int = 120) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Wrap a function with ``@spaces.GPU(duration=...)`` when running on Hugging Face ZeroGPU.

    On ``kaggle_gpu`` or ``local_cpu`` where the ``spaces`` package is not installed, returns the
    function unchanged as a no-op decorator.
    """
    def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        try:
            import spaces  # type: ignore[import-not-found]

            return spaces.GPU(duration=duration)(fn)
        except ImportError:
            @functools.wraps(fn)
            def _passthrough(*args: Any, **kwargs: Any) -> Any:
                return fn(*args, **kwargs)

            return _passthrough

    return _decorator


def build_aya_chat_generator(
    *,
    task: AyaTaskName | str = "literature_extraction",
    model_name: str | None = None,
    backend: BackendName = "auto",
    temperature: float = 0.2,
    max_new_tokens: int = 1024,
    use_curatorkit_if_available: bool = True,
) -> Callable[[str, str], str]:
    """Create a callable ``generate(system_prompt, user_prompt) -> response_text`` powered by Aya-Expanse.

    Execution order:
      1. If CuratorKIT's generator adapter is installed and ``use_curatorkit_if_available=True``,
         routes through CuratorKIT with the selected ``CohereForAI/aya-expanse-*`` model.
      2. Otherwise loads ``CohereForAI/aya-expanse-8b`` or ``32b`` via Hugging Face ``transformers``
         using the resolved backend (`kaggle_gpu`, `hf_zerogpu` with ``@spaces.GPU``, or `local_cpu`).
    """
    rt = detect_runtime_backend(backend)
    resolved_model, reason = select_aya_expanse_model(task, runtime=rt, override_model=model_name)
    logger.info(
        "Configured Aya-Expanse generator: model=%s | backend=%s | device=%s | 4bit=%s (%s)",
        resolved_model,
        rt.backend,
        rt.device,
        rt.load_in_4bit,
        reason,
    )

    state: dict[str, Any] = {}

    def _load_pipeline() -> tuple[Any, Any]:
        if "model" in state and "tokenizer" in state:
            return state["tokenizer"], state["model"]

        import torch  # type: ignore[import-not-found]
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]

        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        tokenizer = AutoTokenizer.from_pretrained(resolved_model, token=token)

        dtype = getattr(torch, rt.torch_dtype, torch.float32)
        model_kwargs: dict[str, Any] = {
            "token": token,
            "low_cpu_mem_usage": True,
        }
        if rt.backend == "local_cpu":
            model_kwargs["torch_dtype"] = torch.float32
        else:
            model_kwargs["torch_dtype"] = dtype
            model_kwargs["device_map"] = rt.device_map or "auto"
            if rt.load_in_4bit or resolved_model == AYA_EXPANSE_32B:
                try:
                    from transformers import BitsAndBytesConfig  # type: ignore[import-not-found]

                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=dtype,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    )
                except ImportError:
                    logger.warning("bitsandbytes not installed; loading %s in %s.", resolved_model, rt.torch_dtype)

        model = AutoModelForCausalLM.from_pretrained(resolved_model, **model_kwargs)
        model.eval()
        state["tokenizer"] = tokenizer
        state["model"] = model
        return tokenizer, model

    @zerogpu_decorator(duration=120)
    def _generate_on_device(system_prompt: str, user_prompt: str) -> str:
        if use_curatorkit_if_available:
            try:
                import curatorkit  # type: ignore[import-not-found]

                if hasattr(curatorkit, "generate"):
                    return str(
                        curatorkit.generate(
                            model=f"huggingface/{resolved_model}",
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                            temperature=temperature,
                            max_tokens=max_new_tokens,
                        )
                    )
            except ImportError:
                pass

        import torch  # type: ignore[import-not-found]

        tokenizer, model = _load_pipeline()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt_text, return_tensors="pt")
        target_device = getattr(model, "device", rt.device)
        inputs = {k: v.to(target_device) for k, v in inputs.items()}

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0.0,
        }
        if temperature > 0.0:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)

        prompt_len = inputs["input_ids"].shape[-1]
        new_tokens = output_ids[0][prompt_len:]
        return str(tokenizer.decode(new_tokens, skip_special_tokens=True)).strip()

    return _generate_on_device
