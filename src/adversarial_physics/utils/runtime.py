"""Multi-backend runtime execution and Aya-Expanse (8B / 32B) model selector.

Supports three execution environments across all inference, synthesis, curation, and training pipelines:
  1. ``kaggle_gpu`` — Kaggle T4 / P100 / 2xT4 GPU environment (CUDA, fp16/bf16 or 4-bit bitsandbytes).
  2. ``hf_gpu``     — Hugging Face GPU execution:
       * When invoked from a machine without local CUDA (e.g. a local laptop with ``HF_TOKEN``),
         executes remotely on Hugging Face's GPU infrastructure via ``huggingface_hub.InferenceClient``
         (routing through ``https://router.huggingface.co`` Inference Providers or a Dedicated GPU
         Inference Endpoint ``HF_ENDPOINT_URL`` on ``l4x1`` / ``t4-small``, billed directly to the
         user's Hugging Face account).
       * When invoked inside an HF GPU container / Job / Space where CUDA is present, executes
         directly on the container's CUDA GPU (recommended hardware flavor: ``l4x1`` 24 GB VRAM
         at $0.80/hr or ``t4-small`` 16 GB VRAM at $0.50/hr).
  3. ``local_cpu``  — Local CPU execution (``torch.float32``).
"""
from __future__ import annotations

import functools
import importlib.util
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Literal

from curatorkit.llm.base import BaseLLM, LLMResponse
from dotenv import load_dotenv

from adversarial_physics.utils.logging import get_logger

logger = get_logger("runtime_backend")

BackendName = Literal["auto", "kaggle_gpu", "hf_gpu", "hf_zerogpu", "local_cpu"]
CanonicalBackendName = Literal["kaggle_gpu", "hf_gpu", "local_cpu"]
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

# Recommended HF Cloud GPU hardware flavors (cost vs VRAM efficiency for 8B / 32B):
# - l4x1 (1x NVIDIA L4 24GB VRAM, ~$0.80/hr): Best price/performance for aya-expanse-8b in native bf16
#   or aya-expanse-32b in 4-bit NF4.
# - t4-small (1x NVIDIA T4 16GB VRAM, ~$0.50/hr): Cheapest dedicated GPU for aya-expanse-8b in 4-bit/fp16.
RECOMMENDED_HF_GPU_FLAVOR = "l4x1"
CHEAPEST_HF_GPU_FLAVOR = "t4-small"

# Cohere model IDs when routed through Hugging Face Inference Providers (`https://router.huggingface.co`)
_HF_COHERE_PROVIDER_MODEL_MAP: dict[str, str] = {
    "CohereForAI/aya-expanse-8b": "c4ai-aya-expanse-8b",
    "CohereLabs/aya-expanse-8b": "c4ai-aya-expanse-8b",
    "CohereForAI/aya-expanse-32b": "c4ai-aya-expanse-32b",
    "CohereLabs/aya-expanse-32b": "c4ai-aya-expanse-32b",
}

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
        AYA_EXPANSE_8B,
        "8B unified teacher: given verified problem + ground-truth step-by-step reference solution, "
        "injects 1 controlled physical/algebraic error from the 5-category taxonomy.",
        False,
    ),
    "directive_generation": (
        AYA_EXPANSE_8B,
        "8B unified teacher: synthesizes concise, self-contained Verifier directives ([[CHALLENGE]] / [[APPROVE]]) "
        "by comparing the candidate trajectory against the ground-truth reference solution.",
        False,
    ),
}


def _normalize_backend_name(name: str) -> Literal["auto", "kaggle_gpu", "hf_gpu", "local_cpu"]:
    cleaned = (name or "auto").strip().lower()
    if cleaned in ("hf_zerogpu", "hf_gpu", "hf"):
        return "hf_gpu"
    if cleaned == "kaggle_gpu":
        return "kaggle_gpu"
    if cleaned == "local_cpu":
        return "local_cpu"
    return "auto"


def _is_accelerate_available() -> bool:
    return importlib.util.find_spec("accelerate") is not None


@dataclass(frozen=True)
class RuntimeConfig:
    """Resolved execution backend and device/quantization settings."""

    backend: CanonicalBackendName
    execution_mode: Literal["local_cuda", "hf_remote_api", "local_cpu"]
    device: str
    torch_dtype: str
    load_in_4bit: bool
    device_map: str | None
    is_kaggle: bool
    is_hf_gpu: bool
    cuda_available: bool
    accelerate_available: bool
    vram_gb: float
    recommended_hf_hardware: str = RECOMMENDED_HF_GPU_FLAVOR

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_runtime_backend(preferred: BackendName = "auto") -> RuntimeConfig:
    """Detect or configure the active runtime backend (`kaggle_gpu`, `hf_gpu`, or `local_cpu`)."""
    load_dotenv()
    norm_preferred = _normalize_backend_name(preferred)

    is_kaggle = bool(
        os.getenv("KAGGLE_KERNEL_RUN_TYPE")
        or os.getenv("KAGGLE_URL_BASE")
        or os.path.exists("/kaggle/working")
    )
    is_hf_env = bool(
        os.getenv("SPACES_ZERO_GPU")
        or os.getenv("ZERO_GPU")
        or os.getenv("SPACE_ID")
        or os.getenv("HF_ENDPOINT_URL")
        or os.getenv("HF_JOB_ID")
    )
    accel_ok = _is_accelerate_available()

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

    if norm_preferred == "auto":
        if is_hf_env:
            resolved: CanonicalBackendName = "hf_gpu"
        elif is_kaggle or cuda_available:
            resolved = "kaggle_gpu"
        else:
            resolved = "local_cpu"
    else:
        resolved = norm_preferred

    if resolved == "local_cpu" or (resolved == "kaggle_gpu" and not cuda_available):
        return RuntimeConfig(
            backend="local_cpu",
            execution_mode="local_cpu",
            device="cpu",
            torch_dtype="float32",
            load_in_4bit=False,
            device_map="cpu" if accel_ok else None,
            is_kaggle=is_kaggle,
            is_hf_gpu=False,
            cuda_available=cuda_available,
            accelerate_available=accel_ok,
            vram_gb=vram_gb,
        )

    if resolved == "hf_gpu":
        # If running on a machine without a local CUDA GPU (e.g. user's laptop), use remote HF GPU API
        force_remote = os.getenv("HF_GPU_MODE", "").strip().lower() == "remote" or bool(os.getenv("HF_ENDPOINT_URL"))
        use_remote_api = force_remote or (not cuda_available and not bool(os.getenv("SPACES_ZERO_GPU")))
        return RuntimeConfig(
            backend="hf_gpu",
            execution_mode="hf_remote_api" if use_remote_api else "local_cuda",
            device="cuda" if not use_remote_api else "hf_remote_gpu",
            torch_dtype="bfloat16" if bf16_supported else "float16",
            load_in_4bit=False,
            device_map="auto" if (accel_ok and not use_remote_api) else None,
            is_kaggle=is_kaggle,
            is_hf_gpu=True,
            cuda_available=cuda_available,
            accelerate_available=accel_ok,
            vram_gb=vram_gb,
            recommended_hf_hardware=RECOMMENDED_HF_GPU_FLAVOR,
        )

    # kaggle_gpu (or generic local CUDA GPU)
    return RuntimeConfig(
        backend="kaggle_gpu",
        execution_mode="local_cuda",
        device="cuda",
        torch_dtype="bfloat16" if bf16_supported else "float16",
        load_in_4bit=vram_gb < 24.0,
        device_map="auto" if accel_ok else None,
        is_kaggle=is_kaggle,
        is_hf_gpu=False,
        cuda_available=cuda_available,
        accelerate_available=accel_ok,
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
    recommended, rationale, _ = TASK_MODEL_RECOMMENDATIONS.get(
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


def hf_gpu_decorator(duration: int = 120) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Wrap a function with ``@spaces.GPU(duration=...)`` when running inside a Hugging Face Space.

    Outside HF Spaces where ``spaces`` is not installed, returns the function unchanged.
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


# Keep alias for backwards compatibility
zerogpu_decorator = hf_gpu_decorator


def _register_hf_cohere_mappings() -> None:
    """Ensure CohereForAI/aya-expanse-8b and 32b are registered in huggingface_hub's provider table."""
    from huggingface_hub.hf_api import InferenceProviderMapping
    from huggingface_hub.inference._providers._common import HARDCODED_MODEL_INFERENCE_MAPPING

    cohere_map = HARDCODED_MODEL_INFERENCE_MAPPING.setdefault("cohere", {})
    for hf_model_id, cohere_id in _HF_COHERE_PROVIDER_MODEL_MAP.items():
        if hf_model_id not in cohere_map:
            cohere_map[hf_model_id] = InferenceProviderMapping(
                provider="cohere",
                hf_model_id=hf_model_id,
                providerId=cohere_id,
                status="live",
                task="conversational",
            )


class AyaExpanseCuratorLLM(BaseLLM):
    """CuratorKIT ``BaseLLM`` implementation backed by ``CohereForAI/aya-expanse-8b`` or ``32b``.

    Integrates directly with CuratorKIT's ``RewardGate``, ``HallucinationGate``,
    ``RewardRefiner``, ``DiagnosticProbe``, and generation tasks across ``kaggle_gpu``,
    ``hf_gpu``, and ``local_cpu`` backends, while tracking per-call and cumulative timing.
    """

    def __init__(
        self,
        *,
        task: AyaTaskName | str = "literature_extraction",
        model_name: str | None = None,
        backend: BackendName = "auto",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        max_retries: int = 1,
        endpoint_url: str | None = None,
    ) -> None:
        self.runtime = detect_runtime_backend(backend)
        resolved_model, reason = select_aya_expanse_model(
            task, runtime=self.runtime, override_model=model_name
        )
        super().__init__(
            model=resolved_model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )
        self.selection_reason = reason
        self._tokenizer: Any = None
        self._hf_model: Any = None
        self._hf_client: Any = None
        self._resolved_endpoint_url: str | None = (endpoint_url or "").strip() or None
        self._resolved_endpoint_model_id: str | None = None
        self.call_count: int = 0
        self.total_latency_seconds: float = 0.0
        logger.info(
            "Configured AyaExpanseCuratorLLM: model=%s | backend=%s | mode=%s | device=%s | accelerate=%s | 4bit=%s (%s)",
            self.model,
            self.runtime.backend,
            self.runtime.execution_mode,
            self.runtime.device,
            self.runtime.accelerate_available,
            self.runtime.load_in_4bit,
            self.selection_reason,
        )

    def get_timing_stats(self) -> dict[str, float | int]:
        """Return cumulative call count, total wall time, and mean latency per call."""
        avg = (self.total_latency_seconds / self.call_count) if self.call_count > 0 else 0.0
        return {
            "llm_call_count": self.call_count,
            "llm_total_seconds": round(self.total_latency_seconds, 3),
            "llm_avg_seconds_per_call": round(avg, 3),
        }

    def _resolve_hf_endpoint_url(self, token: str) -> str:
        """Resolve (or auto-discover / auto-resume / auto-create) the HF GPU Inference Endpoint URL for `self.model`."""
        if self._resolved_endpoint_url:
            return self._resolved_endpoint_url

        lower_model = self.model.lower()
        if "tiny-aya" in lower_model:
            specific_env = "HF_ENDPOINT_URL_TINY_AYA"
        elif "32b" in lower_model:
            specific_env = "HF_ENDPOINT_URL_32B"
        else:
            specific_env = "HF_ENDPOINT_URL_8B"

        specific_val = (os.getenv(specific_env) or "").strip()
        if specific_val:
            self._resolved_endpoint_url = specific_val
            return specific_val

        # Auto-discover an existing Hugging Face Inference Endpoint matching `self.model`
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        try:
            endpoints = api.list_inference_endpoints()
            for ep in endpoints:
                if ep.repository.lower() == lower_model:
                    logger.info(
                        "Found existing HF GPU Inference Endpoint %r for %s (status=%s)",
                        ep.name,
                        self.model,
                        ep.status,
                    )
                    if str(ep.status) in ("paused", "scaledToZero"):
                        logger.info("Resuming HF GPU Endpoint %r and waiting for it to become ready...", ep.name)
                        ep.resume()
                    if str(ep.status) != "running" or not ep.url:
                        ep.wait(timeout=600)
                    if ep.url:
                        self._resolved_endpoint_url = ep.url
                        return ep.url
        except Exception as exc:
            logger.warning("Could not list existing HF Inference Endpoints: %s", exc)

        generic_url = (os.getenv("HF_ENDPOINT_URL") or "").strip()
        if generic_url:
            self._resolved_endpoint_url = generic_url
            return generic_url

        # Optionally auto-create the endpoint if HF_AUTO_CREATE_ENDPOINT=1
        if os.getenv("HF_AUTO_CREATE_ENDPOINT", "").strip() in ("1", "true", "yes"):
            ep_name = "aya-expanse-32b-gpu" if is_32b else "aya-expanse-8b-gpu"
            instance_size = "x4" if is_32b else "x1"
            logger.info(
                "Auto-creating HF GPU Inference Endpoint %r for %s on aws/us-east-1/nvidia-l4 (%s, scale-to-zero=15m)...",
                ep_name,
                self.model,
                instance_size,
            )
            ep = api.create_inference_endpoint(
                ep_name,
                repository=self.model,
                framework="pytorch",
                task="text-generation",
                accelerator="gpu",
                vendor="aws",
                region="us-east-1",
                type="protected",
                instance_size=instance_size,
                instance_type="nvidia-l4",
                min_replica=0,
                max_replica=1,
                scale_to_zero_timeout=15,
            )
            ep.wait(timeout=900)
            if ep.url:
                self._resolved_endpoint_url = ep.url
                return ep.url

        raise RuntimeError(
            f"No active HF GPU Inference Endpoint found for {self.model} (and Cohere's serverless API retired Aya-Expanse).\n"
            f"Either:\n"
            f"  1. Set {specific_env}=https://<your-endpoint>.endpoints.huggingface.cloud in .env, OR\n"
            f"  2. Create a Dedicated Endpoint for {self.model} at https://ui.endpoints.huggingface.co/ "
            f"(it will be auto-discovered by repository name), OR\n"
            f"  3. Set HF_AUTO_CREATE_ENDPOINT=1 in .env to let the script create it automatically."
        )

    def _call_hf_remote_gpu(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> LLMResponse:
        """Execute chat completion remotely on a Hugging Face Dedicated GPU Inference Endpoint (`nvidia-l4`)."""
        from huggingface_hub import InferenceClient
        from huggingface_hub.utils import get_token

        load_dotenv()
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or get_token()
        if not token:
            raise RuntimeError(
                "HF_TOKEN is required for --backend hf_gpu when running without a local CUDA GPU. "
                "Add HF_TOKEN=hf_... to your .env file or run `hf auth login`."
            )

        endpoint_url = self._resolve_hf_endpoint_url(token)
        if not self._resolved_endpoint_model_id:
            # Query /v1/models on the vLLM endpoint to get the exact served model ID (avoids huggingface_hub sending 'dummy')
            served_id = self.model
            try:
                from huggingface_hub.utils import get_session

                models_url = f"{endpoint_url.rstrip('/')}/v1/models"
                resp = get_session().get(models_url, headers={"Authorization": f"Bearer {token}"}, timeout=15)
                if resp.status_code == 200:
                    data = resp.json().get("data") or []
                    if data and isinstance(data[0], dict) and data[0].get("id"):
                        served_id = str(data[0]["id"])
                        logger.info("Detected vLLM served model ID on HF GPU Endpoint: %s", served_id)
            except Exception as exc:
                logger.debug("Could not query /v1/models (%s); defaulting payload model to %s", exc, served_id)
            self._resolved_endpoint_model_id = served_id

        bill_to = (os.getenv("HF_BILL_TO") or "").strip() or None
        temp = float(kwargs.get("temperature", self.temperature))
        max_new = int(kwargs.get("max_tokens", self.max_tokens))

        t0 = time.perf_counter()
        client = InferenceClient(
            model=endpoint_url,
            token=token,
            bill_to=bill_to,
            timeout=self.timeout,
        )
        response = client.chat_completion(
            model=self._resolved_endpoint_model_id,
            messages=messages,
            temperature=max(temp, 1e-3) if temp > 0 else 0.0,
            max_tokens=max_new,
        )

        elapsed = time.perf_counter() - t0
        choice = response.choices[0]
        content = getattr(choice.message, "content", "") or ""
        usage = getattr(response, "usage", None)
        prompt_toks = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        comp_toks = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
        total_toks = int(getattr(usage, "total_tokens", prompt_toks + comp_toks) or 0) if usage else 0

        self.call_count += 1
        self.total_latency_seconds += elapsed
        return LLMResponse(
            text=str(content).strip(),
            model=self.model,
            prompt_tokens=prompt_toks,
            completion_tokens=comp_toks,
            total_tokens=total_toks,
            latency_seconds=elapsed,
            metadata={"backend": "hf_gpu", "execution_mode": "hf_remote_api", "endpoint_url": endpoint_url},
        )

    def _load_pipeline(self) -> tuple[Any, Any]:
        if self._tokenizer is not None and self._hf_model is not None:
            return self._tokenizer, self._hf_model

        import torch  # type: ignore[import-not-found]
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]

        load_dotenv()
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        t0 = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(self.model, token=token)

        dtype = getattr(torch, self.runtime.torch_dtype, torch.float32)
        accel_ok = _is_accelerate_available()

        model_kwargs: dict[str, Any] = {
            "token": token,
        }
        if accel_ok:
            model_kwargs["low_cpu_mem_usage"] = True

        if self.runtime.backend == "local_cpu" or not self.runtime.cuda_available:
            model_kwargs["torch_dtype"] = torch.float32
        else:
            model_kwargs["torch_dtype"] = dtype
            if accel_ok and self.runtime.device_map:
                model_kwargs["device_map"] = self.runtime.device_map
            if self.runtime.load_in_4bit or self.model == AYA_EXPANSE_32B:
                try:
                    from transformers import BitsAndBytesConfig  # type: ignore[import-not-found]

                    if accel_ok:
                        model_kwargs["quantization_config"] = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=dtype,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_use_double_quant=True,
                        )
                except ImportError:
                    logger.warning(
                        "bitsandbytes not installed; loading %s in %s.",
                        self.model,
                        self.runtime.torch_dtype,
                    )

        hf_model = AutoModelForCausalLM.from_pretrained(self.model, **model_kwargs)
        if not (accel_ok and model_kwargs.get("device_map")) and self.runtime.cuda_available:
            hf_model = hf_model.to("cuda")
        hf_model.eval()
        load_sec = time.perf_counter() - t0
        logger.info("Loaded %s on %s in %.2fs", self.model, self.runtime.device, load_sec)
        self._tokenizer = tokenizer
        self._hf_model = hf_model
        return tokenizer, hf_model

    def _call(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> LLMResponse:
        if self.runtime.execution_mode == "hf_remote_api":
            return self._call_hf_remote_gpu(messages, **kwargs)

        @hf_gpu_decorator(duration=120)
        def _run_generation() -> LLMResponse:
            import torch  # type: ignore[import-not-found]

            t0 = time.perf_counter()
            tokenizer, hf_model = self._load_pipeline()
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(prompt_text, return_tensors="pt")
            target_device = getattr(hf_model, "device", "cuda" if self.runtime.cuda_available else "cpu")
            inputs = {k: v.to(target_device) for k, v in inputs.items()}

            temp = float(kwargs.get("temperature", self.temperature))
            max_new = int(kwargs.get("max_tokens", self.max_tokens))

            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new,
                "do_sample": temp > 0.0,
            }
            if temp > 0.0:
                gen_kwargs["temperature"] = temp

            with torch.no_grad():
                output_ids = hf_model.generate(**inputs, **gen_kwargs)

            elapsed = time.perf_counter() - t0
            prompt_len = int(inputs["input_ids"].shape[-1])
            new_tokens = output_ids[0][prompt_len:]
            completion_len = int(new_tokens.shape[-1])
            text = str(tokenizer.decode(new_tokens, skip_special_tokens=True)).strip()
            self.call_count += 1
            self.total_latency_seconds += elapsed
            return LLMResponse(
                text=text,
                model=self.model,
                prompt_tokens=prompt_len,
                completion_tokens=completion_len,
                total_tokens=prompt_len + completion_len,
                latency_seconds=elapsed,
                metadata={"backend": self.runtime.backend, "execution_mode": self.runtime.execution_mode},
            )

        return _run_generation()


def build_aya_chat_generator(
    *,
    task: AyaTaskName | str = "literature_extraction",
    model_name: str | None = None,
    backend: BackendName = "auto",
    temperature: float = 0.2,
    max_new_tokens: int = 1024,
    endpoint_url: str | None = None,
) -> Callable[[str, str], str]:
    """Create a callable ``generate(system_prompt, user_prompt) -> response_text`` backed by ``AyaExpanseCuratorLLM``."""
    llm = AyaExpanseCuratorLLM(
        task=task,
        model_name=model_name,
        backend=backend,
        temperature=temperature,
        max_tokens=max_new_tokens,
        endpoint_url=endpoint_url,
    )

    def _generate(system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        resp = llm.generate(messages, temperature=temperature, max_tokens=max_new_tokens)
        return resp.text

    return _generate
