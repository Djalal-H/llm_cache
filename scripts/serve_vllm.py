"""Run with .venv-vllm/bin/python; resolve and record the actual model/template revision."""

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("VLLM_CACHE_ROOT", str(ROOT / ".cache/vllm"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache/serving"))
MODEL = "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4"


def main():
    import torch
    from huggingface_hub import HfApi, hf_hub_download

    if importlib.metadata.version("vllm") != "0.29.0":
        raise RuntimeError("Expected the pinned vllm==0.29.0 serving environment")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no CPU or model fallback will be used")
    revision = HfApi().model_info(MODEL).sha
    tokenizer = Path(hf_hub_download(MODEL, "tokenizer_config.json", revision=revision))
    template = json.loads(tokenizer.read_text())["chat_template"]
    gpu = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    command = [
        str(ROOT / ".venv-vllm/bin/vllm"),
        "serve",
        MODEL,
        "--revision",
        revision,
        "--tokenizer-revision",
        revision,
        "--served-model-name",
        "cachewise-model",
        "--host",
        "127.0.0.1",
        "--port",
        "8001",
        "--quantization",
        "gptq",
        "--dtype",
        "half",
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "1",
        "--max-num-batched-tokens",
        "2048",
        "--gpu-memory-utilization",
        "0.85",
        "--enforce-eager",
        "--enable-prefix-caching",
        "--generation-config",
        "vllm",
        "--enable-prompt-tokens-details",
    ]
    metadata = {
        "recorded_at": datetime.now(UTC).isoformat(),
        "model": MODEL,
        "model_revision": revision,
        "tokenizer_revision": revision,
        "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "vllm_version": "0.29.0",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "hardware": gpu,
        "weight_quantization": "GPTQ Int4",
        "compute_dtype": "float16",
        "context_length": 2048,
        "max_num_seqs": 1,
        "gpu_memory_utilization": 0.85,
        "prefix_caching": True,
        "served_model_name": "cachewise-model",
        "command": command,
        "status": "startup_requested_not_readiness_evidence",
    }
    output = ROOT / "artifacts/serving.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)
    os.execv(command[0], command)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"vLLM startup failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
