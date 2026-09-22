"""Revision-pinned inference adapters for Transformers Whisper and NeMo ASR."""

from __future__ import annotations

import importlib
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

EGYPTALK_ID = "NAMAA-Space/EgypTalk-ASR-v2"
EGYPTALK_FILE = "asr-egyptian-nemo-v2.0.nemo"


@dataclass
class BaselineRuntime:
    kind: str
    model: Any
    processor: Any
    torch: Any
    device: str
    dtype: Any
    language: str | None = None

    def transcribe(
        self, samples: np.ndarray, sample_rate: int, temporary_dir: Path
    ) -> tuple[str, float]:
        import time

        if self.kind == "transformers":
            inputs = self.processor(samples, sampling_rate=sample_rate, return_tensors="pt")
            features = inputs.input_features.to(device=self.device, dtype=self.dtype)
            started = time.perf_counter()
            with self.torch.inference_mode():
                generate_kwargs = {"task": "transcribe"}
                if self.language is not None:
                    generate_kwargs["language"] = self.language
                predicted = self.model.generate(features, **generate_kwargs)
            if self.device.startswith("cuda"):
                self.torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            text = self.processor.batch_decode(predicted, skip_special_tokens=True)[0].strip()
            return text, elapsed
        temporary_dir.mkdir(parents=True, exist_ok=True)
        soundfile = importlib.import_module("soundfile")
        with tempfile.NamedTemporaryFile(suffix=".wav", dir=temporary_dir, delete=False) as handle:
            path = Path(handle.name)
        try:
            soundfile.write(path, samples, sample_rate)
            started = time.perf_counter()
            result = self.model.transcribe([str(path)])
            if self.device.startswith("cuda"):
                self.torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            first = result[0]
            text = first if isinstance(first, str) else str(getattr(first, "text", first))
            return text.strip(), elapsed
        finally:
            path.unlink(missing_ok=True)


def load_runtime(
    model_id: str,
    revision: str,
    *,
    adapter: str | None = None,
    allow_cpu: bool = False,
) -> BaselineRuntime:  # pragma: no cover - external model integration
    torch = importlib.import_module("torch")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and not allow_cpu:
        raise SystemExit("batch inference requires CUDA unless --allow-cpu is explicit")
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    if model_id == EGYPTALK_ID:
        if adapter is not None:
            raise ValueError("PEFT adapters are not supported for the NeMo baseline")
        hub = importlib.import_module("huggingface_hub")
        nemo_models = importlib.import_module("nemo.collections.asr.models")
        archive = hub.hf_hub_download(repo_id=model_id, filename=EGYPTALK_FILE, revision=revision)
        model = nemo_models.ASRModel.restore_from(archive)
        model = model.to(device)
        model.eval()
        return BaselineRuntime("nemo", model, None, torch, device, dtype)
    transformers = importlib.import_module("transformers")
    processor = transformers.AutoProcessor.from_pretrained(model_id, revision=revision)
    model = transformers.AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, revision=revision, torch_dtype=dtype, low_cpu_mem_usage=True
    )
    if adapter:
        peft = importlib.import_module("peft")
        model = peft.PeftModel.from_pretrained(model, adapter)
    model.to(device)
    model.eval()
    language = "ar" if model_id == "Seif-Eldeen-Sameh/whisper-medium-arabic-codeswitched" else None
    return BaselineRuntime("transformers", model, processor, torch, device, dtype, language)
