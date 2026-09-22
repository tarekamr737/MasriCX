"""Local Whisper/PEFT transcription CLI with timing metadata."""

from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["build_parser", "main", "transcribe_audio"]


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    try:
        soundfile = importlib.import_module("soundfile")
    except ImportError as exc:
        raise RuntimeError("soundfile is required for inference") from exc
    audio, sample_rate = soundfile.read(path, dtype="float32", always_2d=False)
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 2:
        array = array.mean(axis=1)
    if array.ndim != 1:
        raise ValueError("audio must decode to mono or stereo waveform")
    if int(sample_rate) != 16000:
        librosa = importlib.import_module("librosa")
        array = np.asarray(
            librosa.resample(array, orig_sr=int(sample_rate), target_sr=16000),
            dtype=np.float32,
        )
        sample_rate = 16000
    return array, int(sample_rate)


def transcribe_audio(
    audio_path: Path,
    model_id: str,
    *,
    adapter: str | None = None,
    language: str | None = None,
    revision: str | None = None,
) -> dict[str, Any]:  # pragma: no cover - model integration
    try:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
    except ImportError as exc:
        raise RuntimeError("torch and transformers are required for inference") from exc
    audio, sample_rate = _load_audio(audio_path)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    processor = transformers.WhisperProcessor.from_pretrained(model_id, revision=revision)
    model = transformers.WhisperForConditionalGeneration.from_pretrained(
        model_id, revision=revision, torch_dtype=dtype
    )
    if adapter:
        try:
            peft = importlib.import_module("peft")
        except ImportError as exc:
            raise RuntimeError("PEFT is required when --adapter is used") from exc
        model = peft.PeftModel.from_pretrained(model, adapter)
    model.to(device)
    inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
    input_features = inputs.input_features.to(device=device, dtype=dtype)
    generation: dict[str, object] = {}
    if language:
        generation["language"] = language
    started = time.perf_counter()
    with torch.inference_mode():
        predicted = model.generate(input_features, **generation)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    text = processor.batch_decode(predicted, skip_special_tokens=True)[0].strip()
    duration = len(audio) / sample_rate
    return {
        "text": text,
        "audio_seconds": duration,
        "processing_seconds": elapsed,
        "rtf": elapsed / duration if duration else None,
        "model": model_id,
        "adapter": adapter,
        "revision": revision,
        "device": device,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transcribe one audio file with MasriCX")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--model", default="tarekamr737/MasriCX-ASR")
    parser.add_argument("--adapter")
    parser.add_argument("--revision")
    parser.add_argument("--language")
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = transcribe_audio(
        args.audio,
        args.model,
        adapter=args.adapter,
        language=args.language,
        revision=args.revision,
    )
    print(
        json.dumps(result, ensure_ascii=False, sort_keys=True) if args.as_json else result["text"]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
