# MasriCX

Egyptian Arabic–English **code-switched ASR**: parameter-efficient fine-tuning of
`openai/whisper-large-v3-turbo` with LoRA/PEFT, evaluated on clean, code-switched,
telephone-degraded, and external Egyptian speech.

> **Status: Phase 0 (repository bootstrap).** All benchmark values are `TBD`;
> no training, evaluation, or data audit has been run yet.

## Why this is hard

Egyptian contact-center speech mixes Arabic dialect with English technical terms
(`"اعمل restart للـ router"`). Generic multilingual ASR often transliterates,
drops, or mistranscribes the English tokens; telephone-band audio makes it worse.
MasriCX targets that gap directly. (See `MasriCX_AGENT_SPEC_DELEGATED.md`.)

## Experiment matrix

| Experiment | Definition |
|---|---|
| E0 | Baselines only (no training) |
| E1 | Core LoRA fine-tuning on the code-switched dataset |
| E2 | E1 + telephone augmentation |
| E3 | Best config + EGYSpeak pseudo-labelled ablation (0%/10%/25%) |

## Results

| Model | Clean WER ↓ | CS-WER ↓ | Telephone WER ↓ | EN Recall ↑ | Casablanca WER ↓ |
|---|---:|---:|---:|---:|---:|
| Whisper Large V3 Turbo | TBD | TBD | TBD | TBD | TBD |
| EgypTalk-ASR-v2 | TBD | TBD | TBD | TBD | TBD |
| Whisper Medium Arabic Code-Switched | TBD | TBD | TBD | TBD | TBD |
| **MasriCX-ASR** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |

All numbers are generated from saved evaluation artifacts. None exist yet.

## Repository layout

```text
MasriCX/
├── configs/          # YAML experiment configs (validated in CI, no model access)
├── src/masricx/      # core package: data, audio, training, evaluation, inference
├── tests/            # unit tests (CPU-only, deterministic, offline)
├── scripts/          # thin shell entry-point wrappers
├── notebooks/        # local analysis notebooks (thin shells)
├── kaggle/           # thin Kaggle notebook shells (clone/install/launch only)
├── reports/          # audit/benchmark/error-analysis/model-card templates
└── artifacts/        # generated run artifacts (git-ignored)
```

## Setup

```bash
python -m venv .runtime/venv
.runtime/venv/Scripts/activate        # Windows (Linux: source .runtime/venv/bin/activate)
pip install -r requirements.txt
pip install -e . --no-deps            # editable install of the masricx package
```

## Tooling gates

```bash
make lint        # ruff check + format check
make typecheck   # mypy src
make test        # pytest
make validate    # config validation CLI (offline, no model/dataset access)
make check       # all of the above
```

## Config validation (offline)

```bash
python -m masricx.config validate \
  configs/pilot.yaml configs/codeswitch.yaml configs/telephone.yaml configs/egyspeak_ablation.yaml
```

## Reproduction (documented commands; results not yet produced)

```bash
python -m masricx.data.audit --config configs/codeswitch.yaml
python -m masricx.evaluation.benchmark --config configs/codeswitch.yaml
python -m masricx.training.train --config configs/telephone.yaml
python -m masricx.inference.transcribe --audio sample.wav --model tarekamr737/MasriCX-ASR
```

The training/evaluation modules listed above are the planned entry points; Phase 0
ships the scaffold, thin script wrappers, and config validation only.

## License and data

- Repository code: Apache-2.0 (see `LICENSE`). This grants no rights to datasets
  or model weights; each dataset/model keeps its own license.
- Dataset revisions, licenses, and redistribution terms: `TBD` until the
  orchestrator's data audit (see `reports/DATA_AUDIT.md`).
- EGYSpeak, if used, is pseudo-labelled machine-generated data and is never
  treated as gold ground truth.
- The external Casablanca Egypt test set is evaluation-only.
