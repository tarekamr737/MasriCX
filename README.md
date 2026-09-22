# MasriCX

[GitHub repository](https://github.com/tarekamr737/MasriCX)

Egyptian Arabic–English **code-switched ASR**: parameter-efficient fine-tuning of
`openai/whisper-large-v3-turbo` with LoRA/PEFT, evaluated on clean, code-switched,
telephone-degraded, and external Egyptian speech.

> **Status: local pipeline implemented and offline-verified.** The canonical audit and
> fixed splits are complete. GPU baselines, training/evaluation, and publication
> have not run; every performance value remains `TBD`.

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

All numbers must be generated from saved prediction artifacts. No GPU prediction artifact exists yet.

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

## Data audit (Phase 1)

Metadata-only smoke audit (registry check, no dataset download, no audio):

```bash
python -m masricx.data.audit --help                 # CLI usage
python -m masricx.config validate configs/pilot.yaml configs/codeswitch.yaml configs/telephone.yaml configs/egyspeak_ablation.yaml
```

Full audit (downloads the pinned dataset and decodes audio by default; requires
`datasets` + `soundfile`/`numpy` locally, never installed during CI):

```bash
python -m masricx.data.audit --config configs/codeswitch.yaml
python -m masricx.data.audit --config configs/codeswitch.yaml --max-examples 500 --streaming
python -m masricx.data.audit --config configs/codeswitch.yaml --no-audio --max-examples 1000
```

Flags: `--max-examples N` (cap examples), `--streaming` (stream rows),
`--no-audio` (skip audio decoding; audio metrics are reported as *unavailable*,
not zero), `--output-json` / `--output-md` (report paths). Outputs:
`artifacts/data_audit.json` (git-ignored) and `reports/DATA_AUDIT.md`.

The full audit requires `datasets` and `soundfile`/`numpy` in the local
environment; they are never downloaded during CI.

The completed canonical audit covers all 45,189 rows and 34.1128 hours at the
pinned revision. Its privacy-safe measured report is `reports/DATA_AUDIT.md`;
the canonical generated JSON remains git-ignored under `artifacts/`.

## Deduplication and fixed splits (Phase 2)

Regenerate the privacy-safe manifests from the pinned primary dataset:

```bash
python -m masricx.data.split --config configs/codeswitch.yaml --audit-json artifacts/data_audit.json --output-dir data/splits
```

The manifests contain stable IDs, exact transcript/audio hashes, and language
categories only—never transcript text or audio. Exact duplicates are removed
before deterministic seed-42 90/5/5 assignment. The 20 audit-reported
near-duplicate pairs are co-located, but are not exhaustive. No reliable
speaker/session/source ID exists, so the splits are explicitly not claimed to
be speaker-disjoint. Counts and leakage results are in `data/splits/metadata.json`.

## Reproduction

All commands are dry-run or local artifact operations unless `--execute` is shown.

```bash
# Verify configs, tests, typing, and formatting
make check

# Rebuild fixed privacy-safe split manifests
make splits

# Inspect the exact E2 plan without loading a model
make training-plan

# On an authorized CUDA/Kaggle runtime: run the 5-hour pilot, then E1/E2
python -m masricx.training.pilot --config configs/pilot.yaml --run P1 --output-dir .runtime/runs/pilot-p1 --split-dir data/splits --execute
python -m masricx.training.train --config configs/codeswitch.yaml --output-dir .runtime/runs/e1 --split-dir data/splits --execute
python -m masricx.training.train --config configs/telephone.yaml --output-dir .runtime/runs/e2 --split-dir data/splits --execute

# Generate resumable internal predictions from a pinned baseline
python -m masricx.evaluation.run_inference --model openai/whisper-large-v3-turbo --revision 41f01f3fe87f28c78e2fbf8b568835947dd65ed9 --output .runtime/predictions/base-clean.jsonl --execute

# Aggregate a saved prediction artifact
python -m masricx.evaluation.benchmark --predictions .runtime/predictions/base-clean.jsonl --output-json artifacts/base-clean.json --output-md reports/BASE_CLEAN.md

# Local single-file inference after a reviewed model release
python -m masricx.inference.transcribe --audio sample.wav --model tarekamr737/MasriCX-ASR
```

The Kaggle notebook is a thin clone/install/launch shell. Set its public
`REPO_URL`, add `HF_TOKEN` through Kaggle Secrets if checkpoint upload is
authorized, select a GPU accelerator, and run `kaggle/setup.ipynb`. It does not
embed credentials or research logic.

GPU training and benchmark generation are intentionally not performed by CI.
Publication remains blocked until measured artifacts pass the promotion gate and
the primary dataset source-chain license review is resolved.

## License and data

- Repository code: Apache-2.0 (see `LICENSE`). This grants no rights to datasets
  or model weights; each dataset/model keeps its own license.
- Dataset revisions, licenses, and redistribution terms: recorded in
  `configs/data_sources.yaml` (verified 2026-09-20) and summarized in
  `reports/DATA_AUDIT.md`. The primary dataset's aggregate is NOT uniformly MIT
  (a ~12,480-clip subset derives from a GPL-tagged source); final model
  publication remains blocked pending source-chain review.
- EGYSpeak, if used, is pseudo-labelled machine-generated data and is never
  treated as gold ground truth; E3 is disabled until the license-chain conflict
  (card CC-BY-4.0 vs upstream Kaggle GPL-3.0) is resolved; the loader fails
  closed on audio/full use.
- The external Casablanca Egypt test set is evaluation-only (CC-BY-NC-ND-4.0).
