# Benchmark Plan and Results

> **Status:** evaluation framework implemented and offline-tested; GPU inference has not
> run. Every performance value remains `TBD` until generated from saved JSONL
> prediction artifacts.

## Environment

- Training/inference environment: TBD.
- Python and package versions: TBD.
- Default compute budget: $0.

## Artifact provenance

Model and dataset IDs, immutable revisions, roles, and licensing caveats are
machine-readable in `configs/baselines.yaml` and `configs/data_sources.yaml`.
Fixed split definitions and hashes are stored in `data/splits/metadata.json`.

## Evaluation rules

Raw WER/CER preserve the reference text as supplied. Normalized WER/CER apply
the deterministic rules implemented in `masricx.evaluation.normalize`.
Code-switched WER is reported on mixed Arabic/Latin references; Arabic-token and
English-token WER use script-filtered token streams. English term recall and
number accuracy return `null` when a reference has no eligible items.
Hallucination rate is automated screening followed by mandatory manual review.
RTF is processing seconds divided by audio seconds.

The clean and deterministic telephone test use identical sample IDs and
transcripts. Telephone robustness delta is `telephone WER - clean WER`.

## Results

| Model / experiment | Clean WER ↓ | CS-WER ↓ | Telephone WER ↓ | EN recall ↑ | Casablanca WER ↓ | RTF ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Whisper Large V3 Turbo (E0) | TBD | TBD | TBD | TBD | TBD | TBD |
| EgypTalk-ASR-v2 (E0) | TBD | TBD | TBD | TBD | TBD | TBD |
| Whisper Medium Arabic Code-Switched (E0) | TBD | TBD | TBD | TBD | TBD | TBD |
| MasriCX E1 | TBD | TBD | TBD | TBD | TBD | TBD |
| MasriCX E2 | TBD | TBD | TBD | TBD | TBD | TBD |
| MasriCX E3 (optional) | TBD | TBD | TBD | TBD | TBD | TBD |

## Required analyses after GPU runs

- Raw and normalized WER/CER, language-token metrics, term/number preservation.
- Clean versus fixed telephone degradation on identical test examples.
- Duration, language-category, clipping, silence, and quality buckets.
- Paired uncertainty/significance caveats; no superiority claim from one metric.
- Casablanca Egypt results labeled external and evaluation-only.
- Privacy-safe manual review of all automated hallucination flags and sampled
  error-analysis candidates.

## Known limitations

Training speech is not private call-center audio, telephone conditions are
simulated, near-duplicate discovery is not exhaustive, and splits are not
speaker-disjoint. E3 remains blocked pending its separate license-provenance
review. Final model publication remains blocked until measured results satisfy
the promotion gate and the primary-source model-weight licensing decision is resolved.
