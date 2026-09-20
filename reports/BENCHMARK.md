# Benchmark Report

> **Status: template (Phase 0).** No evaluation has been run. Every number is
> `TBD`. Tables here will be generated from saved evaluation artifacts only;
> nothing is ever hand-invented.

## Environment

- OS / GPU: TBD
- Python: 3.11
- Stack versions (transformers/peft/torch/datasets/accelerate): TBD

## Dataset revisions

| Dataset | Revision pinned at eval time |
|---|---|
| Seif-Eldeen-Sameh/asr_codeswitched_dataset | TBD |
| UBC-NLP/Casablanca (Egypt subset, eval-only) | TBD |
| MohamedGomaa30/EGYSpeak (pseudo-labelled, if used) | TBD |

## Split definitions

- Train 90% / Validation 5% / Internal Test 5%, seed 42 (fixed, persisted in
  `data/splits/`; not yet generated).
- No speaker-disjoint splitting is claimed unless speaker metadata enables it
  (status: TBD).

## Normalization rules

- Raw evaluation: trim + collapse whitespace.
- Normalized evaluation: diacritics, tatweel, punctuation, whitespace, English
  case. English tokens, numbers, and dialect spelling are never deleted or
  transliterated.

## Metric definitions

WER, normalized WER, CER, CS-WER, Arabic-token WER, English-token WER, English
Term Recall, Number Accuracy, Hallucination Rate, RTF. Exact semantics TBD by
the orchestrator (Phase 3).

## Baseline results (E0)

| Model | Clean WER ↓ | CS-WER ↓ | Telephone WER ↓ | EN Recall ↑ | Casablanca WER ↓ |
|---|---:|---:|---:|---:|---:|
| Whisper Large V3 Turbo | TBD | TBD | TBD | TBD | TBD |
| EgypTalk-ASR-v2 | TBD | TBD | TBD | TBD | TBD |
| Whisper Medium Arabic Code-Switched | TBD | TBD | TBD | TBD | TBD |

## E1 (core fine-tuning)

TBD.

## E2 (telephone augmentation)

TBD.

## E3 (EGYSpeak pseudo-labelled ablation, 0%/10%/25%)

TBD.

## Internal test / telephone test / Casablanca

TBD.

## Telephone degradation delta

`telephone WER - clean WER` on identical examples: TBD.

## Latency / RTF

TBD.

## Significance caveats

TBD.

## Known limitations

TBD.
