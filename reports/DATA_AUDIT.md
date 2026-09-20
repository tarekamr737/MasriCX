# Data Audit

> **Status: template (Phase 0).** The audit has not been run. All values are
> `TBD` until the orchestrator executes `scripts/audit.sh` in Phase 1.

Outputs: `reports/DATA_AUDIT.md` (this file, filled) and
`artifacts/data_audit.json` (git-ignored).

## Datasets covered

| Dataset | Revision | License | Redistribution | Notes |
|---|---|---|---|---|
| Seif-Eldeen-Sameh/asr_codeswitched_dataset (primary, gold) | TBD | TBD | TBD | ~45,189 clips, ~34.1h (expected) |
| MohamedGomaa30/EGYSpeak (optional, **pseudo-labelled**) | TBD | TBD | TBD | machine-generated transcripts; not gold |
| UBC-NLP/Casablanca Egypt subset (external, eval-only) | TBD | TBD | TBD | never trained on |

## Audit metrics

| Metric | Value |
|---|---|
| number of examples | TBD |
| total audio duration | TBD |
| duration distribution | TBD |
| sampling rate distribution | TBD |
| empty transcripts | TBD |
| corrupted audio | TBD |
| silent clips | TBD |
| clipped audio | TBD |
| duplicate transcripts | TBD |
| exact duplicate audio | TBD |
| near-duplicate transcripts | TBD |
| Arabic character ratio | TBD |
| English character/token ratio | TBD |
| code-switch frequency | TBD |
| number frequency | TBD |
| symbol frequency | TBD |
| unusually long transcripts | TBD |
| audio/transcript mismatch outliers | TBD |

## Machine-generated / pseudo-labelled content disclosure

- EGYSpeak (if used): **pseudo-labelled** — transcripts produced by another ASR
  model. Never treated as gold; used only in the E3 controlled ablation at
  0%/10%/25%.

## Source limitations

TBD (filled by the orchestrator after reading dataset cards).
