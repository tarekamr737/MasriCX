# Error Analysis

> **Status:** deterministic privacy-safe candidate generation is implemented,
> but no GPU prediction artifact exists. Counts and findings remain `TBD`.

Manual review will sample systematically from saved evaluation artifacts. Raw
audio and transcript text are not serialized into the candidate artifact.

## Taxonomy

| Category | Count | Notes |
|---|---:|---|
| English term dropped | TBD | |
| English transliterated | TBD | |
| Egyptian word converted to MSA | TBD | |
| Number error | TBD | |
| Proper-name error | TBD | |
| Hallucination | TBD | |
| Repetition | TBD | |
| Noise failure | TBD | |
| Short-utterance failure | TBD | |
| Long-utterance failure | TBD | |
| Clipping failure | TBD | |
| Telephone-bandwidth failure | TBD | |

## Reproduction

```bash
python -m masricx.evaluation.error_analysis --predictions .runtime/predictions/e2-clean.jsonl --output-json artifacts/error_candidates.json
```

The generated IDs, categories, and error rates guide manual review; they are not
final findings. Representative reviewed examples remain TBD.
