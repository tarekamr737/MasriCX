# MasriCX Data Audit

## Card-declared/verified metadata

Source retrieval date: 2026-09-20

| Dataset | Role | Revision | Declared license | Rows | Pseudo-labelled | Release blocker |
|---|---|---|---|---|---|---|
| Seif-Eldeen-Sameh/asr_codeswitched_dataset | primary_training | 712de01079517771f95bcdecee68ca232b979628 | MIT (aggregate card metadata) | 45189 | no | YES: final model publication/license claims blocked pending source-chain review |
| MohamedGomaa30/EGYSpeak | optional_pseudo_labelled_supplement | 59c40fc6833382a743c180f50addcda5edaa47c3 | CC-BY-4.0 (card) | 147979 | yes | YES: E3 disabled/not runnable until the CC-BY-4.0 vs GPL-3.0 license-chain conflict is resolved; audio/full use fail-closed in the loader |
| UBC-NLP/Casablanca | external_evaluation_only | 8951b1b88e28c1107142ced57967b8d16350951d | CC-BY-NC-ND-4.0 | test: 846; validation: 846 | no | YES: data redistribution blocked by CC-BY-NC-ND-4.0; aggregate metrics publication allowed with attribution |

## Measured audit results

| Metric | Value |
|---|---|
| examples | 45189 |
| total duration (s) | 122806 |
| total duration (h) | 34.1128 |
| duration p25 (s) | 1.56 |
| median duration (s) | 2.2 |
| duration p75 (s) | 2.8 |
| duration p95 (s) | 7.23 |
| min duration (s) | 0.021 |
| max duration (s) | 24.94 |
| sampling-rate distribution | 16000: 45189 |
| empty transcripts | 1 |
| corrupted audio | 0 |
| silent clips | 1 |
| clipped audio | 66 |
| mismatch outliers | 53 |
| unusually long transcripts | 168 |
| Arabic letter ratio | 0.886674 |
| Latin letter ratio | 0.113326 |
| language categories | AR_EN_CODE_SWITCHED: 10451; AR_ONLY: 34550; EN_ONLY: 170; OTHER: 18 |

### Numbers and symbols

| Metric | Value |
|---|---|
| number frequency by structural bucket | decimal: 11; integer_digits_1: 283; integer_digits_2_3: 333; integer_digits_4_plus: 140 |
| number bucket definition | privacy-safe structural number buckets (raw numeric tokens are PII-sensitive and never serialized): 'integer_digits_1' = integer of 1 digit, 'integer_digits_2_3' = integer of 2-3 digits, 'integer_digits_4_plus' = integer of 4+ digits, 'decimal' = number containing a decimal point or comma. Each matched number token contributes to exactly one bucket: 'decimal' if it contains '.' or ',', otherwise by integer digit count. |
| symbol frequency | !: 738; ": 2170; #: 7; $: 2; %: 33; &: 26; ': 265; (: 290; ): 290; *: 2; +: 102; ,: 529; -: 372; .: 7287; /: 38; :: 15; =: 9; ?: 141; @: 3; [: 5; \: 1; ]: 5; ^: 2; _: 2; «: 102; »: 102; ،: 8893; ؟: 2259; ’: 11 |

### Duplicates

| Metric | Value |
|---|---|
| exact duplicate transcript groups | 684 |
| exact duplicate transcript extra instances | 2022 |
| near-duplicate pairs reported (capped at 20) | 20 |
| near-duplicate pair comparisons | 191239 |
| near-duplicate global comparison budget exhausted | no |
| near-duplicate oversized buckets skipped | 1017 |
| near-duplicate method | per-bigram hash bucketing (candidate generation; buckets over 64 members skipped as uninformative) + exact bigram Jaccard; deterministic and idempotent; worst case strictly bounded by MAX_TOTAL_COMPARISONS and the bucket-member cap, NOT mathematically subquadratic in text count and NOT exhaustive (can miss high-similarity pairs whose shared bigrams are all ultra-common: documented false negatives); reported pairs are exact-Jaccard verified (no false positives) |
| exact duplicate audio groups | 0 |
| exact duplicate audio extra instances | 0 |
| audio duplicate pair comparisons | 0 |
| audio duplicate oversized buckets skipped | 256 |

No audio or PII is included in this report: counts and IDs only; raw numeric tokens are reported only as structural buckets.

## Interpretation and decisions

- Audio coverage: 45,189/45,189 audio rows decoded; total 122806.18 s = 34.1128 h (consistent with the card's 34.1 h claim).
- All measured audio at 16000 Hz sampling rate.
- The 0.021 s minimum duration is suspicious; review it in Phase 2 before trusting it as real speech.
- Language categories: 10451 AR_EN_CODE_SWITCHED (23.13%), 34550 AR_ONLY (76.46%), 170 EN_ONLY (0.38%), 18 OTHER (0.04%). This validates meaningful code-switch coverage but strong Arabic dominance; preserve stratified code-switch metadata in splitting/evaluation.
- Quality triage: 1 empty transcript (0.00%); 1 silent clip (0.00%); 0 corrupted decodes (0.00%); 66 clipped clips (0.15%); 53 duration/transcript mismatch outliers (0.12%); 168 unusually long transcripts (0.37%). These are review/filter candidates, not automatic deletions until IDs/rules are inspected in Phase 2.
- Deduplication: 684 exact transcript groups / 2022 extra instances (4.47% of rows) require deduplication before splitting.
- Exact audio duplicate groups: 0 under the canonical sampling-rate + float32-PCM SHA-256 hash.
- Near-duplicate output is diagnostic only: 20 reported pairs (capped at 20) is the reporting cap, not total prevalence; 1017 oversized buckets were skipped and the global comparison budget was not exhausted. Do not infer a dataset-wide near-duplicate rate from this diagnostic output.
- Number frequencies are privacy-safe structural buckets only (integer_digits_1, integer_digits_2_3, integer_digits_4_plus, decimal); raw numeric tokens were not serialized.
- Governance: primary data may be audited and used for provisional research but not redistributed; final model publication/license claims remain blocked pending source-chain review; EGYSpeak E3 remains disabled pending license-chain resolution; Casablanca remains evaluation-only.
