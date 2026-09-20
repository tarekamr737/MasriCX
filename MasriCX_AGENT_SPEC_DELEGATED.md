# MasriCX — End-to-End Agent Execution Specification

> **Project:** MasriCX  
> **Purpose:** Build, fine-tune, evaluate, document, and publish an Egyptian Arabic–English code-switched ASR model for contact-center style speech.  
> **Primary goal:** Produce a portfolio-grade fine-tuning project with reproducible training, rigorous evaluation, external testing, public GitHub code, and a Hugging Face model release.  
> **Budget rule:** **$0 by default.** Use Kaggle free GPU compute and Hugging Face for datasets/checkpoints/model hosting. Only consider paid compute if free training is empirically proven insufficient.  
> **Agent mode:** Execute the project end-to-end using installed Kaggle and Hugging Face MCP integrations wherever possible.

---

# 1. Project Objective

MasriCX is an Egyptian Arabic–English **code-switched automatic speech recognition (ASR)** project aimed at contact-center and BPO-style speech.

The project is **not** primarily a chatbot, CRM, voice agent, or customer-service automation system.

The core research question is:

> **Can parameter-efficient fine-tuning of Whisper Large V3 Turbo improve Egyptian Arabic–English code-switched speech recognition while remaining robust under simulated telephone/contact-center audio degradation?**

The main deliverable is a fine-tuned ASR model, benchmark suite, reproducible training pipeline, external evaluation, error analysis, and public model release.

---

# 2. Example Use Case

Input audio:

> "أنا كلمت customer service امبارح عشان الـ package اتجددت بس الـ internet لسه مش شغال."

Expected ASR output:

```json
{
  "transcript": "أنا كلمت customer service امبارح عشان الـ package اتجددت بس الـ internet لسه مش شغال.",
  "language": "ar-en"
}
```

Future product layers may optionally add:

```json
{
  "english_translation": "I contacted customer service yesterday because the package was renewed, but the internet is still not working.",
  "summary": "Customer reports loss of internet service after package renewal.",
  "issue": "internet_connectivity",
  "department": "technical_support"
}
```

However, these product layers are **not part of the initial research contribution**.

---

# 3. Hard Scope Rules

## Must build

- Egyptian Arabic–English code-switched ASR
- Fine-tuned Whisper Large V3 Turbo
- Parameter-efficient fine-tuning using LoRA/PEFT
- Clean-data benchmark
- Code-switched benchmark
- Telephone-degraded benchmark
- External Egyptian benchmark
- Error analysis
- Reproducible training
- Kaggle-compatible training pipeline
- Hugging Face checkpoint/model publishing
- Public GitHub repository
- Model card
- Benchmark report
- Minimal local inference/demo path

## Do NOT build before the ASR benchmark is complete

- Voice bot
- TTS
- CRM integrations
- RAG
- Live calling system
- Autonomous support agent
- Persistent production GPU deployment
- Paid inference infrastructure
- Full customer support SaaS
- Large frontend product
- Complex multi-agent architecture

Optional features may only start after the ASR model passes its promotion gates.

---

# 4. Compute and Cost Policy

## Primary compute

Use **Kaggle free GPU notebooks**.

Current working assumption:

- Kaggle GPU sessions may be time-limited
- Free GPU quota is finite and replenishes
- T4-class GPUs should be treated as the main target
- Training must therefore be resumable and checkpoint-safe

## Model/checkpoint storage

Use **Hugging Face Hub**.

Store:

- LoRA adapters
- trainer state when required
- optimizer state when needed for resumption
- scheduler state
- config files
- experiment metadata
- final model/model card
- benchmark files if useful

Do **not** repeatedly upload full copies of the 809M-parameter base model during intermediate checkpoints.

## Paid compute

Do not use RunPod or any paid GPU unless:

1. the pipeline has already been validated,
2. free Kaggle compute has been tried properly,
3. the limitation is documented,
4. training is shown to be impractical or impossible using the free plan.

Paid compute is a last resort, not part of the default plan.

---

# 5. MCP Usage Policy

The agent has Kaggle and Hugging Face MCP integrations installed.

Use them whenever appropriate.

## Kaggle MCP should be used for

- creating or updating training notebooks
- selecting GPU runtime
- launching training runs
- monitoring run state
- stopping failed runs
- retrieving notebook outputs
- retrieving logs
- retrieving generated artifacts
- checking available datasets if needed
- restarting/resuming experiments

## Hugging Face MCP should be used for

- inspecting model cards
- inspecting dataset cards
- downloading/loading datasets
- creating model repositories
- creating private training/checkpoint repositories
- uploading LoRA adapters/checkpoints
- uploading final model artifacts
- updating model card
- versioning releases
- reading model/dataset metadata
- checking licenses before use

## Agent safety with MCP tools

Never:

- expose secrets in GitHub
- commit Hugging Face tokens
- store Kaggle secrets in repository files
- upload private credentials in notebooks
- overwrite final model checkpoints without versioning
- publish a model before evaluation is complete

All secrets must be handled through environment variables, Kaggle Secrets, or secure MCP authentication.

---

# 6. Base Model

Primary model:

```text
openai/whisper-large-v3-turbo
```

Why:

- multilingual ASR
- strong Arabic baseline
- 809M parameters
- faster than full Whisper Large V3
- compatible with Hugging Face Transformers
- supports PEFT/LoRA approaches
- suitable for realistic fine-tuning research

Training strategy:

```text
8-bit base model + LoRA/PEFT + FP16
```

Do not perform full fine-tuning initially.

---

# 7. Baseline Models

The benchmark must include at least:

1. `openai/whisper-large-v3-turbo`
2. `Seif-Eldeen-Sameh/whisper-medium-arabic-codeswitched`
3. `NAMAA-Space/EgypTalk-ASR-v2`
4. `MasriCX-ASR` — final project model

Optional additional baseline:

- Qwen3-ASR if inference is practical on available hardware

The purpose is to compare MasriCX against:

- a strong generic multilingual model
- an existing code-switched specialist
- an Egyptian specialist
- optionally a modern alternative ASR architecture

---

# 8. Primary Training Dataset

Use:

```text
Seif-Eldeen-Sameh/asr_codeswitched_dataset
```

Expected characteristics:

- ~45,189 audio clips
- ~34.1 hours
- Egyptian Arabic dominant
- Arabic + English code switching
- 16 kHz audio
- real transcript pairs
- suitable for Egyptian technical/code-switched speech

This is the **primary ground-truth training source**.

Do not replace it with generated text/audio.

---

# 9. Supplementary Egyptian Dataset

Use optionally:

```text
MohamedGomaa30/EGYSpeak
```

Important limitation:

- transcripts are machine-generated by another ASR model
- therefore this is pseudo-labelled data
- do not treat it as gold ground truth

Use only for controlled ablation.

Suggested conditions:

```text
0% EGYSpeak
10% filtered EGYSpeak
25% filtered EGYSpeak
```

The experiment should answer:

> Does pseudo-labelled Egyptian speech improve dialect robustness, or does it damage code-switched ASR quality?

Do not allow EGYSpeak to dominate the primary training set.

---

# 10. External Test Dataset

Use the Egyptian subset of:

```text
UBC-NLP/Casablanca
```

Rules:

- never train on it
- never augment it into training
- use it only for external evaluation
- respect its license
- do not redistribute if license terms prohibit derivatives

Purpose:

- evaluate generalization to unseen Egyptian speech
- prove improvements are not tied only to the primary training source

---

# 11. Data Governance

Before any training:

1. read all dataset cards
2. record dataset revisions
3. record licenses
4. document redistribution restrictions
5. document synthetic/pseudo-labelled content
6. document source limitations
7. create `reports/DATA_AUDIT.md`

Do not publish a transformed dataset unless licensing explicitly allows it.

---

# 12. Data Audit

Create an automated audit that reports:

- number of examples
- total audio duration
- duration distribution
- sampling rate distribution
- empty transcripts
- corrupted audio
- silent clips
- clipped audio
- duplicate transcripts
- exact duplicate audio
- near-duplicate transcripts
- Arabic character ratio
- English character/token ratio
- code-switch frequency
- number frequency
- symbol frequency
- unusually long transcripts
- audio/transcript mismatch outliers

Output:

```text
reports/DATA_AUDIT.md
artifacts/data_audit.json
```

No training should begin before the audit completes.

---

# 13. Deduplication

Perform at least:

- normalized transcript exact deduplication
- audio fingerprint or hash deduplication
- suspicious near-duplicate transcript checks

Do not allow identical audio/transcript examples to cross train/validation/test boundaries.

---

# 14. Data Splits

Use deterministic fixed splits.

Default:

```text
Train: 90%
Validation: 5%
Internal Test: 5%
Seed: 42
```

If reliable speaker/session/source IDs become available, prefer group-aware splitting.

If such metadata is not available, do not falsely claim speaker-disjoint splitting.

Persist split IDs in files so every experiment uses the same samples.

Example:

```text
data/splits/train.jsonl
data/splits/validation.jsonl
data/splits/test.jsonl
```

---

# 15. Text Normalization

Implement two evaluation modes.

## Raw evaluation

Minimal normalization:

- trim whitespace
- collapse repeated whitespace

## Normalized evaluation

Normalize:

- Arabic diacritics
- tatweel
- punctuation
- repeated whitespace
- English case

Do not:

- convert English words to Arabic transliteration
- convert Egyptian dialect to MSA
- delete English tokens
- delete numbers
- aggressively rewrite dialect spelling

English term preservation is part of the benchmark.

---

# 16. Code-Switch Detection

Create a heuristic code-switch classifier for each transcript.

Possible categories:

```text
AR_ONLY
EN_ONLY
AR_EN_CODE_SWITCHED
```

A sample is code-switched when it contains meaningful Arabic and English lexical content.

Store this metadata for evaluation.

Do not rely solely on Unicode character count if a better token-level method can be implemented.

---

# 17. Telephone / Contact-Center Augmentation

The augmentation pipeline is a major MasriCX contribution.

Use **real audio + original transcript**.

Do not generate training labels.

Apply randomized signal-only degradation to approximately 30–40% of training examples.

Candidate augmentations:

| Augmentation | Suggested probability |
|---|---:|
| 300–3400 Hz telephone bandpass | 0.30 |
| 16k → 8k → 16k resampling | 0.30 |
| µ-law / A-law simulation | 0.15 |
| Gain variation | 0.20 |
| Mild clipping | 0.10 |
| Light Gaussian/pink noise | 0.15 |
| Speed perturbation 0.95–1.05 | 0.10 |
| Light reverberation | optional |

Rules:

- do not apply all augmentations simultaneously
- preserve transcript labels exactly
- log augmentation seed/config
- create deterministic telephone test corruption for evaluation
- distinguish training-time random augmentation from test-time fixed degradation

---

# 18. Telephone Robustness Test Set

Create:

```text
test_clean
test_telephone
```

They must contain the same examples and transcripts.

The only difference is waveform degradation.

This allows direct measurement of robustness loss.

Report:

```text
Telephone degradation delta = Telephone WER - Clean WER
```

---

# 19. Training Strategy

Use:

```text
8-bit Whisper base
+
LoRA
+
gradient checkpointing
+
FP16
+
dynamic padding
+
duration bucketing where practical
```

Initial config:

```yaml
model:
  id: openai/whisper-large-v3-turbo
  load_in_8bit: true
  dtype: float16

lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules:
    - q_proj
    - v_proj

training:
  learning_rate: 1.0e-4
  epochs: 2
  warmup_ratio: 0.05
  weight_decay: 0.01
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  gradient_checkpointing: true
  fp16: true
  eval_strategy: steps
  eval_steps: 250
  save_strategy: steps
  save_steps: 250
  save_total_limit: 3
  seed: 42
```

These are starting values only.

---

# 20. Hyperparameter Search

Do not run a large grid search.

Use a small pilot subset of approximately 5 hours.

Test:

| Run | LoRA rank | LR |
|---|---:|---:|
| P1 | 8 | 1e-4 |
| P2 | 16 | 1e-4 |
| P3 | 16 | 5e-5 |

Choose using:

- validation WER
- English-term recall
- stability
- VRAM
- training speed

Then freeze the selected configuration.

---

# 21. Experiment Matrix

## E0 — Baselines

No training.

Evaluate all baseline models on:

- internal test
- code-switched subset
- telephone-degraded internal test
- Casablanca Egypt external test

---

## E1 — Core MasriCX

Train:

```text
Whisper Large V3 Turbo
+
LoRA
+
primary 34.1h code-switched dataset
```

No telephony augmentation.

Purpose:

> Measure pure fine-tuning gain.

---

## E2 — MasriCX Telephone

Train:

```text
E1
+
telephone augmentation
```

Purpose:

> Measure whether the model becomes more robust to contact-center style audio while preserving clean accuracy.

Likely final model candidate.

---

## E3 — EGYSpeak Ablation

Train or continue from the best configuration with controlled pseudo-labelled Egyptian data.

Suggested comparisons:

```text
E2 + 0% EGYSpeak
E2 + 10% EGYSpeak
E2 + 25% EGYSpeak
```

Purpose:

> Determine whether pseudo-labelled Egyptian-only speech helps generalization.

Do not automatically promote E3 unless metrics justify it.

---

# 22. Metrics

Required metrics:

- WER
- normalized WER
- CER
- code-switched WER
- Arabic-token WER
- English-token WER
- English Term Recall
- Number Accuracy
- Hallucination Rate
- Real-Time Factor (RTF)

Optional:

- named-entity accuracy
- punctuation quality
- confidence calibration if practical

---

# 23. English Term Recall

This is a core MasriCX metric.

Reference:

```text
اعمل restart للـ router وبعدها افتح الـ application
```

If prediction preserves:

```text
restart ✓
router ✓
application ✗
```

Then:

```text
English Term Recall = 2 / 3 = 66.7%
```

The metric should be computed automatically.

Do not manually cherry-pick examples.

---

# 24. Number Accuracy

Track preservation of:

- phone numbers
- quantities
- IDs
- monetary values
- package names containing numbers
- model numbers

Create a deterministic extraction/evaluation procedure.

---

# 25. Hallucination Analysis

Track obvious ASR hallucinations such as:

- text when speech is silent
- repeated phrases
- invented English terms
- unrelated phrases
- long generated output for short audio

Create an automated heuristic plus manual error review.

---

# 26. External Evaluation

After E1 and E2:

Evaluate on Casablanca Egypt.

Do not tune using Casablanca test performance.

If validation data exists separately, keep it conceptually separate from the final test.

The final report must clearly identify:

- internal results
- telephone results
- external results

---

# 27. Baseline Benchmark Table

The final benchmark must resemble:

| Model | Clean WER ↓ | CS-WER ↓ | Telephone WER ↓ | EN Recall ↑ | Casablanca WER ↓ |
|---|---:|---:|---:|---:|---:|
| Whisper Large V3 Turbo | TBD | TBD | TBD | TBD | TBD |
| EgypTalk-ASR-v2 | TBD | TBD | TBD | TBD | TBD |
| Whisper Medium Arabic Code-Switched | TBD | TBD | TBD | TBD | TBD |
| **MasriCX-ASR** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |

Never invent results.

The table must be generated from saved evaluation artifacts.

---

# 28. Promotion Gate

Do not publish a candidate as the official MasriCX model unless:

```text
Clean CS-WER <= base Whisper
AND
Telephone WER < base Whisper
AND
English Term Recall > base Whisper
AND
External Casablanca performance does not regress unacceptably
```

After baseline runs, replace qualitative gates with exact numeric thresholds.

---

# 29. Error Analysis

Manually inspect approximately 100–200 representative failures.

Categorize:

- English term dropped
- English transliterated
- Egyptian word converted to MSA
- number error
- proper-name error
- hallucination
- repetition
- noise failure
- short-utterance failure
- long-utterance failure
- clipping failure
- telephone-bandwidth failure

Output:

```text
reports/ERROR_ANALYSIS.md
```

Include representative examples but avoid exposing any personally identifiable or sensitive audio.

---

# 30. Kaggle Training Workflow

The Kaggle notebook should remain thin.

It should:

1. clone GitHub repository
2. install dependencies
3. authenticate to Hugging Face securely
4. select YAML config
5. launch `train.py`
6. periodically push checkpoint
7. evaluate
8. export results
9. terminate cleanly

Do not place core training logic inside notebook cells.

Training logic belongs in Python modules under `src/`.

---

# 31. Kaggle Session Recovery

Training must be resumable by design.

Workflow:

```text
Kaggle Session 1
  ↓
train
  ↓
checkpoint
  ↓
push adapter + trainer state to HF
  ↓
session ends

Kaggle Session 2
  ↓
fetch latest checkpoint
  ↓
resume_from_checkpoint
  ↓
continue training
```

The agent must verify the checkpoint is complete before terminating a session.

---

# 32. Hugging Face Repository Plan

Create:

## Final public model repository

```text
tarekamr737/MasriCX-ASR
```

## Training/checkpoint repository

Prefer private during training:

```text
tarekamr737/MasriCX-ASR-training
```

Only make checkpoints public if useful and clean.

---

# 33. Checkpoint Contents

Persist:

- LoRA adapter weights
- adapter config
- trainer state
- optimizer state when needed
- scheduler state
- training config
- experiment metadata
- metrics
- git commit hash
- dataset revision
- seed

Avoid uploading redundant full base-model weights each time.

---

# 34. Experiment Metadata

Every experiment must save something equivalent to:

```json
{
  "experiment": "E2",
  "git_commit": "<commit>",
  "base_model": "openai/whisper-large-v3-turbo",
  "dataset_revision": "<revision>",
  "seed": 42,
  "learning_rate": 0.0001,
  "lora_rank": 16,
  "lora_alpha": 32,
  "augmentation": "telephone-v1",
  "training_examples": 0,
  "started_at": "<timestamp>"
}
```

Never rely on memory for experiment provenance.

---

# 35. Repository Structure

```text
MasriCX/
│
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── Makefile
│
├── configs/
│   ├── pilot.yaml
│   ├── codeswitch.yaml
│   ├── telephone.yaml
│   └── egyspeak_ablation.yaml
│
├── src/
│   └── masricx/
│       ├── data/
│       │   ├── load_codeswitch.py
│       │   ├── load_egyspeak.py
│       │   ├── clean.py
│       │   ├── deduplicate.py
│       │   ├── split.py
│       │   └── audit.py
│       │
│       ├── audio/
│       │   ├── augmentation.py
│       │   ├── telephony.py
│       │   └── preprocessing.py
│       │
│       ├── training/
│       │   ├── train.py
│       │   ├── lora.py
│       │   ├── collator.py
│       │   └── checkpoint.py
│       │
│       ├── evaluation/
│       │   ├── normalize.py
│       │   ├── wer.py
│       │   ├── cer.py
│       │   ├── code_switch.py
│       │   ├── english_recall.py
│       │   ├── numbers.py
│       │   ├── hallucination.py
│       │   └── benchmark.py
│       │
│       └── inference/
│           └── transcribe.py
│
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_baselines.ipynb
│   └── 03_error_analysis.ipynb
│
├── kaggle/
│   ├── setup.ipynb
│   └── README.md
│
├── tests/
│   ├── test_normalization.py
│   ├── test_augmentation.py
│   ├── test_metrics.py
│   └── test_data.py
│
├── scripts/
│   ├── train.sh
│   ├── evaluate.sh
│   └── push_model.sh
│
├── reports/
│   ├── DATA_AUDIT.md
│   ├── BENCHMARK.md
│   ├── ERROR_ANALYSIS.md
│   └── MODEL_CARD.md
│
└── artifacts/
    └── figures/
```

---

# 36. Engineering Standards

Use:

- Ruff
- mypy or Pyright
- pytest
- pre-commit
- GitHub Actions
- deterministic random seeds
- structured configs
- logging
- typed Python where practical

CI must run:

```text
lint
type check
unit tests
config validation
tiny smoke inference/test
```

CI must NOT attempt full model training.

---

# 37. Reproducibility Requirements

The repository must make it possible to reproduce:

- dataset audit
- split generation
- baseline evaluation
- pilot training
- E1
- E2
- E3
- external evaluation
- benchmark table
- error analysis inputs

Document exact commands.

Example:

```bash
python -m masricx.data.audit --config configs/codeswitch.yaml

python -m masricx.evaluation.benchmark \
  --config configs/codeswitch.yaml

python -m masricx.training.train \
  --config configs/telephone.yaml
```

---

# 38. Minimal Inference Interface

Provide a local inference command:

```bash
python -m masricx.inference.transcribe \
  --audio sample.wav \
  --model tarekamr737/MasriCX-ASR
```

Optional local UI:

- Gradio
- Streamlit

Do not require a paid hosted GPU endpoint.

---

# 39. Model Card Requirements

The final Hugging Face model card must contain:

- model name
- model description
- intended use
- out-of-scope uses
- base model
- training datasets
- dataset licenses
- training methodology
- LoRA configuration
- augmentation methodology
- compute environment
- evaluation datasets
- evaluation metrics
- benchmark table
- external evaluation
- error analysis
- limitations
- ethical considerations
- reproduction instructions
- citation
- GitHub link

Disclose clearly:

- the model is Egyptian-dominant
- training audio is not real private call-center data
- telephone conditions are simulated
- ASR hallucinations remain possible
- performance on other Arabic dialects is not guaranteed
- EGYSpeak is pseudo-labelled if used
- model should not be treated as perfect transcription for high-stakes legal/medical use

---

# 40. GitHub README Structure

Recommended order:

```text
MasriCX
├── One-line value proposition
├── Demo/example
├── Results
├── Why Egyptian code-switching is difficult
├── Benchmark table
├── Base vs MasriCX examples
├── Architecture
├── Dataset
├── Fine-tuning method
├── Telephone augmentation
├── External evaluation
├── Error analysis
├── Reproduction
├── Inference
├── Limitations
└── License / citation
```

Results should appear near the top.

---

# 41. Final Benchmark Report

Create:

```text
reports/BENCHMARK.md
```

It must include:

- environment
- dataset revisions
- split definitions
- normalization rules
- metric definitions
- baseline results
- E1 results
- E2 results
- E3 results
- internal test results
- telephone test results
- Casablanca results
- latency/RTF
- significance caveats
- known limitations

---

# 42. Build Order

Execute in this order:

1. Create GitHub repository
2. Add repository skeleton
3. Configure CI
4. Add requirements/pyproject
5. Implement primary dataset loader
6. Implement audit
7. Run full data audit
8. Implement deduplication
9. Generate fixed splits
10. Implement text normalization
11. Implement evaluation metrics
12. Benchmark base Whisper
13. Benchmark EgypTalk
14. Benchmark code-switched Whisper
15. Build 5-hour pilot subset
16. Run LoRA pilot
17. Compare small hyperparameter set
18. Freeze training config
19. Train E1
20. Evaluate E1
21. Perform preliminary error analysis
22. Implement telephony augmentation
23. Train E2
24. Evaluate E2
25. Evaluate E1/E2 on Casablanca
26. Decide whether E3 is worthwhile
27. If yes, filter EGYSpeak
28. Run E3 ablation
29. Select final model using promotion gate
30. Export/push final model
31. Write model card
32. Generate benchmark report
33. Complete error analysis
34. Add inference command
35. Create minimal demo/video
36. Final README
37. Final CI validation
38. Final Hugging Face release
39. Final CV/LinkedIn project summary

Do not skip benchmark-before-training.

---

# 43. Definition of Done

MasriCX is finished only when all items below are complete.

| Deliverable | Required |
|---|---|
| Public GitHub repository | ✅ |
| Reproducible training code | ✅ |
| Dataset audit | ✅ |
| Fixed train/val/test split | ✅ |
| ≥3 meaningful baselines | ✅ |
| Whisper Large V3 Turbo LoRA fine-tuning | ✅ |
| Telephone augmentation | ✅ |
| Clean benchmark | ✅ |
| Code-switched benchmark | ✅ |
| Telephone benchmark | ✅ |
| External Egyptian benchmark | ✅ |
| Error analysis | ✅ |
| At least one ablation experiment | ✅ |
| Hugging Face final model | ✅ |
| Detailed model card | ✅ |
| Local inference command | ✅ |
| CI/tests | ✅ |
| No fabricated data/results | ✅ |
| $0 default compute path | ✅ |

---

# 44. Agent Decision Rules

The agent may make implementation decisions autonomously when they are low-risk and consistent with this spec.

The agent should NOT ask for confirmation for ordinary implementation details.

Examples:

- exact Python package version compatible with the stack
- minor refactors
- retrying failed Kaggle runs
- resuming checkpoints
- fixing lint/test failures
- selecting equivalent library APIs
- batching optimizations
- safe VRAM optimizations

The agent MUST preserve:

- project scope
- dataset integrity
- evaluation integrity
- cost policy
- external test isolation
- no fabricated results
- no hidden synthetic labels

---

# 45. Stop Conditions

Stop training and investigate if:

- loss becomes NaN
- WER sharply degrades across multiple evaluations
- model output becomes empty/repetitive
- corrupted data is detected
- train/test leakage is discovered
- checkpoint upload fails and session termination is near
- VRAM errors repeat after safe batch/gradient changes
- metrics appear implausibly high
- a dataset license blocks intended use

Do not silently continue through data leakage or invalid evaluation.

---

# 46. Free-Compute Optimization Rules

Because the target budget is $0:

1. debug locally/CPU where possible
2. benchmark before training
3. use small subsets before full runs
4. use LoRA
5. use quantized loading if compatible
6. use gradient checkpointing
7. use dynamic padding
8. bucket by audio duration where useful
9. save resumable checkpoints
10. resume across Kaggle sessions
11. do not run unnecessary hyperparameter sweeps
12. do not retrain when evaluation-only work is sufficient

---

# 47. Success Criteria

The project is successful if MasriCX demonstrates a meaningful improvement in at least one important target dimension while remaining competitive elsewhere.

Preferred success pattern:

```text
MasriCX:
- lower code-switched WER than base Whisper
- better English-term recall than base Whisper
- lower telephone WER than base Whisper
- acceptable or improved Casablanca external WER
```

The project may still be publishable if one metric does not improve, provided the trade-off is honestly reported and analyzed.

---

# 48. Non-Negotiable Research Integrity

Never:

- invent benchmark numbers
- tune on the final external test set
- claim speaker-disjoint splits without evidence
- claim real call-center training data
- hide pseudo-labelled data use
- cherry-pick only successful examples
- delete failed runs from the experiment record to misrepresent progress
- describe synthetic telephone degradation as real call-center recordings
- claim production readiness without evidence

---

# 49. Final CV Bullet Template

Do not fill in metrics until measured.

Template:

> **MasriCX — Egyptian Arabic–English Code-Switched ASR:** Fine-tuned Whisper Large V3 Turbo using 8-bit LoRA on 34+ hours of Egyptian-dominant code-switched speech, developed a telephony-robust augmentation and evaluation pipeline, and benchmarked against strong multilingual and Egyptian ASR baselines across WER, code-switched WER, English-term preservation, telephone robustness, and external Egyptian speech; released the model and reproducible training stack on Hugging Face and GitHub.

Add actual measured improvements only after final evaluation.

---

# 50. Final Agent Instruction

Build MasriCX end-to-end according to this specification.

Use the installed **Kaggle MCP** for free GPU execution and the installed **Hugging Face MCP** for datasets, checkpoint/model storage, and final publication.

Default to **$0 infrastructure cost**.

Only use paid GPU compute if the free workflow is empirically shown to be insufficient.

Prioritize:

1. data quality
2. reproducibility
3. evaluation integrity
4. efficient free-compute training
5. external validation
6. honest reporting
7. strong documentation

Do not expand project scope until the core ASR benchmark and final model are complete.

---

# 51. Delegation Architecture — GPT-5.6 Sol Medium + OpenCode GLM-5.3-Flash

MasriCX uses a two-tier agent architecture.

## Orchestrator

**Model:** GPT-5.6 Sol, Medium thinking

Sol is the project owner, technical lead, scientific reviewer, MCP operator, and final decision-maker.

Sol must retain every task where a mistake could invalidate the research, waste scarce GPU quota, publish incorrect information, or create data leakage.

## Implementer

**CLI:** OpenCode  
**Requested model:** `glm-5.3-flash` from provider `sovereignEG`

The exact OpenCode `provider/model` identifier MUST be discovered from the installed CLI before configuring the lane.

Run:

```bash
opencode models
```

Find the entry corresponding to:

```text
provider: sovereignEG
model: glm-5.3-flash
```

Use the **exact identifier returned by OpenCode**.

Do not guess the provider/model string.

If OpenCode reports the identifier as:

```text
sovereignEG/glm-5.3-flash
```

then use exactly that value.

---

# 52. Core Delegation Principle

The split is intentionally asymmetric.

```text
GPT-5.6 Sol Medium
        │
        ├── Think
        ├── Research
        ├── Decide
        ├── Use MCPs
        ├── Design experiments
        ├── Review scientific validity
        ├── Review diffs
        ├── Run final gates
        └── Commit / publish
                 │
                 ▼
        OpenCode delegate
        GLM-5.3-Flash
                 │
        ├── Mechanical implementation
        ├── Boilerplate
        ├── Test implementation
        ├── Config implementation
        ├── Documentation formatting
        ├── Repetitive refactors
        ├── CLI wrappers
        └── Other bounded coding work
```

OpenCode is a **bounded implementer**, not a co-orchestrator.

Sol owns judgment.

GLM owns typing where the expected result is objectively reviewable.

---

# 53. Delegate Skill

Use the installed:

```text
opencode-delegate
```

from:

```text
amElnagdy/delegate-skills
```

The delegate loop is:

```text
1. Sol writes a self-contained brief
2. Sol dispatches to OpenCode
3. OpenCode edits the working tree
4. Sol reads result.json
5. Sol reviews git diff
6. Sol independently runs gates
7. Sol either:
   a. accepts and commits
   b. sends a delta brief with --resume-last
   c. rejects/reverts the work
```

OpenCode MUST NOT commit.

The reviewer/orchestrator commits only after verification.

---

# 54. Recommended Delegate Fleet

Use a project-scoped delegate fleet for MasriCX.

The project should have approximately four OpenCode lanes:

| Lane | Implementer | Purpose |
|---|---|---|
| `implementation` | OpenCode + GLM-5.3-Flash | Straightforward production Python implementation |
| `tests` | OpenCode + GLM-5.3-Flash | Unit/integration tests after Sol defines expected behavior |
| `mechanical` | OpenCode + GLM-5.3-Flash | Refactors, formatting, repetitive edits, repo hygiene |
| `docs` | OpenCode + GLM-5.3-Flash | README/report/model-card scaffolding from verified facts |

All four may use the same GLM model because the goal is to shift high-volume deterministic work away from Sol.

Sol itself is **not** a delegate fleet lane.

Sol is the orchestrator sitting above the fleet.

---

# 55. Delegate Fleet Config Template

First discover the exact OpenCode model ID.

Then create a project configuration equivalent to:

```json
{
  "version": "delegate-fleet.v1",
  "lanes": {
    "implementation": {
      "implementer": "opencode",
      "model": "<EXACT_SOVEREIGNEG_GLM_5_3_FLASH_MODEL_ID>",
      "timeout": "2h"
    },
    "tests": {
      "implementer": "opencode",
      "model": "<EXACT_SOVEREIGNEG_GLM_5_3_FLASH_MODEL_ID>",
      "timeout": "2h"
    },
    "mechanical": {
      "implementer": "opencode",
      "model": "<EXACT_SOVEREIGNEG_GLM_5_3_FLASH_MODEL_ID>",
      "timeout": "2h"
    },
    "docs": {
      "implementer": "opencode",
      "model": "<EXACT_SOVEREIGNEG_GLM_5_3_FLASH_MODEL_ID>",
      "timeout": "2h"
    }
  }
}
```

Do not add OpenCode `variant` unless the user explicitly chooses one.

OpenCode requires a model in `provider/model` form.

The delegate setup must validate the exact model token before writing the project fleet.

---

# 56. Sol-Owned Work — Never Delegate Blindly

GPT-5.6 Sol Medium MUST personally own the following categories.

## A. Research and external information

Sol handles:

- Hugging Face dataset/model discovery
- dataset-card inspection
- model-card inspection
- license research
- benchmark research
- checking upstream limitations
- checking latest library/model changes
- selecting candidate baselines

Reason:

These tasks depend on interpretation, current information, and research integrity.

---

## B. MCP operations

Sol handles all:

- Kaggle MCP actions
- Hugging Face MCP actions
- GitHub MCP actions where project state matters
- external-resource creation
- checkpoint publishing
- final model publishing
- dataset metadata retrieval
- remote notebook execution decisions

OpenCode must not independently perform these remote actions unless Sol explicitly delegates a narrowly scoped operation and the required MCP is actually available to it.

Default policy:

```text
MCP work = Sol
```

---

## C. Dataset decisions

Sol decides:

- which datasets may be used
- train/validation/test composition
- external test isolation
- license compatibility
- pseudo-label policy
- filtering thresholds
- leakage rules
- whether EGYSpeak is included
- exact split policy
- split seed
- which examples are excluded

GLM may implement the code after Sol specifies the rules.

---

## D. Evaluation definitions

Sol owns the definition of:

- WER
- normalized WER
- CER
- CS-WER
- Arabic-token WER
- English-token WER
- English Term Recall
- Number Accuracy
- hallucination criteria
- RTF
- normalization rules
- promotion thresholds

GLM may implement these metrics only from a precise Sol-authored specification.

---

## E. Experiment design

Sol owns:

- E0/E1/E2/E3 definitions
- model selection
- LoRA strategy
- learning-rate choices
- rank choices
- batch strategy
- augmentation probabilities
- early stopping
- checkpoint cadence
- ablation decisions
- whether a failed run should be retried
- whether a result warrants another experiment

---

## F. Scientific interpretation

Sol personally analyzes:

- baseline results
- training curves
- overfitting
- regressions
- data leakage signals
- external-test differences
- code-switch failures
- telephone degradation
- English-token errors
- error categories
- promotion decision

GLM must never declare a model “better” based only on one metric without Sol review.

---

## G. High-impact code review

Sol personally reviews code that affects:

- dataset splits
- deduplication
- metric computation
- label integrity
- checkpoint restoration
- external-test isolation
- model loading
- training state
- Hugging Face publishing
- Kaggle execution
- experiment provenance

GLM may author these files, but Sol reviews every relevant line and re-runs tests.

---

## H. Final outputs

Sol owns:

- final benchmark numbers
- benchmark interpretation
- final README claims
- final model card facts
- limitations
- CV claims
- LinkedIn claims
- release version
- model promotion
- Git commits
- releases

No performance claim written by GLM should be published without Sol verifying the underlying artifact.

---

# 57. OpenCode / GLM-5.3-Flash Work

OpenCode should receive work that is bounded, well specified, and objectively checkable.

## Excellent delegation candidates

### Repository scaffolding

Examples:

- create Python package folders
- add `__init__.py`
- create configs
- create Makefile
- create shell wrappers
- add `.gitignore`
- build CI skeleton
- add pre-commit config

Sol specifies architecture first.

---

### Straightforward Python implementation

Examples:

- file loaders
- serializers
- config parsing
- deterministic utilities
- logging helpers
- path helpers
- dataset wrappers
- CLI argument parsing
- report writers

---

### Unit tests

Once Sol defines expected behavior, delegate:

- normalization tests
- augmentation tests
- metric tests
- loader tests
- config validation tests
- CLI smoke tests
- checkpoint utility tests

GLM should write many test cases efficiently.

Sol runs them independently.

---

### Mechanical refactors

Examples:

- rename modules
- update imports
- move code
- remove duplicate functions
- type-hint cleanup
- lint cleanup
- docstring sweep
- common helper extraction
- formatting
- dead-code removal

---

### Telephony augmentation implementation

Sol defines:

- allowed transforms
- probabilities
- ranges
- transcript-preservation requirement
- deterministic test mode

GLM may implement:

- bandpass utility
- resampling utility
- µ-law/A-law wrappers
- gain
- clipping
- noise
- speed perturbation
- pipeline composition
- tests

Sol validates the signal-processing assumptions.

---

### Data-audit implementation

Sol defines the audit fields.

GLM can implement:

- counters
- statistics aggregation
- histogram generation
- JSON export
- Markdown report rendering
- duplicate summaries
- duration summaries

Sol interprets the results.

---

### Metric implementation

Sol defines the exact metric semantics.

GLM can implement:

- token separation
- English-token extraction
- number extraction
- per-bucket aggregation
- metric serialization
- benchmark-table generation

Sol validates outputs against hand-worked examples.

---

### Training boilerplate

Sol defines the training strategy.

GLM can implement:

- Trainer/DataCollator wiring
- PEFT configuration construction
- config-to-code plumbing
- checkpoint callbacks
- logging hooks
- resume helpers
- command-line interface
- environment validation

Sol reviews all training-critical behavior.

---

### Documentation scaffolding

GLM can generate structure for:

- README sections
- benchmark report template
- error-analysis template
- model-card template
- reproduction instructions
- CLI usage docs

But GLM must use:

```text
TBD
```

for any unverified result.

It must never invent metrics.

---

# 58. Tasks Sol Should Usually Do Directly

Do not delegate a task merely because GLM could technically do it.

Sol should directly handle tasks where delegation/review overhead is greater than implementation effort.

Examples:

- changing one configuration value
- fixing one obvious typo
- writing a tiny conditional
- inspecting one failed log
- adjusting one MCP argument
- changing an experiment threshold
- deciding whether a run is valid
- a small research-critical bug

Delegate only when it meaningfully saves Sol tokens or implementation time.

---

# 59. Delegation Decision Test

Before delegating, Sol asks:

```text
1. Is the task bounded?
2. Can I describe the expected outcome precisely?
3. Can correctness be checked through tests/diff?
4. Does it avoid independent scientific judgment?
5. Does it avoid MCP/remote side effects?
6. Will delegation save meaningful Sol tokens?
```

If most answers are YES:

```text
Delegate to GLM.
```

If any of these are true:

```text
research ambiguity
scientific decision
MCP operation
license interpretation
dataset selection
metric definition
model promotion
final claim
```

then:

```text
Sol owns it.
```

---

# 60. Required Delegate Brief Format

Every OpenCode dispatch must be self-contained.

OpenCode has no access to the orchestrator's chat history.

Use a brief similar to:

```markdown
# Task

Implement deterministic English-term recall evaluation for MasriCX.

## Context

MasriCX evaluates Egyptian Arabic-English code-switched ASR.

The orchestrator has already defined the metric semantics below.

## Goal

Implement the metric exactly as specified.

## Allowed files

- src/masricx/evaluation/english_recall.py
- tests/test_english_recall.py

Do not edit other files unless required for imports.

## Required behavior

1. Extract Latin-script lexical tokens from the normalized reference.
2. Preserve duplicates if they represent separate reference occurrences.
3. Match prediction tokens after English lowercase normalization.
4. Do not transliterate Arabic tokens.
5. Return:
   - reference_count
   - matched_count
   - recall
6. Empty English-reference case must return null/NaN according to project convention, not 1.0.

## Tests

Add tests for:
- all terms matched
- partial match
- repeated term
- no English terms
- mixed punctuation
- Arabic transliteration does not count as English preservation

## Gates

Run:

```bash
pytest tests/test_english_recall.py
ruff check src tests
```

## Constraints

- Do not change metric semantics.
- Do not add dependencies unless unavoidable.
- Do not commit.
- Do not modify unrelated code.

## Report

Return:
- files changed
- implementation summary
- test results
- assumptions
- anything requiring orchestrator review
```

This is the expected level of specificity.

---

# 61. OpenCode Dispatch Pattern

Use the installed `opencode-delegate` relay.

Conceptually:

```bash
node "<opencode-delegate-skill-dir>/scripts/relay.mjs" \
  --brief brief.txt \
  --lane implementation \
  --cd /path/to/MasriCX
```

For direct model selection:

```bash
node "<opencode-delegate-skill-dir>/scripts/relay.mjs" \
  --brief brief.txt \
  --model "<EXACT_SOVEREIGNEG_GLM_MODEL_ID>" \
  --cd /path/to/MasriCX
```

For follow-up fixes:

```bash
node "<opencode-delegate-skill-dir>/scripts/relay.mjs" \
  --brief delta-brief.txt \
  --resume-last \
  --cd /path/to/MasriCX
```

Do not restate the full original brief when using `--resume-last`.

Use a concise delta brief.

---

# 62. Review Contract

Never trust OpenCode's self-report.

After every delegated implementation:

## Sol must inspect

```bash
git status
git diff
```

and the delegate:

```text
result.json
```

Review:

- touched files
- unexpected scope
- changed APIs
- dependencies added
- hidden assumptions
- TODOs
- disabled tests
- hardcoded paths
- fabricated data
- network calls
- unrequested architecture changes

---

# 63. Gate Re-Execution

Sol independently re-runs all relevant gates.

Examples:

```bash
ruff check .
pytest
pyright
```

For a narrower task:

```bash
pytest tests/test_normalization.py
```

Do not accept:

```text
"tests passed"
```

from the delegate as sufficient evidence.

---

# 64. Commit Boundary

OpenCode must never commit.

The sequence is:

```text
GLM changes files
    ↓
Sol reviews diff
    ↓
Sol runs gates
    ↓
Sol decides
    ↓
Sol commits
```

If changes are wrong:

```text
Sol writes delta brief
    ↓
OpenCode --resume-last
    ↓
Sol reviews again
```

---

# 65. Work Allocation by MasriCX Phase

## Phase 0 — Repository Bootstrap

### Sol

- choose architecture
- define dependencies
- define Python version
- define CI gates
- define repository standards

### GLM

- create repository skeleton
- pyproject
- Ruff configuration
- type-check configuration
- pytest setup
- pre-commit
- GitHub Actions
- Makefile
- shell scripts

### Sol

- review
- run gates
- commit

---

## Phase 1 — Dataset Discovery and Audit

### Sol

Uses Hugging Face MCP to:

- inspect primary dataset
- inspect EGYSpeak
- inspect Casablanca
- record revisions
- verify licenses
- inspect schemas
- decide allowed fields
- decide filtering rules

### GLM

Implements:

- loaders
- audit engine
- report writer
- audio statistics
- transcript statistics
- duplicate utilities
- tests

### Sol

Runs the real audit and interprets the output.

---

## Phase 2 — Deduplication and Splits

### Sol

Defines:

- normalization for dedupe
- exact leakage rules
- split policy
- split seed
- source grouping rules if metadata exists

### GLM

Implements:

- transcript hashes
- audio hashes/fingerprints
- fixed split generator
- split persistence
- leakage checks
- tests

### Sol

Verifies split statistics and leakage results.

---

## Phase 3 — Evaluation Framework

### Sol

Defines every metric mathematically.

### GLM

Implements:

- WER wrappers
- CER wrappers
- code-switch categorization
- English-term recall
- number accuracy
- bucket aggregation
- benchmark JSON/Markdown generation
- tests

### Sol

Hand-checks metric outputs.

---

## Phase 4 — Baselines

### Sol

Uses Kaggle/HF MCP to:

- launch baseline jobs
- choose inference settings
- monitor runs
- collect artifacts
- investigate failures
- store benchmark results

### GLM

May implement:

- baseline runner scripts
- batching utilities
- cache layer
- inference CLI
- result serialization

GLM does not decide baseline interpretation.

---

## Phase 5 — Pilot Training

### Sol

Decides:

- 5-hour subset
- LoRA ranks
- LR candidates
- batch strategy
- checkpoint cadence

Uses Kaggle MCP to run pilots.

### GLM

May implement:

- training CLI
- PEFT setup
- config loading
- callbacks
- resume logic
- smoke tests

### Sol

Chooses the final pilot configuration.

---

## Phase 6 — E1 Core Training

### Sol

- launches job
- monitors loss/eval
- handles checkpoint decisions
- resumes sessions
- records experiment metadata

### GLM

Only modifies code if a bounded implementation bug is identified.

GLM must not independently tune hyperparameters.

---

## Phase 7 — Telephone Augmentation / E2

### Sol

Defines signal-processing policy.

### GLM

Implements augmentation pipeline and tests.

### Sol

Validates waveform behavior.

Sol uses Kaggle MCP to launch E2.

---

## Phase 8 — External Evaluation

### Sol

- keeps Casablanca isolated
- runs final external benchmark
- verifies no tuning leakage
- compares E1/E2
- interprets results

### GLM

Can format result files/tables.

GLM must not select the winning model.

---

## Phase 9 — EGYSpeak Ablation

### Sol

Decides whether E3 is worth scarce GPU quota.

If yes:

- defines filtering
- determines 10%/25% conditions
- performs MCP dataset/run actions

### GLM

Can implement deterministic filtering once rules are fixed.

---

## Phase 10 — Error Analysis

### Sol

Selects representative errors and derives the taxonomy.

### GLM

Can implement tools that:

- sample errors
- group records
- render Markdown tables
- generate counts
- export CSV/JSON

Sol writes the actual interpretation.

---

## Phase 11 — Final Release

### Sol

- selects final model
- verifies promotion gate
- creates final benchmark
- confirms every metric
- uses Hugging Face MCP
- publishes adapter/model
- approves model card
- approves README claims
- publishes release

### GLM

Can help prepare:

- model-card skeleton
- README formatting
- reproduction commands
- tables
- code comments

GLM must leave unverified fields as `TBD`.

---

# 66. Sol Token-Saving Strategy

Use GLM specifically to reduce Sol token expenditure on:

- repetitive source-code generation
- repetitive unit-test generation
- config files
- wrappers
- serializers
- CLIs
- docstrings
- report templates
- Markdown formatting
- mechanical refactors
- lint fixes
- type fixes
- obvious bug fixes with known expected behavior

Do NOT spend Sol reasoning tokens typing hundreds of lines of predictable code.

Sol should instead spend tokens on:

- understanding results
- selecting experiments
- reading model/dataset cards
- designing correct metrics
- reviewing tricky diffs
- diagnosing unusual failures
- making final decisions

---

# 67. GLM Context Minimization

Do not dump the entire 1,000+ line MasriCX specification into every delegated task.

Each brief should contain only:

```text
task-specific goal
relevant architecture
exact required behavior
allowed files
constraints
gate commands
report format
```

This improves GLM performance and reduces unnecessary token use.

---

# 68. Parallelism Rules

Do not allow two delegates to edit overlapping files simultaneously.

Safe parallel examples:

```text
Task A:
tests/test_normalization.py

Task B:
docs / README template
```

Unsafe:

```text
Task A:
src/masricx/evaluation/benchmark.py

Task B:
src/masricx/evaluation/benchmark.py
```

When in doubt:

```text
delegate sequentially
```

Training and MCP tasks remain coordinated by Sol.

---

# 69. Branch / Worktree Recommendation

For larger delegated tasks, prefer a dedicated branch/worktree.

Example:

```text
feat/data-audit
feat/evaluation
feat/telephony
```

OpenCode edits the selected worktree.

Sol reviews the diff.

Sol commits only after verification.

This prevents one bad delegation from contaminating unrelated work.

---

# 70. Files OpenCode Must Not Change Without Explicit Brief Authorization

By default, OpenCode must not modify:

```text
data/splits/*
reports/final/*
release/*
LICENSE
dataset license records
final benchmark JSON
final benchmark tables
Hugging Face release metadata
Kaggle remote configuration
secrets
.env
.git/config
```

It may modify these only if Sol explicitly authorizes the exact task.

---

# 71. Remote Side-Effect Policy

OpenCode should be treated as a local code implementer.

Default:

```text
No remote side effects.
```

OpenCode should not:

- publish HF models
- delete HF artifacts
- start paid compute
- mutate Kaggle datasets
- create remote releases
- push Git commits
- change remote branches
- expose secrets

Sol handles those operations.

---

# 72. Failure / Escalation Rules

GLM should stop and report back rather than improvise when:

- required behavior is ambiguous
- dataset schema differs from the brief
- implementing correctly requires changing project architecture
- new dependencies appear necessary
- tests expose a conflict in metric semantics
- training code requires changing experiment design
- remote access is required
- a license issue appears
- the task exceeds allowed files substantially

Sol then decides how to proceed.

---

# 73. Required Review Severity

Use three review levels.

## Level 1 — Mechanical

Examples:

- docs
- formatting
- import cleanup
- simple tests

Review:

```text
diff + relevant gate
```

## Level 2 — Functional

Examples:

- loaders
- augmentation
- CLI
- report generation

Review:

```text
diff + unit tests + smoke test
```

## Level 3 — Research Critical

Examples:

- split logic
- deduplication
- metric implementation
- training state
- checkpoint resume

Review:

```text
line-by-line diff
unit tests
hand-worked validation examples
integration test
full relevant gate
```

Sol owns all Level 3 review.

---

# 74. Example Task Allocation Table

| Task | Sol | GLM |
|---|---:|---:|
| Research latest dataset/model facts | ✅ Owner | ❌ |
| Hugging Face MCP | ✅ Owner | ❌ |
| Kaggle MCP | ✅ Owner | ❌ |
| Dataset license decisions | ✅ Owner | ❌ |
| Experiment design | ✅ Owner | ❌ |
| Define WER/CS-WER semantics | ✅ Owner | ❌ |
| Implement metric code | Review/spec | ✅ Implement |
| Repository scaffolding | Design/review | ✅ Implement |
| Tests | Define critical cases/review | ✅ Implement |
| Data loader | Define schema/review | ✅ Implement |
| Data audit | Interpret | ✅ Implement tooling |
| Dedup code | Define rules/review | ✅ Implement |
| Split code | Define policy/review | ✅ Implement |
| Telephone augmentation | Define policy/review | ✅ Implement |
| Training boilerplate | Define config/review | ✅ Implement |
| Launch Kaggle training | ✅ | ❌ |
| Analyze training curves | ✅ | ❌ |
| Choose best checkpoint | ✅ | ❌ |
| Error-analysis tooling | Define taxonomy | ✅ Implement |
| Error interpretation | ✅ | ❌ |
| README formatting | Verify facts | ✅ Draft |
| Model card formatting | Verify facts | ✅ Draft |
| Final claims | ✅ | ❌ |
| Git commit | ✅ | ❌ |
| HF publish | ✅ | ❌ |

---

# 75. Agent Operating Loop

For every MasriCX milestone:

```text
Sol:
  Understand current state
      ↓
  Decide next bounded unit of work
      ↓
  If implementation-heavy and deterministic:
      delegate to GLM
      ↓
  Read result.json
      ↓
  Inspect diff
      ↓
  Run gates independently
      ↓
  Fix/redelegate if needed
      ↓
  Commit verified change
      ↓
  If remote experiment required:
      Sol uses MCP
      ↓
  Interpret results
      ↓
  Decide next milestone
```

This is the default operating model for the entire project.

---

# 76. Final Delegation Instruction to the MasriCX Agent

You are GPT-5.6 Sol Medium and the **orchestrator** of MasriCX.

Use your reasoning budget on:

- architecture
- research
- scientific decisions
- experiment design
- MCP orchestration
- debugging difficult failures
- evaluation integrity
- review
- final reporting

Use the installed `opencode-delegate` skill with the user's `sovereignEG` `glm-5.3-flash` model for implementation work that is:

- bounded
- repetitive
- token-heavy
- objectively testable
- straightforward for a capable coding model

Never delegate away scientific ownership.

OpenCode is an implementer.

Sol is the reviewer and decision-maker.

Every delegated change follows:

```text
brief → OpenCode → diff → Sol review → Sol gates → Sol commit
```

Never:

```text
brief → OpenCode → trust → publish
```

Use Kaggle and Hugging Face MCP integrations directly from Sol for all remote dataset, training, checkpoint, and release operations.

Maintain the project-wide default budget:

```text
$0
```

Do not use paid infrastructure unless the free route is empirically proven insufficient.

