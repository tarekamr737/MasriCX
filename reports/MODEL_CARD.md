# MasriCX-ASR Model Card (pre-release draft)

> **Status:** code-complete draft for the planned
> `tarekamr737/MasriCX-ASR` release. No trained adapter or performance result
> has been approved. Publication is blocked until evaluation, promotion-gate,
> and primary-source model-weight licensing review are complete.

## Model details

- **Task:** Egyptian Arabic-English code-switched automatic speech recognition.
- **Architecture:** `openai/whisper-large-v3-turbo` with LoRA/PEFT adapters.
- **Base revision:** recorded in the validated experiment config.
- **Final adapter revision:** TBD.
- **License:** repository code is Apache-2.0; model-weight license remains TBD
  until the evaluated release candidate is selected.

## Intended use

Transcription research for Egyptian-dominant Arabic-English mixed speech, with
special attention to English technical terms and simulated telephone audio.

Out of scope: unreviewed legal, medical, safety-critical, surveillance, speaker
identification, or guaranteed production transcription; performance on other
Arabic dialects and real call-center traffic is not established.

## Training

- 8-bit base loading, LoRA on `q_proj`/`v_proj`, gradient checkpointing, FP16.
- Pilot candidates and final E1/E2 settings are declared in `configs/`.
- E1 uses the full primary aggregate for internal research. Its verified
  GPL-tagged tail contains most code-switched training examples, so public
  weight licensing remains unresolved rather than silently narrowing the task.
  E2 adds seeded signal-only telephone
  augmentation while preserving transcripts.
- E3 is optional and disabled pending licensing review. If used, EGYSpeak is
  pseudo-labelled machine-generated data and is never treated as gold truth.
- Intended compute path: Kaggle free GPU; training is resumable and checkpoint-safe.

Exact dataset revisions, roles, and licensing caveats are recorded in
`configs/data_sources.yaml`; exact run provenance is emitted beside checkpoints.

## Evaluation

Required evaluation includes raw/normalized WER and CER, code-switched WER,
Arabic/English token WER, English-term recall, number accuracy, hallucination
screening plus manual review, RTF, fixed telephone degradation, and Casablanca
Egypt external evaluation. All values are TBD; see `reports/BENCHMARK.md`.

## Limitations and ethics

Training data is public research speech rather than private contact-center audio.
Telephone conditions are simulated. Fixed splits are not claimed to be
speaker-disjoint. ASR can hallucinate, omit English terms, and mishandle names or
numbers. Reports must remain privacy-safe and must not reproduce raw audio or
transcripts. Casablanca is evaluation-only.

## Reproduction

Use the validated configs, persisted split manifests, training CLI, resumable
prediction runners, and benchmark aggregator documented in `README.md`. The
Kaggle notebook is a thin launcher and expects credentials only through Kaggle
Secrets.

## Citation and links

- GitHub repository: `https://github.com/tarekamr737/MasriCX`.
- Hugging Face model: `tarekamr737/MasriCX-ASR` (not yet released).
- Citation: TBD after release.
