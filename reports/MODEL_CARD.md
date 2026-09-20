# Model Card (draft template)

> **Status: template (Phase 0).** This card is drafted for the future
> `tarekamr737/MasriCX-ASR` HF release. All performance values are `TBD`; none
> will be published before evaluation is complete and reviewed by the
> orchestrator.

- **Model name:** MasriCX-ASR (TBD final name)
- **Model description:** TBD
- **Intended use:** Egyptian Arabic-English code-switched ASR transcription.
- **Out-of-scope uses:** high-stakes legal/medical transcription; other Arabic
  dialects (performance not guaranteed); production deployment claims.
- **Base model:** openai/whisper-large-v3-turbo (809M parameters)
- **Training datasets:** Seif-Eldeen-Sameh/asr_codeswitched_dataset (revision
  TBD, license TBD); EGYSpeak only if E3 promotes it — **pseudo-labelled,
  machine-generated, not gold**.
- **Training methodology:** 8-bit base + LoRA + gradient checkpointing + FP16.
- **LoRA configuration:** r=16, alpha=32, dropout=0.05, targets q_proj/v_proj
  (starting candidate; final pilot-frozen selection: TBD).
- **Augmentation methodology:** simulated telephone degradation on ~30-40% of
  training examples (signal-only; transcripts preserved exactly).
- **Compute environment:** Kaggle free GPU (T4-class); $0 budget.
- **Evaluation datasets:** internal test, telephone test, Casablanca Egypt
  (external, evaluation-only).
- **Evaluation metrics:** TBD.
- **Benchmark table:** TBD (see reports/BENCHMARK.md).
- **External evaluation:** TBD.
- **Error analysis:** TBD (reports/ERROR_ANALYSIS.md).
- **Limitations:** Egyptian-dominant; training audio is not real private
  call-center data; telephone conditions are simulated; hallucinations remain
  possible; not suitable for unreviewed high-stakes use.
- **Ethical considerations:** dataset licenses and redistribution TBD; no PII
  shared in reports.
- **Reproduction instructions:** TBD (commands documented in README).
- **Citation:** TBD.
- **GitHub link:** TBD.
- **License:** repository code Apache-2.0; model weights released under a
  license TBD at publish time.
