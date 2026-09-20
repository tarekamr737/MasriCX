# MasriCX — Agent Operating Rules

Rules for any coding agent (OpenCode/GLM or human) working in this repository.

## Scope

- MasriCX is a research project for Egyptian Arabic–English code-switched ASR
  (fine-tuned Whisper Large V3 Turbo with LoRA/PEFT).
- The authoritative specification is `MasriCX_AGENT_SPEC_DELEGATED.md`. Do not edit
  that file without orchestrator authorization.
- Core logic (data, training, evaluation, inference) belongs under `src/masricx/`.
  Notebooks (Kaggle or local) must stay thin shells that clone/install/launch the
  modules in `src/`; never place research-critical logic in notebook cells.

## Working tree and artifacts

- All generated, runtime, cache, checkpoint, download, and temporary artifacts must
  stay inside `D:\MasriCX`, preferably under `D:\MasriCX\.runtime\` (which is
  git-ignored). Never write caches or scratch files to `C:\`.
- The orchestrator maintains `.gitignore`; agents extend it only when adding a new
  artifact class, and never remove the `.runtime/` entry.
- Do not create data files, checkpoints, downloaded models, or credentials as part
  of code changes.

## Integrity

- **No secrets.** Never commit or hardcode Hugging Face tokens, Kaggle API keys, or
  any credentials. Secrets live only in environment variables, Kaggle Secrets, or
  secure MCP authentication.
- **No remote side effects.** Agents do not push git, publish Hugging Face
  artifacts, start remote compute, mutate Kaggle datasets, or create remote
  releases. The orchestrator (Sol) owns all remote operations.
- **No fabricated metrics.** Every unverified number, revision, license, or result
  is written as `TBD` or `null`. Never invent benchmark values or dataset facts.
- **No training in CI or local gates.** CI is CPU-only, deterministic, and
  network-independent after dependency installation; it never downloads models or
  datasets.

## Git

- Agents **never** run `git add` or `git commit`. Changes are left uncommitted for
  orchestrator review (brief → implement → diff review → gates → orchestrator
  commit).

## Tooling gates

Run before declaring work done:

```text
python -m ruff check .
python -m ruff format --check .
python -m mypy src
python -m pytest -q
python -m masricx.config validate configs/pilot.yaml configs/codeswitch.yaml configs/telephone.yaml configs/egyspeak_ablation.yaml
```

All caches go under repo-local `.runtime/` subdirectories:
`.runtime/pytest` (pytest cache_dir), `.runtime/mypy` (MYPY_CACHE_DIR),
`.runtime/ruff` (RUFF_CACHE_DIR), `.runtime/pycache` (PYTHONPYCACHEPREFIX),
`.runtime/pip` (PIP_CACHE_DIR), `.runtime/pre-commit` (PRE_COMMIT_HOME), and
`.runtime/venv`. The `make clean` target removes only those named cache
subdirectories; never delete all of `.runtime` because it also holds
orchestrator credentials and relay artifacts.

## Experiments

- E0 = baselines (no training), E1 = core fine-tuning, E2 = E1 + telephone
  augmentation, E3 = EGYSpeak pseudo-labelled ablation. Preserve those
  distinctions in configs, reports, and provenance metadata.
- EGYSpeak data is machine-generated (pseudo-labelled) and must always be marked
  as such; it may never be treated as gold ground truth.
- The external Casablanca Egypt test set is evaluation-only: never train on it,
  never augment it into training.
