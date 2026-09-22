# Kaggle

Thin Kaggle notebook shells only. Core logic lives in `src/masricx/`; notebook
cells must never contain training logic.

Provided:
- `setup.ipynb` — minimal valid shell documenting the clone/install/launch flow
  (thin launcher; contains no research-critical training logic).

Workflow: set the authorized public GitHub URL, install dependencies, read the
Hugging Face token from Kaggle Secrets, select a YAML config, and launch the
resumable training CLI. Checkpoint upload, evaluation, and publication remain
separate orchestrator-controlled steps.
