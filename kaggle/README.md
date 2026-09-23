# Kaggle

Thin Kaggle notebook shells only. Core logic lives in `src/masricx/`; notebook
cells must never contain training logic.

Provided:
- `setup.ipynb` — minimal valid shell documenting the clone/install/launch flow
  (thin launcher; contains no research-critical training logic).

Workflow: set the authorized public GitHub URL and exact pushed
`EXPECTED_COMMIT`, install dependencies, read the
Hugging Face token from Kaggle Secrets, select a YAML config, and launch the
resumable training CLI. After the orchestrator creates and authorizes a private
Hub model repository, set `CHECKPOINT_REPO` to its `OWNER/NAME`. Every complete
checkpoint and the final adapter are then uploaded under a commit- and
experiment-specific path. Evaluation and final publication remain separate
orchestrator-controlled steps.

Set `MAX_STEPS = 2` for a fresh numerical smoke run before any full pilot. The
trainer logs every smoke step and stops immediately on non-finite gradients or
trainable weights. Do not resume a checkpoint from a failed numerical smoke run.
