# Kaggle

Thin Kaggle notebook shells only. Core logic lives in `src/masricx/`; notebook
cells must never contain training logic.

Provided:
- `setup.ipynb` — minimal valid shell documenting the clone/install/launch flow
  (Phase 0; contains no training logic).

Planned workflow (spec §30): clone repo → install deps → authenticate to HF
securely (Kaggle Secrets) → select YAML config → launch `train.py` → push
checkpoints → evaluate → export → terminate cleanly. Training must be
resumable across sessions (spec §31).
