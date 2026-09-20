# MasriCX quality gates. All caches and temp files stay inside .runtime/
# under the repository (never C:\, and never outside D:\MasriCX).

PYTHON ?= python
CONFIGS := configs/pilot.yaml configs/codeswitch.yaml configs/telephone.yaml configs/egyspeak_ablation.yaml

# Tool caches under D:\MasriCX\.runtime (repo-local).
export PIP_CACHE_DIR ?= .runtime/pip
export PRE_COMMIT_HOME ?= .runtime/pre-commit
export MYPY_CACHE_DIR  ?= .runtime/mypy
export RUFF_CACHE_DIR  ?= .runtime/ruff
export PYTHONPYCACHEPREFIX ?= .runtime/pycache

.PHONY: check lint format typecheck test validate clean

check: lint format typecheck test validate

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format --check .

typecheck:
	$(PYTHON) -m mypy src

test:
	$(PYTHON) -m pytest -q

validate:
	$(PYTHON) -m masricx.config validate $(CONFIGS)

# Remove ONLY explicitly named cache subdirectories. Never delete all of
# .runtime: it also holds ignored OpenCode credentials and relay artifacts.
clean:
	$(PYTHON) -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ['.runtime/pycache', '.runtime/pytest', '.runtime/mypy', '.runtime/ruff', '.runtime/pip', '.runtime/pre-commit']]"
