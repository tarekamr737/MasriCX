"""Allow ``python -m masricx.config``."""

from masricx.config.loader import main

if __name__ == "__main__":
    raise SystemExit(main())
