"""Module entry: `python -m Neural_Networks.analyzer ...`
Also works as `python __main__.py` or `python analyzer/__main__.py`.
"""
from __future__ import annotations

import sys

# Bootstrap: when run directly (not via -m) relative imports fail.
# A simple try/except lets both execution modes share one code path.
try:
    from .cli import main
except ImportError:
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from Neural_Networks.analyzer.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
