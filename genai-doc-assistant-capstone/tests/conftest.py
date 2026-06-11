"""
Test bootstrap: ensure monorepo root is on sys.path so `import pycorekit` works
when running capstone tests in CI or local dev without a prior editable install.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
