from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

# Ensure src-layout package imports work even when the project is not installed.
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
