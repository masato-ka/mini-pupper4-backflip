import sys
from pathlib import Path

# Make the helpers in scripts/ importable from the tests.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
