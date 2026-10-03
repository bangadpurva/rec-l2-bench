"""Kept for old instructions: same as `python scripts/api_ping.py --model clef`."""
import runpy
import sys
from pathlib import Path

sys.argv = [sys.argv[0], "--model", "clef"]
runpy.run_path(str(Path(__file__).with_name("api_ping.py")), run_name="__main__")
