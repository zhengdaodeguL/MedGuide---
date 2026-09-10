from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MEDGUIDE_MODE", "test")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
