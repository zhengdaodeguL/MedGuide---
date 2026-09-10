"""Single source of truth for the released MedGuide version.

Keep this in step with ``frontend/package.json`` and the top entry of
``CHANGELOG.md``.  ``scripts/check_version.py`` verifies the three agree.
"""

from __future__ import annotations

__version__ = "0.2.0"

__all__ = ["__version__"]
