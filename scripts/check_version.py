"""Fail the build when the three version declarations drift apart.

The released version lives in ``backend/app/version.py``.  ``frontend/package.json``
and the newest ``CHANGELOG.md`` entry must agree with it, otherwise a release can
ship a changelog or UI build that advertises a different version than the API
reports at ``/api/health``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_VERSION_RE = re.compile(r'^__version__\s*=\s*"([^"]+)"', re.MULTILINE)


def backend_version() -> str:
    text = (ROOT / "backend" / "app" / "version.py").read_text(encoding="utf-8")
    match = _VERSION_RE.search(text)
    if not match:
        raise ValueError("backend/app/version.py does not declare __version__")
    return match.group(1)


def frontend_version() -> str:
    payload = json.loads((ROOT / "frontend" / "package.json").read_text(encoding="utf-8"))
    version = payload.get("version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("frontend/package.json does not declare a version")
    return version


def changelog_version() -> str:
    """Return the newest released heading, ignoring ``[Unreleased]``."""
    for line in (ROOT / "CHANGELOG.md").read_text(encoding="utf-8").splitlines():
        match = re.match(r"^##\s+\[?([0-9]+\.[0-9]+\.[0-9]+)\]?", line.strip())
        if match:
            return match.group(1)
    raise ValueError("CHANGELOG.md has no released version heading")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check that version declarations agree")
    parser.add_argument("--expect", help="assert every declaration equals this version")
    args = parser.parse_args(argv)

    found = {
        "backend/app/version.py": backend_version(),
        "frontend/package.json": frontend_version(),
        "CHANGELOG.md": changelog_version(),
    }
    distinct = set(found.values())
    if len(distinct) != 1:
        for source, version in found.items():
            print(f"  {source}: {version}", file=sys.stderr)
        print("version declarations disagree", file=sys.stderr)
        return 1
    version = distinct.pop()
    if args.expect and version != args.expect:
        print(f"expected {args.expect}, found {version}", file=sys.stderr)
        return 1
    print(json.dumps({"version": version, "sources": sorted(found)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
