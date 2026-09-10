from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from app.main import app
from app.version import __version__


ROOT = Path(__file__).resolve().parents[2]


def test_released_version_matches_frontend_and_changelog() -> None:
    """The API, the UI bundle and the changelog must advertise one version."""
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_version.py"), "--expect", __version__],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["version"] == __version__


def test_version_is_semantic() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", __version__), __version__


@pytest.mark.parametrize("root_entry", [False, True])
def test_version_check_rejects_lockfile_drift(tmp_path, monkeypatch, root_entry):
    import importlib.util

    spec = importlib.util.spec_from_file_location("version_check", ROOT / "scripts/check_version.py")
    check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check)
    (tmp_path / "frontend").mkdir()
    payload = {"version": __version__, "packages": {"": {"version": __version__}}}
    if root_entry:
        payload["packages"][""]["version"] = "0.0.0"
    else:
        payload["version"] = "0.0.0"
    (tmp_path / "frontend/package-lock.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(check, "ROOT", tmp_path)
    monkeypatch.setattr(check, "backend_version", lambda: __version__)
    monkeypatch.setattr(check, "frontend_version", lambda: __version__)
    monkeypatch.setattr(check, "changelog_version", lambda: __version__)
    assert check.main([]) == 1


def test_fastapi_application_uses_the_single_version_source() -> None:
    assert app.version == __version__


def test_health_reports_the_released_version() -> None:
    """/api/health is the only unauthenticated surface an operator can poll."""
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["version"] == __version__


def test_changelog_declares_the_current_version() -> None:
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    # A released heading (not [Unreleased]) must name the shipping version.
    headings = re.findall(r"^##\s+\[([0-9]+\.[0-9]+\.[0-9]+)\]", text, re.MULTILINE)
    assert headings, "CHANGELOG.md has no released version heading"
    assert headings[0] == __version__


@pytest.mark.parametrize("path", ["backend/app/version.py", "scripts/check_version.py"])
def test_version_sources_are_present(path: str) -> None:
    assert (ROOT / path).is_file()
