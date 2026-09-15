import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("ORITATAMI_HOME", str(tmp_path / "home"))
    # exports_dir() defaults to ~/Downloads/Oritatami and does NOT follow ORITATAMI_HOME,
    # so the endpoint walk in test_endpoints.py was writing real files into the user's
    # Downloads folder on every run. 15 stray exports before anyone noticed.
    monkeypatch.setenv("ORITATAMI_EXPORT_DIR", str(tmp_path / "exports"))
    import oritatami.config as config

    config._settings = None
    yield tmp_path / "home"
    config._settings = None


DATA = ROOT / "tests" / "data"


@pytest.fixture
def db_and_jobs(tmp_path):
    from oritatami.db import Database

    return Database(tmp_path / "regime.sqlite3")
