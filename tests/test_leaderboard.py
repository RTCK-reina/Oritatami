"""The leaderboard must rank every prediction, not just the newest page of them."""
import time

import pytest
from oritatami import app as app_mod
from oritatami.db import Database


@pytest.fixture
def db(tmp_path, monkeypatch):
    d = Database(tmp_path / "t.sqlite3")
    monkeypatch.setattr(app_mod.state, "db", d)
    return d


def _succeed(db, title, plddt, parent=None):
    j = db.insert_job(kind="predict", title=title, spec={}, parent_id=parent, origin="user")
    db.update_job(j["id"], status="succeeded",
                  result={"models": [{"plddt": {"A": [plddt]}, "ligand_plddt": {},
                                      "confidence": {}}]},
                  finished_at=time.time())
    return db.get_job(j["id"])


def test_the_leader_is_the_best_overall_not_the_best_of_the_newest(db):
    """The peak is usually an old row; a newest-first window would hide it."""
    peak = _succeed(db, "頂点", 95.6)
    for i in range(10):
        _succeed(db, f"あと {i}", 90.0 + i * 0.1)

    top = app_mod.leaderboard(metric="mean_plddt", limit=3)
    assert top["best_id"] == peak["id"]
    assert top["rows"][0]["id"] == peak["id"]
    assert top["rows"][0]["rank"] == 1
    assert len(top["rows"]) == 3, "limit は返す行数を絞るだけ"
    assert top["count"] == 11, "件数は全体を数えること"
    assert top["returned"] == 3


def test_children_of_finds_a_child_however_old(db):
    parent = _succeed(db, "親", 90.0)
    child = db.insert_job(kind="predict", title="子", spec={"autopilot_mutation": "K48R"},
                          parent_id=parent["id"], origin="autopilot_variant")
    for i in range(50):
        _succeed(db, f"無関係 {i}", 80.0)
    assert [c["id"] for c in db.children_of(parent["id"])] == [child["id"]]
    assert db.children_of(child["id"]) == []


def test_succeeded_predictions_unlimited_returns_everything(db):
    for i in range(30):
        _succeed(db, f"j{i}", 80.0)
    assert len(db.succeeded_predictions(limit=None)) == 30
    assert len(db.succeeded_predictions(limit=5)) == 5
