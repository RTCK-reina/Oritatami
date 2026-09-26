"""The MSA endpoint must find boltz's CSVs under both layouts it writes.

Fresh runs keep them in out/boltz_results_complex/msa; a reused cache lands in input/msa.
"""
import time

import pytest
from fastapi.testclient import TestClient
from oritatami import app as app_mod
from oritatami.db import Database

CSV = "key,sequence\nquery,MKR\nhit1,MK-\nhit2,---\n"


@pytest.fixture
def client(tmp_path, monkeypatch):
    with TestClient(app_mod.app) as c:
        d = Database(tmp_path / "t.sqlite3")
        monkeypatch.setattr(app_mod.state, "db", d)
        yield c


@pytest.fixture
def job(client):
    d = app_mod.state.db
    j = d.insert_job(kind="predict", title="t", spec={}, parent_id=None, origin="user")
    d.update_job(j["id"], status="succeeded", result={}, finished_at=time.time())
    return j


def _write_msa(job_id, isolated_home, rel):
    d = isolated_home / "jobs" / job_id / rel
    d.mkdir(parents=True)
    (d / "msa_A.csv").write_text(CSV)


def test_msa_endpoint_reports_coverage_and_identity(job, client, isolated_home):
    _write_msa(job["id"], isolated_home, "out/boltz_results_complex/msa")
    body = client.get(f"/api/jobs/{job['id']}/msa").json()
    assert len(body["alignments"]) == 1
    a = body["alignments"][0]
    assert a["query"] == "MKR" and a["depth"] == 2 and a["columns"] == 3
    assert a["coverage"] == [0.5, 0.5, 0], "hit2 は全面ギャップ"
    assert a["identity"] == [0.5, 0.5, 0], "hit1 は M と K がクエリと一致"
    assert {s["key"] for s in a["sample"]} == {"hit1", "hit2"}


def test_msa_endpoint_finds_reused_csvs_too(job, client, isolated_home):
    _write_msa(job["id"], isolated_home, "input/msa")
    body = client.get(f"/api/jobs/{job['id']}/msa").json()
    assert len(body["alignments"]) == 1


def test_msa_endpoint_is_empty_not_broken_without_csvs(job, client):
    body = client.get(f"/api/jobs/{job['id']}/msa").json()
    assert body["alignments"] == []


def test_methods_txt_responds_in_both_languages(job, client):
    for lang in ("ja", "en"):
        r = client.get(f"/api/jobs/{job['id']}/methods.txt", params={"lang": lang})
        assert r.status_code == 200 and "attachment" in r.headers["content-disposition"]
        assert r.text.strip(), "空の methods は書き出せない"
    assert client.get(f"/api/jobs/{job['id']}/methods.txt",
                      params={"lang": "zz"}).status_code == 400
