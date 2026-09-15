"""Tests for 0.2 features: change long-poll, retry, variant batches, estimates, export, resilience."""

import io
import json
import time
import zipfile

import pytest
from conftest import DATA
from fastapi.testclient import TestClient

UBQ = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"


@pytest.fixture
def client(monkeypatch):
    from oritatami import app as app_module

    monkeypatch.setattr(app_module, "resolve_boltz_bin", lambda: "/usr/bin/true")
    monkeypatch.setattr("oritatami.llm.unload_model", lambda: None)  # keep tests away from a real Ollama
    with TestClient(app_module.app) as c:
        yield c


def _wait(client, job_id, statuses=("succeeded", "failed", "cancelled"), timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in statuses:
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} stuck in {job['status']}")


def _spec():
    return {"name": "UBQ", "workbench_name": "UBQ",
            "components": [{"type": "protein", "chains": ["A"], "label": "ubiquitin", "sequence": UBQ, "msa": "single"}]}


def test_changes_long_poll_wakes_on_submit(client):
    from oritatami.app import state

    state.jobs.handlers["scan"] = lambda ctx: {"pseudo_perplexity": 1.0, "chain": "A"}
    rev = client.get("/api/jobs/changes", params={"rev": 0, "timeout": 0}).json()["rev"]
    idle = client.get("/api/jobs/changes", params={"rev": rev, "timeout": 0.2}).json()
    assert idle == {"rev": rev, "changed": False}
    client.post("/api/jobs/scan", json={"sequence": UBQ, "chain": "A"})
    started = time.time()
    moved = client.get("/api/jobs/changes", params={"rev": rev, "timeout": 5}).json()
    assert moved["changed"] and moved["rev"] > rev and time.time() - started < 2


def test_retry_failed_job_with_overrides(client):
    from oritatami.app import state

    def boom(ctx):
        raise RuntimeError("MPS backend out of memory (MPS allocated: 18 GB)")

    state.jobs.handlers["predict"] = boom
    job = client.post("/api/jobs/predict", json={"spec": _spec()}).json()
    failed = _wait(client, job["id"])
    assert failed["status"] == "failed" and failed["error_kind"] == "oom"

    state.jobs.handlers["predict"] = lambda ctx: {"models": [], "chains": []}
    r = client.post(f"/api/jobs/{job['id']}/retry", json={"accelerator": "cpu", "diffusion_samples": 1})
    assert r.status_code == 200, r.text
    again = _wait(client, r.json()["id"])
    assert again["status"] == "succeeded"
    assert again["spec"]["params"]["accelerator"] == "cpu"
    assert "再実行" in again["title"] and "CPU" in again["title"]
    assert client.post(f"/api/jobs/{job['id']}/retry", json={"accelerator": "gpu"}).status_code == 422


def test_batch_variants_are_atomic(client):
    from oritatami.app import state

    state.jobs.handlers["predict"] = lambda ctx: {"models": [], "chains": []}
    bad = client.post("/api/jobs/predict/batch", json={"spec": _spec(), "variants": [
        {"chain": "A", "mutations": ["K48R"]}, {"chain": "A", "mutations": ["A48R"]}]})
    assert bad.status_code == 400
    assert client.get("/api/jobs").json() == []

    ok = client.post("/api/jobs/predict/batch", json={"spec": _spec(), "parent_id": "job_parent", "variants": [
        {"chain": "A", "mutations": ["K48R"]}, {"chain": "A", "mutations": ["K63R", "L73P"]}]})
    assert ok.status_code == 200, ok.text
    jobs = ok.json()["jobs"]
    assert [j["title"] for j in jobs] == ["UBQ + K48R", "UBQ + K63R/L73P"]
    full = client.get(f"/api/jobs/{jobs[1]['id']}").json()
    comp = full["spec"]["components"][0]
    assert comp["mutations"] == ["K63R", "L73P"] and comp["parent_sequence"] == UBQ
    assert comp["sequence"][62] == "R" and full["parent_id"] == "job_parent"


def test_job_summary_omits_sequences(client):
    from oritatami.app import state

    state.jobs.handlers["predict"] = lambda ctx: {"models": [{"index": 0, "file": "x.cif", "confidence": {"ptm": 0.9},
                                                              "plddt": {"A": [80.0, 90.0]}}], "chains": []}
    job = client.post("/api/jobs/predict", json={"spec": _spec()}).json()
    _wait(client, job["id"])
    row = next(j for j in client.get("/api/jobs").json() if j["id"] == job["id"])
    assert "sequence" not in json.dumps(row["spec"])
    assert row["result"]["mean_plddt"] == 85.0 and row["result"]["confidence"]["ptm"] == 0.9


def test_estimate_endpoint(client):
    r = client.post("/api/estimate", json={"spec": _spec()})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["tokens"] == len(UBQ) and body["low"] <= body["seconds"] <= body["high"]
    assert body["needs_msa_search"] is False and body["basis"] == "default"
    assert client.post("/api/estimate", json={"spec": {"components": []}}).status_code == 400


def test_export_zip_for_scan(client):
    from oritatami.app import state

    matrix = [[0.0] * 20 for _ in UBQ]
    state.jobs.handlers["scan"] = lambda ctx: {"sequence": UBQ, "alphabet": "ACDEFGHIKLMNPQRSTVWY", "matrix": matrix,
                                               "position_tolerance": [0.0] * len(UBQ), "pseudo_perplexity": 2.0}
    job = client.post("/api/jobs/scan", json={"sequence": UBQ, "chain": "A", "label": "UBQ"}).json()
    _wait(client, job["id"])
    r = client.get(f"/api/jobs/{job['id']}/export.zip")
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    names = zipfile.ZipFile(io.BytesIO(r.content)).namelist()
    assert {"summary.json", "scan_matrix.csv", "sequences.fasta", "README.md"} <= set(names)


def test_structure_to_pdb():
    from oritatami import structure

    text = structure.to_pdb(DATA / "ubq_prediction" / "complex_model_0.cif")
    assert "ATOM" in text and text.rstrip().endswith("END")


def test_classify_boltz_failures():
    from oritatami.engines.boltz import classify_failure

    assert classify_failure("RuntimeError: MPS backend out of memory", 1, "structure") == "oom"
    assert classify_failure("", -9, "structure") == "oom"
    assert classify_failure("Calling MSA server\nException: MMseqs2 API is giving errors.", 1, "msa") == "msa"
    assert classify_failure("Failed to download model from all URLs.", 1, "download") == "download"
    assert classify_failure("ValueError: Unable to parse filetype", 1, "preprocess") == "input"


def test_cleanup_keeps_results(tmp_path):
    from oritatami.engines.boltz import cleanup_intermediate

    root = tmp_path / "out" / "boltz_results_complex"
    (root / "processed" / "structures").mkdir(parents=True)
    (root / "processed" / "structures" / "x.npz").write_bytes(b"x" * 1000)
    (root / "msa" / "complex_unpaired_tmp_env").mkdir(parents=True)
    (root / "msa" / "complex_unpaired_tmp_env" / "uniref.a3m").write_bytes(b"y" * 500)
    (root / "msa" / "complex_0.csv").write_text("key,sequence\n")
    (root / "predictions" / "complex").mkdir(parents=True)
    (root / "predictions" / "complex" / "complex_model_0.cif").write_text("data_x\n")
    assert cleanup_intermediate(tmp_path) == 1500
    assert (root / "predictions" / "complex" / "complex_model_0.cif").exists()
    assert (root / "msa" / "complex_0.csv").exists()
    assert not (root / "processed").exists()


def test_broken_settings_file_falls_back_to_defaults(isolated_home):
    import oritatami.config as config

    isolated_home.mkdir(parents=True, exist_ok=True)
    (isolated_home / "settings.json").write_text("{not json", "utf-8")
    s = config.get_settings()
    assert s.diffusion_samples == 1
    assert list(isolated_home.glob("settings.broken-*.json"))


def test_second_instance_detects_lock(isolated_home):
    from oritatami import instance

    first = instance.acquire()
    assert not isinstance(first, str)
    first.publish("http://127.0.0.1:9", "test")  # nothing listens there
    try:
        with pytest.raises(RuntimeError):
            instance.acquire(wait_for_url=0.5)
    finally:
        first.release()
    again = instance.acquire()
    assert not isinstance(again, str)
    again.release()


def test_lock_is_taken_over_when_previous_instance_exits(isolated_home):
    import threading

    from oritatami import instance

    first = instance.acquire()
    assert not isinstance(first, str)
    threading.Timer(0.6, first.release).start()  # the old instance finishes shutting down
    second = instance.acquire(wait_for_url=5)
    assert not isinstance(second, str)
    second.release()


def test_supervisor_stops_command_when_app_exits(tmp_path):
    import os
    import subprocess
    import sys
    import textwrap

    from conftest import ROOT

    marker = tmp_path / "child.pid"
    worker = tmp_path / "worker.py"
    worker.write_text(textwrap.dedent(f"""
        import os, time
        open({str(marker)!r}, "w").write(str(os.getpid()))
        time.sleep(120)
    """))
    app = tmp_path / "app.py"
    app.write_text(textwrap.dedent(f"""
        import os, subprocess, sys, time
        subprocess.Popen([sys.executable, "-m", "oritatami.engines.supervise", str(os.getpid()), "--",
                          sys.executable, {str(worker)!r}], start_new_session=True)
        for _ in range(100):
            time.sleep(0.1)
            if os.path.exists({str(marker)!r}):
                break
    """))
    env = {**os.environ, "PYTHONPATH": str(ROOT / "backend")}
    subprocess.run([sys.executable, str(app)], env=env, timeout=30, check=True)  # the "app" exits here
    pid = int(marker.read_text())
    deadline = time.time() + 15
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.2)
    os.kill(pid, 9)
    raise AssertionError("supervised command survived the app")
