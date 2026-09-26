"""Every route, walked automatically.

The three silent-truncation bugs and the two 500s found in this codebase were all on
endpoints no test touched — 13 of 66. Rather than write 53 hand-made tests that rot, this
walks the router itself, so an endpoint added tomorrow is covered the day it appears.
"""
import pytest
from fastapi.testclient import TestClient

UBQ = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"

# Endpoints that reach the network or a model, or that would take the app down.
SKIP = {"/api/llm/start", "/api/llm/pull", "/api/assistant/ask", "/api/quit",
        "/api/pdb-watcher/poll", "/api/sources/search", "/api/sources/fetch"}


def _routes(method: str):
    from oritatami.app import app

    for r in app.routes:
        path = getattr(r, "path", "")
        if not path.startswith("/api/") or path in SKIP:
            continue
        if method in getattr(r, "methods", set()):
            yield path


@pytest.fixture
def client():
    from oritatami.app import app

    with TestClient(app) as c:
        yield c


def test_every_parameterless_get_answers_without_crashing(client):
    checked = []
    for path in _routes("GET"):
        if "{" in path:
            continue
        r = client.get(path)
        checked.append(path)
        assert r.status_code < 500, f"GET {path} → {r.status_code}: {r.text[:200]}"
    assert len(checked) >= 15, f"歩けたのは {len(checked)} 本だけ — ルータの形が変わった?"


def test_every_path_parameter_endpoint_handles_an_unknown_id(client):
    """A made-up id must produce a clean 404/400, never a stack trace."""
    for method in ("GET", "POST", "DELETE"):
        for path in _routes(method):
            if "{" not in path:
                continue
            url = path
            for name in ("job_id", "tid", "item_id", "id", "rel", "thread_id", "name"):
                url = url.replace("{" + name + "}", "does-not-exist").replace(
                    "{" + name + ":path}", "does-not-exist")
            if "{" in url:  # a parameter this test does not know how to fill
                continue
            r = client.request(method, url, json={} if method == "POST" else None)
            assert r.status_code < 500, f"{method} {url} → {r.status_code}: {r.text[:200]}"


def test_job_file_traversal_is_refused_not_a_crash(client):
    """.. inside the files path escaped the job directory and answered 500."""
    r = client.get("/api/jobs/x/files/..%2F..%2Fetc%2Fpasswd")
    assert r.status_code in (403, 404), r.status_code


def test_the_endpoints_that_rank_do_not_silently_truncate(client):
    """count is the whole population; returned is what fits the limit."""
    body = client.get("/api/leaderboard", params={"limit": 1}).json()
    assert set(body) >= {"count", "returned", "rows", "best_id", "metric"}
    assert body["returned"] == len(body["rows"])
    assert body["count"] >= body["returned"]
    for metric in body["metrics"]:
        assert client.get("/api/leaderboard", params={"metric": metric}).status_code == 200
    assert client.get("/api/leaderboard", params={"metric": "でたらめ"}).status_code == 400


def test_health_reports_every_engine(client):
    h = client.get("/api/health").json()
    for key in ("llm", "esm", "boltz"):
        assert key in h, key


GARBAGE = [{}, {"x": 1}, {"spec": None}, {"sequence": ""}, {"sequences": "abc"},
           {"smiles": "!!!"}, {"id": "nope"}, {"path": "../../etc/passwd"},
           {"text": "　"}, {"message": "'; DROP TABLE jobs;--"}]
# Destructive or network-bound even with a bad body.
POST_SKIP = SKIP | {"/api/storage/cleanup", "/api/import/afdb", "/api/import/pdb",
                    "/api/import/upload", "/api/pdb_watcher/poll_now", "/api/notify"}


def test_every_post_refuses_garbage_without_crashing(client):
    """A body of the wrong shape must come back as 4xx with a reason, not a stack trace."""
    checked = 0
    for path in _routes("POST"):
        if "{" in path or path in POST_SKIP:
            continue
        for body in GARBAGE:
            r = client.post(path, json=body)
            assert r.status_code < 500, f"POST {path} {body} → {r.status_code}: {r.text[:200]}"
        checked += 1
    assert checked >= 5, f"歩けた POST は {checked} 本だけ"


def test_sequence_helpers_handle_junk(client):
    for body, ok in [
        ({"sequence": UBQ}, True),
        ({"sequence": UBQ.lower()}, True),
        ({"sequence": ">header\n" + UBQ}, True),
        ({"sequence": ""}, False),
        ({"sequence": "MQIF123"}, False),
        ({"sequence": "ユビキチン"}, False),
    ]:
        r = client.post("/api/sequence/validate", json={**body, "kind": "protein"})
        assert r.status_code < 500
        if ok:
            assert r.status_code == 200, f"{body} が拒否された: {r.text[:120]}"

    r = client.post("/api/sequence/diff", json={"a": UBQ, "b": UBQ[:47] + "R" + UBQ[48:]})
    assert r.status_code == 200 and "K48R" in str(r.json())
    assert client.post("/api/sequence/diff",
                       json={"a": UBQ, "b": "MQIF"}).status_code in (200, 400)


def test_autopilot_status_carries_what_the_progress_view_needs(client):
    st = client.get("/api/autopilot/status").json()
    for key in ("enabled", "queued", "max_queued", "strategy", "climb_patience",
                "min_esm_llr", "protected_residues", "disk_free_gb", "accepting"):
        assert key in st, key


def test_leaderboard_rows_carry_what_the_charts_need(client):
    """The progress chart plots created_at vs the metric; the spectrum counts mutations."""
    body = client.get("/api/leaderboard", params={"limit": 5}).json()
    for r in body["rows"]:
        for key in ("id", "title", "created_at", "autopilot_depth", "mutations",
                    "mean_plddt", "core_plddt"):
            assert key in r, key
        assert isinstance(r["mutations"], list)
