import pytest
from fastapi.testclient import TestClient

UBQ = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"


@pytest.fixture
def client():
    from oritatami.app import app

    with TestClient(app) as c:
        yield c


def test_settings_roundtrip(client):
    r = client.get("/api/settings")
    assert r.status_code == 200
    r = client.patch("/api/settings", json={"diffusion_samples": 3})
    assert r.json()["diffusion_samples"] == 3
    r = client.patch("/api/settings", json={"nope": 1})
    assert r.status_code == 400


def test_mutate_endpoint(client):
    r = client.post("/api/sequence/mutate", json={"sequence": UBQ, "mutations": "K48R"})
    assert r.status_code == 200 and r.json()["sequence"][47] == "R"
    r = client.post("/api/sequence/mutate", json={"sequence": UBQ, "mutations": "A48R"})
    assert r.status_code == 400 and "K" in r.json()["detail"]


def test_chem_describe(client):
    r = client.post("/api/chem/describe", json={"smiles": "CC(=O)Oc1ccccc1C(=O)O"})
    body = r.json()
    assert r.status_code == 200 and body["formula"] == "C9H8O4" and body["svg"].startswith("<?xml")
    assert client.post("/api/chem/describe", json={"smiles": "C1CC("}).status_code == 400


def test_predict_rejects_bad_spec_before_queue(client):
    r = client.post("/api/jobs/predict", json={"spec": {"components": []}})
    assert r.status_code == 400
    assert client.get("/api/jobs").json() == []


def test_job_lifecycle_with_fake_handler(client):
    from oritatami.app import state

    def fake(ctx):
        ctx.set_phase("work", "作業中", 0.5)
        return {"ok": True}

    state.jobs.handlers["scan"] = fake
    r = client.post("/api/jobs/scan", json={"sequence": UBQ, "chain": "A"})
    job_id = r.json()["id"]
    import time

    for _ in range(50):
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] == "succeeded":
            break
        time.sleep(0.1)
    assert j["status"] == "succeeded" and j["result"] == {"ok": True}
    assert "作業中" in client.get(f"/api/jobs/{job_id}/log").text
    assert client.delete(f"/api/jobs/{job_id}").status_code == 200


def test_assistant_verification_moves_a_mutation_onto_the_residue_it_names():
    """A48R on ubiquitin: position 48 is K, and the nearest A is 46.

    This used to be rejected outright. The model names the residue it means and misses the
    index often enough that throwing the proposal away was the common outcome, so it is moved
    instead — and the move is on the card, because A46R is not the same claim as A48R.
    """
    from oritatami import assistant

    wb = {"components": [{"type": "protein", "chains": ["A"], "sequence": UBQ}]}
    p = assistant.verify_proposal({"type": "mutation_set", "title": "t", "rationale": "r", "chain": "A",
                                   "mutations": ["A48R"]}, wb, "A")
    assert p["status"] == "warning"
    assert p["mutations"] == ["A46R"]
    assert any("A48R → A46R" in note for note in p["repaired"])
    assert "位置 48 は K" in p["repaired"][0]


def test_spa_fallback_without_build(client, monkeypatch):
    r = client.get("/api/unknown-endpoint")
    assert r.status_code == 404


def test_autopilot_analyses_do_not_pile_up_chat_threads(tmp_path, monkeypatch):
    """An overnight run once left ~700 machine-made threads in the picker."""
    from oritatami import assistant, llm
    from oritatami.db import Database

    db = Database(tmp_path / "t.sqlite3")
    monkeypatch.setattr(llm, "chat", lambda *a, **k: '{"reply": "ok", "proposals": []}')
    wb = {"name": "t", "components": [{"type": "protein", "label": "U", "chains": ["A"],
                                       "sequence": "MQIFVKTL"}]}
    job = {"id": "j1", "result": {"models": [{"plddt": {"A": [90.0]}}]}}

    r = assistant.ask(db, thread_id=None, mode="explain", message="解説して", workbench=wb,
                      job=job, scan=None, scan_chain=None, focus_chain="A", count=2,
                      persist=False)
    assert r["thread_id"] is None
    assert r["reply"] == "ok"
    assert db.list_threads() == [], "自動解析はスレッドを残さないこと"

    # a person asking still gets a thread they can reopen
    r2 = assistant.ask(db, thread_id=None, mode="explain", message="解説して", workbench=wb,
                       job=job, scan=None, scan_chain=None, focus_chain="A", count=2)
    assert r2["thread_id"]
    assert len(db.list_threads()) == 1
    assert len(db.get_thread(r2["thread_id"])["messages"]) == 2


# ---- hostile input ----------------------------------------------------------
def test_malformed_spec_is_a_400_not_a_crash(client):
    """Both of these reached the user as a bare 500 before: a dict where a list belongs
    iterates as strings, and a nested chain id is unhashable."""
    bad = [
        {"components": {"a": 1}},
        {"components": [{"type": "protein", "label": "x", "sequence": UBQ, "chains": [["A"]]}]},
        {"components": [{"type": "protein", "label": "x", "sequence": UBQ, "chains": [""]}]},
        {"components": [{"type": "protein", "label": "x", "sequence": UBQ, "chains": "A"}]},
        {"components": ["ユビキチン"]},
        {"components": [{"type": "protein", "label": "x", "sequence": UBQ,
                         "chains": ["A"], "copies": "たくさん"}]},
        {"components": [{"type": "protein", "label": f"c{i}", "sequence": "MQIFVKTL",
                         "chains": [chr(65 + i % 26)]} for i in range(200)]},
    ]
    for spec in bad:
        for path in ("/api/jobs/predict", "/api/estimate"):
            r = client.post(path, json={"spec": spec})
            assert r.status_code == 400, f"{path} {str(spec)[:60]} → {r.status_code}"
            assert r.json().get("detail"), "理由を返すこと"


def test_text_settings_are_bounded_and_shaped(client):
    """A model name of '<script>…' or a megabyte of control characters used to be
    accepted and written to settings.json, breaking the assistant until someone noticed."""
    before = client.get("/api/settings").json()
    for key, value in [
        ("llm_model", "<script>alert(1)</script>"),
        ("llm_model", ""),
        ("llm_model", "x" * 5000),
        ("llm_model", "qwen\x00evil"),
        ("ollama_url", "javascript:alert(1)"),
        ("ollama_url", "notaurl"),
        ("msa_server_url", "file:///etc/passwd"),
        ("autopilot_protected_residues", "'; DROP TABLE jobs;--"),
        ("autopilot_experiment", "e" * 500),
        ("esm_model", "../../../../etc/passwd"),
    ]:
        r = client.patch("/api/settings", json={key: value})
        assert r.status_code == 400, f"{key}={value[:30]!r} が通ってしまった"
    # and the good values still work
    for key, value in [("llm_model", "qwen3.5:9b"), ("ollama_url", "http://127.0.0.1:11434"),
                       ("autopilot_protected_residues", "K48, K63, G76"),
                       ("autopilot_experiment", "R2"), ("autopilot_experiment", "")]:
        assert client.patch("/api/settings", json={key: value}).status_code == 200, key
    client.patch("/api/settings", json={k: before[k] for k in
                                        ("llm_model", "ollama_url", "autopilot_experiment",
                                         "autopilot_protected_residues")})


def test_unknown_ids_and_odd_paths_are_404_or_400(client):
    for path in ("/api/jobs/nope", "/api/jobs/nope/log", "/api/jobs/nope/autopilot",
                 "/api/assistant/threads/nope"):
        assert client.get(path).status_code in (400, 404), path
    for path in ("/api/jobs/nope/cancel", "/api/jobs/nope/retry"):
        assert client.post(path, json={}).status_code in (400, 404), path
    assert client.delete("/api/jobs/nope").status_code in (200, 400, 404)
    assert client.patch("/api/settings", json={}).status_code == 200
    assert client.patch("/api/settings", json={"autopilot_enabled": "はい"}).status_code == 400


def test_boolean_settings_reject_ambiguous_strings(client):
    """`bool("false")` is True. The old coercion turned every non-empty string on, so a
    client sending "false" or "off" switched the autopilot ON."""
    for value in ("false", "off", "no", "0", ""):
        r = client.patch("/api/settings", json={"autopilot_enabled": value})
        assert r.status_code == 200 and r.json()["autopilot_enabled"] is False, value
    for value in ("true", "on", "1"):
        r = client.patch("/api/settings", json={"autopilot_enabled": value})
        assert r.status_code == 200 and r.json()["autopilot_enabled"] is True, value
    for value in ("はい", "maybe", "2", [], {}):
        assert client.patch("/api/settings",
                            json={"autopilot_enabled": value}).status_code == 400, value
    client.patch("/api/settings", json={"autopilot_enabled": True})


def test_model_names_cannot_be_paths(client):
    """esm_model is handed to a loader that also accepts local directories."""
    for value in ("../../../../etc/passwd", "/etc/passwd", "~/secrets", "a/../../b"):
        for key in ("esm_model", "llm_model"):
            assert client.patch("/api/settings", json={key: value}).status_code == 400, value
    assert client.patch("/api/settings",
                        json={"esm_model": "facebook/esm2_t33_650M_UR50D"}).status_code == 200


# ---- prose accuracy ---------------------------------------------------------
def test_context_counts_residues_so_the_model_does_not_have_to():
    """9B answered "リジンは 10 個" for a sequence holding 7, and 27B was no better.
    Counting belongs in the context, not in the model."""
    from oritatami import assistant

    ubq = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
    wb = {"name": "u", "components": [{"type": "protein", "label": "U", "chains": ["A"],
                                       "sequence": ubq}]}
    ctx = assistant.build_context(wb, None, None, None)
    assert "K (リジン): 7 個 — 6, 11, 27, 29, 33, 48, 63" in ctx
    assert "C 末端の 5 残基: R L R G G (最後は 76 番の G)" in ctx
    assert "長さ: 76 残基" in ctx


def test_facts_block_is_skipped_for_an_empty_sequence():
    from oritatami import assistant

    assert assistant._sequence_facts("") == []
    assert assistant._sequence_facts("   ") == []


def test_a_reply_contradicting_the_sequence_is_sent_back_for_correction(tmp_path, monkeypatch):
    """The detector already knows which claims are false; handing them back costs one
    round trip and fixes what a larger model does not."""
    from oritatami import assistant, llm
    from oritatami.db import Database

    db = Database(tmp_path / "t.sqlite3")
    wb = {"name": "t", "components": [{"type": "protein", "label": "U", "chains": ["A"],
                                       "sequence": "MQIFVKTLTG"}]}
    calls: list[list[dict]] = []

    def fake_chat(messages, **kwargs):
        calls.append(messages)
        if len(calls) == 1:
            return '{"reply": "位置 3 は E なので E3K を薦めます", "proposals": []}'
        return '{"reply": "位置 3 は I です", "proposals": []}'

    monkeypatch.setattr(llm, "chat", fake_chat)
    r = assistant.ask(db, thread_id=None, mode="chat", message="3番目は？", workbench=wb,
                      job=None, scan=None, scan_chain=None, focus_chain="A", count=1,
                      persist=False)
    assert len(calls) == 2, "矛盾があれば訂正パスを回すこと"
    assert "位置 3 の残基は E ではなく" in calls[1][-1]["content"]
    assert r["corrected"] is True
    assert r["reply_issues"] == []


def test_the_autopilot_does_not_pay_for_the_correction_pass(tmp_path, monkeypatch):
    """Prose nobody reads is not worth a second generation in the loop."""
    from oritatami import assistant, llm
    from oritatami.db import Database

    db = Database(tmp_path / "t.sqlite3")
    wb = {"name": "t", "components": [{"type": "protein", "label": "U", "chains": ["A"],
                                       "sequence": "MQIFVKTLTG"}]}
    calls: list[int] = []

    def fake_chat(messages, **kwargs):
        calls.append(1)
        return '{"reply": "E3K を薦めます", "proposals": []}'

    monkeypatch.setattr(llm, "chat", fake_chat)
    r = assistant.ask(db, thread_id=None, mode="chat", message="?", workbench=wb, job=None,
                      scan=None, scan_chain=None, focus_chain="A", count=1, persist=False,
                      verify_reply=False)
    assert len(calls) == 1
    assert r["corrected"] is False
    assert r["reply_issues"], "検出自体は続けること"


def test_the_heavy_model_is_only_used_when_asked(tmp_path, monkeypatch):
    from oritatami import assistant, llm
    from oritatami.config import update_settings
    from oritatami.db import Database

    db = Database(tmp_path / "t.sqlite3")
    update_settings({"llm_model": "small:1b", "llm_model_heavy": "big:27b"})
    wb = {"name": "t", "components": [{"type": "protein", "label": "U", "chains": ["A"],
                                       "sequence": "MQIFVKTLTG"}]}
    seen: list[str | None] = []

    def fake_chat(messages, **kwargs):
        seen.append(kwargs.get("model"))
        return '{"reply": "ok", "proposals": []}'

    monkeypatch.setattr(llm, "chat", fake_chat)
    common = dict(thread_id=None, mode="chat", message="?", workbench=wb, job=None, scan=None,
                  scan_chain=None, focus_chain="A", count=1, persist=False)
    r = assistant.ask(db, **common)
    assert seen == [None] and r["model"] == "small:1b"
    r = assistant.ask(db, model="big:27b", **common)
    assert seen[-1] == "big:27b" and r["model"] == "big:27b"


def test_a_malformed_json_answer_is_repaired_once(tmp_path, monkeypatch):
    """A sentence around the object is the common small-model failure; throwing the
    whole answer away loses a generation of autopilot ideas."""
    from oritatami import assistant, llm
    from oritatami.db import Database

    db = Database(tmp_path / "t.sqlite3")
    wb = {"name": "t", "components": [{"type": "protein", "label": "U", "chains": ["A"],
                                       "sequence": "MQIFVKTLTG"}]}
    outs = iter(["もちろんです: <<<not json>>>",
                 '{"reply": "E3K を薦めます", "proposals": []}'])
    asks: list[list[dict]] = []

    def fake_chat(messages, **kwargs):
        asks.append(messages)
        return next(outs)

    monkeypatch.setattr(llm, "chat", fake_chat)
    r = assistant.ask(db, thread_id=None, mode="chat", message="?", workbench=wb, job=None,
                      scan=None, scan_chain=None, focus_chain="A", count=1, persist=False,
                      verify_reply=False)
    assert len(asks) == 2
    assert "JSON だけを出力し直してください" in asks[1][-1]["content"]
    assert asks[1][-2]["role"] == "assistant", "壊れた出力を見せて直させる"
    assert r["reply"] == "E3K を薦めます"


def test_an_unrepairable_json_answer_still_fails(tmp_path, monkeypatch):
    """The repair gets one try, then the original error stands — retries must not loop."""
    from oritatami import assistant, llm
    from oritatami.db import Database

    db = Database(tmp_path / "t.sqlite3")
    wb = {"name": "t", "components": [{"type": "protein", "label": "U", "chains": ["A"],
                                       "sequence": "MQIFVKTLTG"}]}
    calls = []
    monkeypatch.setattr(llm, "chat", lambda messages, **kwargs: calls.append(1) or "not json at all")
    with pytest.raises(llm.LlmError):
        assistant.ask(db, thread_id=None, mode="chat", message="?", workbench=wb, job=None,
                      scan=None, scan_chain=None, focus_chain="A", count=1, persist=False,
                      verify_reply=False)
    assert len(calls) == 2


def test_a_settings_change_is_logged(caplog):
    """The autopilot switched itself off overnight and the log said nothing."""
    import logging

    from oritatami.config import update_settings

    with caplog.at_level(logging.INFO, logger="oritatami.config"):
        update_settings({"autopilot_enabled": False})
    messages = [r.getMessage() for r in caplog.records]
    assert any("設定を変更しました" in m for m in messages), messages
    assert any("False" in m for m in messages), messages


def test_prediction_defaults_share_the_spec_bounds():
    """A default outside the submit-time caps saved cleanly, then every prediction was
    rejected at the workbench with a contradiction the settings dialog itself created."""
    from oritatami.config import update_settings

    for key, bad in (("diffusion_samples", 11), ("recycling_steps", 0),
                     ("sampling_steps", 5), ("sampling_steps", 501)):
        with pytest.raises(ValueError):
            update_settings({key: bad})
    s = update_settings({"diffusion_samples": 2, "recycling_steps": 4, "sampling_steps": 100})
    assert s.diffusion_samples == 2 and s.sampling_steps == 100


def test_a_scan_of_an_older_sequence_does_not_offer_dead_substitutions():
    """35 of 50 proposals in a live run were rejected, all of them because the reused
    ESM-2 scan still listed E24A for a chain whose 24 was already A."""
    from oritatami import assistant

    base = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
    now = base[:23] + "A" + base[24:]          # E24A already applied
    wb = {"name": "u", "components": [{"type": "protein", "label": "U", "chains": ["A"],
                                       "sequence": now, "mutations": ["E24A"]}]}
    scan = {"pseudo_perplexity": 3.1, "sequence": base,
            "top_substitutions": [{"mutation": "E24A", "llr": 2.7},
                                  {"mutation": "R54K", "llr": 2.0},
                                  {"mutation": "T55K", "llr": 1.5}],
            "position_tolerance": [0.0] * len(base)}
    ctx = assistant.build_context(wb, None, scan, "A")
    assert "E24A(+2.7)" not in ctx, "既に適用済みの置換を候補として出さないこと"
    assert "T55K(+1.5)" in ctx
    assert "1 件は今の配列で元残基が変わっている" in ctx
    assert "同じ変異コードを再提案しないこと" in ctx


def test_a_matching_scan_is_passed_through_untouched():
    from oritatami import assistant

    seq = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
    wb = {"name": "u", "components": [{"type": "protein", "label": "U", "chains": ["A"],
                                       "sequence": seq}]}
    scan = {"pseudo_perplexity": 3.1, "sequence": seq,
            "top_substitutions": [{"mutation": "E24A", "llr": 2.7}],
            "position_tolerance": [0.0] * len(seq)}
    ctx = assistant.build_context(wb, None, scan, "A")
    assert "E24A(+2.7)" in ctx
    assert "除外しました" not in ctx


# ---- call log and search history --------------------------------------------
def test_every_exchange_is_logged_including_the_autopilots(tmp_path, monkeypatch):
    """An overnight run left no record of the prompts behind its proposals."""
    from oritatami import assistant, llm
    from oritatami.db import Database

    db = Database(tmp_path / "t.sqlite3")
    monkeypatch.setattr(llm, "chat", lambda *a, **k: '{"reply": "ok", "proposals": []}')
    wb = {"name": "t", "components": [{"type": "protein", "label": "U", "chains": ["A"],
                                       "sequence": "MQIFVKTLTG"}]}
    common = dict(thread_id=None, message="?", workbench=wb, job=None, scan=None,
                  scan_chain=None, focus_chain="A", count=1, persist=False)
    assistant.ask(db, mode="chat", **common)
    assistant.ask(db, mode="mutations", origin="autopilot", **common)

    calls = db.list_llm_calls(with_messages=True)
    assert len(calls) == 2
    assert {c["origin"] for c in calls} == {"user", "autopilot"}
    assert db.list_llm_calls(origin="autopilot")[0]["mode"] == "mutations"
    prompt = calls[0]["messages"]
    assert prompt[0]["role"] == "system" and "MQIFVKTLTG" in prompt[0]["content"]
    assert db.count_llm_calls() == 2


def test_a_failed_call_is_logged_too(tmp_path, monkeypatch):
    from oritatami import assistant, llm
    from oritatami.db import Database

    db = Database(tmp_path / "t.sqlite3")

    def boom(*a, **k):
        raise llm.LlmError("Ollama が落ちています")

    monkeypatch.setattr(llm, "chat", boom)
    wb = {"name": "t", "components": [{"type": "protein", "label": "U", "chains": ["A"],
                                       "sequence": "MQIFVKTLTG"}]}
    with pytest.raises(llm.LlmError):
        assistant.ask(db, thread_id=None, mode="chat", message="?", workbench=wb, job=None,
                      scan=None, scan_chain=None, focus_chain="A", count=1, persist=False)
    (call,) = db.list_llm_calls()
    assert "Ollama が落ちています" in call["error"]


def test_the_log_is_capped(tmp_path, monkeypatch):
    from oritatami.db import Database

    db = Database(tmp_path / "t.sqlite3")
    for i in range(5):
        db.record_llm_call(model="m", mode="chat", origin="user", job_id=None, thread_id=None,
                           messages=[{"role": "user", "content": str(i)}], raw="{}", reply=str(i),
                           proposals=[], reply_issues=[], corrected=False, elapsed_sec=0.1, keep=3)
    kept = db.list_llm_calls()
    assert [c["reply"] for c in kept] == ["4", "3", "2"]


def test_a_search_is_remembered_once_per_query_and_records_what_was_added(tmp_path):
    from oritatami.db import Database

    db = Database(tmp_path / "t.sqlite3")
    db.record_search(source="pdb", query="ubiquitin", hits=2,
                     top=[{"id": "1UBQ", "title": "Ubiquitin"}, {"id": "1AAR", "title": "Diubiquitin"}])
    db.record_search(source="pdb", query="ubiquitin", hits=2,
                     top=[{"id": "1UBQ", "title": "Ubiquitin"}])
    rows = db.list_searches()
    assert len(rows) == 1, "同じ検索語で行が増えないこと"

    db.note_search_pick(source="pdb", item_id="1UBQ", title="Ubiquitin")
    assert db.list_searches()[0]["picked"]["id"] == "1UBQ"

    # an accession typed in directly still leaves a trace
    db.note_search_pick(source="uniprot", item_id="P0CG48", title="Polyubiquitin-C")
    typed = [r for r in db.list_searches() if r["source"] == "uniprot"]
    assert typed and typed[0]["query"] == "P0CG48"


def test_search_endpoints_write_history(client, monkeypatch):
    from oritatami import sources

    monkeypatch.setattr(sources, "pdb_search", lambda q: [{"id": "1UBQ", "title": "Ubiquitin"}])
    assert client.get("/api/pdb/search?q=ubiquitin").status_code == 200
    rows = client.get("/api/searches").json()
    assert rows[0]["source"] == "pdb" and rows[0]["query"] == "ubiquitin"
    assert rows[0]["top"][0]["id"] == "1UBQ"
    assert client.delete("/api/searches").json()["deleted"] == 1
    assert client.get("/api/searches").json() == []


def test_unloading_the_model_stops_the_server(monkeypatch):
    """One process serves one model — releasing memory for Boltz is the process ending."""
    from oritatami import llm

    stopped = []
    up = {"v": True}
    monkeypatch.setattr(llm, "server_up", lambda: up["v"])

    def stop():
        stopped.append(True)
        up["v"] = False

    monkeypatch.setattr(llm, "_stop_server", stop)
    llm.unload_model()
    assert stopped, "予測の前にサーバーを止めてメモリを明け渡す"


def test_loading_a_different_model_is_a_restart(monkeypatch):
    """A same-family model at another tag is another file; serving it means restarting —
    gemma3:12b and gemma3:27b cannot share one process."""
    from pathlib import Path

    from oritatami import llm

    monkeypatch.setattr(llm, "_proc", object())
    monkeypatch.setattr(llm, "_proc_path", Path("/m/gemma3-12b.gguf"))
    monkeypatch.setattr(llm, "_proc_model", "gemma3:12b")
    monkeypatch.setattr(llm, "resolve_model",
                        lambda name: {"name": name,
                                      "path": Path(f"/m/{name.replace(':', '-')}.gguf")})
    stopped: list[bool] = []
    monkeypatch.setattr(llm, "_stop_server_locked", lambda: stopped.append(True))

    llm.make_room_for("gemma3:12b")
    assert not stopped, "同じファイルなら起動済みのまま"
    llm.make_room_for("gemma3:27b")
    assert stopped == [True], "別タグは別モデル — 27b を載せるには 12b を終了する"


def test_the_tagless_name_is_an_alias_of_latest(monkeypatch, tmp_path):
    """`gemma3` resolves to the file a `gemma3:latest` pull wrote; without the alias the
    resolution misses and the very model it wants gets treated as absent."""
    from oritatami import llm

    gguf = tmp_path / "gemma3-latest.gguf"
    gguf.write_bytes(b"GGUF")
    monkeypatch.setattr(llm, "models_dir", lambda: tmp_path)
    monkeypatch.setattr(llm, "_registry",
                        lambda: {"gemma3-latest": {"name": "gemma3:latest"}})
    monkeypatch.setattr(llm, "_ollama_blob", lambda name: None)
    assert llm.resolve_model("gemma3")["path"] == gguf
    assert llm.resolve_model("gemma3:latest")["path"] == gguf


def test_think_and_context_window_follow_what_the_model_can_do(monkeypatch):
    """A model whose template has no thinking switch gets no enable_thinking kwarg, and
    one trained on a smaller window does not get a 32k KV cache."""
    from oritatami import llm
    from oritatami.config import update_settings

    update_settings({"llm_think": True})
    info = {
        "gemma3:4b": {"capabilities": ["completion"], "context_length": 4096},
        "qwen3.5:9b": {"capabilities": ["completion", "thinking"], "context_length": 262144},
    }
    monkeypatch.setattr(llm, "model_info", lambda name: info.get(name, {}))
    monkeypatch.setattr(llm, "_ensure_running", lambda name, est_tokens=0.0: None)
    payloads: list[dict] = []
    monkeypatch.setattr(llm, "_post_chat", lambda payload, timeout, client=None: payloads.append(payload) or "")

    llm.chat([{"role": "user", "content": "hi"}], model="gemma3:4b")
    assert "chat_template_kwargs" not in payloads[-1], "思考に非対応のモデルには送らない"
    assert llm._needed_ctx("gemma3:4b", 0) == 4096, "学習窓より広い KV は取らない"

    llm.chat([{"role": "user", "content": "hi"}], model="qwen3.5:9b")
    assert payloads[-1]["chat_template_kwargs"]["enable_thinking"] is True
    assert llm._needed_ctx("qwen3.5:9b", 0) == llm.NUM_CTX


def test_strict_mps_turns_a_silent_cpu_fallback_into_a_failure():
    """Off, an op with no Metal kernel just makes every run slower and says nothing."""
    from oritatami.config import update_settings
    from oritatami.engines import boltz

    update_settings({"mps_strict": False})
    assert boltz.run_env("mps")["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"

    update_settings({"mps_strict": True})
    assert boltz.run_env("mps")["PYTORCH_ENABLE_MPS_FALLBACK"] == "0"
    assert boltz.run_env("cpu")["PYTORCH_ENABLE_MPS_FALLBACK"] == "1", \
        "CPU 指定のときは厳格モードに意味がない"


def test_no_user_facing_string_still_says_qwen():
    """The panel is model-agnostic now; only the credits and the default model name may
    name Qwen."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    allowed = {
        root / "backend" / "oritatami" / "llm.py",     # module docstring names the default
    }
    offenders = []
    for path in list((root / "backend").rglob("*.py")) + list((root / "frontend" / "src").rglob("*.ts*")):
        if "__pycache__" in str(path) or path in allowed:
            continue
        for n, line in enumerate(path.read_text("utf-8").splitlines(), 1):
            if re.search(r"Qwen", line) and "qwen3" not in line and "'Qwen'" not in line:
                offenders.append(f"{path.relative_to(root)}:{n}: {line.strip()[:80]}")
    assert not offenders, "表示文字列に Qwen が残っています:\n" + "\n".join(offenders)


# ---- whole virus particles ---------------------------------------------------
def test_a_capsid_needs_the_assembly_file_not_the_deposited_entry(monkeypatch):
    """An icosahedral virus is deposited as one wedge plus the operators that build the
    other 59, so the plain entry shows a fragment. Measured: MS2 0.36 MB → 21.6 MB."""
    from oritatami import sources

    asked: list[str] = []

    def fake_download(url, *, what):
        asked.append(url)
        return b"data_2MS2\n"

    monkeypatch.setattr(sources, "_download_structure", fake_download)
    monkeypatch.setattr(sources, "_get", lambda *a, **k: None)

    got = sources.fetch_pdb("2ms2", assembly=True)
    assert asked == ["https://files.rcsb.org/download/2MS2-assembly1.cif"]
    assert got["source"]["assembly"] is True
    assert got["path"].name == "pdb_2MS2_assembly1.cif"
    assert "生物学的単位" in got["title"]

    asked.clear()
    sources.fetch_pdb("2ms2")
    assert asked == ["https://files.rcsb.org/download/2MS2.cif"]


def test_an_entry_without_an_assembly_falls_back_and_says_so(monkeypatch):
    from oritatami import sources

    def fake_download(url, *, what):
        return None if "assembly" in url else b"data_X\n"

    monkeypatch.setattr(sources, "_download_structure", fake_download)
    monkeypatch.setattr(sources, "_get", lambda *a, **k: None)
    got = sources.fetch_pdb("1ABC", assembly=True)
    assert got["source"]["assembly"] is False
    assert "非対称単位を読み込みました" in got["note"]
    assert got["path"].name == "pdb_1ABC.cif"


def test_a_structure_too_large_to_open_is_refused(monkeypatch):
    """6CGV expands to 693 MB; loading that would take the viewer down with it."""
    import httpx
    from oritatami import sources

    class FakeResponse:
        status_code = 200
        headers = {"content-length": str(700_000_000)}

        def iter_bytes(self, size):
            yield b"x" * size

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def stream(self, method, url):
            return FakeResponse()

    monkeypatch.setattr(httpx, "Client", FakeClient)
    with pytest.raises(sources.SourceError) as exc:
        sources._download_structure("https://example.invalid/x.cif", what="テスト")
    assert "上限" in str(exc.value)


def test_the_import_endpoint_passes_the_assembly_flag(client, monkeypatch, tmp_path):
    from oritatami import app as app_module
    from oritatami import sources

    seen: dict[str, object] = {}

    def fake_fetch(pdb_id, assembly=False):
        seen["id"] = pdb_id
        seen["assembly"] = assembly
        return {"path": tmp_path / "x.cif", "title": "T", "note": None, "bytes": 12,
                "source": {"db": "PDB", "id": pdb_id, "assembly": assembly}}

    monkeypatch.setattr(sources, "fetch_pdb", fake_fetch)
    monkeypatch.setattr(app_module, "_import_response",
                        lambda path, title, source, plddt=False: {"title": title, "source": source})
    r = client.post("/api/import/pdb", json={"id": "2MS2", "assembly": True})
    assert r.status_code == 200, r.text
    assert seen == {"id": "2MS2", "assembly": True}


def test_an_empty_log_leaves_no_file_behind(client, monkeypatch, tmp_path):
    """15 zero-byte exports piled up in the user's Downloads folder before anyone looked:
    the file was opened first and the row count checked never."""
    from oritatami import app as app_module

    out = tmp_path / "exports"
    out.mkdir()
    monkeypatch.setattr(app_module, "exports_dir", lambda: out)

    r = client.post("/api/llm/calls/export")
    assert r.status_code == 400
    assert list(out.iterdir()) == [], "空でもファイルを作らないこと"

    app_module.state.db.record_llm_call(
        model="m", mode="chat", origin="user", job_id=None, thread_id=None,
        messages=[{"role": "user", "content": "hi"}], raw="{}", reply="ok",
        proposals=[], reply_issues=[], corrected=False, elapsed_sec=0.2)
    r = client.post("/api/llm/calls/export")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] == 1 and body["bytes"] > 0
    files = list(out.iterdir())
    assert len(files) == 1 and files[0].suffix == ".jsonl"


def test_the_test_suite_never_writes_into_the_users_downloads_folder():
    """exports_dir() ignores ORITATAMI_HOME, so isolation has to be set explicitly."""
    import os

    from oritatami.config import exports_dir

    assert os.environ.get("ORITATAMI_EXPORT_DIR"), "conftest が書き出し先を隔離していない"
    assert "Downloads" not in str(exports_dir())
