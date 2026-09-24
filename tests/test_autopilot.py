"""Autopilot, governor and job-retry behaviour.

These cover the autonomous loop's brakes and its selection logic without touching the
network: the whole point of the loop is that it runs unattended, so a regression here is
one nobody is watching for.
"""
import time

import pytest
from oritatami import autopilot, config, governor, system
from oritatami import jobs as jobs_mod
from oritatami.db import Database

UBQ = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"


@pytest.fixture
def db(tmp_path):
    return Database(tmp_path / "t.sqlite3")


def _result(plddt_by_chain, conf=None):
    return {"models": [{"plddt": plddt_by_chain, "ligand_plddt": {}, "confidence": conf or {}}]}


def _succeed(db, title, result, parent=None, origin="user"):
    j = db.insert_job(kind="predict", title=title, spec={}, parent_id=parent, origin=origin)
    db.update_job(j["id"], status="succeeded", result=result, finished_at=time.time())
    return db.get_job(j["id"])


# ---- the module still has its pieces ---------------------------------------
def test_autopilot_public_surface():
    """A block edit once silently deleted _analyze; nothing in the suite noticed."""
    for name in ("start", "_analyze", "_auto_submit_variants", "_collect_variant_candidates",
                 "_notify_if_improved", "mean_plddt", "metric_value", "resolve_metric",
                 "_branch_point", "_ranked_jobs", "_climb_from_peak", "_submit_candidates"):
        assert callable(getattr(autopilot, name)), name


# ---- governor ---------------------------------------------------------------
def test_daily_budget_counts_only_autonomous_jobs(db):
    config.update_settings({"autopilot_enabled": True, "autopilot_daily_budget": 2,
                            "autopilot_min_disk_gb": 0.0})
    assert governor.check(db).allowed
    for i in range(2):
        db.insert_job(kind="predict", title=f"a{i}", spec={}, parent_id=None,
                      origin="autopilot_variant")
    assert not governor.check(db).allowed
    assert governor.check(db).reason == "daily_budget"

    # A job the user started must not eat the autonomous allowance.
    before = governor.used_today(db)
    db.insert_job(kind="predict", title="mine", spec={}, parent_id=None, origin="user")
    assert governor.used_today(db) == before


def test_disk_floor_and_disabled_switch(db):
    config.update_settings({"autopilot_daily_budget": 99, "autopilot_min_disk_gb": 10**9})
    assert governor.check(db).reason == "low_disk"
    config.update_settings({"autopilot_min_disk_gb": 0.0, "autopilot_enabled": False})
    assert governor.check(db).reason == "autopilot_disabled"
    config.update_settings({"autopilot_enabled": True})


# ---- improvement metric -----------------------------------------------------
def test_auto_metric_picks_iptm_only_for_complexes():
    config.update_settings({"autopilot_improvement_metric": "auto"})
    mono = _result({"A": [80.0]}, {"iptm": 0.0, "ptm": 0.9})
    cplx = _result({"A": [80.0], "B": [78.0]}, {"iptm": 0.62, "ptm": 0.85})
    assert autopilot.resolve_metric(mono) == "mean_plddt"
    assert autopilot.resolve_metric(cplx) == "iptm"
    config.update_settings({"autopilot_improvement_metric": "ptm"})
    assert autopilot.resolve_metric(cplx) == "ptm"
    config.update_settings({"autopilot_improvement_metric": "auto"})


def test_metric_values_share_one_scale():
    """0–1 metrics are lifted onto the 0–100 pLDDT scale so one threshold covers all."""
    assert autopilot.metric_value(_result({"A": [80.0, 90.0]}), "mean_plddt") == 85.0
    assert autopilot.metric_value(_result({"A": [80.0]}, {"iptm": 0.62}), "iptm") == 62.0
    assert autopilot.metric_value(_result({"A": [80.0]}, {}), "iptm") is None


def test_bad_metric_name_is_rejected():
    with pytest.raises(ValueError):
        config.update_settings({"autopilot_improvement_metric": "nonsense"})


@pytest.mark.parametrize("child_iptm,expected", [(0.66, True), (0.63, False), (0.55, False)])
def test_notifies_only_on_a_real_gain(db, monkeypatch, child_iptm, expected):
    sent = []
    monkeypatch.setattr(system, "notify", lambda *a, **k: sent.append(a) or True)
    config.update_settings({"autopilot_notify_improvement": True,
                            "autopilot_improvement_delta": 2.0,
                            "autopilot_improvement_metric": "auto"})
    conf = {"iptm": 0.62, "ptm": 0.85}
    parent = _succeed(db, "parent", _result({"A": [80.0], "B": [78.0]}, conf))
    child = _succeed(db, "child", _result({"A": [80.0], "B": [78.0]},
                                          {**conf, "iptm": child_iptm}), parent=parent["id"])
    autopilot._notify_if_improved(child, db)
    assert bool(sent) is expected


def test_no_parent_means_no_notification(db, monkeypatch):
    sent = []
    monkeypatch.setattr(system, "notify", lambda *a, **k: sent.append(a) or True)
    autopilot._notify_if_improved(_succeed(db, "orphan", _result({"A": [99.0]})), db)
    assert not sent


# ---- proposal selection -----------------------------------------------------
def _job_with(seq=UBQ):
    comp = {"type": "protein", "label": "P", "sequence": seq, "chains": ["A"]}
    spec = {"name": "base", "workbench_name": "base", "autopilot_depth": 0,
            "workbench": {"name": "base", "components": [dict(comp)]},
            "components": [dict(comp, msa="server")]}
    return {"id": "job_t", "title": "base", "spec": spec, "parent_id": None}


def _proposal(muts, llr, status="ok"):
    return {"status": status, "apply": {"action": "mutate", "chain": "A", "mutations": muts},
            "esm": None if llr is None else {"total_llr": llr, "per_mutation": []}}


def test_candidates_are_deduped_and_filtered():
    results = {"analyses": [
        {"proposals": [_proposal(["K48R"], -2.1), _proposal(["T9A"], 1.0),
                       _proposal(["K48A"], 5.0, status="rejected")]},
        # the same mutation proposed again by the other analysis mode
        {"proposals": [_proposal(["K48R"], -2.1), _proposal(["Q40V"], -10.0)]},
    ]}
    got = autopilot._collect_variant_candidates(_job_with(), results)
    names = [c["mut_str"] for c in got]
    assert names.count("K48R") == 1, "duplicate proposals must collapse"
    assert "K48A" not in names, "a rejected proposal must not become a job"
    assert set(names) == {"K48R", "T9A", "Q40V"}
    for c in got:
        assert c["spec"]["autopilot_depth"] == 1
        assert c["spec"]["components"][0]["sequence"] != UBQ


def test_ranking_prefers_higher_esm_and_scored_over_unscored():
    results = {"analyses": [{"proposals": [
        _proposal(["Q40V"], -10.0), _proposal(["T9A"], 1.0),
        _proposal(["K48R"], -2.1), _proposal(["L50M"], None)]}]}
    got = autopilot._collect_variant_candidates(_job_with(), results)
    kept = [c for c in got if c["llr"] is None or c["llr"] >= -5.0]
    kept.sort(key=lambda c: (c["llr"] is not None, c["llr"] if c["llr"] is not None else 0.0),
              reverse=True)
    assert [c["mut_str"] for c in kept][:2] == ["T9A", "K48R"]
    assert kept[-1]["llr"] is None, "an unscored proposal ranks below every scored one"


def _fake_jobs(sink):
    return type("J", (), {"submit": lambda *a, **k: sink.append(a),
                          "touch": lambda s: None})()


def test_depth_limit_stops_the_chain(db, monkeypatch):
    config.update_settings({"autopilot_max_depth": 1, "autopilot_min_disk_gb": 0.0,
                            "autopilot_daily_budget": 0})
    job = _job_with()
    job["spec"]["autopilot_depth"] = 1
    submitted = []
    autopilot._auto_submit_variants(job, {"analyses": [{"proposals": [_proposal(["K48R"], 1.0)]}]},
                                    _fake_jobs(submitted), db)
    assert not submitted


def test_depth_zero_means_unlimited(db):
    """Continuous operation: the chain must not stop itself at any generation."""
    config.update_settings({"autopilot_max_depth": 0, "autopilot_min_disk_gb": 0.0,
                            "autopilot_daily_budget": 0, "autopilot_max_variants_per_job": 1})
    assert not governor.depth_exhausted(0)
    assert not governor.depth_exhausted(50)
    job = _job_with()
    job["spec"]["autopilot_depth"] = 42
    submitted = []
    autopilot._auto_submit_variants(job, {"analyses": [{"proposals": [_proposal(["K48R"], 1.0)]}]},
                                    _fake_jobs(submitted), db)
    assert len(submitted) == 1, "深さ無制限なら世代が進み続ける"
    _self, kind, vspec, _title = submitted[0]
    assert kind == "predict"
    assert vspec["autopilot_depth"] == 43


def test_zero_budget_means_unlimited(db):
    config.update_settings({"autopilot_daily_budget": 0, "autopilot_min_disk_gb": 0.0,
                            "autopilot_max_queued": 8})
    for i in range(50):
        j = db.insert_job(kind="predict", title=f"a{i}", spec={}, parent_id=None,
                          origin="autopilot_variant")
        db.update_job(j["id"], status="succeeded")  # counted against the budget, not the queue
    assert governor.used_today(db) == 50
    assert governor.check(db).allowed, "0 は無制限を意味する"
    config.update_settings({"autopilot_daily_budget": 50})
    assert not governor.check(db).allowed, "0 以外なら従来どおり効く"


def test_queue_cap_holds_an_unlimited_loop_back(db):
    """With depth unlimited the queue cap is what keeps the lane from being buried."""
    config.update_settings({"autopilot_daily_budget": 0, "autopilot_min_disk_gb": 0.0,
                            "autopilot_max_queued": 3})
    for i in range(3):
        db.insert_job(kind="predict", title=f"q{i}", spec={}, parent_id=None,
                      origin="autopilot_variant")  # inserted as 'queued'
    d = governor.check(db)
    assert not d.allowed and d.reason == "queue_full"
    assert "待機中" in governor.describe_reason(d)


# ---- retry ------------------------------------------------------------------
@pytest.mark.parametrize("kind,expected", [("network", True), ("oom", False),
                                           ("missing_dependency", False), ("error", False)])
def test_only_transient_failures_retry(db, monkeypatch, kind, expected):
    config.update_settings({"job_auto_retry": True, "job_max_retries": 2})
    monkeypatch.setattr(jobs_mod, "RETRY_BASE_DELAY_SEC", 0.05)
    mgr = jobs_mod.JobManager(db)
    mgr.register("predict", lambda ctx: {}, lane="predict")
    got = []
    monkeypatch.setattr(mgr, "submit", lambda k, s, t, **kw: got.append((t, s.get("_retry_attempt"))))
    mgr._maybe_retry({"id": "j", "kind": "predict", "title": "t", "spec": {},
                      "parent_id": None, "origin": "user"}, kind)
    time.sleep(0.3)
    assert bool(got) is expected


def test_retry_stops_at_the_limit_without_stacking_titles(db, monkeypatch):
    config.update_settings({"job_auto_retry": True, "job_max_retries": 2})
    monkeypatch.setattr(jobs_mod, "RETRY_BASE_DELAY_SEC", 0.05)
    mgr = jobs_mod.JobManager(db)
    mgr.register("predict", lambda ctx: {}, lane="predict")
    got = []
    monkeypatch.setattr(mgr, "submit", lambda k, s, t, **kw: got.append((t, s.get("_retry_attempt"))))
    base = {"id": "j", "kind": "predict", "parent_id": None, "origin": "user"}
    mgr._maybe_retry({**base, "title": "t (自動再試行 1/2)", "spec": {"_retry_attempt": 1}}, "network")
    time.sleep(0.3)
    assert got == [("t (自動再試行 2/2)", 2)]
    got.clear()
    mgr._maybe_retry({**base, "title": "t", "spec": {"_retry_attempt": 2}}, "network")
    time.sleep(0.2)
    assert not got


# ---- prose grounding --------------------------------------------------------
# Measured against this model: asked to name the residue at a given position it is right
# 1/10 of the time from a raw sequence and 4/10 with a numbered table. Proposals are
# verified residue by residue; these checks do the same for the explanation text.
def _wb():
    return {"components": [{"type": "protein", "label": "U", "sequence": UBQ, "chains": ["A"]}]}


@pytest.mark.parametrize("text,expected", [
    # clean prose stays clean
    ("K48R は表面の塩橋を保ちます。位置 24 のグルタミン酸も重要です。", 0),
    ("位置 48 のリジンは重要です。", 0),
    ("グルタミン (Q40) は表面にあります。", 0),
    # numbers that are not residue references must not trip it
    ("pLDDT は 93.8、ipTM は 0.62、200 ステップで計算しました。", 0),
    # the two mistakes this model actually made
    ("表面露出領域のグルタミン酸 (Q40) をバリンに置換します。", 1),
    ("C 末端の柔軟性を減少させるため、位置 73 のグリシンを置換します。", 1),
    # wrong wild-type letters, either order, and out of range
    ("T68A と G48A を提案します。", 2),
    ("Q40 (グルタミン酸) を置換します。", 1),
    ("位置 60 のアルギニンを置換します。", 1),
    ("L999A が効きます。", 1),
])
def test_reply_claims_are_checked_against_the_sequence(text, expected):
    from oritatami.assistant import check_reply_claims
    assert len(check_reply_claims(text, _wb())) == expected


def test_reply_claims_need_a_protein_to_check_against():
    from oritatami.assistant import check_reply_claims
    assert check_reply_claims("T68A", {"components": []}) == []
    assert check_reply_claims("", _wb()) == []


def test_numbered_residues_is_indexable():
    """The table is what the model counts from, so its numbering has to be exact."""
    from oritatami.assistant import numbered_residues
    rows = numbered_residues(UBQ).splitlines()
    assert rows[0].split()[0] == "1-10"
    assert "".join(rows[0].split()[1:]) == UBQ[:10]
    last = rows[-1]
    assert last.split()[0].endswith(str(len(UBQ)))
    assert "".join(last.split()[1:]) == UBQ[(len(UBQ) - 1) // 10 * 10:]


# ---- context window ---------------------------------------------------------
# Measured prompt sizes: bare 76-residue chain ~900 tokens, one annotated 1100-residue
# chain ~5,200, a four-chain annotated complex ~10,400. Generation speed is identical at
# 16k/32k/64k/128k (44.7 tok/s), so the window costs only memory — but it must stay
# paired with an output cap, since in thinking mode the window was the only thing
# stopping a runaway (16,304 tokens, 405 s, no answer).
def test_context_and_output_caps_are_set_together():
    from oritatami import llm
    assert llm.NUM_CTX >= 16384
    assert 512 <= llm.NUM_PREDICT < llm.NUM_CTX, (
        "an output cap below the window is what bounds a runaway generation")


def test_chat_payload_and_spawn_carry_both_caps(monkeypatch):
    """The window moved to spawn args (--ctx-size); the output cap is a request field."""
    from oritatami import llm
    seen = {}
    spawns = []

    def fake_post(payload, timeout):
        seen.update(payload)
        return '{"reply":"ok","proposals":[]}'

    monkeypatch.setattr(llm, "_post_chat", fake_post)
    monkeypatch.setattr(llm, "_ensure_running",
                        lambda name, est_tokens=0.0: spawns.append(llm._needed_ctx(name, est_tokens)))
    monkeypatch.setattr(llm, "model_info", lambda name: {})
    llm.chat([{"role": "user", "content": "hi"}])
    assert seen["max_tokens"] == llm.NUM_PREDICT
    assert spawns == [llm.NUM_CTX], "小さいプロンプトでは既定の窓で起動する"


def test_oversized_prompt_is_warned_about(monkeypatch, caplog):
    """A prompt near the window is about to force a respawn or fail — the only signal."""
    import logging

    from oritatami import llm
    monkeypatch.setattr(llm, "_post_chat", lambda payload, timeout: "ok")
    monkeypatch.setattr(llm, "_ensure_running", lambda *a, **k: None)
    monkeypatch.setattr(llm, "model_info", lambda name: {})
    huge = "あ" * int(llm.NUM_CTX * llm._CHARS_PER_TOKEN * 0.9)
    with caplog.at_level(logging.WARNING, logger="oritatami.llm"):
        llm.chat([{"role": "user", "content": huge}])
    assert any("コンテキスト窓" in r.message for r in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="oritatami.llm"):
        llm.chat([{"role": "user", "content": "短い"}])
    assert not [r for r in caplog.records if "コンテキスト窓" in r.message]


# ---- watchdog ---------------------------------------------------------------
# A five-hour continuous run died because one mutations reply was truncated by the
# output cap: no proposal, no successor, chain over, silently. These cover the two
# fixes — truncation is now its own error, and a stalled continuous loop restarts
# from the best result rather than from wherever the walk had drifted to.
def test_truncated_reply_is_its_own_error(monkeypatch):
    import httpx
    from oritatami import llm

    class R:
        status_code = 200

        @staticmethod
        def json():
            return {"choices": [{"finish_reason": "length",
                                 "message": {"content": '{"reply": "切れ'}}]}

    monkeypatch.setattr(httpx, "post", lambda *a, **k: R())
    monkeypatch.setattr(llm, "_ensure_running", lambda *a, **k: None)
    monkeypatch.setattr(llm, "model_info", lambda name: {})
    with pytest.raises(llm.TruncatedError):
        llm.chat([{"role": "user", "content": "hi"}])


def test_output_cap_leaves_room_under_the_window():
    from oritatami import llm
    assert llm.NUM_PREDICT >= 8192, "8192 未満だと深い世代の mutations 応答が切れる"
    assert llm.NUM_PREDICT < llm.NUM_CTX


def _succeed_protein(db, title, result):
    """A finished prediction a chain could actually be resumed from."""
    j = db.insert_job(kind="predict", title=title,
                      spec={"components": [{"type": "protein", "label": "U",
                                            "sequence": UBQ, "chains": ["A"]}]},
                      parent_id=None, origin="autopilot_variant")
    db.update_job(j["id"], status="succeeded", result=result, finished_at=time.time())
    return db.get_job(j["id"])


def test_watchdog_resumes_from_the_best_not_the_latest(db):
    """The walk drifts downhill; a restart should pull it back to the top."""
    older = _succeed_protein(db, "gen3", _result({"A": [93.4]}))
    time.sleep(0.01)
    _succeed_protein(db, "gen51", _result({"A": [91.4]}))  # newest, but worse
    best = autopilot._best_succeeded_predict(db)
    assert best["id"] == older["id"]


def test_watchdog_only_fires_when_everything_is_quiet(db):
    _succeed(db, "done", _result({"A": [90.0]}))
    db.update_job(db.list_jobs()[0]["id"], finished_at=time.time())
    assert not autopilot._loop_is_idle(db), "直後は待つ"
    db.insert_job(kind="predict", title="q", spec={}, parent_id=None, origin="autopilot_variant")
    assert not autopilot._loop_is_idle(db), "待機中のジョブがあれば介入しない"


def test_watchdog_is_scoped_to_continuous_mode():
    """With a finite depth the chain is meant to end; the watchdog must not revive it."""
    config.update_settings({"autopilot_max_depth": 1})
    assert governor.max_depth() != 0
    config.update_settings({"autopilot_max_depth": 0})
    assert governor.max_depth() == 0


def test_analyze_falls_back_to_spec_components(db, monkeypatch):
    """UI-queued jobs have no `workbench` key; the mutations pass must still find chains."""
    seen = {}

    def fake_ask(_db, **kw):
        seen.setdefault('chains', []).append(kw.get('focus_chain'))
        return {'thread_id': 't', 'reply': '', 'proposals': [], 'elapsed_sec': 0.1}

    monkeypatch.setattr(autopilot.assistant, "ask", fake_ask)
    monkeypatch.setattr(autopilot.llm, "ensure_server", lambda **kw: {"running": True})
    monkeypatch.setattr(autopilot, "_chain_scan", lambda *a, **k: None)
    job = {"id": "job_ui", "title": "ユビキチン", "parent_id": None,
           "spec": {"name": "ユビキチン", "components": [
               {"type": "protein", "label": "U", "sequence": UBQ,
                "chains": ["A"], "msa": "server"}]}}
    autopilot._analyze(job, db, type("J", (), {"touch": lambda s: None})())
    assert "A" in (seen.get('chains') or []), "explain だけで終わってはいけない"


def test_watchdog_skips_jobs_it_cannot_continue_from(db):
    """A job with no protein cannot seed a chain, so it must not be picked."""
    ligand_only = db.insert_job(kind="predict", title="リガンドのみ",
                                spec={"components": [{"type": "ligand", "ccd": "ZN"}]},
                                parent_id=None, origin="user")
    db.update_job(ligand_only["id"], status="succeeded",
                  result=_result({"A": [99.0]}), finished_at=time.time())
    usable = db.insert_job(kind="predict", title="タンパク質あり",
                           spec={"components": [{"type": "protein", "sequence": UBQ,
                                                 "chains": ["A"]}]},
                           parent_id=None, origin="user")
    db.update_job(usable["id"], status="succeeded",
                  result=_result({"A": [80.0]}), finished_at=time.time())
    best = autopilot._best_succeeded_predict(db)
    assert best is not None and best["id"] == usable["id"]


def test_watchdog_excludes_candidates_it_already_tried(db):
    a = _succeed_protein(db, "高い", _result({"A": [95.0]}))
    b = _succeed_protein(db, "低い", _result({"A": [90.0]}))
    assert autopilot._best_succeeded_predict(db)["id"] == a["id"]
    assert autopilot._best_succeeded_predict(db, exclude={a["id"]})["id"] == b["id"]


# ---- hill climb -------------------------------------------------------------
# Measured over 51 generations of "continue from the newest": best 93.37 at generation 3,
# 91.43 by generation 51, 26 mutations from wild type. Half the steps beat their own
# parent, but never returning to the peak turned it into a downhill random walk.
def _peak_setup(db, tmp_path, monkeypatch, peak_plddt=95.0, other_plddt=90.0):
    monkeypatch.setattr(autopilot, "jobs_dir", lambda: tmp_path)
    peak = _succeed_protein(db, "頂点", _result({"A": [peak_plddt]}))
    worse = _succeed_protein(db, "外れ", _result({"A": [other_plddt]}))
    return peak, worse


def test_climb_branches_from_the_peak_not_the_latest(db, tmp_path, monkeypatch):
    config.update_settings({"autopilot_strategy": "climb", "autopilot_max_depth": 0,
                            "autopilot_max_variants_per_job": 1,
                            "autopilot_min_disk_gb": 0.0, "autopilot_daily_budget": 0})
    peak, worse = _peak_setup(db, tmp_path, monkeypatch)
    # the peak has a stored analysis with a candidate left to try
    (tmp_path / peak["id"]).mkdir(parents=True, exist_ok=True)
    (tmp_path / peak["id"] / "autopilot.json").write_text(
        __import__("json").dumps({"analyses": [{"proposals": [_proposal(["K48R"], 2.0)]}]}),
        "utf-8")

    submitted = []
    fake = type("J", (), {"submit": lambda s, k, sp, t, **kw: submitted.append((sp, kw)),
                          "touch": lambda s: None})()
    # the job that just finished is the WORSE one — the climb must ignore it
    autopilot._auto_submit_variants(
        worse, {"analyses": [{"proposals": [_proposal(["T9A"], 5.0)]}]}, fake, db)
    assert len(submitted) == 1
    spec, kw = submitted[0]
    assert kw["parent_id"] == peak["id"], "頂点から分岐すること"
    assert spec["autopilot_mutation"] == "K48R", "頂点の候補を使うこと"


def test_climb_does_not_retry_a_mutation_it_already_spent(db, tmp_path, monkeypatch):
    config.update_settings({"autopilot_strategy": "climb", "autopilot_max_depth": 0,
                            "autopilot_max_variants_per_job": 1,
                            "autopilot_min_disk_gb": 0.0, "autopilot_daily_budget": 0})
    peak = _succeed_protein(db, "頂点", _result({"A": [95.0]}))
    child = db.insert_job(kind="predict", title="頂点 + K48R",
                          spec={"autopilot_mutation": "K48R"},
                          parent_id=peak["id"], origin="autopilot_variant")
    db.update_job(child["id"], status="succeeded", result=_result({"A": [90.0]}),
                  finished_at=time.time())
    assert autopilot._tried_mutations(db, peak["id"]) == {"K48R"}

    submitted = []
    fake = type("J", (), {"submit": lambda s, k, sp, t, **kw: submitted.append(sp),
                          "touch": lambda s: None})()
    cands = autopilot._ranked_candidates(
        peak, {"analyses": [{"proposals": [_proposal(["K48R"], 9.0),
                                           _proposal(["T9A"], 1.0)]}]})
    n = autopilot._submit_candidates(fake, db, peak, cands, limit=1)
    assert n == 1
    assert submitted[0]["autopilot_mutation"] == "T9A", "最高スコアでも試行済みなら飛ばす"


def test_walk_still_continues_from_the_newest(db, tmp_path, monkeypatch):
    config.update_settings({"autopilot_strategy": "walk", "autopilot_max_depth": 0,
                            "autopilot_max_variants_per_job": 1,
                            "autopilot_min_disk_gb": 0.0, "autopilot_daily_budget": 0})
    monkeypatch.setattr(autopilot, "jobs_dir", lambda: tmp_path)
    _succeed_protein(db, "頂点", _result({"A": [95.0]}))
    worse = _succeed_protein(db, "最新", _result({"A": [90.0]}))
    submitted = []
    fake = type("J", (), {"submit": lambda s, k, sp, t, **kw: submitted.append(kw),
                          "touch": lambda s: None})()
    autopilot._auto_submit_variants(
        worse, {"analyses": [{"proposals": [_proposal(["T9A"], 5.0)]}]}, fake, db)
    assert submitted and submitted[0]["parent_id"] == worse["id"]
    config.update_settings({"autopilot_strategy": "climb"})


def test_bad_strategy_is_rejected():
    with pytest.raises(ValueError):
        config.update_settings({"autopilot_strategy": "ワープ"})


# ---- climb patience ---------------------------------------------------------
def _spend(db, parent, n):
    """Give `parent` n children that all scored worse than it."""
    for i in range(n):
        c = db.insert_job(kind="predict", title=f"{parent['title']} + M{i}A",
                          spec={"autopilot_mutation": f"M{i}A"},
                          parent_id=parent["id"], origin="autopilot_variant")
        db.update_job(c["id"], status="succeeded", result=_result({"A": [80.0]}),
                      finished_at=time.time())


def test_climb_gives_up_on_a_peak_it_has_spent_its_patience_on(db):
    """Wild-type ubiquitin beat all 72 variants tried against it; the search must move on."""
    config.update_settings({"autopilot_strategy": "climb", "autopilot_climb_patience": 3})
    peak = _succeed_protein(db, "頂点", _result({"A": [95.0]}))
    second = _succeed_protein(db, "次点", _result({"A": [93.0]}))
    assert autopilot._branch_point(db)["id"] == peak["id"]
    _spend(db, peak, 3)
    assert autopilot._branch_point(db)["id"] == second["id"], "我慢の限界で次点に降りること"


def test_climb_returns_to_the_peak_once_every_candidate_is_spent(db):
    config.update_settings({"autopilot_strategy": "climb", "autopilot_climb_patience": 2})
    peak = _succeed_protein(db, "頂点", _result({"A": [95.0]}))
    second = _succeed_protein(db, "次点", _result({"A": [93.0]}))
    _spend(db, peak, 2)
    _spend(db, second, 2)
    # the children themselves are not resumable branch points (no protein in their spec),
    # so with everything spent the loop must fall back rather than stall.
    assert autopilot._branch_point(db)["id"] == peak["id"]


def test_climb_patience_zero_never_gives_up(db):
    config.update_settings({"autopilot_strategy": "climb", "autopilot_climb_patience": 0})
    peak = _succeed_protein(db, "頂点", _result({"A": [95.0]}))
    _succeed_protein(db, "次点", _result({"A": [93.0]}))
    _spend(db, peak, 20)
    assert autopilot._branch_point(db)["id"] == peak["id"]
    config.update_settings({"autopilot_climb_patience": 8})


def test_walk_ignores_patience(db):
    config.update_settings({"autopilot_strategy": "walk", "autopilot_climb_patience": 1})
    peak = _succeed_protein(db, "頂点", _result({"A": [95.0]}))
    _succeed_protein(db, "次点", _result({"A": [93.0]}))
    _spend(db, peak, 5)
    assert autopilot._branch_point(db)["id"] == peak["id"], "酔歩に我慢の概念はない"
    config.update_settings({"autopilot_strategy": "climb", "autopilot_climb_patience": 8})


def test_bad_patience_is_rejected():
    with pytest.raises(ValueError):
        config.update_settings({"autopilot_climb_patience": -1})


# ---- protected residues -----------------------------------------------------
def _job_with_plddt(db, title, per_res, parent=None, spec=None):
    j = db.insert_job(kind="predict", title=title,
                      spec=spec or {"components": [{"type": "protein", "label": "U",
                                                    "sequence": UBQ, "chains": ["A"]}]},
                      parent_id=parent, origin="autopilot_variant")
    db.update_job(j["id"], status="succeeded", finished_at=time.time(),
                  result={"models": [{"plddt": {"A": per_res}, "ligand_plddt": {},
                                      "confidence": {}}]})
    return db.get_job(j["id"])


def test_explicit_protected_residues_are_never_proposed(db):
    """G76 is what makes ubiquitin ubiquitin; 91% of a 491-run search deleted it."""
    config.update_settings({"autopilot_protected_residues": "K63, G75, G76",
                            "autopilot_protect_disordered": False,
                            "autopilot_min_esm_llr": -15.0})
    job = _job_with_plddt(db, "野生型", [95.0] * 76)
    results = {"analyses": [{"proposals": [_proposal(["G76C"], 5.0), _proposal(["K63N"], 4.0),
                                           _proposal(["T9E"], 1.0)]}]}
    kept = autopilot._ranked_candidates(job, results, db)
    assert [c["mut_str"] for c in kept] == ["T9E"]
    # without the mask the search takes the highest-scoring one, which is the lethal one
    config.update_settings({"autopilot_protected_residues": ""})
    assert autopilot._ranked_candidates(job, results, db)[0]["mut_str"] == "G76C"


def test_disordered_residues_are_protected_from_the_lineage_root(db):
    """The tail was 49.5 pLDDT in the wild type — the cheapest confidence in the molecule."""
    config.update_settings({"autopilot_protected_residues": "",
                            "autopilot_protect_disordered": True,
                            "autopilot_disorder_plddt": 70.0})
    root = _job_with_plddt(db, "根", [95.0] * 72 + [68.0, 62.0, 55.0, 49.0])
    # the child already rigidified the tail; the root is what decides, not the child
    child = _job_with_plddt(db, "子", [96.0] * 76, parent=root["id"])
    assert autopilot._protected_positions(db, child, "A") == {73, 74, 75, 76}
    results = {"analyses": [{"proposals": [_proposal(["G75C"], 9.0), _proposal(["T9E"], 0.5)]}]}
    assert [c["mut_str"] for c in autopilot._ranked_candidates(child, results, db)] == ["T9E"]
    config.update_settings({"autopilot_protect_disordered": False})


def test_a_multi_mutation_proposal_is_dropped_if_any_part_is_protected(db):
    config.update_settings({"autopilot_protected_residues": "76",
                            "autopilot_protect_disordered": False})
    job = _job_with_plddt(db, "野生型", [95.0] * 76)
    results = {"analyses": [{"proposals": [_proposal(["T9E", "G76C"], 9.0)]}]}
    assert autopilot._ranked_candidates(job, results, db) == []
    config.update_settings({"autopilot_protected_residues": ""})


def test_bad_disorder_threshold_is_rejected():
    with pytest.raises(ValueError):
        config.update_settings({"autopilot_disorder_plddt": 120.0})


def test_core_plddt_ignores_the_worst_residues():
    """The plain mean is dominated by the flexible tail — the cheapest thing to buy."""
    ordered = [95.0] * 72 + [60.0, 55.0, 50.0, 45.0]
    res = {"models": [{"plddt": {"A": ordered}, "ligand_plddt": {}, "confidence": {}}]}
    assert round(autopilot.mean_plddt(res), 2) == round(sum(ordered) / 76, 2)
    # 76 residues, 10% trimmed = 7 dropped: the four tail residues and three of the 95s
    assert autopilot.core_plddt(res) == 95.0

    rigid_tail = [95.0] * 72 + [90.0, 90.0, 90.0, 90.0]
    res2 = {"models": [{"plddt": {"A": rigid_tail}, "ligand_plddt": {}, "confidence": {}}]}
    gain_mean = autopilot.mean_plddt(res2) - autopilot.mean_plddt(res)
    gain_core = autopilot.core_plddt(res2) - autopilot.core_plddt(res)
    assert gain_mean > 1.5, "平均は末尾を固めるだけで大きく上がる"
    assert gain_core == 0.0, "コアは末尾を固めても動かない"


def test_core_plddt_is_selectable_as_the_improvement_metric():
    config.update_settings({"autopilot_improvement_metric": "core_plddt"})
    res = {"models": [{"plddt": {"A": [95.0] * 72 + [50.0] * 4}, "ligand_plddt": {},
                       "confidence": {}}]}
    assert autopilot.resolve_metric(res) == "core_plddt"
    assert autopilot.metric_value(res, "core_plddt") == 95.0
    config.update_settings({"autopilot_improvement_metric": "auto"})


def test_an_experiment_tag_keeps_two_searches_apart(db):
    """Run 2 started and immediately branched from run 1's champion, and from a
    hand-run one-off that happened to score well. A search has to stay in its own run."""
    config.update_settings({"autopilot_experiment": "R2", "autopilot_strategy": "climb",
                            "autopilot_climb_patience": 0})
    old = db.insert_job(kind="predict", title="旧チャンピオン",
                        spec={"autopilot_experiment": "R1",
                              "components": [{"type": "protein", "label": "U",
                                              "sequence": UBQ, "chains": ["A"]}]},
                        parent_id=None, origin="autopilot_variant")
    db.update_job(old["id"], status="succeeded", finished_at=time.time(),
                  result=_result({"A": [99.0]}))
    stray = _succeed_protein(db, "手で回した実験", _result({"A": [98.0]}))
    mine = db.insert_job(kind="predict", title="R2 種",
                         spec={"autopilot_experiment": "R2",
                               "components": [{"type": "protein", "label": "U",
                                               "sequence": UBQ, "chains": ["A"]}]},
                         parent_id=None, origin="user")
    db.update_job(mine["id"], status="succeeded", finished_at=time.time(),
                  result=_result({"A": [90.0]}))
    assert stray["id"] and old["id"]
    assert autopilot._branch_point(db)["id"] == mine["id"], "点が低くても自分の実験から伸ばす"

    config.update_settings({"autopilot_experiment": ""})
    assert autopilot._branch_point(db)["id"] == old["id"], "タグなしなら従来どおり全体の最良"
    config.update_settings({"autopilot_climb_patience": 8})


# ---- cost of the explain pass ----------------------------------------------
def test_explain_can_be_switched_off(db, tmp_path, monkeypatch):
    """It is 27% of a generation's wall clock and nothing reads its prose."""
    calls = []

    def fake_ask(_db, **kw):
        calls.append(kw["mode"])
        # one usable proposal, so the analysis submits and the cycle ends there
        return {"thread_id": None, "reply": "",
                "proposals": [_proposal([f"T9{'AE'[len(calls) % 2]}"], 1.0)],
                "reply_issues": [], "elapsed_sec": 0.1}

    monkeypatch.setattr(autopilot, "jobs_dir", lambda: tmp_path)
    monkeypatch.setattr(autopilot.llm, "ensure_server", lambda **kw: {"running": True})
    monkeypatch.setattr(autopilot.assistant, "ask", fake_ask)
    monkeypatch.setattr(autopilot, "_chain_scan", lambda *a, **k: None)
    config.update_settings({"autopilot_protected_residues": "", "autopilot_protect_disordered": False,
                            "autopilot_min_disk_gb": 0.0, "autopilot_daily_budget": 0,
                            "autopilot_max_variants_per_job": 1})
    job = _succeed_protein(db, "元", _result({"A": [90.0]}))
    (tmp_path / job["id"]).mkdir(parents=True, exist_ok=True)
    fake = type("J", (), {"submit": lambda *a, **k: None, "touch": lambda s: None})()

    config.update_settings({"autopilot_explain": True})
    autopilot._analyze(job, db, fake)
    assert calls == ["explain", "mutations"]

    calls.clear()
    (tmp_path / job["id"] / "autopilot.json").unlink(missing_ok=True)
    config.update_settings({"autopilot_explain": False})
    autopilot._analyze(job, db, fake)
    assert calls == ["mutations"], "explain だけ止まり、変異提案は残ること"
    config.update_settings({"autopilot_explain": True})


def test_proposals_per_call_is_bounded():
    config.update_settings({"autopilot_proposals_per_call": 6})
    assert governor.proposals_per_call() == 6
    with pytest.raises(ValueError):
        config.update_settings({"autopilot_proposals_per_call": 99})
    config.update_settings({"autopilot_proposals_per_call": 3})


def test_an_unsatisfiable_mask_does_not_loop_forever(db, tmp_path, monkeypatch):
    """Every proposal blocked used to mean: re-analyse the same job, forever, at full GPU."""
    analysed = []

    def fake_ask(_db, **kw):
        analysed.append(kw["mode"])
        return {"thread_id": None, "reply": "", "proposals": [_proposal(["G76C"], 5.0)],
                "reply_issues": [], "elapsed_sec": 0.1}

    monkeypatch.setattr(autopilot, "jobs_dir", lambda: tmp_path)
    monkeypatch.setattr(autopilot.llm, "ensure_server", lambda **kw: {"running": True})
    monkeypatch.setattr(autopilot.assistant, "ask", fake_ask)
    monkeypatch.setattr(autopilot, "_chain_scan", lambda *a, **k: None)
    config.update_settings({"autopilot_protected_residues": "G76", "autopilot_explain": False,
                            "autopilot_protect_disordered": False, "autopilot_strategy": "climb",
                            "autopilot_min_disk_gb": 0.0, "autopilot_daily_budget": 0})
    job = _succeed_protein(db, "元", _result({"A": [90.0]}))
    fake = type("J", (), {"submit": lambda *a, **k: None, "touch": lambda s: None})()

    autopilot._analyze(job, db, fake)
    assert len(analysed) <= autopilot._MAX_REANALYSIS_PER_CYCLE + 1, \
        f"解析し直しが止まらない ({len(analysed)} 回)"
    config.update_settings({"autopilot_protected_residues": "", "autopilot_explain": True})


def test_transient_engine_failures_are_retried(db, monkeypatch):
    """The one failure in an 857-prediction run was a ColabFold timeout, and auto-retry
    never fired because only "network" counted as transient."""
    from oritatami import jobs as jobs_mod

    assert "msa" in jobs_mod.RETRYABLE_ERROR_KINDS
    assert "download" in jobs_mod.RETRYABLE_ERROR_KINDS
    # a genuinely hopeless failure must not be retried forever
    for kind in ("oom", "input", "error"):
        assert kind not in jobs_mod.RETRYABLE_ERROR_KINDS, kind


def test_msa_timeout_is_classified_as_msa():
    from oritatami.engines.boltz import classify_failure

    log = ("Running MSA generation\nToo many failed attempts for the MSA server\n"
           "Exception: MMseqs2 API is giving errors\n")
    assert classify_failure(log, 1, "msa") == "msa"
    assert classify_failure("out of memory", 1, "structure") == "oom"
    assert classify_failure("Invalid SMILES", 1, "preprocess") == "input"


def test_forbidden_residues_are_marked_in_the_table_the_model_reads():
    """A sentence saying "do not touch 63, 76" was ignored by 28% of 901 proposals —
    G76C and K63N were the two most-proposed mutations of all. The ban has to appear
    where the model picks positions from."""
    from oritatami.assistant import numbered_residues

    plain = numbered_residues(UBQ)
    assert "[" not in plain and "禁止" not in plain

    marked = numbered_residues(UBQ, fixed={63, 76})
    assert "変更禁止" in marked.splitlines()[0], "凡例を先頭に出すこと"
    # K63 and G76 bracketed, their neighbours not
    body = "\n".join(marked.splitlines()[1:])
    assert "[K]" in body and "[G]" in body
    assert body.count("[") == 2, "指定した2残基だけを囲むこと"
    # the table still spells out every residue in order
    assert "".join(c for c in body if c.isupper()) == UBQ


def test_the_experiment_tag_is_stamped_on_every_variant(tmp_path, monkeypatch):
    """It was only ever read, never written: a variant carried the tag solely because the
    spec was deep-copied from its parent. Rename the experiment mid-run and every child
    kept the old tag, so _ranked_jobs filtered them all out and the climb quietly stalled."""
    from oritatami import autopilot, config

    config.update_settings({"autopilot_experiment": "R3"})
    spec = {"name": "ユビキチン", "workbench_name": "ユビキチン",
            "autopilot_experiment": "R2", "autopilot_depth": 4,
            "components": [{"type": "protein", "label": "U", "chains": ["A"],
                            "sequence": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"}]}
    results = {"analyses": [{"mode": "mutations", "chain": "A", "proposals": [
        {"type": "mutation_set", "status": "ok", "chain": "A",
         "apply": {"action": "mutate", "chain": "A", "mutations": ["A:T7A"]}}]}]}
    cands = autopilot._ranked_candidates({"spec": spec, "id": "j1"}, results)
    assert cands, "候補が作られること"
    assert cands[0]["spec"]["autopilot_experiment"] == "R3", "設定の実験名で上書きすること"
    assert cands[0]["spec"]["autopilot_depth"] == 5

    config.update_settings({"autopilot_experiment": ""})
    cands = autopilot._ranked_candidates({"spec": spec, "id": "j1"}, results)
    assert cands
    assert "autopilot_experiment" not in cands[0]["spec"], "実験名が空なら札を外すこと"


def test_the_suite_never_waits_on_a_live_server(db, tmp_path, monkeypatch):
    """_analyze calls ensure_server, which used to be a daemon wait. A test that forgot
    the patch would spawn a real llama-server (or block on a pull); pin the skip path."""
    monkeypatch.setattr(autopilot, "jobs_dir", lambda: tmp_path)
    monkeypatch.setattr(autopilot.llm, "ensure_server",
                        lambda **kw: {"running": False, "error": "なし"})
    called = []
    monkeypatch.setattr(autopilot.assistant, "ask", lambda *a, **k: called.append(1))

    job = _succeed_protein(db, "元", _result({"A": [90.0]}))
    started = time.time()
    autopilot._analyze(job, db, type("J", (), {"touch": lambda s: None})())
    assert time.time() - started < 5.0, "LLM 不在でブロックしないこと"
    assert called == [], "LLM が無いなら問い合わせないこと"
    assert not (tmp_path / job["id"] / "autopilot.json").exists(), "解析結果を書かないこと"


def test_every_analyze_call_in_this_file_patches_ensure_server():
    """A new test that forgets the patch reintroduces a real server spawn silently."""
    import pathlib
    import re

    src = pathlib.Path(__file__).read_text("utf-8")
    bodies = re.split(r"\ndef (test_\w+)", src)
    for name, body in zip(bodies[1::2], bodies[2::2], strict=False):
        if "_analyze(" not in body or name == "test_every_analyze_call_in_this_file_patches_ensure_server":
            continue
        assert 'ensure_server' in body, \
            f"{name} が llm.ensure_server を差し替えていない — 実サーバーが起きる"


def test_interface_residues_are_protected_in_a_complex(db, monkeypatch):
    """Boltz computes the contact list on every prediction and only the prose pass read
    it. In a complex those residues are the one part whose job is known."""
    config.update_settings({"autopilot_protect_interfaces": True,
                            "autopilot_protect_disordered": False,
                            "autopilot_protected_residues": ""})
    job = {"id": "j1", "spec": {}, "result": {
        "models": [{"plddt": {"A": [90.0] * 20, "B": [90.0] * 20}}],
        "interfaces": {"cutoff": 5.0, "interfaces": [
            {"chains": ["A", "B"], "residues": {"A": ["L8", "I44", "V70"], "B": ["R42"]}}]},
    }}
    assert autopilot._protected_positions(db, job, "A") == {8, 44, 70}
    assert autopilot._protected_positions(db, job, "B") == {42}

    config.update_settings({"autopilot_protect_interfaces": False})
    assert autopilot._protected_positions(db, job, "A") == set()
    config.update_settings({"autopilot_protect_interfaces": True})


def test_a_single_chain_prediction_has_no_interface_to_protect(db):
    config.update_settings({"autopilot_protect_interfaces": True,
                            "autopilot_protect_disordered": False,
                            "autopilot_protected_residues": ""})
    job = {"id": "j1", "spec": {}, "result": {"models": [{"plddt": {"A": [90.0] * 20}}],
                                              "interfaces": {"interfaces": []}}}
    assert autopilot._protected_positions(db, job, "A") == set()
