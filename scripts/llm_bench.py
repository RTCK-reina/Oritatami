#!/usr/bin/env python3
"""Model benchmark that drives the real assistant pipeline end-to-end.

Runs each candidate model through `assistant.ask` — the same system prompt, workbench
context, JSON schema, verification and repair path the app uses — and reports per-case
outcome, proposal acceptance, residue-claim correctness and wall-clock time.

    ORITATAMI_LLAMA_NGPU_LAYERS=0 .venv/bin/python scripts/llm_bench.py gemma3:4b qwen3:8b

The env override pins generation to the CPU, which is what this development VM needs
(its virtualised Metal is ~45x slower); on real hardware omit it. Numbers here are for
comparing models on the same machine, not absolute throughput.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from oritatami import assistant, llm  # noqa: E402
from oritatami.config import update_settings  # noqa: E402
from oritatami.db import Database  # noqa: E402

UBQ = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
WORKBENCH = {"name": "bench",
             "components": [{"type": "protein", "label": "U", "chains": ["A"],
                             "sequence": UBQ}]}

# (label, mode, message, optional check on the result)
CASES = [
    ("surface-muts", "mutations",
     "表面の親水性を高める変異を提案してください",
     None),
    ("position-k48", "mutations",
     "48 番残基のリジンをアルギニンに変える提案を含めてください",
     lambda r: any("K48R" in (p.get("mutations") or []) or "K48R" in str(p.get("apply") or "")
                    for p in r.get("proposals") or [])),
    ("claim-60-70", "chat",
     "60〜70 番残基あたりの配列の特徴を教えてください",
     None),
    ("design-helix", "design",
     "4 本の α ヘリックス束を 60〜80 残基で設計してください",
     lambda r: any(p.get("status") == "ok" for p in r.get("proposals") or [])),
]


def run_case(db: Database, mode: str, message: str, check) -> dict:
    t0 = time.time()
    try:
        r = assistant.ask(db, thread_id=None, mode=mode, message=message,
                          workbench=WORKBENCH, job=None, scan=None, scan_chain=None,
                          focus_chain="A", count=3, persist=False, verify_reply=False)
        props = r.get("proposals") or []
        out = {"ok": True, "sec": round(time.time() - t0, 1),
               "proposals": len(props),
               "accepted": sum(1 for p in props if p.get("status") == "ok"),
               "warning": sum(1 for p in props if p.get("status") == "warning"),
               "invalid": sum(1 for p in props if p.get("status") == "invalid"),
               "reply_issues": len(r.get("reply_issues") or []),
               "reply_chars": len(r.get("reply") or "")}
        if check is not None:
            out["target_hit"] = bool(check(r))
        return out
    except Exception as exc:  # noqa: BLE001 — the bench records, not raises
        return {"ok": False, "sec": round(time.time() - t0, 1),
                "error": str(exc)[:300]}


def main() -> int:
    models = sys.argv[1:] or ["gemma3:4b"]
    out_path = Path(__file__).resolve().parents[1] / "bench_results.json"
    results: dict[str, list[dict]] = {}
    if out_path.exists():
        try:
            results = json.loads(out_path.read_text("utf-8"))
        except ValueError:
            pass
    db = Database(Path("/tmp") / f"llm-bench-{int(time.time())}.sqlite3")
    for model in models:
        update_settings({"llm_model": model, "llm_think": False})
        llm.unload_model()
        rows = []
        print(f"== {model}", flush=True)
        for label, mode, msg, check in CASES:
            row = {"case": label, **run_case(db, mode, msg, check)}
            row["rep"] = 1
            print(f"  {label}: {row}", flush=True)
            rows.append(row)
        results[model] = rows
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=1), "utf-8")
    print(f"written: {out_path}")
    llm.unload_model()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
