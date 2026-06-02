"""Step 60 — the watch engine (`tensorguard verify --watch`).

The watch loop's two interesting pieces — change detection and a single
verification pass — are pure functions, so these tests drive them with stub
mtime/verify callbacks (no real filesystem watcher, no infinite loop) and also
run one real verification pass against torch to prove the wiring.
"""

import torch  # noqa: F401  (ensures the torch-backed verify path is importable)

from src.api import verify_architecture
from src.watch_mode import (
    WatchResult,
    format_watch_result,
    poll_once,
    run_pass,
    run_verification,
    snapshot_mtimes,
)


def test_poll_once_reports_only_increased_mtimes():
    times = {"a.py": 1.0, "b.py": 1.0}
    prev = {"a.py": 1.0, "b.py": 1.0}
    changed, new = poll_once(
        ["a.py", "b.py"], prev, mtime_fn=lambda p: times[p]
    )
    assert changed == []  # nothing moved
    # bump a.py
    times["a.py"] = 2.0
    changed, new = poll_once(["a.py", "b.py"], new, mtime_fn=lambda p: times[p])
    assert changed == ["a.py"]
    assert new["a.py"] == 2.0


def test_poll_once_first_sighting_not_reported():
    times = {"a.py": 5.0}
    # prev is empty: first sighting records but does NOT report a change.
    changed, new = poll_once(["a.py"], {}, mtime_fn=lambda p: times[p])
    assert changed == []
    assert new["a.py"] == 5.0


def test_poll_once_ignores_missing_files():
    def mt(p):
        raise OSError("gone")

    changed, new = poll_once(["x.py"], {"x.py": 1.0}, mtime_fn=mt)
    assert changed == []
    assert new == {"x.py": 1.0}


def test_snapshot_mtimes_skips_unreadable():
    def mt(p):
        if p == "bad":
            raise OSError
        return 3.0

    snap = snapshot_mtimes(["good", "bad"], mtime_fn=mt)
    assert snap == {"good": 3.0}


def test_run_verification_captures_exceptions():
    def boom(_p):
        raise SyntaxError("invalid syntax")

    wr = run_verification("m.py", boom)
    assert isinstance(wr, WatchResult)
    assert wr.ok is False
    assert wr.error is not None and "invalid syntax" in wr.error


def test_run_verification_counts_error_bugs():
    class FakeBug:
        def __init__(self, sev):
            self.severity = sev

    class FakeResult:
        bugs = [FakeBug("error"), FakeBug("warning"), FakeBug("error")]
        duration_ms = 12.5

    wr = run_verification("m.py", lambda _p: FakeResult())
    assert wr.bug_count == 2  # warnings are not errors
    assert wr.ok is False
    assert wr.duration_ms == 12.5


def test_format_watch_result_variants():
    safe = WatchResult(path="a/m.py", ok=True, bug_count=0, duration_ms=9.0)
    bad = WatchResult(path="a/m.py", ok=False, bug_count=1, duration_ms=9.0)
    err = WatchResult(path="a/m.py", ok=False, error="bad import")
    assert "verified safe" in format_watch_result(safe)
    assert "1 issue" in format_watch_result(bad)
    assert "could not verify" in format_watch_result(err)
    # color path wraps with ANSI
    assert "\033[" in format_watch_result(safe, use_color=True)
    # basename only
    assert "m.py" in format_watch_result(safe) and "a/" not in format_watch_result(safe)


def test_run_pass_against_real_torch_models(tmp_path):
    safe_src = (
        "import torch.nn as nn\n"
        "class Net(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.fc = nn.Linear(4, 2)\n"
        "    def forward(self, x):\n"
        "        return self.fc(x)\n"
    )
    bug_src = safe_src.replace("nn.Linear(4, 2)", "nn.Linear(7, 2)")
    safe_f = tmp_path / "safe.py"
    bug_f = tmp_path / "bug.py"
    safe_f.write_text(safe_src)
    bug_f.write_text(bug_src)

    def verify_fn(path):
        src = open(path).read()
        return verify_architecture(src, input_shapes={"x": ("batch", 4)})

    results = run_pass([str(safe_f), str(bug_f)], verify_fn)
    by_name = {r.path.split("/")[-1]: r for r in results}
    assert by_name["safe.py"].ok is True
    assert by_name["bug.py"].ok is False
    assert by_name["bug.py"].bug_count >= 1
