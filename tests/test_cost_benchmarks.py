"""Step 53 -- import / analysis cost benchmarks.

Proves the load-bearing dependency claim (verifying a model is torch-free) and
the import-time ceiling, plus manifest byte-reproducibility.
"""
import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from evaluation import cost_benchmarks as cb  # noqa: E402


def test_manifest_is_deterministic():
    a = cb._dumps(cb.manifest())
    b = cb._dumps(cb.manifest())
    assert a == b
    man = cb.manifest()
    assert man["invariants"]["import_is_torch_free"] is True
    assert man["invariants"]["analysis_is_torch_free"] is True


def test_committed_manifest_up_to_date():
    # The committed JSON/MD must match a freshly rendered manifest.
    assert cb.run(check=True) == 0


def test_probe_reports_torch_free_import_and_analysis():
    data = cb._probe()
    assert data["torch_after_import"] is False
    assert data["torch_after_analysis"] is False
    assert data["import_s"] >= 0.0
    assert data["small_analysis_s"] >= 0.0
    assert data["medium_analysis_s"] >= 0.0


def test_import_is_actually_torch_free_in_fresh_process():
    # Independent of the harness: a clean interpreter that imports the analysis
    # API must not have torch loaded afterwards.
    script = (
        "import sys; sys.path.insert(0, %r);"
        "from src.model_checker import verify_model;"
        "print('torch' in sys.modules)" % ROOT
    )
    proc = subprocess.run([sys.executable, "-c", script],
                          capture_output=True, text=True, timeout=120)
    assert proc.stdout.strip().splitlines()[-1] == "False"


def test_gate_passes_live():
    # The live gate (real timings) should pass on the dev machine: import and
    # analysis are well under their generous ceilings and torch-free.
    assert cb.gate() == 0


def test_ceilings_are_sane():
    man = cb.manifest()
    c = man["ceilings_s"]
    assert c["import"] <= c["small_analysis"] <= c["medium_analysis"]
