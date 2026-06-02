"""Tests for the training-loop hazard analyzer (Step 96, Phase 10)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.training_loop_checks import (  # noqa: E402
    analyze_training_loop, summarize, HazardKind, Confidence,
)
import reproducibility.training_loop_hazards as tlh  # noqa: E402

VOLATILE_TOKENS = ("time", "elapsed", "timestamp", "wall", "clock", "_ms",
                   "seconds", "duration", "date")


def _walk_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _walk_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_keys(v)


CLEAN = (
    "def train(m, loader, opt):\n"
    "    for x, y in loader:\n"
    "        opt.zero_grad()\n"
    "        loss = ((m(x) - y) ** 2).mean()\n"
    "        loss.backward()\n"
    "        opt.step()\n"
)


def test_clean_loop_has_no_hazard():
    assert analyze_training_loop(CLEAN) == []


def test_detach_flags_gradient_flow_break():
    src = CLEAN.replace("m(x)", "m(x).detach()")
    kinds = {h.kind for h in analyze_training_loop(src)}
    assert HazardKind.GRADIENT_FLOW_BREAK in kinds


def test_no_grad_block_flags_gradient_flow_break():
    src = (
        "def train(m, loader, opt):\n"
        "    for x, y in loader:\n"
        "        opt.zero_grad()\n"
        "        with torch.no_grad():\n"
        "            loss = ((m(x) - y) ** 2).mean()\n"
        "        loss.backward()\n"
        "        opt.step()\n"
    )
    kinds = {h.kind for h in analyze_training_loop(src)}
    assert HazardKind.GRADIENT_FLOW_BREAK in kinds


def test_missing_zero_grad():
    src = CLEAN.replace("        opt.zero_grad()\n", "")
    kinds = {h.kind for h in analyze_training_loop(src)}
    assert HazardKind.MISSING_ZERO_GRAD in kinds


def test_missing_optimizer_step():
    src = CLEAN.replace("        opt.step()\n", "")
    kinds = {h.kind for h in analyze_training_loop(src)}
    assert HazardKind.MISSING_OPTIMIZER_STEP in kinds


def test_backward_before_zero_grad():
    src = (
        "def train(m, loader, opt):\n"
        "    for x, y in loader:\n"
        "        loss = ((m(x) - y) ** 2).mean()\n"
        "        loss.backward()\n"
        "        opt.zero_grad()\n"
        "        opt.step()\n"
    )
    kinds = {h.kind for h in analyze_training_loop(src)}
    assert HazardKind.BACKWARD_BEFORE_ZERO_GRAD in kinds


def test_scheduler_step_is_not_optimizer_step():
    """scheduler.step() must not satisfy the optimizer.step() requirement."""
    src = (
        "def train(m, loader, opt, scheduler):\n"
        "    for x, y in loader:\n"
        "        opt.zero_grad()\n"
        "        loss = ((m(x) - y) ** 2).mean()\n"
        "        loss.backward()\n"
        "        scheduler.step()\n"
    )
    kinds = {h.kind for h in analyze_training_loop(src)}
    assert HazardKind.MISSING_OPTIMIZER_STEP in kinds


def test_amp_fp16_without_scaler_is_heuristic():
    src = (
        "def train(m, loader, opt):\n"
        "    for x, y in loader:\n"
        "        opt.zero_grad()\n"
        "        with torch.autocast('cuda', dtype=torch.float16):\n"
        "            loss = ((m(x) - y) ** 2).mean()\n"
        "        loss.backward()\n"
        "        opt.step()\n"
    )
    hs = analyze_training_loop(src)
    amp = [h for h in hs if h.kind is HazardKind.AMP_MISSING_GRAD_SCALER]
    assert amp and amp[0].confidence is Confidence.HEURISTIC


def test_amp_with_scaler_idiom_is_clean():
    src = (
        "def train(m, loader, opt, scaler):\n"
        "    for x, y in loader:\n"
        "        opt.zero_grad()\n"
        "        with torch.autocast('cuda', dtype=torch.float16):\n"
        "            loss = ((m(x) - y) ** 2).mean()\n"
        "        scaler.scale(loss).backward()\n"
        "        scaler.step(opt)\n"
        "        scaler.update()\n"
    )
    kinds = {h.kind for h in analyze_training_loop(src)}
    assert HazardKind.AMP_MISSING_GRAD_SCALER not in kinds


def test_summarize_counts():
    s = summarize(CLEAN.replace("m(x)", "m(x).detach()").replace(
        "        opt.step()\n", ""))
    assert s["n_hazards"] >= 2
    assert s["sound"] >= 2


# --- reproducibility harness: static verdict vs real torch -----------------
def test_static_matches_runtime_on_all_cases():
    data = tlh.measure()
    assert data["all_ok"] is True
    for c in data["cases"]:
        assert c["static_ok"], f"{c['name']} static mismatch"
        assert c["live_ok"], f"{c['name']} runtime mismatch"


def test_artifact_is_byte_deterministic():
    assert tlh.run(check=True) == 0


def test_artifact_has_no_volatile_fields():
    data = json.loads(tlh.OUT_JSON.read_text())
    for key in _walk_keys(data):
        low = key.lower()
        for tok in VOLATILE_TOKENS:
            assert tok not in low, f"volatile key token {tok!r} in {key!r}"
