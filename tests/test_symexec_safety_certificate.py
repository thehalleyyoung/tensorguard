"""Tests for the proof-carrying **safety certificate** (certified absence of
forced failures) — even_more.md "quantum leap": find → certify.

Pins:
  1. the certificate is *built faithfully* from a result (verdict, obligations,
     coverage, abstain boundary);
  2. it is *replayable* — re-derivable from source, tamper-evident on both the
     source and the fingerprint;
  3. it is *honest* — every COMPLETE_FOR kind has an obligation backed by a Lean
     refutation theorem that actually appears in the audited soundness set, and a
     buggy program is never certified safe;
  4. it *round-trips* through JSON.
"""

from __future__ import annotations

import ast

import pytest

from src.symexec import (
    LEAN_REFUTATION_FOR,
    certify_safety,
    dumps_safety_certificate,
    loads_safety_certificate,
    render_safety_certificate,
    safety_certificate_from_dict,
    safety_certificate_to_dict,
    verify_safety_certificate,
)
from src.symexec.completeness_contract import COMPLETE_FOR
from src.symexec.engine import analyze_source

# Import the audited-theorem list straight from the Lean regression test so the
# registry can never silently cite a theorem that is not actually audited.
from tests.test_lean_soundness import _AUDITED_THEOREMS

SAFE = """import torch, torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    def forward(self, x):
        return self.fc(x)
"""

BUGGY = """import torch
if __name__ == "__main__":
    a = torch.randn(2, 3)
    b = torch.randn(4, 5)
    c = a @ b
"""


def _cert(source, filename="<unknown>"):
    return certify_safety(analyze_source(source), source, filename=filename)


# --------------------------------------------------------------------------- #
# 1. Faithful construction.                                                     #
# --------------------------------------------------------------------------- #
def test_safe_program_is_certified():
    cert = _cert(SAFE, "m.py")
    assert cert.proven_safe
    assert cert.sound_bug_count == 0
    assert cert.all_obligations_discharged
    assert cert.filename == "m.py"
    assert len(cert.fingerprint) == 64


def test_obligations_cover_every_complete_for_kind():
    cert = _cert(SAFE)
    complete_kinds = {c.kind for c in COMPLETE_FOR}
    obl_kinds = {o.kind for o in cert.obligations}
    assert obl_kinds == complete_kinds
    # Each obligation carries the contract's precondition.
    by_kind = {c.kind: c for c in COMPLETE_FOR}
    for o in cert.obligations:
        assert o.predicate == by_kind[o.kind].predicate
        assert o.witness_condition == by_kind[o.kind].condition


def test_buggy_program_is_not_certified():
    cert = _cert(BUGGY)
    assert not cert.proven_safe
    assert cert.sound_bug_count >= 1
    # The matmul obligation is *not* discharged (it was reported).
    matmul = next(o for o in cert.obligations if o.kind == "matmul_dim_mismatch")
    assert matmul.reported_count >= 1
    assert not matmul.discharged


# --------------------------------------------------------------------------- #
# 2. Replayability & tamper-evidence.                                           #
# --------------------------------------------------------------------------- #
def test_certificate_verifies_against_its_source():
    cert = _cert(SAFE)
    v = verify_safety_certificate(cert, SAFE)
    assert v.verified, v.reasons()
    assert all(ok for _, ok, _ in v.checks)


def test_verification_rejects_tampered_source():
    cert = _cert(SAFE)
    v = verify_safety_certificate(cert, SAFE + "\n# sneaky edit\n")
    assert not v.verified
    assert any("source" in r for r in v.reasons())


def test_verification_rejects_tampered_fingerprint():
    cert = _cert(SAFE)
    import dataclasses

    forged = dataclasses.replace(cert, fingerprint="0" * 64)
    v = verify_safety_certificate(forged, SAFE)
    assert not v.verified
    assert any("fingerprint" in r for r in v.reasons())


def test_verification_rejects_forged_safe_verdict():
    """A certificate that *claims* a buggy program is safe must not verify."""
    import dataclasses

    cert = _cert(BUGGY)
    forged = dataclasses.replace(cert, proven_safe=True)
    v = verify_safety_certificate(forged, BUGGY)
    assert not v.verified


# --------------------------------------------------------------------------- #
# 3. Honesty: the Lean trust base is real and in-sync.                          #
# --------------------------------------------------------------------------- #
def test_every_complete_for_kind_has_a_lean_refutation():
    for clause in COMPLETE_FOR:
        assert clause.kind in LEAN_REFUTATION_FOR, (
            f"COMPLETE_FOR kind {clause.kind} has no Lean refutation in the "
            f"safety-certificate registry"
        )


def test_cited_lean_theorems_are_actually_audited():
    """Every theorem the registry cites must appear in the Lean soundness audit
    (so a safety certificate can never name an unproven/unaudited theorem)."""
    audited = set(_AUDITED_THEOREMS)
    for kind, thm in LEAN_REFUTATION_FOR.items():
        assert thm in audited, f"{kind} cites non-audited Lean theorem {thm!r}"


def test_obligations_record_their_lean_refutation():
    cert = _cert(SAFE)
    for o in cert.obligations:
        assert o.lean_refutation == LEAN_REFUTATION_FOR.get(o.kind)
        assert o.lean_refutation is not None


def test_trusted_axioms_are_recorded():
    cert = _cert(SAFE)
    assert set(cert.trusted_axioms) == {"propext", "Classical.choice", "Quot.sound"}


# --------------------------------------------------------------------------- #
# 4. Serialization round-trip + rendering.                                      #
# --------------------------------------------------------------------------- #
def test_json_round_trip_preserves_verification():
    cert = _cert(SAFE, "m.py")
    text = dumps_safety_certificate(cert)
    back = loads_safety_certificate(text)
    assert safety_certificate_to_dict(back) == safety_certificate_to_dict(cert)
    assert verify_safety_certificate(back, SAFE).verified


def test_dict_round_trip_is_stable():
    cert = _cert(SAFE)
    d = safety_certificate_to_dict(cert)
    again = safety_certificate_to_dict(safety_certificate_from_dict(d))
    assert d == again


def test_render_is_deterministic_markdown():
    cert = _cert(SAFE, "m.py")
    a = render_safety_certificate(cert)
    b = render_safety_certificate(cert)
    assert a == b
    assert a.startswith("# Safety certificate for `m.py`")
    assert "Certified" in a
    assert "machine-checked" in a


def test_symresult_safety_certificate_method():
    r = analyze_source(SAFE)
    cert = r.safety_certificate(SAFE, filename="m.py")
    assert cert.proven_safe
    assert verify_safety_certificate(cert, SAFE).verified


def test_render_for_buggy_program_is_not_certified():
    cert = _cert(BUGGY)
    out = render_safety_certificate(cert)
    assert "Not certified" in out


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
