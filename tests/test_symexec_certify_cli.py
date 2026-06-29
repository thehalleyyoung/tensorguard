"""Tests for the safety-certifier *product surface*: file-level certification and
the ``python -m src.symexec.certify`` CLI (even_more.md quantum leap)."""

from __future__ import annotations

import io

import pytest

from src.symexec import certify_file, verify_certificate_file, verify_safety_certificate
from src.symexec.certify import main

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


@pytest.fixture()
def safe_file(tmp_path):
    p = tmp_path / "safe.py"
    p.write_text(SAFE)
    return p


@pytest.fixture()
def buggy_file(tmp_path):
    p = tmp_path / "buggy.py"
    p.write_text(BUGGY)
    return p


# --------------------------------------------------------------------------- #
# File-level helpers.                                                           #
# --------------------------------------------------------------------------- #
def test_certify_file_certifies_safe_file(safe_file):
    cert = certify_file(str(safe_file))
    assert cert.proven_safe
    assert cert.filename == str(safe_file)
    assert verify_certificate_file(cert, str(safe_file)).verified


def test_certify_file_rejects_buggy_file(buggy_file):
    cert = certify_file(str(buggy_file))
    assert not cert.proven_safe
    assert cert.sound_bug_count >= 1


def test_verify_certificate_file_detects_edit(safe_file):
    cert = certify_file(str(safe_file))
    safe_file.write_text(SAFE + "\n# edited\n")
    assert not verify_certificate_file(cert, str(safe_file)).verified


# --------------------------------------------------------------------------- #
# CLI: check.                                                                   #
# --------------------------------------------------------------------------- #
def test_cli_check_safe_exits_zero(safe_file):
    out = io.StringIO()
    rc = main(["check", str(safe_file)], out=out)
    assert rc == 0
    assert "CERTIFIED" in out.getvalue()
    assert "All files certified safe." in out.getvalue()


def test_cli_check_buggy_exits_nonzero(safe_file, buggy_file):
    out = io.StringIO()
    rc = main(["check", str(safe_file), str(buggy_file)], out=out)
    assert rc == 1
    text = out.getvalue()
    assert "NOT CERTIFIED" in text
    assert "Not all files could be certified." in text


# --------------------------------------------------------------------------- #
# CLI: emit + verify round-trip.                                                #
# --------------------------------------------------------------------------- #
def test_cli_emit_then_verify(tmp_path, safe_file):
    cert_path = tmp_path / "safe.cert"
    out = io.StringIO()
    rc = main(["emit", str(safe_file), "-o", str(cert_path)], out=out)
    assert rc == 0
    assert cert_path.exists()

    out2 = io.StringIO()
    rc2 = main(["verify", str(safe_file), str(cert_path)], out=out2)
    assert rc2 == 0
    assert "VERIFIED" in out2.getvalue()


def test_cli_emit_buggy_exits_nonzero(buggy_file):
    out = io.StringIO()
    rc = main(["emit", str(buggy_file)], out=out)
    assert rc == 1


def test_cli_verify_detects_tamper(tmp_path, safe_file):
    cert_path = tmp_path / "safe.cert"
    main(["emit", str(safe_file), "-o", str(cert_path)], out=io.StringIO())
    safe_file.write_text(SAFE + "\n# tampered\n")
    out = io.StringIO()
    rc = main(["verify", str(safe_file), str(cert_path)], out=out)
    assert rc == 1
    assert "NOT VERIFIED" in out.getvalue()


def test_cli_emit_to_stdout_is_loadable(safe_file):
    from src.symexec.safety_certificate import loads_safety_certificate

    out = io.StringIO()
    main(["emit", str(safe_file)], out=out)
    # The stdout content (minus a possible trailing newline) is valid JSON cert.
    cert = loads_safety_certificate(out.getvalue())
    assert verify_safety_certificate(cert, SAFE).verified


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
