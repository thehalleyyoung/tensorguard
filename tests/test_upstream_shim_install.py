"""Step 176 — proposed-upstream ``torch.nn.utils.verify_module`` shim.

``install()`` grafts the *exact* Phase-1 API surface from
``docs/upstream/pytorch_proposal.md`` onto the real ``torch`` namespace, so a
user can call the proposed upstream helpers today with **no core PyTorch
changes**.  We prove against real torch that, once installed:

* ``torch.nn.utils.verify_module(model, ...)`` returns SAFE on a clean module
  and UNSAFE on a buggy one;
* ``torch.nn.utils.attach_verifier`` raises ``torch.nn.ShapeVerificationError``
  on the first forward of a buggy module, *before* the crashing kernel;
* ``@torch.nn.verifiable(...)`` verifies at construction-time;
* ``install`` is idempotent and refuses to clobber a pre-existing core name
  without ``force``; ``uninstall`` restores the namespace.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.upstream_hook import ShapeVerificationError, install, uninstall


class CleanNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(8, 16)
        self.b = nn.Linear(16, 4)

    def forward(self, x):
        return self.b(self.a(x)).relu()


class BuggyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(8, 16)
        self.b = nn.Linear(24, 4)  # expects 24, gets 16 -> real shape bug

    def forward(self, x):
        return self.b(self.a(x))


@pytest.fixture()
def installed():
    names = install()
    try:
        yield names
    finally:
        uninstall()


def test_install_grafts_proposed_names(installed):
    assert "torch.nn.utils.verify_module" in installed
    assert "torch.nn.verifiable" in installed
    assert hasattr(torch.nn.utils, "verify_module")
    assert hasattr(torch.nn, "ShapeVerificationError")


def test_verify_module_on_real_torch_namespace(installed):
    clean = torch.nn.utils.verify_module(CleanNet(), input_shapes={"x": (2, 8)})
    buggy = torch.nn.utils.verify_module(BuggyNet(), input_shapes={"x": (2, 8)})
    assert clean.verdict == "SAFE"
    assert buggy.verdict == "UNSAFE"


def test_attach_verifier_raises_before_kernel(installed):
    model = BuggyNet()
    torch.nn.utils.attach_verifier(model, input_shapes={"x": (2, 8)})
    with pytest.raises(torch.nn.ShapeVerificationError):
        model(torch.randn(2, 8))


def test_verifiable_decorator_via_torch_namespace(installed):
    @torch.nn.verifiable(input_shapes={"x": (2, 8)})
    class Decorated(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(8, 16)
            self.b = nn.Linear(24, 4)  # bug

        def forward(self, x):
            return self.b(self.a(x))

    with pytest.raises(ShapeVerificationError):
        Decorated()(torch.randn(2, 8))


def test_install_is_idempotent_and_reversible():
    first = install()
    second = install()  # no error: marked names are re-installable
    assert set(first) == set(second)
    removed = uninstall()
    assert "torch.nn.utils.verify_module" in removed
    assert not hasattr(torch.nn.utils, "verify_module")
    assert not hasattr(torch.nn, "verifiable")


def test_install_refuses_to_clobber_core_name():
    uninstall()
    sentinel = object()
    torch.nn.utils.verify_module = sentinel  # pretend core already defines it
    try:
        with pytest.raises(RuntimeError):
            install()
        # force=True overrides
        install(force=True)
        assert torch.nn.utils.verify_module is not sentinel
    finally:
        uninstall()
        if hasattr(torch.nn.utils, "verify_module"):
            del torch.nn.utils.verify_module
